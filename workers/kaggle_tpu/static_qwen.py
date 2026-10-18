"""Fixed-layout, no-reference Qwen 2.1 execution for the one-rank TPU gate.

Only sequence assembly and device-specific attention/RoPE are adapted. All
projections, modulation, normalization and blocks remain the toolkit model.
Dense SDPA is a correctness gate, not a memory-efficient training claim.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class StaticAttention:
    def __call__(self, attn, hidden_states, rotary_emb=None, **kwargs):
        if kwargs.get("layer_cache") is not None or kwargs.get("kv_cache_mode") is not None:
            raise ValueError("TPU smoke does not support inference KV caches")
        q, k, v = [projection(hidden_states).unflatten(-1, (attn.heads, -1))
                   for projection in (attn.to_q, attn.to_k, attn.to_v)]
        q, k = attn.norm_q(q).to(v.dtype), attn.norm_k(k).to(v.dtype)
        cos, sin = rotary_emb

        def rotate(x):
            real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
            return torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1).flatten(3).to(x.dtype)

        q, k, v = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2), v.transpose(1, 2)
        prefix = kwargs["segments"][-1][1]
        valid = kwargs.get("key_valid")
        # Text is causal; every target query sees all valid prefix+target keys.
        text_mask = torch.ones(prefix, prefix, device=q.device, dtype=torch.bool).tril()[None, None]
        if valid is not None:
            text_mask = text_mask & valid[:, None, None, :prefix]
        text = F.scaled_dot_product_attention(q[:, :, :prefix], k[:, :, :prefix], v[:, :, :prefix],
                                             attn_mask=text_mask, dropout_p=0.0)
        target = F.scaled_dot_product_attention(q[:, :, prefix:], k, v,
                                               attn_mask=None if valid is None else valid[:, None, None, :],
                                               dropout_p=0.0)
        out = torch.cat((text, target), 2).transpose(1, 2).flatten(2, 3).to(q.dtype)
        return attn.to_out[1](attn.to_out[0](out))


class StaticQwen(torch.nn.Module):
    """One paired cache layout, prepared entirely on CPU before XLA placement."""

    def __init__(self, transformer, prompt_mask, height, width, *, checkpoint_blocks=True):
        super().__init__()
        if not transformer.config.causal_condition or prompt_mask.ndim != 2 or prompt_mask.shape[0] != 1:
            raise ValueError("TPU smoke requires causal Qwen 2.1 and per-device batch one")
        if height % 2 or width % 2 or not bool(prompt_mask.any()):
            raise ValueError("Qwen target latent dimensions must be even and prompt must have a valid token")
        self.transformer = transformer
        self.height, self.width = height, width
        self.prefix = prompt_mask.shape[1]
        self.checkpoint_blocks = checkpoint_blocks
        image_mask = torch.cat((torch.zeros(self.prefix, dtype=torch.bool),
                                torch.ones(height * width, dtype=torch.bool)))
        rope = transformer.pos_embed([(1, height, width)], image_mask, device=torch.device("cpu"))
        self.register_buffer("rope_cos", rope.real[None, :, None, :].contiguous())
        self.register_buffer("rope_sin", rope.imag[None, :, None, :].contiguous())
        self.register_buffer("target_mask", image_mask)
        self.register_buffer("key_valid", torch.cat((prompt_mask.bool(), torch.ones(1, height * width, dtype=torch.bool)), 1))
        for block in transformer.transformer_blocks:
            block.attn.set_processor(StaticAttention())

    def forward(self, latents, timestep, prompt):
        model = self.transformer
        image = model.img_in(latents.flatten(2).transpose(1, 2))
        text = model.txt_in(prompt)
        hidden = torch.cat((text, image), 1)
        t = torch.cat((timestep.to(hidden.dtype), timestep.new_zeros(1).to(hidden.dtype)))
        temb = model.time_text_embed(t, hidden)
        modulation = model.modulation(temb)
        rotary = (self.rope_cos, self.rope_sin)
        segments = [(0, self.prefix, True)]
        for block in model.transformer_blocks:
            arguments = (hidden, modulation, rotary, None, self.target_mask, None, None, None, segments, self.key_valid)
            if self.checkpoint_blocks and torch.is_grad_enabled():
                if hidden.device.type == "xla":
                    # XLA 2.8 requires its own RNG handling and optimization
                    # barrier, and supports only the reentrant implementation.
                    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
                    hidden = xla_checkpoint(block.forward, *arguments, use_reentrant=True, preserve_rng_state=True)
                else:
                    hidden = checkpoint(block, *arguments, use_reentrant=True, preserve_rng_state=True)
            else:
                hidden = block(*arguments)
        output = model.proj_out(model.norm_out(hidden, temb, self.target_mask))[:, self.prefix:]
        return output.transpose(1, 2).reshape(latents.shape[0], -1, self.height, self.width)
