"""Audio-free MiniMax H3 Image DiT on the shared H3 attention and AdaLN blocks.

The original image transformer has one output head. Comfy repacks may instead
carry six PDD heads in ``final_layer.video_out``: head zero is complete and the
remaining five are deltas from it. A separate official fused-head file carries
six complete heads. Training always uses the original head when available, or
head zero of a Comfy repack; inference uses the six published PDD heads.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...minimax_h3.src.transformer import (
    MODALITY_NUM,
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)


class MiniMaxH3ImageTransformer(MiniMaxH3Transformer):
    @classmethod
    def aitk_config_from_state_dict(cls, state_dict):
        if "audio_patch_proj.weight" in state_dict or "final_layer.audio_out.weight" in state_dict:
            raise ValueError("MiniMax H3 Image needs an audio-free image checkpoint")
        params = super().aitk_config_from_state_dict(state_dict)
        output = state_dict["final_layer.video_out.weight"].shape[0]
        expected = params.latents_dim * 4
        if output not in (expected, expected * 6):
            raise ValueError(f"MiniMax H3 Image output has {output} rows; expected {expected} or {expected * 6}")
        params.pdd_steps = output // expected
        return params

    def __init__(self, params: MiniMaxH3TransformerParams | None = None):
        if params is None:
            params = MiniMaxH3TransformerParams()
        super().__init__(params)
        del self.audio_patch_proj
        del self.final_layer.audio_out
        self.pdd_steps = getattr(params, "pdd_steps", 1)
        if self.pdd_steps not in (1, 6):
            raise ValueError("MiniMax H3 Image supports a base head or six PDD heads")
        if self.pdd_steps == 6:
            self.final_layer.video_out = nn.Linear(
                params.hidden_size, params.latents_dim * 4 * 6, bias=True
            )
        # A separate fused-head bank is needed for inference from original
        # single-head weights. It is not a trainable parameter or checkpoint key.
        self.register_buffer("pdd_fused_weight", None, persistent=False)
        self.register_buffer("pdd_fused_bias", None, persistent=False)

    def attach_pdd_fused_heads(self, weight: torch.Tensor, bias: torch.Tensor):
        size = self.params.latents_dim * 4
        expected_weight = (6, size, self.params.hidden_size)
        if tuple(weight.shape) != expected_weight or tuple(bias.shape) != (6, size):
            raise ValueError("MiniMax H3 Image PDD fused heads have incompatible dimensions")
        self.pdd_fused_weight = weight.float().contiguous()
        self.pdd_fused_bias = bias.float().contiguous()

    def _head(self, h: torch.Tensor, pdd_step: int | None):
        head = self.final_layer.video_out
        rows = self.params.latents_dim * 4
        if pdd_step is None:
            # The published base transformer has one head. For Comfy's packed
            # PDD checkpoint, head zero is used as the training hypothesis.
            weight = head.weight[:rows]
            bias = head.bias[:rows]
        elif self.pdd_fused_weight is not None:
            if not 0 <= pdd_step < 6:
                raise ValueError("MiniMax H3 Image PDD step must be in 0..5")
            weight = self.pdd_fused_weight[pdd_step]
            bias = self.pdd_fused_bias[pdd_step]
        elif self.pdd_steps == 6:
            if not 0 <= pdd_step < 6:
                raise ValueError("MiniMax H3 Image PDD step must be in 0..5")
            weight = head.weight[:rows]
            bias = head.bias[:rows]
            if pdd_step:
                start = pdd_step * rows
                weight = weight + head.weight[start : start + rows]
                bias = bias + head.bias[start : start + rows]
        else:
            raise ValueError("MiniMax H3 Image needs the six published PDD heads for sampling")
        # MemoryManager may keep the frozen output head on CPU while the block
        # activations are on CUDA. This functional projection bypasses the
        # module's paging hook, so stage only the selected FP32 head here.
        weight = weight.to(device=h.device, non_blocking=True)
        bias = bias.to(device=h.device, non_blocking=True)
        return F.linear(h.to(weight.dtype), weight, bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        row_timesteps: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        text_indices: torch.Tensor,
        pdd_step: int | None = None,
    ) -> torch.Tensor:
        """Predict clean-minus-noise rows for the image and optional image refs."""
        if hidden_states.shape[0] != 1:
            raise ValueError("MiniMax H3 Image currently requires batch size 1")
        if bool((token_tags == 2).any()):
            raise ValueError("MiniMax H3 Image cannot pack audio rows")
        rotary_emb = self.rope(position_ids)
        video_embeds = self.video_patch_proj(hidden_states.to(self.video_patch_proj.weight.dtype))
        text_embeds = self.condition_proj(encoder_hidden_states.to(self.dtype))
        is_pad = token_tags < 0
        attn_mask = text_attn_mask = None
        if bool(is_pad.any()):
            live = ~is_pad
            attn_mask = live[:, None, None, :]
            text_attn_mask = live[:, text_indices][:, None, None, :]
        text_embeds = self.token_refiner(text_embeds, text_attn_mask)
        x = text_embeds.new_zeros((token_tags.shape[0], token_tags.shape[1], text_embeds.shape[-1]))
        x = x.index_copy(1, text_indices, text_embeds)
        x = x.index_copy(1, video_indices, video_embeds.to(x.dtype))

        unique_t, inverse = torch.unique(row_timesteps.float(), sorted=True, return_inverse=True)
        temb = self._time_embedding(unique_t)
        adaln_indices = inverse * MODALITY_NUM + token_tags.clamp(min=0)
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(block, x, temb, adaln_indices, rotary_emb, attn_mask, None, use_reentrant=False)
            else:
                x = block(x, temb, adaln_indices, rotary_emb, attn_mask, None)

        shift, scale = self.final_layer.adaln_proj(temb)
        # The released image head performs modulation in fp32 after the
        # block-stack RMSNorm; the projection itself is also an fp32 island.
        h = self.final_layer.norm(x).float()
        h = h * (1.0 + scale[inverse].float()) + shift[inverse].float()
        return self._head(h.index_select(1, video_indices), pdd_step)
