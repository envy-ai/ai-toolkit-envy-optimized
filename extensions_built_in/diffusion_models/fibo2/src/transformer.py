# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Training implementation of the Apache FIBO2 equations, with Comfy checkpoint names.

Reference: muhammadali-afridi/diffusers@ab96e2b58c89adb9df418af74d81cc8b1e1be074,
src/diffusers/models/transformers/transformer_bria_fibo2.py. No Comfy runtime/code
is imported. Canonical inputs are normalized FLUX.2 packed BCHW latents. All
residuals are out of place, including checkpoint recomputation.
"""
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from toolkit.models.v2._mixin import OstrisModelMixin


@contextmanager
def adapter_strengths(states):
    """Restore the captured per-example adapter strengths during checkpoint replay."""
    previous = []
    try:
        for network, scalar, tensor in states:
            previous.append((network, network._multiplier, network.torch_multiplier))
            network._multiplier, network.torch_multiplier = scalar, tensor
        yield
    finally:
        for network, scalar, tensor in reversed(previous):
            network._multiplier, network.torch_multiplier = scalar, tensor


@dataclass
class Fibo2Params:
    in_channels: int = 128
    dim: int = 3584
    n_heads: int = 28
    n_layers: int = 44
    n_refiner_layers: int = 2
    injection_layers: tuple = (6, 14, 22, 30, 38)
    cap_feat_dim: int = 7680
    perceiver_layers: int = 2
    min_gist_tokens: int = 256
    max_gist_tokens: int = 512
    gist_step: int = 64
    gist_min_text_len: int = 256
    gist_max_text_len: int = 2048
    ffn_dim: int = 9728
    norm_eps: float = 1e-5
    axes_dims: tuple = (32, 48, 48)
    rope_theta: float = 256.0


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).to(x.dtype) * self.weight


def rotary(x, phases):
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * phases[None, :, None]).flatten(-2).to(x.dtype)


def attention(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                        v.transpose(1, 2)).transpose(1, 2).flatten(2)


class JointAttention(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.heads = p.n_heads
        self.qkv = nn.Linear(p.dim, 3 * p.dim, bias=False)
        self.out = nn.Linear(p.dim, p.dim, bias=False)
        self.q_norm = RMSNorm(p.dim // p.n_heads, p.norm_eps)
        self.k_norm = RMSNorm(p.dim // p.n_heads, p.norm_eps)

    def forward(self, x, phases):
        q, k, v = self.qkv(x).unflatten(-1, (3, self.heads, -1)).unbind(2)
        return self.out(attention(rotary(self.q_norm(q), phases),
                                  rotary(self.k_norm(k), phases), v))


class Attention(nn.Module):
    def __init__(self, p, context_width):
        super().__init__()
        self.heads = p.n_heads
        self.to_q = nn.Linear(p.dim, p.dim, bias=False)
        self.to_kv = nn.Linear(context_width, 2 * p.dim, bias=False)
        self.to_out = nn.Linear(p.dim, p.dim, bias=False)
        self.norm_q = RMSNorm(p.dim // p.n_heads, p.norm_eps)
        self.norm_k = RMSNorm(p.dim // p.n_heads, p.norm_eps)

    def forward(self, x, context):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k, v = self.to_kv(context).unflatten(-1, (2, self.heads, -1)).unbind(2)
        return self.to_out(attention(q, self.norm_k(k), v))


class FeedForward(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.w13 = nn.Linear(p.dim, 2 * p.ffn_dim, bias=False)
        self.w2 = nn.Linear(p.ffn_dim, p.dim, bias=False)

    def forward(self, x):
        gate, value = self.w13(x).chunk(2, -1)
        return self.w2(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self, p, inject=False):
        super().__init__()
        self.attention = JointAttention(p)
        self.feed_forward = FeedForward(p)
        for name in ('attention_norm1', 'attention_norm2', 'ffn_norm1', 'ffn_norm2'):
            setattr(self, name, RMSNorm(p.dim, p.norm_eps))
        self.adaLN_modulation = nn.Sequential(nn.Linear(min(p.dim, 256), 4 * p.dim))
        self.cross_attention = Attention(p, p.cap_feat_dim) if inject else None
        if inject:
            self.cross_attn_norm1 = RMSNorm(p.dim, p.norm_eps)
            self.cross_attn_norm2 = RMSNorm(p.dim, p.norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, phases, noisy, clean, start, end, context=None):
        # Only target-image tokens have nonzero time; gist/reference tokens are clean.
        mn = self.adaLN_modulation(noisy).unsqueeze(1)
        mc = self.adaLN_modulation(clean).unsqueeze(1)
        mods = torch.cat((mc.expand(-1, start, -1), mn.expand(-1, end-start, -1),
                          mc.expand(-1, x.shape[1]-end, -1)), dim=1)
        sa, ga, sf, gf = mods.chunk(4, -1)
        x = x + ga.tanh() * self.attention_norm2(self.attention(self.attention_norm1(x) * (1 + sa), phases))
        if self.cross_attention is not None:
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(
                self.cross_attention(self.cross_attn_norm1(x), context))
        return x + gf.tanh() * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * (1 + sf)))


class PerceiverLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.cross_attn_norm = RMSNorm(p.dim, p.norm_eps)
        self.cross_attention = Attention(p, p.cap_feat_dim)
        self.self_attn_norm = RMSNorm(p.dim, p.norm_eps)
        self.self_attention = Attention(p, p.dim)
        self.ffn_norm = RMSNorm(p.dim, p.norm_eps)
        self.ffn = FeedForward(p)

    def forward(self, x, context):
        x = x + self.cross_attention(self.cross_attn_norm(x), context)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y)
        return x + self.ffn(self.ffn_norm(x))


class Perceiver(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(p.max_gist_tokens, p.dim) * .02)
        self.layers = nn.ModuleList([PerceiverLayer(p) for _ in range(p.perceiver_layers)])
        self.norm_out = RMSNorm(p.dim, p.norm_eps)


class TimestepEmbedder(nn.Module):
    def __init__(self, output_width=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, 1024), nn.SiLU(), nn.Linear(1024, output_width))

    def forward(self, t, dtype):
        freqs = torch.exp(-math.log(10000) * torch.arange(128, device=t.device, dtype=torch.float32) / 128)
        args = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat((args.cos(), args.sin()), -1).to(dtype))


class FinalLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.norm_final = nn.LayerNorm(p.dim, eps=1e-6, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(p.dim, 256), p.dim))
        self.linear = nn.Linear(p.dim, p.in_channels)

    def forward(self, x, time):
        return self.linear(self.norm_final(x) * (1 + self.adaLN_modulation(time)[:, None]))


def convert_checkpoint(state_dict):
    """Strict structural conversion of float native split or canonical Comfy weights.

    Packed quantized linears are already canonical and are never concatenated.
    Missing pairs fail immediately; final loader validates all remaining tensors.
    """
    out = dict(state_dict)
    for key in list(out):
        if key.endswith('.comfy_attention.config'):
            config = json.loads(bytes(out.pop(key).tolist()))
            if config != {'attention': 'comfy_kitchen_int8'}:
                raise ValueError(f'Unsupported FIBO attention metadata: {key}')
    native = any(key.endswith(('.attention.to_q.weight', '.to_k.weight', '.w1.weight')) for key in out)
    if native and any(key.endswith('.comfy_quant') for key in out):
        raise ValueError('Quantized split FIBO checkpoints must be repacked before loading')
    for key in list(out):
        if key.endswith('.attention.to_q.weight'):
            root = key[:-len('to_q.weight')]
            q, k, v = [out.pop(root + name + '.weight') for name in ('to_q', 'to_k', 'to_v')]
            out[root + 'qkv.weight'] = torch.cat((q, k, v), 0)
            out[root + 'out.weight'] = out.pop(root + 'to_out.0.weight')
            out[root + 'q_norm.weight'] = out.pop(root + 'norm_q.weight')
            out[root + 'k_norm.weight'] = out.pop(root + 'norm_k.weight')
    for key in list(out):
        if key.endswith('.to_k.weight'):
            root = key[:-len('to_k.weight')]
            out[root + 'to_kv.weight'] = torch.cat((out.pop(key), out.pop(root + 'to_v.weight')), 0)
        if key.endswith('.to_out.0.weight'):
            out[key.replace('.to_out.0.weight', '.to_out.weight')] = out.pop(key)
        if key.endswith('.w1.weight'):
            root = key[:-len('w1.weight')]
            w1, w3, w2 = out.pop(key), out.pop(root + 'w3.weight'), out.pop(root + 'w2.weight')
            pad = 9728 - 9536 if w1.shape == (9536, 3584) else 0
            out[root + 'w13.weight'] = torch.cat((F.pad(w1, (0, 0, 0, pad)), F.pad(w3, (0, 0, 0, pad))), 0)
            out[root + 'w2.weight'] = F.pad(w2, (0, pad))
    return out


class Fibo2Transformer(nn.Module, OstrisModelMixin):
    aitk_subfolder = 'transformer'
    aitk_cast_quantized_load = True

    def __init__(self, params=None):
        super().__init__()
        p = self.params = params or Fibo2Params()
        if p.dim % p.n_heads or sum(p.axes_dims) != p.dim // p.n_heads or any(d % 2 for d in p.axes_dims):
            raise ValueError('FIBO head width must equal the sum of even RoPE axis widths')
        if any(i < 0 or i >= p.n_layers for i in p.injection_layers):
            raise ValueError('FIBO injection block is outside the transformer')
        self.x_embedder = nn.Linear(p.in_channels, p.dim)
        self.t_embedder = TimestepEmbedder(min(p.dim, 256))
        self.noise_refiner = nn.ModuleList([Block(p) for _ in range(p.n_refiner_layers)])
        self.perceiver = Perceiver(p)
        self.layers = nn.ModuleList([Block(p, i in p.injection_layers) for i in range(p.n_layers)])
        self.final_layer = FinalLayer(p)
        self.gradient_checkpointing = False

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        return convert_checkpoint(state_dict)

    @classmethod
    def load_from_state_dict(cls, state_dict, dtype, **kwargs):
        # Validate execution metadata before the shared loader removes it.
        return super().load_from_state_dict(convert_checkpoint(state_dict), dtype, **kwargs)

    @classmethod
    def aitk_config_from_state_dict(cls, state_dict):
        dim, channels = state_dict['x_embedder.weight'].shape
        heads = dim // state_dict['layers.0.attention.q_norm.weight'].numel()
        count = lambda prefix, suffix: sum(key.startswith(prefix) and key.endswith(suffix) for key in state_dict)
        injections = tuple(sorted(int(key.split('.')[1]) for key in state_dict if key.startswith('layers.') and key.endswith('.cross_attn_gate')))
        head = dim // heads
        axes = (32, 48, 48) if head == 128 else (0, head // 2, head - head // 2)
        queries = state_dict['perceiver.latent_queries'].shape[0]
        return Fibo2Params(in_channels=channels, dim=dim, n_heads=heads,
            n_layers=count('layers.', '.attention_norm1.weight'),
            n_refiner_layers=count('noise_refiner.', '.attention_norm1.weight'),
            injection_layers=injections, cap_feat_dim=state_dict['perceiver.layers.0.cross_attention.to_kv.weight'].shape[1],
            perceiver_layers=count('perceiver.layers.', '.cross_attn_norm.weight'),
            ffn_dim=state_dict['layers.0.feed_forward.w2.weight'].shape[1], max_gist_tokens=queries,
            min_gist_tokens=min(256, queries), gist_step=min(64, queries), axes_dims=axes)

    @classmethod
    def aitk_from_pretrained(cls, source, subfolder=None, dtype=None, **kwargs):
        from safetensors.torch import load_file
        if not os.path.isdir(source):
            from huggingface_hub import snapshot_download
            source = snapshot_download(source, allow_patterns=[f'{subfolder or "transformer"}/*.safetensors'])
        folder = os.path.join(source, subfolder or '')
        if not os.path.isdir(folder):
            folder = source
        state = {}
        for name in sorted(os.listdir(folder)):
            if name.endswith('.safetensors'):
                for key, value in load_file(os.path.join(folder, name)).items():
                    if key in state:
                        raise ValueError(f'Duplicate FIBO checkpoint tensor: {key}')
                    state[key] = value
        if not state:
            raise FileNotFoundError(f'No FIBO transformer safetensors in {folder}')
        return cls.load_from_state_dict(state, dtype)

    @classmethod
    def get_transformer_block_names(cls):
        return ['noise_refiner', 'perceiver.layers', 'layers']

    @classmethod
    def get_quantization_exclude_modules(cls):
        return ['x_embedder', 't_embedder*', 'final_layer*', '*.adaLN_modulation*']

    @property
    def device(self):
        return self.x_embedder.weight.device

    @property
    def dtype(self):
        return self.x_embedder.weight.dtype

    def enable_gradient_checkpointing(self, enable=True):
        self.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _call(self, module, *args):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            states = tuple(getattr(self, '_adapter_replay_states', ()))
            return checkpoint(module, *args, use_reentrant=False,
                context_fn=lambda: (adapter_strengths(states), adapter_strengths(states)))
        return module(*args)

    def _phases(self, ids):
        angles = []
        for axis, width in enumerate(self.params.axes_dims):
            if width:
                freq = 1 / (self.params.rope_theta ** (torch.arange(0, width, 2, dtype=torch.float64, device=ids.device) / width))
                angles.append((ids[:, axis:axis+1].double() * freq).float())
        return torch.polar(torch.ones_like(torch.cat(angles, -1)), torch.cat(angles, -1))

    def _one(self, x, time, bundles, refs):
        p = self.params
        _, channels, height, width = x.shape
        length = bundles.shape[2]
        ratio = min(1., max(0., (length-p.gist_min_text_len) / (p.gist_max_text_len-p.gist_min_text_len)))
        gist_count = min(p.max_gist_tokens, max(p.min_gist_tokens,
            math.ceil((p.min_gist_tokens + ratio*(p.max_gist_tokens-p.min_gist_tokens)) / p.gist_step)*p.gist_step))
        tokens, positions = [], []
        plane = p.max_gist_tokens + 1
        for index, latent in enumerate([x] + refs):
            tokens.append(latent.flatten(2).transpose(1, 2))
            h, w = latent.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
            positions.append(torch.stack((torch.full_like(yy, plane+index), yy, xx), -1).flatten(0, 1))
        phases = self._phases(torch.cat(positions))
        img = self.x_embedder(torch.cat(tokens, 1))
        noisy = self.t_embedder(time.float() * 1000, x.dtype)
        clean = self.t_embedder(torch.zeros_like(time), x.dtype)
        target_end = height * width
        for block in self.noise_refiner:
            img = self._call(block, img, phases, noisy, clean, 0, target_end)
        gist = self.perceiver.latent_queries[:gist_count][None].expand(1, -1, -1)
        for layer in self.perceiver.layers:
            gist = self._call(layer, gist, bundles[:, 0])
        gist = self.perceiver.norm_out(gist)
        hidden = torch.cat((gist, img), 1)
        gist_ids = torch.zeros(gist_count, 3, device=x.device)
        gist_ids[:, 0] = torch.arange(1, gist_count+1, device=x.device)
        phases = self._phases(torch.cat((gist_ids, torch.cat(positions)), 0))
        by_layer = dict(zip(p.injection_layers, bundles.unbind(1)[1:]))
        for index, block in enumerate(self.layers):
            hidden = self._call(block, hidden, phases, noisy, clean, gist_count, gist_count+target_end, by_layer.get(index))
        return self.final_layer(hidden[:, gist_count:gist_count+target_end], noisy).transpose(1, 2).reshape(1, channels, height, width)

    def forward(self, x, timestep, context, mask=None, ref_latents=None):
        """Packed BCHW and sigma [0,1]; six bundles B,6,L,D (or B,L,6D)."""
        if context.ndim == 3:
            context = context.unflatten(-1, (1+len(self.params.injection_layers), self.params.cap_feat_dim)).permute(0, 2, 1, 3)
        if context.shape[1] != 1 + len(self.params.injection_layers):
            raise ValueError('FIBO conditioning bundle count does not match injection blocks')
        times = timestep.reshape(-1).expand(x.shape[0])
        outputs = []
        # Each example keeps its real token/gist count, without padded attention.
        for i in range(x.shape[0]):
            embeds = context[i:i+1]
            if mask is not None:
                embeds = embeds[:, :, mask[i].bool()]
            refs = [] if ref_latents is None else ref_latents[i]
            refs = [ref[None] if ref.ndim == 3 else ref for ref in refs]
            refs = [ref.to(x.device, x.dtype) for ref in refs]
            outputs.append(self._one(x[i:i+1], times[i:i+1], embeds, refs))
        return torch.cat(outputs)
