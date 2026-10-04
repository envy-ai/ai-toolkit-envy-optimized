"""LoHa/DoHa adapters with bounded dense-weight temporaries.

Uses the standard LyCORIS ``hada_*`` checkpoint layout without upgrading the
legacy LyCORIS dependency used by LoCon. Only inputs and low-rank factors are
retained for backward; dense Hadamard products are recomputed a row chunk at
a time. The frozen (possibly quantized/offloaded) base forward is unchanged.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from optimum.quanto import QTensor

from toolkit.network_mixins import ToolkitModuleMixin, broadcast_and_multiply


class _ChunkedLoHa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, c, d, chunk_size, conv):
        ctx.save_for_backward(x, a, b, c, d)
        ctx.chunk_size, ctx.conv = chunk_size, conv
        outputs = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, a.shape[0], chunk_size):
                end = start + chunk_size
                weight = (a[start:end] @ b) * (c[start:end] @ d)
                if conv is None:
                    outputs.append(F.linear(x, weight))
                else:
                    kernel, stride, padding, dilation = conv
                    outputs.append(F.conv2d(x, weight.reshape(-1, *kernel),
                                            stride=stride, padding=padding, dilation=dilation))
        return torch.cat(outputs, dim=-1 if conv is None else 1)

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b, c, d = ctx.saved_tensors
        # Accumulate cross-chunk reductions in fp32 for bf16/fp16 training.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        ga, gb, gc, gd = [torch.zeros_like(p, dtype=dtype) for p in (a, b, c, d)]
        gx = torch.zeros_like(x, dtype=dtype) if ctx.needs_input_grad[0] else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            bf, df = b.to(dtype), d.to(dtype)
            xf = x.to(dtype)
            for start in range(0, a.shape[0], ctx.chunk_size):
                end = min(start + ctx.chunk_size, a.shape[0])
                af, cf = a[start:end].to(dtype), c[start:end].to(dtype)
                u, v = af @ bf, cf @ df
                if ctx.conv is None:
                    go = grad_output[..., start:end].to(dtype).reshape(-1, end - start)
                    gw = go.T @ xf.reshape(-1, x.shape[-1])
                    if gx is not None:
                        gx.add_((go @ (u * v)).reshape_as(x))
                else:
                    kernel, stride, padding, dilation = ctx.conv
                    go = grad_output[:, start:end].to(dtype).contiguous()
                    shape = (end - start, *kernel)
                    gw = torch.nn.grad.conv2d_weight(xf, shape, go, stride, padding, dilation).flatten(1)
                    if gx is not None:
                        gx.add_(torch.nn.grad.conv2d_input(x.shape, (u * v).reshape(shape),
                                                        go, stride, padding, dilation))
                gu, gv = gw * v, gw * u
                ga[start:end] = gu @ bf.T
                gb.add_(af.T @ gu)
                gc[start:end] = gv @ df.T
                gd.add_(cf.T @ gv)
        return (None if gx is None else gx.to(x.dtype), ga.to(a.dtype), gb.to(b.dtype),
                gc.to(c.dtype), gd.to(d.dtype), None, None)


class LoHaModule(ToolkitModuleMixin, nn.Module):
    def __init__(self, lora_name, org_module, multiplier=1.0, lora_dim=4, alpha=1,
                 dropout=None, rank_dropout=None, module_dropout=None, network=None,
                 use_bias=False, use_dora=False, loha_chunk_size=128, **kwargs):
        ToolkitModuleMixin.__init__(self, network=network)
        nn.Module.__init__(self)
        if use_bias:
            raise ValueError("LoHa does not support training adapter biases")
        self.lora_name, self.lora_dim = lora_name, int(lora_dim)
        if self.lora_dim <= 0:
            raise ValueError("LoHa rank must be positive")
        self.org_module = [org_module]  # do not register/train the base weights
        self.multiplier = multiplier
        self.dropout, self.rank_dropout, self.module_dropout = dropout, rank_dropout, module_dropout
        self.is_checkpointing, self.can_merge_in = False, False
        self.use_dora = bool(use_dora)
        self.chunk_size = int(loha_chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("loha_chunk_size must be positive")
        self.conv = None
        if hasattr(org_module, "kernel_size"):
            if org_module.groups != 1 or org_module.padding_mode != "zeros":
                raise ValueError("LoHa supports ungrouped, zero-padded Conv2d layers only")
            out_dim = org_module.out_channels
            kernel = (org_module.in_channels, *org_module.kernel_size)
            in_dim = math.prod(kernel)
            self.conv = (kernel, org_module.stride, org_module.padding, org_module.dilation)
        else:
            out_dim, in_dim = org_module.out_features, org_module.in_features
        self.hada_w1_a = nn.Parameter(torch.empty(out_dim, self.lora_dim))
        self.hada_w1_b = nn.Parameter(torch.empty(self.lora_dim, in_dim))
        self.hada_w2_a = nn.Parameter(torch.empty(out_dim, self.lora_dim))
        self.hada_w2_b = nn.Parameter(torch.empty(self.lora_dim, in_dim))
        for parameter in (self.hada_w1_a, self.hada_w1_b, self.hada_w2_a):
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        nn.init.zeros_(self.hada_w2_b)
        if isinstance(alpha, torch.Tensor):
            alpha = float(alpha.detach().float().item())
        alpha = self.lora_dim if alpha is None or alpha == 0 else float(alpha)
        self.register_buffer("alpha", torch.tensor(alpha))
        self._set_runtime_scale(alpha / self.lora_dim)
        if self.use_dora:
            self.magnitude = nn.Parameter(self._weight_norms(adapted=False))

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    @torch.no_grad()
    def _weight_norms(self, adapted):
        weight = self.org_module[0].weight
        if hasattr(weight, "dequantize"):
            weight = weight.dequantize()
        weight = weight.detach().flatten(1)
        ref = self.hada_w1_a
        dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
        norms = torch.empty(ref.shape[0], device=ref.device, dtype=dtype)
        with torch.autocast(device_type=ref.device.type, enabled=False):
            for start in range(0, ref.shape[0], self.chunk_size):
                end = start + self.chunk_size
                chunk = weight[start:end].to(device=ref.device, dtype=dtype)
                if adapted:
                    u = ref[start:end].to(dtype) @ self.hada_w1_b.to(dtype)
                    v = self.hada_w2_a[start:end].to(dtype) @ self.hada_w2_b.to(dtype)
                    chunk = chunk + (u * v) * self.scale
                norms[start:end] = chunk.norm(dim=1)
        return norms.clamp_min(torch.finfo(dtype).tiny)

    @torch.no_grad()
    def comfy_magnitude(self):
        # Comfy's output-axis DoRA uses the BASE row norm, whereas training
        # uses the detached ADAPTED row norm. Convert rather than exporting
        # the raw magnitude (which would change the trained adapter).
        value = self.magnitude * (self._weight_norms(False) + torch.finfo(torch.float32).eps)
        value = value / self._weight_norms(True)
        return value.reshape(-1, *([1] * (3 if self.conv else 1)))

    @torch.no_grad()
    def load_comfy_magnitude(self):
        self.magnitude.mul_(self._weight_norms(True) /
                            (self._weight_norms(False) + torch.finfo(torch.float32).eps))

    def forward(self, x, *args, **kwargs):
        network = self.network_ref()
        base = self.org_forward(x, *args, **kwargs)
        if not network.is_active or network.is_merged_in or network._multiplier == 0.0:
            return base
        if self.training and self.module_dropout and torch.rand((), device=x.device) < self.module_dropout:
            return base
        if isinstance(x, QTensor) or x.is_quantized:
            x = x.dequantize()
        x = x.to(self.hada_w1_a.dtype)
        if self.training and self.dropout:
            x = F.dropout(x, p=self.dropout)
        delta = _ChunkedLoHa.apply(x, self.hada_w1_a, self.hada_w1_b,
                                  self.hada_w2_a, self.hada_w2_b, self.chunk_size, self.conv)
        delta = delta * self._runtime_scale
        if self.training and self.rank_dropout:
            # LyCORIS LoHa's rank dropout masks output channels.
            mask = F.dropout(torch.ones(self.hada_w1_a.shape[0], device=delta.device,
                                       dtype=delta.dtype), p=self.rank_dropout)
            delta = delta * self._channel_view(mask, delta.ndim)
        if self.use_dora:
            ratio = self.magnitude / self._weight_norms(True)
            ratio = self._channel_view(ratio.to(delta.dtype), delta.ndim)
            bias = self.org_module[0].bias
            unbiased = base.to(delta.dtype)
            if bias is not None:
                unbiased = unbiased - self._channel_view(
                    bias.to(device=delta.device, dtype=delta.dtype), base.ndim)
            delta = unbiased * (ratio - 1) + delta * ratio
        strength = network.torch_multiplier.to(device=delta.device, dtype=delta.dtype)
        if strength.ndim == 1 and strength.numel() > 1 and delta.shape[0] != strength.numel():
            if delta.shape[0] % strength.numel():
                raise ValueError("LoHa strengths must match the batch (including CFG duplication)")
            strength = strength.repeat_interleave(delta.shape[0] // strength.numel())
        return base + broadcast_and_multiply(delta, strength).to(base.dtype)

    def _channel_view(self, value, ndim):
        return value.reshape(1, -1, *([1] * (ndim - 2))) if self.conv else value

    @torch.no_grad()
    def reset_weights(self):
        self.hada_w2_b.zero_()
        if self.use_dora:
            self.magnitude.copy_(self._weight_norms(False))

    def merge_in(self, *args, **kwargs):
        # Quantized base weights must not be overwritten by generic LoRA merge.
        return

    def merge_out(self, *args, **kwargs):
        return
