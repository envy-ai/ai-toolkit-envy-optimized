"""CPU-backed per-expert int8 quantization for Nucleus MoE layers."""

import os
from types import MethodType
from typing import Optional, Union

import torch
import torch.nn.functional as F


NUCLEUS_MOE_EXPERT_CHUNK_SIZE = int(
    os.environ.get("AI_TOOLKIT_NUCLEUS_MOE_EXPERT_CHUNK_SIZE", "8")
)


class QuantizedNucleusMoEWeight:
    def __init__(self, qweight, scale, original_dtype, keep_on_cpu=False):
        self.qweight = qweight
        self.scale = scale
        self.original_dtype = original_dtype
        self.keep_on_cpu = keep_on_cpu

    @classmethod
    @torch.no_grad()
    def from_tensor(cls, tensor, keep_on_cpu=False):
        qweights, scales = [], []
        for expert_weight in tensor.detach():
            expert_weight = expert_weight.to(device="cpu", dtype=torch.float32)
            max_abs = expert_weight.abs().amax(dim=0, keepdim=True)
            scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
            qweights.append(torch.round(expert_weight / scale).clamp(-127, 127).to(torch.int8).contiguous())
            scales.append(scale.contiguous())
        return cls(
            torch.stack(qweights).contiguous(),
            torch.stack(scales).contiguous(),
            tensor.dtype,
            keep_on_cpu,
        )

    def dequantize(self, expert_idx=None, device=None, dtype=None):
        dtype = dtype or self.original_dtype
        qweight = self.qweight if expert_idx is None else self.qweight[expert_idx]
        scale = self.scale if expert_idx is None else self.scale[expert_idx]
        return qweight.to(device=device, dtype=dtype) * scale.to(device=device, dtype=dtype)

    def storage_nbytes(self):
        return self.qweight.nbytes + self.scale.nbytes

    def apply_(self, fn):
        if self.keep_on_cpu:
            self.scale = fn(self.scale).to("cpu")
        else:
            self.qweight = fn(self.qweight)
            self.scale = fn(self.scale)
        return self


def _run_quantized_nucleus_moe_for_loop(self, x, num_tokens_per_expert):
    if (
        num_tokens_per_expert.numel() == self.num_experts
        and x.shape[0] % self.num_experts == 0
        and bool(torch.all(num_tokens_per_expert == num_tokens_per_expert[0]).item())
    ):
        x_per_expert = x.reshape(self.num_experts, x.shape[0] // self.num_experts, x.shape[-1])
        outputs = []
        for start in range(0, self.num_experts, max(1, NUCLEUS_MOE_EXPERT_CHUNK_SIZE)):
            stop = min(start + max(1, NUCLEUS_MOE_EXPERT_CHUNK_SIZE), self.num_experts)
            indices = slice(start, stop)
            gate_up = torch.bmm(
                x_per_expert[indices],
                self.gate_up_proj.dequantize(indices, device=x.device, dtype=x.dtype),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            outputs.append(torch.bmm(
                F.silu(gate) * up,
                self.down_proj.dequantize(indices, device=x.device, dtype=x.dtype),
            ))
        return torch.cat(outputs, dim=0).reshape(x.shape[0], -1)

    counts = num_tokens_per_expert.tolist()
    real_tokens = sum(counts)
    outputs = []
    for expert_idx, x_expert in enumerate(torch.split(x[:real_tokens], counts, dim=0)):
        gate, up = torch.matmul(
            x_expert,
            self.gate_up_proj.dequantize(expert_idx, device=x.device, dtype=x.dtype),
        ).chunk(2, dim=-1)
        outputs.append(torch.matmul(
            F.silu(gate) * up,
            self.down_proj.dequantize(expert_idx, device=x.device, dtype=x.dtype),
        ))
    out = torch.cat(outputs, dim=0)
    padding = x.shape[0] - real_tokens
    return torch.vstack((out, out.new_zeros((padding, out.shape[-1])))) if padding else out


def _apply_quantized_nucleus_moe(self, fn, recurse=True):
    try:
        result = self._nucleus_moe_orig_apply(fn, recurse=recurse)
    except TypeError:
        result = self._nucleus_moe_orig_apply(fn)
    for name in ("gate_up_proj", "down_proj"):
        packed = getattr(self, name)
        if isinstance(packed, QuantizedNucleusMoEWeight):
            packed.apply_(fn)
    return result


def _is_nucleus_moe_experts_module(module):
    return (
        module.__class__.__name__ == "SwiGLUExperts"
        and isinstance(getattr(module, "gate_up_proj", None), torch.nn.Parameter)
        and isinstance(getattr(module, "down_proj", None), torch.nn.Parameter)
        and module.gate_up_proj.ndim == 3
        and module.down_proj.ndim == 3
    )


def quantize_nucleus_moe_experts(model, keep_on_cpu=False):
    count = 0
    for module in model.modules():
        if not _is_nucleus_moe_experts_module(module):
            continue
        gate_up_proj = module._parameters.pop("gate_up_proj")
        down_proj = module._parameters.pop("down_proj")
        module.gate_up_proj = QuantizedNucleusMoEWeight.from_tensor(gate_up_proj, keep_on_cpu)
        module.down_proj = QuantizedNucleusMoEWeight.from_tensor(down_proj, keep_on_cpu)
        module.use_grouped_mm = False
        module._run_experts_for_loop = MethodType(_run_quantized_nucleus_moe_for_loop, module)
        if not hasattr(module, "_nucleus_moe_orig_apply"):
            module._nucleus_moe_orig_apply = module._apply
            module._apply = MethodType(_apply_quantized_nucleus_moe, module)
        count += 1
    return count


def move_nucleus_moe_quantized_weights(model, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None, non_blocking=True):
    count = 0
    device = torch.device(device)
    for module in model.modules():
        for name in ("gate_up_proj", "down_proj"):
            packed = getattr(module, name, None)
            if not isinstance(packed, QuantizedNucleusMoEWeight):
                continue
            packed.keep_on_cpu = False
            packed.qweight = packed.qweight.to(device=device, non_blocking=non_blocking)
            packed.scale = packed.scale.to(device=device, dtype=dtype, non_blocking=non_blocking)
            count += 1
    return count
