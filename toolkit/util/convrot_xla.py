"""Strict XLA execution of frozen ConvRot8 W8A8 linears.

The only device-specific dependency is PyTorch/XLA's private int8 dot-general
entry point. Kaggle's installed wheel must pass the runtime probe before a
training job can use it. No floating-point matmul substitute is permitted here.
"""

from __future__ import annotations

import torch
import os


def _reject_global_bf16_downcast() -> None:
    for name in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16"):
        if os.environ.get(name, "").lower() not in ("", "0", "false", "no"):
            raise RuntimeError(f"{name} changes ConvRot FP32 scale arithmetic; unset it")


def prepare_convrot8_scales_for_xla_(root: torch.nn.Module) -> int:
    """Decode checkpoint byte-view scales on CPU before moving the model to XLA.

    Call after checkpoint import and dtype conversion, immediately before
    ``root.to(xla_device)``. The ordinary byte-view buffer remains intact for
    existing checkpoint export and CUDA paths.
    """
    count = 0
    for module in root.modules():
        if not hasattr(module, "cr8_qdata"):
            continue
        if module.cr8_qdata.dtype != torch.int8:
            raise TypeError("XLA ConvRot8 requires unpacked int8 weight codes")
        raw = module.cr8_scales
        if raw.device.type != "cpu" or raw.dtype != torch.uint8:
            raise ValueError("prepare ConvRot8 scales from CPU byte-view buffers before XLA placement")
        if raw.numel() != module.cr8_qdata.shape[0] * 4:
            raise ValueError("ConvRot8 scale bytes do not match the output row count")
        scales = raw.contiguous().view(torch.float32).clone()
        if not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("ConvRot8 scales must be finite and positive")
        module.register_buffer("cr8_xla_scales", scales, persistent=False)
        count += 1
    return count


def _int8_dot(a: torch.Tensor, qdata: torch.Tensor) -> torch.Tensor:
    """Version-checked adapter for XLA's int8 dot with int32 accumulation."""
    try:
        import torch_xla
    except ImportError as exc:
        raise RuntimeError("ConvRot8 XLA requires a matching torch_xla wheel") from exc
    dot = getattr(getattr(torch_xla, "_XLAC", None), "_xla_dot_general", None)
    if not callable(dot):
        raise RuntimeError("This torch_xla wheel lacks _XLAC._xla_dot_general")
    try:
        result = dot(a, qdata, (([-1], [-1]), ()), preferred_element_type=torch.int32)
    except Exception as exc:
        raise RuntimeError("XLA INT8 dot with INT32 accumulation failed; W8A8 is unavailable") from exc
    if result.dtype != torch.int32:
        raise RuntimeError(f"XLA INT8 dot returned {result.dtype}, expected torch.int32")
    return result


class _ConvRot8LinearSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, qdata, scales, bias, act_qmax, quantize_rows):
        _reject_global_bf16_downcast()
        if x2d.device.type != "xla" or qdata.device != x2d.device or scales.device != x2d.device:
            raise ValueError("ConvRot8 XLA inputs, codes, and scales must share an XLA device")
        if qdata.requires_grad or scales.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError("ConvRot8 XLA supports frozen base weights and bias only")
        if scales.dtype != torch.float32:
            raise TypeError("ConvRot8 XLA scales must remain FP32")
        aq, a_scales = quantize_rows(x2d, act_qmax)
        i32 = _int8_dot(aq, qdata)
        # Match the existing non-Triton epilogue's multiply and cast order.
        out = i32.float() * scales
        out = out * a_scales.unsqueeze(1)
        if bias is not None:
            out = out + bias.float()
        ctx.save_for_backward(qdata, scales)
        return out.to(x2d.dtype)

    @staticmethod
    def backward(ctx, grad):
        qdata, scales = ctx.saved_tensors
        w = qdata.to(grad.dtype) * scales.to(grad.dtype).unsqueeze(1)
        return grad @ w, None, None, None, None, None


def convrot8_linear_xla(
    module, x: torch.Tensor, rot: int, act_qmax: int, rotate_fn, quantize_rows
) -> torch.Tensor:
    """Apply the existing rotation, strict W8A8 forward, and compact STE backward."""
    if x.device.type != "xla":
        raise ValueError("convrot8_linear_xla requires XLA activations")
    scales = getattr(module, "cr8_xla_scales", None)
    if scales is None:
        raise RuntimeError("prepare_convrot8_scales_for_xla_ must run before model placement")
    if scales.device != x.device:
        raise RuntimeError("ConvRot8 XLA scales have not been placed on the activation device")
    if module.cr8_qdata.device != x.device:
        raise RuntimeError("ConvRot8 XLA weight codes have not been placed on the activation device")
    if module.bias is not None and module.bias.device != x.device:
        raise RuntimeError("ConvRot8 XLA bias has not been placed on the activation device")
    in_f = module.in_features
    x2d = rotate_fn(x, rot).reshape(-1, in_f)
    y2d = _ConvRot8LinearSTE.apply(
        x2d, module.cr8_qdata, scales, module.bias, act_qmax, quantize_rows
    )
    return y2d.reshape(*x.shape[:-1], module.out_features)
