# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pedro/Comfy codebook W4A8 storage, distinct from NVFP4 and ConvRot W4A4.

Forward matches comfy_kitchen's W4->INT8 reconstruction and dynamic INT8
activation quantization. Backward uses the frozen reconstructed weight as a
straight-through surrogate, including the orthogonal rotation. Only packed
state is saved; no input or dequantized weight is retained for backward.
Arithmetic reference: comfy_kitchen/backends/eager/w4a8_int8.py.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from toolkit.util.ostris_quant import OstrisQuantizer
from toolkit.util.convrot_quant import rotate

FORMAT_VERSION = 1


def _decode_int8_rows(packed, relative, codebook, group_size):
    n, half = packed.shape
    codes = torch.empty((n, half * 2), device=packed.device, dtype=torch.long)
    byte = packed.to(torch.uint8)
    codes[:, 0::2] = (byte & 15).long()
    codes[:, 1::2] = (byte >> 4).long()
    values = codebook.float()[codes] if codebook is not None else codes.float() - 8
    values = values.reshape(n, -1, group_size) * relative.float().unsqueeze(-1)
    return values.reshape(n, half * 2).round().clamp(-127, 127).to(torch.int8)


def decode_int8(packed, relative, codebook, group_size):
    # Bound LUT indices, FP32 levels and rounding scratch independently of
    # projection size. Never expand an expert bank. The final INT8 projection
    # remains available for integer GEMM and is much smaller than this scratch.
    rows, half = packed.shape
    chunk_rows = max(1, (128 * 1024 * 1024) // max(1, half * 2 * 32))
    if rows <= chunk_rows:
        return _decode_int8_rows(packed, relative, codebook, group_size)
    output = torch.empty((rows, half * 2), device=packed.device, dtype=torch.int8)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        output[start:stop].copy_(_decode_int8_rows(packed[start:stop], relative[start:stop], codebook, group_size))
    return output


def _relative(raw, fp32):
    return raw.view(torch.float32 if fp32 else torch.float8_e4m3fn)


@torch.library.custom_op('ostris::comfy_w4a8_linear', mutates_args=())
def w4a8_linear(x: torch.Tensor, packed: torch.Tensor, relative_raw: torch.Tensor,
                channel_raw: torch.Tensor, codebook_raw: Optional[torch.Tensor],
                correction_raw: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                group_size: int, rotation: int, relative_fp32: bool) -> torch.Tensor:
    relative = _relative(relative_raw, relative_fp32)
    channel = channel_raw.view(torch.float32)
    codebook = None if codebook_raw is None else codebook_raw.view(torch.float32)
    weight = decode_int8(packed, relative, codebook, group_size)
    if correction_raw is not None:
        # The published correction format uses its dequantized linear fallback.
        correction = correction_raw.view(torch.float32).t().unsqueeze(-1)
        folded = weight.float().reshape(weight.shape[0], -1, group_size)
        folded = folded * channel[:, None, None] + correction
        folded = rotate(folded.flatten(1).to(x.dtype), rotation)
        return F.linear(x, folded, None if bias is None else bias.to(x.dtype))
    rotated = rotate(x, rotation)
    scale = (rotated.float().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-30)
    native_scale = scale.to(rotated.dtype)
    native_scale = torch.where(native_scale == 0, torch.finfo(rotated.dtype).tiny, native_scale)
    activation = (rotated / native_scale).round().clamp(-128, 127).to(torch.int8)
    if x.is_cuda:
        rows = activation.shape[0]
        padded = F.pad(activation, (0, 0, 0, (-rows) % 32))
        result = torch._int_mm(padded, weight.t().contiguous())[:rows]
    else:
        result = activation.to(torch.int32) @ weight.t().to(torch.int32)
    result = (result.float() * (scale * channel[None])).to(x.dtype)
    if bias is not None:
        result = result + bias.to(x.dtype)
    return result


@w4a8_linear.register_fake
def _fake(x, packed, relative_raw, channel_raw, codebook_raw, correction_raw,
          bias, group_size, rotation, relative_fp32):
    return x.new_empty((x.shape[0], packed.shape[0]))


def _setup(ctx, inputs, output):
    x, packed, rel, channel, book, correction, bias, group, rotation, fp32 = inputs
    ctx.save_for_backward(packed, rel, channel, book, correction)
    ctx.group, ctx.rotation, ctx.relative_fp32 = group, rotation, fp32


def _backward(ctx, gradient):
    packed, rel, channel, book, correction = ctx.saved_tensors
    integer = decode_int8(packed, _relative(rel, ctx.relative_fp32),
                          None if book is None else book.view(torch.float32), ctx.group)
    weight = integer.float().reshape(integer.shape[0], -1, ctx.group)
    weight = weight * channel.view(torch.float32)[:, None, None]
    if correction is not None:
        weight = weight + correction.view(torch.float32).t().unsqueeze(-1)
    dx_rotated = gradient @ weight.flatten(1).to(gradient.dtype)
    return (rotate(dx_rotated, ctx.rotation),) + (None,) * 9


w4a8_linear.register_autograd(_backward, setup_context=_setup)


class ComfyW4A8Quantizer(OstrisQuantizer):
    wants_fp32_weight = False

    @staticmethod
    def attach_(module, packed, relative, channel, codebook=None, correction=None,
                group_size=16, rotation=256):
        rows, cols = module.out_features, module.in_features
        if cols % 2 or group_size <= 0 or rotation <= 0:
            raise ValueError('W4A8 requires even input width and positive group/rotation sizes')
        if packed.dtype != torch.int8 or tuple(packed.shape) != (rows, cols // 2):
            raise ValueError('W4A8 packed weight must be int8 [out,in/2]')
        if cols % group_size or cols % rotation or rotation < 1 or rotation & (rotation - 1) or (rotation.bit_length() - 1) % 2:
            raise ValueError('W4A8 group/rotation does not divide the projection')
        if tuple(relative.shape) != (rows, cols // group_size) or tuple(channel.shape) != (rows,):
            raise ValueError('W4A8 scale shape mismatch')
        if relative.dtype == torch.uint8:
            relative = relative.view(torch.float8_e4m3fn)
        if relative.dtype not in (torch.float8_e4m3fn, torch.float32):
            raise ValueError('W4A8 relative scales must be e4m3 or float32')
        if codebook is not None and tuple(codebook.shape) != (16,):
            raise ValueError('W4A8 codebook must contain 16 levels')
        if correction is not None and tuple(correction.shape) != (cols // group_size, rows):
            raise ValueError('W4A8 correction shape mismatch')
        for name, tensor in [('cw4_packed', packed), ('cw4_relative', relative),
                             ('cw4_channel', channel.float()),
                             ('cw4_codebook', None if codebook is None else codebook.float()),
                             ('cw4_correction', None if correction is None else correction.float())]:
            module.register_buffer(name, None if tensor is None else tensor.detach().contiguous().view(torch.uint8), persistent=False)
        module.cw4_group_size, module.cw4_rotation = group_size, rotation
        module.cw4_relative_fp32 = relative.dtype == torch.float32

    def quantize_(self, module, weight_fp32):
        raise ValueError('comfy_w4a8 imports published packed checkpoints; use convrot8 for fresh BF16 quantization')

    def requantize_(self, module, fp_weight):
        raise ValueError('W4A8 requantization is unsupported; keep adapters factorized')

    def dequantize(self, module):
        integer = decode_int8(module.cw4_packed.view(torch.int8),
                              _relative(module.cw4_relative, module.cw4_relative_fp32),
                              None if module.cw4_codebook is None else module.cw4_codebook.view(torch.float32),
                              module.cw4_group_size)
        weight = integer.float().reshape(module.out_features, -1, module.cw4_group_size)
        weight = weight * module.cw4_channel.view(torch.float32)[:, None, None]
        if module.cw4_correction is not None:
            weight += module.cw4_correction.view(torch.float32).t().unsqueeze(-1)
        return rotate(weight.flatten(1), module.cw4_rotation)

    def forward(self, module, x):
        shape = x.shape
        result = w4a8_linear(x.reshape(-1, shape[-1]), module.cw4_packed.view(torch.int8),
                             module.cw4_relative, module.cw4_channel, module.cw4_codebook,
                             module.cw4_correction, module.bias, module.cw4_group_size,
                             module.cw4_rotation, module.cw4_relative_fp32)
        return result.reshape(*shape[:-1], module.out_features)
