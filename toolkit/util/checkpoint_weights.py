"""Read individual logical weights from explicit safetensors storage formats.

Descriptors inspect headers and small quantization metadata, never instantiate
models or read full weight payloads. Callers admit their working-memory budget
before decoding a layer. The actual toolkit quantizers reconstruct quantized
weights in their original input basis on CPU.
"""

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from types import SimpleNamespace

import torch
from safetensors import SafetensorError

from toolkit.util.comfy_quant_import import parse_comfy_quant_blob
from toolkit.util.convrot_quant import ConvRotInt8Quantizer, rotate
from toolkit.util.float8_quant import Float8Quantizer


WEIGHT_DECODER = 'comfy_logical_weight_v2'
_FLOAT_DTYPES = {'F16', 'BF16', 'F32', 'F64'}


@dataclass(frozen=True)
class CheckpointWeight:
    key: str
    shape: tuple[int, ...]
    storage_dtype: str
    quant_format: str = 'floating'
    scale_key: str | None = None
    marker_key: str | None = None
    rotation: int = 1
    group_size: int = 16
    auxiliary_keys: tuple[str, ...] = ()

    def report(self):
        return dict(decoder=WEIGHT_DECODER, **asdict(self))

    @property
    def decode_working_bytes(self):
        if self.quant_format == 'floating':
            return 0
        rows, columns = self.shape
        # FP32 row scaling, rotation buffers and construction/cached Hadamard.
        # Add this to the caller's layer/SVD estimate before reading payloads.
        return 4*(16*rows*columns + 3*self.rotation*self.rotation + rows)


def _header(handle, key, dtypes, shapes=None):
    try:
        value = handle.get_slice(key)
    except SafetensorError as error:
        raise ValueError(f'Missing required quantization tensor {key}') from error
    shape = tuple(value.get_shape())
    if value.get_dtype() not in dtypes or (shapes is not None and shape not in shapes):
        raise ValueError(f'{key}: invalid quantization layout {value.get_dtype()} {shape}; expected dtype {dtypes}, shapes {shapes}')
    return shape


def _rotation(config, columns, default=1):
    size = config.get('convrot_groupsize', default)
    if type(size) is not int or size < 1:
        raise ValueError('convrot_groupsize must be 1 or a positive power of 4')
    remaining = size
    while remaining > 1 and remaining % 4 == 0:
        remaining //= 4
    if remaining != 1 or columns % size:
        raise ValueError(f'convrot_groupsize must be 1 or a power of 4 dividing input width {columns}')
    return size


def _configuration(handle, stem, keys, metadata=None):
    marker_key = stem+'.comfy_quant'
    if marker_key in keys:
        shape = _header(handle, marker_key, {'U8'})
        if len(shape) != 1 or not 0 < shape[0] <= 65536:
            raise ValueError(f'{marker_key}: comfy_quant marker must be a nonempty 1D uint8 JSON tensor of at most 64 KiB')
        try:
            config = parse_comfy_quant_blob(handle.get_tensor(marker_key))
        except (ValueError, UnicodeError, TypeError) as error:
            raise ValueError(f'{marker_key}: malformed comfy_quant JSON marker') from error
        if not isinstance(config, dict):
            raise ValueError(f'{marker_key}: comfy_quant JSON must be an object')
        return config, marker_key
    raw = (metadata if metadata is not None else handle.metadata() or {}).get('_quantization_metadata')
    if raw is not None:
        try:
            layers = _header_layers(raw)
            if stem in layers:
                if not isinstance(layers[stem], dict):
                    raise ValueError(f'{stem}: quantization config must be an object')
                return layers[stem], '_quantization_metadata:'+stem
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError('Malformed _quantization_metadata checkpoint header') from error
    if 'scaled_fp8' in keys and stem+'.scale_weight' in keys:
        return {'format': 'float8_e4m3fn', 'full_precision_matrix_mult': True}, 'scaled_fp8'
    return None, None


@lru_cache(maxsize=1)
def _header_layers(raw):
    layers = json.loads(raw).get('layers', {})
    if not isinstance(layers, dict):
        raise ValueError('layers must be an object')
    return layers


def describe_checkpoint_weight(handle, key, keys=None, metadata=None):
    """Validate supported storage metadata without loading a dense weight."""
    keys = set(handle.keys()) if keys is None else keys
    stored = handle.get_slice(key)
    shape, dtype = tuple(stored.get_shape()), stored.get_dtype()
    stem = key.removesuffix('.weight')
    config, marker_key = _configuration(handle, stem, keys, metadata)
    scale_key = stem+('.scale_weight' if marker_key == 'scaled_fp8' else '.weight_scale')
    auxiliaries = {stem+suffix for suffix in (
        '.scale', '.scale_weight', '.weight_scale', '.weight_scale_2', '.pre_quant_scale',
        '.weight_s_rel', '.weight_s_channel', '.weight_codebook', '.weight_correction',
        '.quant_state', '.weight._data',
    )} & keys
    if config is None:
        if dtype not in _FLOAT_DTYPES or auxiliaries:
            raise ValueError(f'{key}: integer/float8/scaled base weights require explicit supported quantization metadata; unsupported auxiliary tensors {sorted(auxiliaries)}')
        return CheckpointWeight(key, shape, dtype)
    if not isinstance(config, dict):
        raise ValueError(f'{marker_key}: comfy_quant JSON must be an object')
    fmt = config.get('format')
    allowed = {
        'int8_tensorwise': {'format', 'convrot', 'convrot_groupsize'},
        'float8_e4m3fn': {'format', 'full_precision_matrix_mult'},
        'nvfp4': {'format', 'full_precision_matrix_mult'},
        'asym_w4a8_int8': {'format', 'group_size', 'convrot_groupsize', 'convrot'},
        'convrot_w4a4': {'format', 'convrot_groupsize'},
    }
    if fmt not in allowed or set(config) - allowed[fmt]:
        raise ValueError(f'{marker_key}: unsupported quantization format/config {config!r}; supported formats: {sorted(allowed)}')
    if 'full_precision_matrix_mult' in config and type(config['full_precision_matrix_mult']) is not bool:
        raise ValueError(f'{marker_key}: full_precision_matrix_mult must be a boolean')
    if len(shape) != 2 or any(size < 1 for size in shape):
        raise ValueError(f'{key}: quantized baking requires a positive 2D logical weight shape')
    rows, columns = shape
    if fmt in ('nvfp4', 'asym_w4a8_int8', 'convrot_w4a4'):
        columns *= 2
    logical_shape = (rows, columns)
    rotation, group_size = 1, 16
    expected_aux = set()
    if fmt in ('int8_tensorwise', 'convrot_w4a4'):
        _header(handle, key, {'I8'} if fmt == 'int8_tensorwise' else {'I8', 'U8'})
        _header(handle, scale_key, {'F32'}, {(rows,), (rows, 1)})
        expected_aux.add(scale_key)
        if type(config.get('convrot', False)) is not bool:
            raise ValueError(f'{marker_key}: convrot must be a boolean')
        if config.get('convrot', False) or fmt == 'convrot_w4a4':
            rotation = _rotation(config, columns, 256)
        elif 'convrot_groupsize' in config:
            raise ValueError(f'{marker_key}: convrot_groupsize without convrot=true is ambiguous')
    elif fmt == 'float8_e4m3fn':
        _header(handle, key, {'F8_E4M3', 'U8', 'I8'})
        _header(handle, scale_key, {'F32'}, {(), (1,)})
        expected_aux.add(scale_key)
    elif fmt == 'nvfp4':
        _header(handle, key, {'U8', 'I8'})
        if columns % 16:
            raise ValueError(f'{key}: NVFP4 logical input width must be divisible by 16')
        block_shape = _header(handle, scale_key, {'F8_E4M3', 'U8', 'I8'})
        padded_size = math.ceil(rows/128)*128 * math.ceil((columns//16)/4)*4
        if math.prod(block_shape) != padded_size:
            raise ValueError(f'{scale_key}: NVFP4 swizzled scale size must be {padded_size}')
        _header(handle, stem+'.weight_scale_2', {'F32'}, {(), (1,)})
        expected_aux.update({scale_key, stem+'.weight_scale_2'})
        if stem+'.pre_quant_scale' in keys:
            _header(handle, stem+'.pre_quant_scale', {'F32'}, {(columns,)})
            expected_aux.add(stem+'.pre_quant_scale')
    elif fmt == 'asym_w4a8_int8':
        _header(handle, key, {'I8'})
        group_size = config.get('group_size', 16)
        if type(group_size) is not int or group_size < 1 or columns % group_size:
            raise ValueError(f'{key}: W4A8 group_size must be a positive divisor of input width')
        rotation = _rotation(config, columns, 256)
        _header(handle, stem+'.weight_s_rel', {'F32', 'F8_E4M3', 'U8'}, {(rows, columns//group_size)})
        _header(handle, stem+'.weight_s_channel', {'F32'}, {(rows,)})
        expected_aux.update({stem+'.weight_s_rel', stem+'.weight_s_channel'})
        scale_key = stem+'.weight_s_channel'
        if stem+'.weight_codebook' in keys:
            _header(handle, stem+'.weight_codebook', {'F32'}, {(16,)})
            expected_aux.add(stem+'.weight_codebook')
        if stem+'.weight_correction' in keys:
            _header(handle, stem+'.weight_correction', {'F32'}, {(columns//group_size, rows)})
            expected_aux.add(stem+'.weight_correction')
    if auxiliaries - expected_aux:
        raise ValueError(f'{key}: unsupported quantization/scaling auxiliary tensors {sorted(auxiliaries-expected_aux)}')
    return CheckpointWeight(key, logical_shape, dtype, fmt, scale_key, marker_key, rotation,
                           group_size, tuple(sorted(expected_aux)))


def decode_checkpoint_weight(handle, descriptor):
    """Return one FP32 logical weight, using the existing CPU codec exactly."""
    weight = handle.get_tensor(descriptor.key)
    if descriptor.quant_format == 'floating':
        return weight
    scales = handle.get_tensor(descriptor.scale_key).float()
    if descriptor.quant_format == 'nvfp4':
        # Block scales may be exactly zero for zero-valued blocks/padding.
        if not torch.isfinite(scales).all() or not (scales >= 0).all():
            raise ValueError(f'{descriptor.scale_key}: NVFP4 scales must be finite and nonnegative')
    elif not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError(f'{descriptor.scale_key}: quantization scales must be finite and strictly positive')
    stem = descriptor.key.removesuffix('.weight')
    if descriptor.quant_format == 'int8_tensorwise':
        stand_in = SimpleNamespace(cr8_qdata=weight.contiguous(),
            cr8_scales=scales.reshape(-1).contiguous().view(torch.uint8),
            cr8_rot_size=descriptor.rotation)
        result = ConvRotInt8Quantizer().dequantize(stand_in)
    elif descriptor.quant_format == 'float8_e4m3fn':
        codes = weight if weight.dtype == torch.float8_e4m3fn else weight.view(torch.float8_e4m3fn)
        stand_in = SimpleNamespace(f8_qdata=codes.contiguous(),
            f8_scale=scales.reshape(1).contiguous().view(torch.uint8))
        result = Float8Quantizer().dequantize(stand_in)
    elif descriptor.quant_format == 'convrot_w4a4':
        # Exported Comfy int4 is signed two's-complement nibbles, unlike the
        # internal ConvRotIntN packed offset codes. Do not reuse that unpacker.
        packed = weight.view(torch.uint8)
        codes = torch.empty(descriptor.shape, dtype=torch.int8)
        for target, nibble in ((0, packed & 15), (1, packed >> 4)):
            codes[:, target::2] = torch.where(nibble >= 8, nibble.to(torch.int16)-16, nibble).to(torch.int8)
        result = rotate(codes.float()*scales.reshape(-1, 1), descriptor.rotation)
    elif descriptor.quant_format == 'nvfp4':
        from toolkit.util.nvfp4_quant import Nvfp4Quantizer, swap_nvfp4_nibbles, unswizzle_nvfp4_scales
        raw = handle.get_tensor(descriptor.scale_key)
        raw = raw if raw.dtype == torch.float8_e4m3fn else raw.view(torch.float8_e4m3fn)
        block = unswizzle_nvfp4_scales(raw, descriptor.shape[0], descriptor.shape[1]//16)
        if not torch.isfinite(block.float()).all() or not (block.float() >= 0).all():
            raise ValueError(f'{descriptor.scale_key}: invalid decoded block scales')
        pts = handle.get_tensor(stem+'.weight_scale_2')
        if not torch.isfinite(pts).all() or not (pts > 0).all():
            raise ValueError(f'{stem}.weight_scale_2: tensor scale must be finite and positive')
        stand_in = SimpleNamespace(nv4_qdata=swap_nvfp4_nibbles(weight.view(torch.uint8)),
            nv4_scales=block.contiguous().view(torch.uint8),
            nv4_pts=pts.reshape(1).contiguous().view(torch.uint8),
            out_features=descriptor.shape[0], in_features=descriptor.shape[1])
        result = Nvfp4Quantizer().dequantize(stand_in)
    elif descriptor.quant_format == 'asym_w4a8_int8':
        from toolkit.util.comfy_w4a8_quant import ComfyW4A8Quantizer
        relative = handle.get_tensor(stem+'.weight_s_rel')
        if relative.dtype == torch.uint8:
            relative = relative.view(torch.float8_e4m3fn)
        if not torch.isfinite(relative.float()).all() or not (relative.float() >= 0).all():
            raise ValueError(f'{stem}.weight_s_rel: relative scales must be finite and nonnegative')
        optional = {}
        for name in ('weight_codebook', 'weight_correction'):
            value = handle.get_tensor(stem+'.'+name) if stem+'.'+name in descriptor.auxiliary_keys else None
            if value is not None and not torch.isfinite(value).all():
                raise ValueError(f'{stem}.{name}: auxiliary tensor must be finite')
            optional[name] = None if value is None else value.contiguous().view(torch.uint8)
        stand_in = SimpleNamespace(cw4_packed=weight.contiguous().view(torch.uint8),
            cw4_relative=relative.contiguous().view(torch.uint8),
            cw4_channel=scales.contiguous().view(torch.uint8),
            cw4_codebook=optional['weight_codebook'], cw4_correction=optional['weight_correction'],
            cw4_group_size=descriptor.group_size, cw4_rotation=descriptor.rotation,
            cw4_relative_fp32=relative.dtype == torch.float32, out_features=descriptor.shape[0])
        result = ComfyW4A8Quantizer().dequantize(stand_in)
    else:
        raise ValueError(f'Unsupported weight descriptor format {descriptor.quant_format}')
    if tuple(result.shape) != descriptor.shape or not torch.isfinite(result).all():
        raise ValueError(f'{descriptor.key}: decoded quantized weight has an invalid shape or nonfinite values')
    return result


def checkpoint_input_scale(handle, descriptor):
    """AWQ input scaling belongs to forward weights, not DoRA normalization."""
    key = descriptor.key.removesuffix('.weight')+'.pre_quant_scale'
    if key not in descriptor.auxiliary_keys:
        return None
    value = handle.get_tensor(key)
    if not torch.isfinite(value).all() or not (value > 0).all():
        raise ValueError(f'{key}: AWQ input scales must be finite and positive')
    return value
