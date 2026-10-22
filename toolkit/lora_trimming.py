"""CPU-only rank reduction of standalone adapter safetensors.

Ordinary/LoHa/LoKr reduction uses factorized updates without a base model.
Opt-in DoRA baking reads local floating base weights and materializes one
memory-budgeted effective-delta layer. No pickle, model code or network calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from toolkit.lora_analysis import _normalized_module_name

ALGORITHM = "qr_core_svd_v2"
_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
# Safetensors orders by descending dtype enum, then tensor name. See
# huggingface/safetensors safetensors/src/tensor.rs: prepare(), Dtype.
_DTYPES = {
    torch.uint8: (0, "U8"), torch.int8: (1, "I8"), torch.int16: (2, "I16"),
    torch.float16: (3, "F16"), torch.bfloat16: (4, "BF16"),
    torch.int32: (5, "I32"), torch.float32: (6, "F32"),
    torch.float64: (7, "F64"), torch.int64: (8, "I64"),
}
_PAIRS = ((".lora_down.weight", ".lora_up.weight"), (".lora_A.weight", ".lora_B.weight"))
_HASH_KEYS = ("sshs_model_hash", "sshs_legacy_hash")
RANK_POLICIES = ('energy', 'relative_error', 'fixed', 'sv_ratio', 'sv_cumulative', 'sv_fro')
_MAGNITUDES = ('.dora_scale', '.magnitude', '.lora_magnitude_vector',
               '.lora_magnitude_vector.weight', '.lora_magnitude_vector.default.weight')


def _options(retained_energy, relative_error, min_rank, max_rank, rank_policy=None, rank_value=None):
    if rank_policy is not None or rank_value is not None:
        if retained_energy is not None or relative_error is not None:
            raise ValueError('rank_policy/rank_value cannot be combined with retained_energy/relative_error')
        if rank_policy not in RANK_POLICIES or rank_value is None:
            raise ValueError(f'rank_policy must be one of {RANK_POLICIES}, with rank_value')
        if isinstance(rank_value, bool) or not isinstance(rank_value, (int, float)) or not math.isfinite(rank_value):
            raise ValueError('rank_value must be a finite number')
        if rank_policy == 'fixed' and (rank_value < 1 or int(rank_value) != rank_value):
            raise ValueError('fixed rank_value must be a positive integer')
        if rank_policy == 'sv_ratio' and rank_value < 1:
            raise ValueError('sv_ratio rank_value must be at least 1')
        if rank_policy in ('energy', 'sv_cumulative', 'sv_fro') and not 0 < rank_value <= 1:
            raise ValueError(f'{rank_policy} rank_value must be in (0, 1]')
        if rank_policy == 'relative_error' and not 0 <= rank_value <= 1:
            raise ValueError('relative_error rank_value must be in [0, 1]')
        if rank_policy == 'energy':
            retained_energy = rank_value
        elif rank_policy == 'relative_error':
            relative_error = rank_value
    if retained_energy is not None and relative_error is not None:
        raise ValueError("Choose retained_energy OR relative_error, not both")
    if retained_energy is None and relative_error is None and rank_policy is None:
        retained_energy = 0.9999
    if retained_energy is not None and (isinstance(retained_energy, bool) or not math.isfinite(retained_energy)
                                        or not 0 < retained_energy <= 1):
        raise ValueError("retained_energy must be finite and in (0, 1]")
    if relative_error is not None and (isinstance(relative_error, bool) or not math.isfinite(relative_error)
                                      or not 0 <= relative_error <= 1):
        raise ValueError("relative_error must be finite and in [0, 1]")
    if type(min_rank) is not int or min_rank < 1:
        raise ValueError("min_rank must be a positive integer")
    if max_rank is not None and (type(max_rank) is not int or max_rank < min_rank):
        raise ValueError("max_rank must be an integer at least min_rank")
    policy = rank_policy or ('energy' if retained_energy is not None else 'relative_error')
    value = rank_value if rank_policy is not None else (retained_energy if retained_energy is not None else relative_error)
    return dict(retained_energy=retained_energy, relative_error=relative_error,
                min_rank=min_rank, max_rank=max_rank, rank_policy=policy, rank_value=value)


def _factor_options(options):
    return {key: options[key] for key in ('min_rank', 'max_rank', 'rank_policy', 'rank_value')}


def _core(up, down):
    qb, rb = torch.linalg.qr(up, mode="reduced")
    qa, ra = torch.linalg.qr(down.T, mode="reduced")
    return qb, qa, rb @ ra.T


def _factor_error(a, b, new_a, new_b, ratio, norm):
    if norm == 0:
        _, _, difference = _core(new_b, new_a)
        return 0.0 if float(torch.linalg.vector_norm(difference)) == 0 else math.inf
    # Difference = [B, -ratio*Bnew] @ [A; Anew]. QR avoids dense output×input.
    _, _, difference = _core(torch.cat((b, -new_b * ratio), 1), torch.cat((a, new_a), 0))
    return float(torch.linalg.vector_norm(difference)) / norm


def trim_lora_factors(down: torch.Tensor, up: torch.Tensor, *, alpha=None,
                      retained_energy=None, relative_error=None, min_rank=1, max_rank=None,
                      rank_policy=None, rank_value=None, output_dtype=None):
    """Return CPU factors and metrics, preserving original storage dtypes/scaling.

    Explicit alpha is kept unchanged: compensate the factor product for its new
    alpha/rank runtime scale. Missing alpha means standalone unit scale.
    """
    options = _options(retained_energy, relative_error, min_rank, max_rank, rank_policy, rank_value)
    if down.device.type != "cpu" or up.device.type != "cpu":
        raise ValueError("LoRA trimming accepts CPU tensors only")
    if down.dtype not in _FLOATS or up.dtype not in _FLOATS:
        raise ValueError("LoRA factors must be float16, bfloat16, float32 or float64")
    if output_dtype is not None and output_dtype not in _FLOATS:
        raise ValueError('output_dtype must be a supported floating dtype')
    down_dtype, up_dtype = output_dtype or down.dtype, output_dtype or up.dtype
    if down.ndim not in (2, 4) or up.ndim != down.ndim:
        raise ValueError("Expected linear factors or Conv2d down/up factors")
    if down.ndim == 4 and tuple(up.shape[2:]) != (1, 1):
        raise ValueError("Only standard convolution LoRA with a 1x1 up kernel is supported; Tucker/mid kernels are not")
    rank = down.shape[0]
    if rank < 1 or up.shape[1] != rank or any(size < 1 for size in (*down.shape, *up.shape)):
        raise ValueError("Invalid LoRA factor dimensions or mismatched ranks")
    if min_rank > rank or (max_rank is not None and max_rank > rank):
        raise ValueError(f"Rank bounds cannot exceed original rank {rank}; ranks are never increased")
    if alpha is not None and (isinstance(alpha, bool) or not math.isfinite(alpha) or alpha == 0):
        raise ValueError("Explicit alpha must be finite and nonzero; alpha=0 has ambiguous loader semantics")
    a, b = down.detach().reshape(rank, -1).double(), up.detach().reshape(up.shape[0], rank).double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("LoRA factors contain nonfinite values")
    qb, qa, core = _core(b, a)
    if not torch.isfinite(core).all():
        raise ValueError("LoRA update exceeds the supported float64 numeric range (nonfinite QR core)")
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    if not torch.isfinite(singular).all():
        raise ValueError("LoRA update exceeds the supported float64 numeric range (nonfinite singular values)")
    norm = float(torch.linalg.vector_norm(singular))
    energy = singular.square()
    total = float(energy.sum())
    if not math.isfinite(norm) or not math.isfinite(total):
        raise ValueError("LoRA update exceeds the supported float64 numeric range (norm/energy overflow)")
    if (norm == 0 or total == 0) and (torch.count_nonzero(core) or torch.count_nonzero(singular)):
        raise ValueError("LoRA update is below the supported float64 numeric range (norm/energy underflow)")
    policy, value = options['rank_policy'], options['rank_value']
    bounded = policy in ('energy', 'relative_error')
    error_limit = (math.sqrt(1 - value) if policy == 'energy' else value) if bounded else None
    cap = rank if max_rank is None else max_rank
    if bounded:
        candidates = range(min_rank, cap + 1)
    else:
        if policy == 'fixed':
            selected = int(value)
        elif total == 0:
            selected = 1
        elif policy == 'sv_ratio':
            selected = max(1, int(torch.count_nonzero(singular > singular[0] / value)))
        else:
            spectrum = singular if policy == 'sv_cumulative' else energy
            target = value if policy == 'sv_cumulative' else value**2
            selected = int(torch.searchsorted(spectrum.cumsum(0) / spectrum.sum(), target)) + 1
        # Rank policies are explicit target/cap policies, not error guarantees.
        selected = max(min_rank, min(selected, cap))
        candidates = (selected,)
    for selected in candidates:
        if selected == rank and down_dtype == down.dtype and up_dtype == up.dtype:
            # Avoid introducing rounding/decomposition error when no reduction
            # satisfies the limit. Original tensors are returned bit-exactly.
            new_down, new_up, error, ideal = down.detach().clone(), up.detach().clone(), 0.0, 1.0
        else:
            tail = float(energy[selected:].sum())
            if bounded and total and math.sqrt(tail / total) > error_limit:
                continue
            count = min(selected, len(singular))
            roots = singular[:count].sqrt()
            compensation = math.sqrt(selected / rank) if alpha is not None else 1.0
            small_b = (qb @ u[:, :count]) * (roots * compensation)[None]
            small_a = (roots * compensation)[:, None] * (vh[:count] @ qa.T)
            if count < selected:
                small_b = torch.nn.functional.pad(small_b, (0, selected-count))
                small_a = torch.nn.functional.pad(small_a, (0, 0, 0, selected-count))
            new_down = small_a.reshape(selected, *down.shape[1:]).to(down_dtype).contiguous()
            new_up = small_b.reshape(up.shape[0], selected, *up.shape[2:]).to(up_dtype).contiguous()
            if not torch.isfinite(new_down).all() or not torch.isfinite(new_up).all():
                if bounded:
                    continue
                raise ValueError('Selected rank cannot be stored in the original factor dtype without overflow')
            ratio = rank / selected if alpha is not None else 1.0
            error = _factor_error(a, b, new_down.reshape(selected, -1).double(),
                                  new_up.reshape(up.shape[0], selected).double(), ratio, norm)
            ideal = 1.0 if total == 0 else float(energy[:selected].sum()) / total
            if not math.isfinite(error) or not math.isfinite(ideal):
                raise ValueError("LoRA error calculation exceeds the supported float64 numeric range")
            if bounded and error > error_limit:
                continue
        old_scale = alpha / rank if alpha is not None else 1.0
        new_scale = alpha / selected if alpha is not None else 1.0
        return new_down, new_up, dict(
            old_rank=rank, new_rank=selected, relative_error=error,
            retained_energy=1-error*error, ideal_retained_energy=ideal,
            dtype_down=str(new_down.dtype).removeprefix('torch.'), dtype_up=str(new_up.dtype).removeprefix('torch.'),
            alpha=alpha, old_scale=old_scale, new_scale=new_scale,
            rank_policy=policy, rank_value=value,
            old_bytes=down.numel()*down.element_size()+up.numel()*up.element_size(),
            new_bytes=new_down.numel()*new_down.element_size()+new_up.numel()*new_up.element_size(),
        )
    raise ValueError(f"Requested error/energy threshold is not achievable within max_rank={cap} after storage-dtype rounding")


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_parts(tensors, metadata):
    """Safetensors header/byte views; no payload-sized serialization buffer."""
    header, offset, chunks = {"__metadata__": metadata}, 0, []
    for name in sorted(tensors, key=lambda key: (-_DTYPES[tensors[key].dtype][0], key)):
        value = tensors[name]
        size = value.numel() * value.element_size()
        header[name] = dict(dtype=_DTYPES[value.dtype][1], shape=list(value.shape), data_offsets=[offset, offset+size])
        offset += size
        chunks.append(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    raw = json.dumps(header, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    raw += b' ' * (-len(raw) % 8)
    return len(raw).to_bytes(8, 'little'), raw, chunks, 8+len(raw)+offset


def _hashes(tensors, metadata):
    # Same additional-networks convention as toolkit.metadata: SHA256 of data,
    # plus the 64KiB legacy window at 1MiB in an ss_*-metadata-only archive.
    prefix, header, chunks, _ = _archive_parts(tensors, {k: v for k, v in metadata.items() if k.startswith('ss_')})
    digest, legacy = hashlib.sha256(), hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    offset = 0
    for chunk in [prefix, header, *chunks]:
        start, end = max(0, 0x100000-offset), min(len(chunk), 0x110000-offset)
        if start < end:
            legacy.update(chunk[start:end])
        offset += len(chunk)
    return dict(sshs_model_hash=digest.hexdigest(), sshs_legacy_hash=legacy.hexdigest()[:8])


def _check_metadata(metadata, formats):
    def enabled(value):
        return value not in (False, None, 0, 'false', 'False', '0')

    def check(key, value):
        if key in ('ss_network_module', 'network_module', 'network_type', 'peft_type', 'algo'):
            label = str(value).lower()
            if any(token in label for token in ('oft', 'ia3', 'loha', 'lokr', 'dora', 'lycoris')):
                family = next((token for token in ('loha', 'lokr', 'dora') if token in label), None)
                if family not in formats and not ('lycoris' in label and formats <= {'loha', 'lokr'}):
                    raise ValueError(f"Adapter metadata {key}={value} conflicts with recognized tensor formats {sorted(formats)}")
        if key.lower() in ('use_rslora', 'rs_lora', 'rank_stabilized', 'use_rank_stabilized_lora') and enabled(value):
            raise ValueError(f"Unsupported rsLoRA scaling in metadata {key}")
        if key.lower() in ('use_dora', 'loha_dora') and enabled(value) and formats != {'dora'}:
            raise ValueError('DoRA metadata requires recognized magnitude tensors; DoHa conversion is unsupported')
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                check(nested_key, nested_value)

    for key, raw in metadata.items():
        if key.startswith('aitk_lora_trimming'):
            continue  # Provenance can mention a previously rejected format.
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            value = raw
        check(key, value)


def _layers(keys):
    """Discover complete logical layers without choosing ambiguous conventions."""
    found, consumed = {}, set()

    def add(name, kind, required):
        if name in found:
            raise ValueError(f'Ambiguous factor conventions for layer {name}')
        missing = set(required) - keys
        if missing:
            raise ValueError(f'Incomplete {kind} factors for {name}: missing {sorted(missing)}')
        extras = {name + '.alpha'} & keys
        magnitudes = {name + suffix for suffix in _MAGNITUDES} & keys
        if len(magnitudes) > 1:
            raise ValueError(f'Ambiguous DoRA magnitude tensors for {name}')
        if magnitudes and kind != 'lora':
            raise ValueError(f'{name}: DoHa/LoKr magnitude conversion is unsupported')
        kind = 'dora' if magnitudes else kind
        layer_keys = set(required) | extras | magnitudes
        consumed.update(layer_keys)
        found[name] = dict(name=name, kind=kind, keys=layer_keys,
                           factors=list(required), magnitude=next(iter(magnitudes), None))

    for key in sorted(keys):
        pair = next((pair for pair in _PAIRS if key.endswith(pair[0])), None)
        if pair:
            name = key[:-len(pair[0])]
            add(name, 'lora', [key, name + pair[1]])
        elif key.endswith('.hada_w1_a'):
            name = key[:-len('.hada_w1_a')]
            add(name, 'loha', [name + '.' + part for part in ('hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b')])
    kron_names = {key.rsplit('.lokr_', 1)[0] for key in keys if '.lokr_' in key}
    for name in sorted(kron_names):
        required = []
        for index in (1, 2):
            full = name + f'.lokr_w{index}'
            a, b = full + '_a', full + '_b'
            if full in keys and ({a, b} & keys):
                raise ValueError(f'{name}: ambiguous full/factored LoKr component {index}')
            required.extend([full] if full in keys else [a, b])
        add(name, 'lokr', required)
    if not found:
        raise ValueError('No supported LoRA/DoRA/LoHa/LoKr factor layers found')
    if keys - consumed:
        raise ValueError('Unrecognized/unsupported tensors (CP/Tucker, biases and scalars are not inferred): '
                         + ', '.join(sorted(keys - consumed)[:8]))
    return list(found.values())


def _conversion_factors(parts, kind, alpha, max_working_mb):
    """Exact factorized 2D LyCORIS update, before storage-dtype rounding."""
    if any(t.ndim != 2 or any(size < 1 for size in t.shape) or t.dtype not in _FLOATS
           for t in parts.values()):
        raise ValueError('LoHa/LoKr conversion supports positive-shape floating 2D factors only; CP/Tucker/convolution layouts are unsupported')
    if any(not torch.isfinite(t).all() for t in parts.values()):
        raise ValueError('Adapter factors contain nonfinite values')
    dtype = next(iter(parts.values())).dtype
    for tensor in parts.values():
        dtype = torch.promote_types(dtype, tensor.dtype)
    if kind == 'loha':
        b1, a1, b2, a2 = (parts[key] for key in ('hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b'))
        if b1.shape[1] != a1.shape[0] or b2.shape[1] != a2.shape[0] or b1.shape[0] != b2.shape[0] or a1.shape[1] != a2.shape[1]:
            raise ValueError('LoHa factor shapes do not describe matching Hadamard matrices')
        if a1.shape[0] != a2.shape[0]:
            raise ValueError('LoHa component ranks differ; alpha/rank loader semantics are ambiguous')
        stored_rank = a1.shape[0]
        rank, rows, columns = a1.shape[0]*a2.shape[0], b1.shape[0], a1.shape[1]
        old_scale = alpha/stored_rank if alpha is not None else 1.0
    else:
        shapes, ranks, stored = [], [], []
        for index in (1, 2):
            prefix = f'lokr_w{index}'
            if prefix in parts:
                tensor = parts[prefix]
                shapes.append(tuple(tensor.shape))
                ranks.append(min(tensor.shape))
            else:
                b, a = parts[prefix+'_a'], parts[prefix+'_b']
                if b.shape[1] != a.shape[0]:
                    raise ValueError(f'LoKr component {index} ranks do not match')
                shapes.append((b.shape[0], a.shape[1]))
                ranks.append(a.shape[0])
                stored.append(a.shape[0])
        if len(set(stored)) > 1:
            raise ValueError('LoKr decomposed ranks differ; alpha/rank loader semantics are ambiguous')
        stored_rank = stored[0] if stored else None
        old_scale = alpha/stored_rank if alpha is not None and stored_rank is not None else 1.0
        rank, rows, columns = ranks[0]*ranks[1], shapes[0][0]*shapes[1][0], shapes[0][1]*shapes[1][1]
    # Includes expanded FP64 factors, QR/SVD and doubled difference-core work;
    # this is a conservative admission estimate, not a host-memory guarantee.
    estimate = 8*(8*(rows+columns)*rank + 32*rank*rank + 4*sum(t.numel() for t in parts.values()))
    if estimate > max_working_mb*1024**2:
        raise ValueError(f'Factorized conversion working-memory estimate {estimate} bytes exceeds max_working_mb={max_working_mb}; raise the explicit budget only if sufficient CPU memory is available')
    if kind == 'loha':
        b1, a1, b2, a2 = (tensor.double() for tensor in (b1, a1, b2, a2))
        up = (b1[:, :, None] * b2[:, None, :]).reshape(rows, rank)
        down = (a1[:, None, :] * a2[None, :, :]).reshape(rank, columns)
    else:
        components = []
        for index in (1, 2):
            prefix = f'lokr_w{index}'
            if prefix in parts:
                u, s, vh = torch.linalg.svd(parts[prefix].double(), full_matrices=False)
                components.append((u*s.sqrt()[None], s.sqrt()[:, None]*vh))
            else:
                components.append((parts[prefix+'_a'].double(), parts[prefix+'_b'].double()))
        up = torch.kron(components[0][0].contiguous(), components[1][0].contiguous())
        down = torch.kron(components[0][1].contiguous(), components[1][1].contiguous())
    # Preserve serialized alpha while converting its input runtime scale to the
    # ordinary-LoRA expanded alpha/rank scale. Missing alpha stays unit scale.
    output_scale = alpha/rank if alpha is not None else 1.0
    up = up * (old_scale/output_scale)
    return down, up, dtype, stored_rank, old_scale, estimate


def _base_index(path):
    """Index exact/dotted and recognized kohya-flat names; never guess a model."""
    index = {}
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        if any(any(marker in key.lower() for marker in ('.comfy_quant', 'weight_scale', '.quant_state', '.weight._', '.scale_weight')) for key in keys):
            raise ValueError('DoRA baking requires a plain floating base; quantization/scaling auxiliary tensors are unsupported')
        for key in keys:
            if not key.endswith('.weight'):
                continue
            if key.removesuffix('.weight')+'.scale' in keys:
                raise ValueError(f'DoRA baking does not infer auxiliary base scale tensors for {key}')
            weight = handle.get_slice(key)
            if weight.get_dtype() not in ('F16', 'BF16', 'F32', 'F64'):
                raise ValueError(f'DoRA baking requires a plain floating base: {key} is {weight.get_dtype()}')
            normalized = _normalized_module_name(key)
            for label in {normalized, normalized.replace('.', '_')}:
                index.setdefault(label, set()).add(key)
    return index


def _base_weight_key(name, index):
    label = name
    for prefix in ('lora_transformer_', 'lora_unet_', 'lycoris_'):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    label = _normalized_module_name(label.replace('$$', '.'))
    matches = index.get(label, set())
    if len(matches) != 1:
        raise ValueError(f'{name}: matching base weight must be unique; found {sorted(matches)}. Use a base checkpoint with matching adapter module names')
    return next(iter(matches))


def _bake_dora_factors(base_path, base_key, down, up, magnitude, alpha, convention, max_working_mb):
    if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
        raise ValueError('DoRA baking currently supports linear factor pairs only')
    rows, columns = up.shape[0], down.shape[1]
    if convention == 'comfy' and tuple(magnitude.shape) != (rows, 1):
        raise ValueError('Comfy DoRA baking requires unambiguous output-axis magnitude shape (output_rows, 1); input-axis/1D layouts are unsupported')
    rank = min(rows, columns)
    estimate = 8*(12*rows*columns + 32*rank*rank + 8*(rows+columns)*rank)
    if estimate > max_working_mb*1024**2:
        raise ValueError(f'DoRA baking working-memory estimate {estimate} bytes exceeds max_working_mb={max_working_mb}; dense effective-delta SVD requires an explicit sufficient CPU budget')
    with safe_open(str(base_path), framework='pt', device='cpu') as handle:
        stored = handle.get_slice(base_key)
        if tuple(stored.get_shape()) != (rows, columns):
            raise ValueError(f'{base_key}: base shape {stored.get_shape()} does not match adapter {(rows, columns)}')
        weight = handle.get_tensor(base_key)
    if not torch.isfinite(weight).all() or not torch.isfinite(down).all() or not torch.isfinite(up).all():
        raise ValueError('DoRA baking inputs contain nonfinite values')
    scale = alpha/down.shape[0] if alpha is not None else 1.0
    adapted = weight.double() + (up.double() @ down.double())*scale
    if convention == 'comfy':
        # Standard Comfy model patching calculates weights in float32, even
        # when the checkpoint stores bf16/fp16. Keep its epsilon explicit.
        denominator = weight.double().norm(dim=1) + torch.finfo(torch.float32).eps
        ratio = magnitude.double().reshape(-1)/denominator
    else:
        norm = adapted.norm(dim=1)
        zero = norm == 0
        denominator = torch.where(zero, torch.ones_like(norm), norm)
        ratio = torch.where(zero, torch.ones_like(norm), magnitude.double().reshape(-1)/denominator)
    effective_delta = adapted*ratio[:, None] - weight.double()
    if not torch.isfinite(effective_delta).all():
        raise ValueError('DoRA effective delta exceeds the supported float64 numeric range')
    u, singular, vh = torch.linalg.svd(effective_delta, full_matrices=False)
    roots = singular.sqrt()
    # Preserve explicit alpha, compensating the ordinary expanded alpha/rank
    # scale. Missing alpha remains the standalone unit-scale convention.
    compensation = rank/alpha if alpha is not None else 1.0
    return roots[:, None]*vh, u*(roots*compensation)[None], torch.promote_types(down.dtype, up.dtype), estimate


def _check_missing_alpha(path, metadata, rank):
    for key in ('ss_network_alpha', 'network_alpha', 'lora_alpha'):
        if key in metadata:
            try:
                alpha = float(metadata[key])
            except ValueError as error:
                raise ValueError(f"Missing per-layer alpha with ambiguous metadata {key}") from error
            if alpha != rank:
                raise ValueError(f"Missing per-layer alpha conflicts with metadata {key}={alpha}; export self-contained factors/scaling first")
    # Historical training configuration is not a standalone adapter contract.
    # Reject contradictory external scaling rather than silently adopting it.
    for filename in ('config.yaml', '.job_config.json'):
        candidate = path.parent / filename
        if not candidate.is_file():
            continue
        if filename.endswith('.yaml'):
            import yaml
            config = yaml.safe_load(candidate.read_text())
        else:
            config = json.loads(candidate.read_text())
        if isinstance(config, dict):
            processes = config.get('config', {}).get('process', [])
            for process in processes:
                network = process.get('network') or {}
                for key in ('linear_alpha', 'conv_alpha'):
                    if network.get(key) is not None and float(network[key]) != rank:
                        raise ValueError(f"Missing per-layer alpha conflicts with {candidate}:{key}; export a self-contained standalone LoRA first")


def _destination(output, source):
    destination = Path(output).expanduser().absolute()
    if destination.suffix.lower() != '.safetensors':
        raise ValueError("Output must be a new .safetensors file")
    if destination.resolve() == source.resolve() or os.path.lexists(destination):
        raise FileExistsError(f"Output already exists or aliases the source: {destination}")
    return destination


def trim_lora(path, *, retained_energy=None, relative_error=None, min_rank=1, max_rank=None,
              output=None, expected_preview=None, progress: Callable[[str], None] | None = None,
              rank_policy=None, rank_value=None, convert_to_lora=False, max_working_mb=512,
              bake_dora_to_lora=False, base_model_path=None, dora_convention=None):
    """Preview or atomically publish a new adapter, never replacing files."""
    options = _options(retained_energy, relative_error, min_rank, max_rank, rank_policy, rank_value)
    if type(convert_to_lora) is not bool:
        raise ValueError('convert_to_lora must be a boolean')
    if type(bake_dora_to_lora) is not bool:
        raise ValueError('bake_dora_to_lora must be a boolean')
    if bake_dora_to_lora:
        if not base_model_path or dora_convention not in ('comfy', 'toolkit'):
            raise ValueError('DoRA baking requires base_model_path and an explicit dora_convention: comfy or toolkit')
        if convert_to_lora:
            raise ValueError('Choose DoRA baking OR LoHa/LoKr conversion, not both')
    elif base_model_path is not None or dora_convention is not None:
        raise ValueError('base_model_path/dora_convention are only valid with bake_dora_to_lora=True')
    if isinstance(max_working_mb, bool) or not isinstance(max_working_mb, (int, float)) or not math.isfinite(max_working_mb) or max_working_mb <= 0:
        raise ValueError('max_working_mb must be a positive finite number')
    options.update(convert_to_lora=convert_to_lora, max_working_mb=max_working_mb,
                   bake_dora_to_lora=bake_dora_to_lora, dora_convention=dora_convention)
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != '.safetensors':
        raise ValueError("Input must be a standalone .safetensors LoRA; pickle/adapter directories are not loaded")
    if (source.parent / 'adapter_config.json').exists():
        raise ValueError("PEFT adapter_config.json sidecars carry external rank/scaling; export a self-contained standalone LoRA first")
    destination = _destination(output, source) if output is not None else None
    if expected_preview is not None and output is None:
        raise ValueError("expected_preview is only valid with an output")
    source_hash = _sha256(source)
    base_info, base_index = None, None
    if bake_dora_to_lora:
        base_path = Path(base_model_path).expanduser().resolve(strict=True)
        if not base_path.is_file() or base_path.suffix.lower() != '.safetensors' or base_path == source:
            raise ValueError('DoRA baking requires a separate local single floating .safetensors base checkpoint')
        base_info = dict(path=str(base_path), sha256=_sha256(base_path))
        base_index = _base_index(base_path)
    token = hashlib.sha256(_json(dict(source_sha256=source_hash, options=options,
        base_model=base_info, algorithm=ALGORITHM)).encode()).hexdigest()
    if expected_preview is not None and expected_preview != token:
        raise ValueError("Preview is stale: source or trimming settings changed; preview again before writing")
    result_tensors, records = {}, []
    with safe_open(str(source), framework='pt', device='cpu') as handle:
        keys, metadata = set(handle.keys()), dict(handle.metadata() or {})
        layers = _layers(keys)
        formats = {layer['kind'] for layer in layers}
        _check_metadata(metadata, formats)
        lycoris_conversion = bool(formats & {'loha', 'lokr'})
        converted = lycoris_conversion or bake_dora_to_lora
        if lycoris_conversion and not convert_to_lora:
            raise ValueError('LoHa/LoKr require explicit convert_to_lora=True (--convert-to-lora); output will be ordinary LoRA')
        if converted and len(formats) != 1:
            raise ValueError('Mixed adapter families cannot be converted with a single unambiguous output-format metadata contract')
        if bake_dora_to_lora and formats != {'dora'}:
            raise ValueError('DoRA baking requires a checkpoint with recognized DoRA magnitude tensors on every layer')
        original_metadata = dict(metadata)
        input_tensor_bytes = 0
        for layer in layers:
            name, kind = layer['name'], layer['kind']
            if progress:
                progress(f"Reducing {name} ({len(records)+1}/{len(layers)}) on CPU")
            tensors = {key: handle.get_tensor(key) for key in layer['keys']}
            input_tensor_bytes += sum(t.numel()*t.element_size() for t in tensors.values())
            alpha_key = name + '.alpha'
            alpha = None
            if alpha_key in tensors:
                alpha_tensor = tensors[alpha_key]
                if alpha_tensor.numel() != 1 or alpha_tensor.dtype not in _DTYPES:
                    raise ValueError(f"Invalid alpha tensor: {alpha_key}")
                alpha = float(alpha_tensor.item())
                if not math.isfinite(alpha) or alpha == 0:
                    raise ValueError(f'{name}: alpha must be finite and nonzero; alpha=0 has ambiguous loader semantics')
            output_name = name
            base_key = None
            if kind in ('loha', 'lokr'):
                parts = {key[len(name)+1:]: tensors[key] for key in layer['factors']}
                down, up, output_dtype, stored_rank, input_scale, estimate = _conversion_factors(parts, kind, alpha, max_working_mb)
                if alpha is None and stored_rank is not None:
                    _check_missing_alpha(source, metadata, stored_rank)
                if name.startswith('lycoris_'):
                    output_name = 'lora_transformer_' + name[len('lycoris_'):]
                down_key, up_key = output_name + '.lora_down.weight', output_name + '.lora_up.weight'
            else:
                down_key, up_key = layer['factors']
                down, up = tensors[down_key], tensors[up_key]
                output_dtype, stored_rank, input_scale, estimate = None, down.shape[0], None, None
                if alpha is None:
                    _check_missing_alpha(source, metadata, stored_rank)
                if kind == 'dora':
                    magnitude = tensors[layer['magnitude']]
                    if down.ndim != 2 or up.ndim != 2 or magnitude.ndim < 1 or magnitude.dtype not in _FLOATS or magnitude.shape[0] != up.shape[0] or magnitude.numel() != up.shape[0] or not torch.isfinite(magnitude).all():
                        raise ValueError(f'{name}: DoRA magnitude must be a finite per-output-channel vector (or singleton trailing dimensions) for linear factors')
                    if bake_dora_to_lora:
                        base_key = _base_weight_key(name, base_index)
                        input_scale = alpha/stored_rank if alpha is not None else 1.0
                        down, up, output_dtype, estimate = _bake_dora_factors(base_path, base_key,
                            down, up, magnitude, alpha, dora_convention, max_working_mb)
                    else:
                        result_tensors[layer['magnitude']] = magnitude.clone().contiguous()
            try:
                new_down, new_up, record = trim_lora_factors(down, up, alpha=alpha,
                    output_dtype=output_dtype, **_factor_options(options))
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
            if down_key in result_tensors or up_key in result_tensors:
                raise ValueError(f'Conversion produces duplicate output layer {output_name}')
            result_tensors[down_key], result_tensors[up_key] = new_down, new_up
            if alpha is not None:
                result_tensors[output_name+'.alpha'] = alpha_tensor.clone().contiguous()
            elif converted:
                # A converted standalone file must describe unit scaling even
                # when historical global alpha/config tags refer to native
                # ranks. This changes no factor values or effective update.
                output_alpha = torch.tensor(record['new_rank'], dtype=torch.float32)
                if float(output_alpha) != record['new_rank']:
                    raise ValueError('Converted rank cannot be represented exactly as a float32 alpha')
                result_tensors[output_name+'.alpha'] = output_alpha
                record['alpha'] = float(output_alpha)
            if converted:
                record['old_bytes'] = sum(tensors[key].numel()*tensors[key].element_size() for key in layer['factors'])
                record['old_scale'] = input_scale
            records.append(dict(name=name, output_name=output_name, input_format=kind,
                output_format='dora' if kind == 'dora' and not bake_dora_to_lora else 'lora',
                metric_scope='pre_normalization_direction' if kind == 'dora' and not bake_dora_to_lora else 'effective_update',
                original_alpha=alpha,
                original_stored_rank=stored_rank, expanded_rank=down.shape[0] if converted else None,
                conversion_working_bytes=estimate, baked=bake_dora_to_lora, base_weight_key=base_key,
                dora_convention=dora_convention,
                normalization_epsilon=float(torch.finfo(torch.float32).eps) if bake_dora_to_lora and dora_convention == 'comfy' else None, **record))
    changed_metadata = []
    if converted:
        # These are loading instructions, not historical user tags. Preserve
        # the exact originals in provenance while describing the actual output.
        replacements = {'ss_network_module': 'networks.lora'}
        for key in ('network_module', 'network_type', 'peft_type', 'algo'):
            if key in metadata:
                replacements[key] = 'networks.lora' if key == 'network_module' else ('LORA' if key == 'peft_type' else 'lora')
        for key in ('ss_network_args', 'network_args'):
            if key in metadata:
                replacements[key] = _json({'algo': 'lora'})
        for key in ('use_dora', 'loha_dora'):
            if key in metadata:
                replacements[key] = 'false'
        for key, value in replacements.items():
            if metadata.get(key) != value:
                changed_metadata.append(dict(key=key, before=metadata.get(key), after=value))
                metadata[key] = value
    originals = {key: metadata[key] for key in _HASH_KEYS if key in metadata}
    if originals:
        replacement_hashes = _hashes(result_tensors, metadata)
        for key, original in originals.items():
            metadata[key] = replacement_hashes[key]
            changed_metadata.append(dict(key=key, before=original, after=metadata[key]))
    provenance_key = 'aitk_lora_trimming'
    while provenance_key in metadata:
        provenance_key += '_previous'
    metadata[provenance_key] = _json(dict(algorithm=ALGORITHM, source_sha256=source_hash,
        options=options, original_hashes=originals,
        base_model=base_info,
        original_format_metadata=original_metadata if converted else None,
        conversions=[{key: record[key] for key in ('name', 'output_name', 'input_format', 'output_format', 'original_stored_rank', 'expanded_rank', 'original_alpha', 'alpha')} for record in records] if converted else [],
        ranks={record['name']: record['new_rank'] for record in records}))
    changed_metadata.append(dict(key=provenance_key, before=None, after=metadata[provenance_key]))
    _, _, _, predicted_bytes = _archive_parts(result_tensors, metadata)
    if _sha256(source) != source_hash:
        raise RuntimeError("Source changed during trimming; no output published")
    if base_info is not None and _sha256(base_path) != base_info['sha256']:
        raise RuntimeError('Base checkpoint changed during baking; no output published')
    report = dict(schema_version=1, algorithm=ALGORITHM, input_path=str(source), source_sha256=source_hash,
        preview_token=token, options=options, input_bytes=source.stat().st_size,
        estimated_output_bytes=predicted_bytes, tensor_bytes_before=input_tensor_bytes,
        tensor_bytes_after=sum(t.numel()*t.element_size() for t in result_tensors.values()),
        layers=records, metadata_changes=changed_metadata,
        input_formats=sorted(formats), output_formats=sorted({r['output_format'] for r in records}),
        base_model=base_info,
        warnings=['Training rank/alpha metadata is preserved as history; actual output ranks are per-layer.',
                  'Numerical energy/error retention does not establish identical images or visual quality; compare normal prompts.'],
        written=False, output_path=None)
    if 'dora' in formats and not bake_dora_to_lora:
        report['warnings'].append('DoRA magnitude tensors are unchanged. Reported retention/error covers only the pre-normalization direction update, not final base-dependent DoRA weights or images.')
    if lycoris_conversion:
        report['warnings'].append('Explicit LoHa/LoKr conversion produces ordinary LoRA, not a resumable native-format adapter. old_rank is the expanded factor rank; original_stored_rank identifies the stored native rank, or null for all-full LoKr.')
    if bake_dora_to_lora:
        report['warnings'].append('DoRA is baked to an ordinary LoRA at full strength using the explicitly selected normalization convention and base checkpoint. Retention/error measures that baked effective delta. Native DoRA strengths can be nonlinear; other strengths and low-precision loader arithmetic are not guaranteed equivalent. old_rank is the full baked factor rank, not the stored direction rank.')
    if options['rank_policy'] not in ('energy', 'relative_error'):
        report['warnings'].append('This spectrum/target-rank policy reports measured rounding error; it does not enforce the energy/error retention guarantee of the bounded policies.')
    if predicted_bytes >= report['input_bytes']:
        report['warnings'].append('The output is not smaller than the input, including metadata and format conversion overhead.')
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
        os.close(descriptor)
        try:
            save_file(result_tensors, temporary, metadata=metadata)
            if os.stat(temporary).st_size != predicted_bytes:
                raise RuntimeError("Safetensors serialized size differs from preview estimate; no output published")
            with open(temporary, 'rb') as prepared:
                os.fsync(prepared.fileno())
            if _sha256(source) != source_hash:
                raise RuntimeError("Source changed before publication; no output published")
            if base_info is not None and _sha256(base_path) != base_info['sha256']:
                raise RuntimeError('Base checkpoint changed before publication; no output published')
            # Hard link publication is atomic and fails if ANY entry already
            # occupies the destination, including a dangling symlink.
            os.link(temporary, destination)
            report.update(written=True, output_path=str(destination))
        finally:
            os.unlink(temporary)
    return report


def format_trimming_text(report):
    lines = [f"{'Wrote' if report['written'] else 'Dry run:'} {report['output_path'] or report['input_path']}",
        f"Bytes: {report['input_bytes']} -> {report['estimated_output_bytes']} (including metadata/header)"]
    for layer in report['layers']:
        lines.append(f"{layer['name']} ({layer['input_format']} -> {layer['output_format']}, {layer['metric_scope']}): "
                     f"factor rank {layer['old_rank']} -> {layer['new_rank']}; "
                     f"relative error {layer['relative_error']:.8g}; "
                     f"energy {layer['retained_energy']:.8g}; {layer['dtype_down']}/{layer['dtype_up']}; "
                     f"alpha {layer['alpha']}; scale {layer['old_scale']:.8g} -> {layer['new_scale']:.8g}")
    if report['base_model'] is not None:
        lines.append(f"Base: {report['base_model']['path']} (SHA256 {report['base_model']['sha256']})")
    lines.extend(report['warnings'])
    return '\n'.join(lines)
