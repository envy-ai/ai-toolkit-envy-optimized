"""CPU-only, error-bounded trimming of standalone ordinary LoRA safetensors.

No base models, pickle deserialization, network calls, or dense B@A updates.
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

ALGORITHM = "qr_core_svd_v1"
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


def _options(retained_energy, relative_error, min_rank, max_rank):
    if retained_energy is not None and relative_error is not None:
        raise ValueError("Choose retained_energy OR relative_error, not both")
    if retained_energy is None and relative_error is None:
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
    return dict(retained_energy=retained_energy, relative_error=relative_error,
                min_rank=min_rank, max_rank=max_rank)


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
                      retained_energy=None, relative_error=None, min_rank=1, max_rank=None):
    """Return CPU factors and metrics, preserving original storage dtypes/scaling.

    Explicit alpha is kept unchanged: compensate the factor product for its new
    alpha/rank runtime scale. Missing alpha means standalone unit scale.
    """
    options = _options(retained_energy, relative_error, min_rank, max_rank)
    if down.device.type != "cpu" or up.device.type != "cpu":
        raise ValueError("LoRA trimming accepts CPU tensors only")
    if down.dtype not in _FLOATS or up.dtype not in _FLOATS:
        raise ValueError("LoRA factors must be float16, bfloat16, float32 or float64")
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
    error_limit = (math.sqrt(1 - options['retained_energy']) if options['retained_energy'] is not None
                   else options['relative_error'])
    cap = rank if max_rank is None else max_rank
    for selected in range(min_rank, cap + 1):
        if selected == rank:
            # Avoid introducing rounding/decomposition error when no reduction
            # satisfies the limit. Original tensors are returned bit-exactly.
            new_down, new_up, error, ideal = down.detach().clone(), up.detach().clone(), 0.0, 1.0
        else:
            tail = float(energy[selected:].sum())
            if total and math.sqrt(tail / total) > error_limit:
                continue
            count = min(selected, len(singular))
            roots = singular[:count].sqrt()
            compensation = math.sqrt(selected / rank) if alpha is not None else 1.0
            small_b = (qb @ u[:, :count]) * (roots * compensation)[None]
            small_a = (roots * compensation)[:, None] * (vh[:count] @ qa.T)
            if count < selected:
                small_b = torch.nn.functional.pad(small_b, (0, selected-count))
                small_a = torch.nn.functional.pad(small_a, (0, 0, 0, selected-count))
            new_down = small_a.reshape(selected, *down.shape[1:]).to(down.dtype).contiguous()
            new_up = small_b.reshape(up.shape[0], selected, *up.shape[2:]).to(up.dtype).contiguous()
            if not torch.isfinite(new_down).all() or not torch.isfinite(new_up).all():
                continue
            ratio = rank / selected if alpha is not None else 1.0
            error = _factor_error(a, b, new_down.reshape(selected, -1).double(),
                                  new_up.reshape(up.shape[0], selected).double(), ratio, norm)
            ideal = 1.0 if total == 0 else float(energy[:selected].sum()) / total
            if not math.isfinite(error) or not math.isfinite(ideal):
                raise ValueError("LoRA error calculation exceeds the supported float64 numeric range")
            if error > error_limit:
                continue
        old_scale = alpha / rank if alpha is not None else 1.0
        new_scale = alpha / selected if alpha is not None else 1.0
        return new_down, new_up, dict(
            old_rank=rank, new_rank=selected, relative_error=error,
            retained_energy=1-error*error, ideal_retained_energy=ideal,
            dtype_down=str(down.dtype).removeprefix('torch.'), dtype_up=str(up.dtype).removeprefix('torch.'),
            alpha=alpha, old_scale=old_scale, new_scale=new_scale,
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


def _check_metadata(metadata):
    def enabled(value):
        return value not in (False, None, 0, 'false', 'False', '0')

    def check(key, value):
        if key in ('ss_network_module', 'network_module', 'network_type', 'peft_type', 'algo') and any(
                token in str(value).lower() for token in ('dora', 'loha', 'lokr', 'lycoris', 'oft')):
            raise ValueError(f"Unsupported adapter type in metadata {key}: {value}; only ordinary LoRA is supported")
        if key.lower() in ('use_dora', 'use_rslora', 'rs_lora', 'rank_stabilized', 'use_rank_stabilized_lora') and enabled(value):
            raise ValueError(f"Unsupported DoRA/rsLoRA scaling in metadata {key}")
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
              output=None, expected_preview=None, progress: Callable[[str], None] | None = None):
    """Preview or atomically publish a new ordinary LoRA, never replacing files."""
    options = _options(retained_energy, relative_error, min_rank, max_rank)
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != '.safetensors':
        raise ValueError("Input must be a standalone .safetensors LoRA; pickle/adapter directories are not loaded")
    if (source.parent / 'adapter_config.json').exists():
        raise ValueError("PEFT adapter_config.json sidecars carry external rank/scaling; export a self-contained standalone LoRA first")
    destination = _destination(output, source) if output is not None else None
    if expected_preview is not None and output is None:
        raise ValueError("expected_preview is only valid with an output")
    source_hash = _sha256(source)
    token = hashlib.sha256(_json(dict(source_sha256=source_hash, options=options, algorithm=ALGORITHM)).encode()).hexdigest()
    if expected_preview is not None and expected_preview != token:
        raise ValueError("Preview is stale: source or trimming settings changed; preview again before writing")
    result_tensors, records, consumed = {}, [], set()
    with safe_open(str(source), framework='pt', device='cpu') as handle:
        keys, metadata = set(handle.keys()), dict(handle.metadata() or {})
        _check_metadata(metadata)
        if any(any(token in key.lower() for token in ('dora_scale', 'magnitude', 'lokr_', 'hada_', 'lora_mid', 'rslora')) for key in keys):
            raise ValueError("DoRA/LoHa/LoKr/rsLoRA/mid factors are unsupported. DoRA needs base-weight-dependent normalization; trim an ordinary LoRA instead")
        pairs, layer_names = [], set()
        for key in sorted(keys):
            suffix_pair = next((pair for pair in _PAIRS if key.endswith(pair[0])), None)
            if suffix_pair is None:
                continue
            name = key[:-len(suffix_pair[0])]
            if name in layer_names:
                raise ValueError(f"Ambiguous factor conventions for layer {name}")
            layer_names.add(name)
            other = name + suffix_pair[1]
            alpha_key = name + '.alpha'
            if other not in keys:
                raise ValueError(f"Incomplete LoRA factor pair for {name}: missing {other}")
            pair_keys = {key, other, *([alpha_key] if alpha_key in keys else [])}
            if consumed.intersection(pair_keys):
                raise ValueError(f"Ambiguous factor conventions for layer {name}")
            consumed.update(pair_keys)
            pairs.append((name, key, other, alpha_key if alpha_key in keys else None))
        if not pairs:
            raise ValueError("No supported ordinary LoRA factor pairs found")
        if keys-consumed:
            raise ValueError(f"Unrecognized/unsupported tensors (bias, scalar and other adapter formats are not inferred): {', '.join(sorted(keys-consumed)[:8])}")
        input_tensor_bytes = 0
        for name, down_key, up_key, alpha_key in pairs:
            if progress:
                progress(f"Trimming {name} ({len(records)+1}/{len(pairs)}) on CPU")
            down, up = handle.get_tensor(down_key), handle.get_tensor(up_key)
            alpha = None
            if alpha_key is not None:
                alpha_tensor = handle.get_tensor(alpha_key)
                if alpha_tensor.numel() != 1 or alpha_tensor.dtype not in _DTYPES:
                    raise ValueError(f"Invalid alpha tensor: {alpha_key}")
                alpha = float(alpha_tensor.item())
                result_tensors[alpha_key] = alpha_tensor.clone().contiguous()
                input_tensor_bytes += alpha_tensor.numel()*alpha_tensor.element_size()
            else:
                _check_missing_alpha(source, metadata, down.shape[0])
            try:
                new_down, new_up, record = trim_lora_factors(down, up, alpha=alpha, **options)
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
            result_tensors[down_key], result_tensors[up_key] = new_down, new_up
            input_tensor_bytes += record['old_bytes']
            records.append(dict(name=name, **record))
    changed_metadata = []
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
        ranks={record['name']: record['new_rank'] for record in records}))
    changed_metadata.append(dict(key=provenance_key, before=None, after=metadata[provenance_key]))
    _, _, _, predicted_bytes = _archive_parts(result_tensors, metadata)
    if _sha256(source) != source_hash:
        raise RuntimeError("Source changed during trimming; no output published")
    report = dict(schema_version=1, algorithm=ALGORITHM, input_path=str(source), source_sha256=source_hash,
        preview_token=token, options=options, input_bytes=source.stat().st_size,
        estimated_output_bytes=predicted_bytes, tensor_bytes_before=input_tensor_bytes,
        tensor_bytes_after=sum(t.numel()*t.element_size() for t in result_tensors.values()),
        layers=records, metadata_changes=changed_metadata,
        warnings=['Training rank/alpha metadata is preserved as history; actual output ranks are per-layer.',
                  'Numerical energy/error retention does not establish identical images or visual quality; compare normal prompts.'],
        written=False, output_path=None)
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
        lines.append(f"{layer['name']}: rank {layer['old_rank']} -> {layer['new_rank']}; "
                     f"relative error {layer['relative_error']:.8g}; "
                     f"energy {layer['retained_energy']:.8g}; {layer['dtype_down']}/{layer['dtype_up']}; "
                     f"alpha {layer['alpha']}; scale {layer['old_scale']:.8g} -> {layer['new_scale']:.8g}")
    lines.extend(report['warnings'])
    return '\n'.join(lines)
