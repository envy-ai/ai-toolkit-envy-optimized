from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
import yaml
from safetensors import safe_open


LORA_A_SUFFIXES = (".lora_A.weight", ".lora_down.weight")
LORA_B_SUFFIXES = (".lora_B.weight", ".lora_up.weight")
BASE_MODEL_METADATA_KEYS = (
    "ss_base_model",
    "base_model",
    "base_model_name_or_path",
)
STRIPPABLE_MODEL_PREFIXES = (
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "unet.",
    "model.",
)


@dataclass(frozen=True)
class LoraLayer:
    name: str
    a_key: str
    b_key: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    alpha_key: Optional[str] = None
    magnitude_key: Optional[str] = None

    @property
    def rank(self) -> int:
        return self.a_shape[0]

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.b_shape[0], *self.a_shape[1:])


@dataclass(frozen=True)
class BaseWeightRef:
    path: Path
    key: str
    shape: tuple[int, ...]
    scale_key: Optional[str] = None


@dataclass(frozen=True)
class BaseCandidate:
    label: str
    files: tuple[Path, ...]
    priority: int


@dataclass
class TrainingContext:
    config_path: Optional[Path] = None
    base_model: Optional[str] = None
    architecture: Optional[str] = None
    network_type: Optional[str] = None
    linear_alpha: Optional[float] = None
    conv_alpha: Optional[float] = None


def _parse_metadata_value(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def load_lora_metadata(path: Path) -> dict[str, Any]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw = handle.metadata() or {}
    return {key: _parse_metadata_value(value) for key, value in raw.items()}


def _shape(handle, key: str) -> tuple[int, ...]:
    return tuple(int(value) for value in handle.get_slice(key).get_shape())


def discover_lora_layers(path: Path) -> tuple[list[LoraLayer], list[str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        layers: list[LoraLayer] = []
        consumed: set[str] = set()

        for key in sorted(keys):
            a_suffix = next((suffix for suffix in LORA_A_SUFFIXES if key.endswith(suffix)), None)
            if a_suffix is None:
                continue
            name = key[:-len(a_suffix)]
            b_suffix = ".lora_B.weight" if a_suffix == ".lora_A.weight" else ".lora_up.weight"
            b_key = name + b_suffix
            if b_key not in keys:
                continue

            alpha_key = name + ".alpha"
            magnitude_candidates = (
                name + ".dora_scale",
                name + ".magnitude",
                name + ".lora_magnitude_vector.weight",
                name + ".lora_magnitude_vector",
            )
            magnitude_key = next(
                (candidate for candidate in magnitude_candidates if candidate in keys),
                None,
            )
            layers.append(
                LoraLayer(
                    name=name,
                    a_key=key,
                    b_key=b_key,
                    a_shape=_shape(handle, key),
                    b_shape=_shape(handle, b_key),
                    alpha_key=alpha_key if alpha_key in keys else None,
                    magnitude_key=magnitude_key,
                )
            )
            consumed.update((key, b_key))
            if alpha_key in keys:
                consumed.add(alpha_key)
            if magnitude_key is not None:
                consumed.add(magnitude_key)

        ignored = sorted(keys - consumed)
    return layers, ignored


def _find_training_process(document: Any) -> Optional[dict[str, Any]]:
    if not isinstance(document, dict):
        return None
    config = document.get("config", {})
    if not isinstance(config, dict):
        return None
    processes = config.get("process", [])
    if isinstance(processes, dict):
        processes = [processes]
    for process in processes:
        if isinstance(process, dict) and process.get("type") == "diffusion_trainer":
            return process
    return next((item for item in processes if isinstance(item, dict)), None)


def load_training_context(lora_path: Path) -> TrainingContext:
    candidates = (
        lora_path.parent / "config.yaml",
        lora_path.parent / ".job_config.json",
    )
    for config_path in candidates:
        if not config_path.is_file():
            continue
        try:
            if config_path.suffix == ".json":
                document = json.loads(config_path.read_text())
            else:
                document = yaml.safe_load(config_path.read_text())
        except (OSError, ValueError, yaml.YAMLError):
            continue
        process = _find_training_process(document)
        if process is None:
            continue
        model = process.get("model", {}) or {}
        network = process.get("network", {}) or {}
        return TrainingContext(
            config_path=config_path,
            base_model=model.get("name_or_path"),
            architecture=model.get("arch"),
            network_type=network.get("type"),
            linear_alpha=_optional_float(network.get("linear_alpha")),
            conv_alpha=_optional_float(network.get("conv_alpha")),
        )
    return TrainingContext()


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_base_model_reference(
    metadata: dict[str, Any],
    context: TrainingContext,
) -> tuple[Optional[str], Optional[str]]:
    for key in BASE_MODEL_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), f"metadata:{key}"
    if context.base_model:
        return context.base_model, f"training_config:{context.config_path}"
    return None, None


def _resolve_existing_reference(reference: str, lora_path: Path) -> Optional[Path]:
    raw = Path(reference).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend((lora_path.parent / raw, Path.cwd() / raw))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_cached_huggingface_snapshot(reference: str) -> Optional[Path]:
    if reference.count("/") != 1:
        return None
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(reference, local_files_only=True)).resolve()
    except Exception:
        return None


def _safetensor_files(path: Path) -> tuple[Path, ...]:
    if path.is_file() and path.suffix == ".safetensors":
        return (path.resolve(),)
    if not path.is_dir():
        return ()
    return tuple(
        sorted(
            item.resolve()
            for item in path.rglob("*.safetensors")
            if item.is_file()
        )
    )


def build_base_candidates(
    lora_path: Path,
    base_reference: Optional[str],
    explicit_base: Optional[str] = None,
    search_roots: Iterable[str | Path] = (),
) -> list[BaseCandidate]:
    candidates: list[BaseCandidate] = []

    def add(label: str, path: Optional[Path], priority: int) -> None:
        if path is None:
            return
        files = _safetensor_files(path)
        if files:
            candidates.append(BaseCandidate(label=label, files=files, priority=priority))

    if explicit_base:
        explicit_path = _resolve_existing_reference(explicit_base, lora_path)
        if explicit_path is None:
            explicit_path = _resolve_cached_huggingface_snapshot(explicit_base)
        add(f"explicit:{explicit_base}", explicit_path, 0)

    if base_reference:
        reference_path = _resolve_existing_reference(base_reference, lora_path)
        if reference_path is None:
            reference_path = _resolve_cached_huggingface_snapshot(base_reference)
        add(f"base_model:{base_reference}", reference_path, 10)

    for index, root in enumerate(search_roots):
        add(f"search_root:{root}", Path(root).expanduser().resolve(), 20 + index)

    toolkit_models = Path(__file__).resolve().parents[1] / "models"
    add(f"ai_toolkit_models:{toolkit_models}", toolkit_models, 100)

    unique: list[BaseCandidate] = []
    seen: set[tuple[Path, ...]] = set()
    for candidate in sorted(candidates, key=lambda item: item.priority):
        signature = candidate.files
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(candidate)
    return unique


def _normalized_module_name(value: str) -> str:
    if value.endswith(".weight"):
        value = value[:-len(".weight")]
    changed = True
    while changed:
        changed = False
        for prefix in STRIPPABLE_MODEL_PREFIXES:
            if value.startswith(prefix):
                value = value[len(prefix):]
                changed = True
                break
    return value


def _find_scale_key(keys: set[str], weight_key: str) -> Optional[str]:
    stem = weight_key[:-len(".weight")] if weight_key.endswith(".weight") else weight_key
    for candidate in (
        stem + ".weight_scale",
        stem + ".scale",
        weight_key + "_scale",
    ):
        if candidate in keys:
            return candidate
    return None


def _index_candidate(candidate: BaseCandidate) -> dict[str, list[BaseWeightRef]]:
    index: dict[str, list[BaseWeightRef]] = defaultdict(list)
    for path in candidate.files:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                for key in keys:
                    if not key.endswith(".weight"):
                        continue
                    ref = BaseWeightRef(
                        path=path,
                        key=key,
                        shape=_shape(handle, key),
                        scale_key=_find_scale_key(keys, key),
                    )
                    index[_normalized_module_name(key)].append(ref)
        except Exception:
            continue
    return index


def select_base_weights(
    layers: list[LoraLayer],
    candidates: list[BaseCandidate],
) -> tuple[Optional[BaseCandidate], dict[str, BaseWeightRef]]:
    best_candidate: Optional[BaseCandidate] = None
    best_matches: dict[str, BaseWeightRef] = {}

    for candidate in candidates:
        index = _index_candidate(candidate)
        matches: dict[str, BaseWeightRef] = {}
        for layer in layers:
            refs = index.get(_normalized_module_name(layer.name), [])
            shape_matches = [ref for ref in refs if ref.shape == layer.output_shape]
            if len(shape_matches) == 1:
                matches[layer.name] = shape_matches[0]
        if len(matches) > len(best_matches):
            best_candidate = candidate
            best_matches = matches
        if len(best_matches) == len(layers):
            break

    return best_candidate, best_matches


def classify_layer(name: str, a_shape: tuple[int, ...]) -> str:
    value = name.lower()
    if len(a_shape) > 2:
        return "convolution"
    if any(token in value for token in ("qkv_proj", "to_qkv", ".qkv.")):
        return "attention_qkv"
    if any(token in value for token in ("add_q_proj", "to_add_q")):
        return "attention_added_query"
    if any(token in value for token in ("add_k_proj", "to_add_k")):
        return "attention_added_key"
    if any(token in value for token in ("add_v_proj", "to_add_v")):
        return "attention_added_value"
    if any(token in value for token in (".q_proj", ".to_q", ".query")):
        return "attention_query"
    if any(token in value for token in (".k_proj", ".to_k", ".key")):
        return "attention_key"
    if any(token in value for token in (".v_proj", ".to_v", ".value")):
        return "attention_value"
    if any(token in value for token in ("out_proj", "to_out", "to_add_out")):
        return "attention_output"
    if any(token in value for token in (
        "adaln", "modulation", "time_embed", "temb", ".img_mod", ".txt_mod"
    )):
        return "conditioning_modulation"
    if any(token in value for token in (
        "gate_proj", "up_proj", ".fc1", "ff.net.0", "mlp.net.0", "mlp.proj_in"
    )):
        return "mlp_input_or_gate"
    if any(token in value for token in (
        "down_proj", ".fc2", "ff.net.2", "mlp.net.2", "mlp.proj_out"
    )):
        return "mlp_output"
    if "embed" in value:
        return "embedding"
    if "norm" in value:
        return "normalization"
    if any(token in value for token in ("proj_in", "input_proj", "patch_proj")):
        return "input_projection"
    if any(token in value for token in ("proj_out", "output_proj")):
        return "output_projection"
    return "other_linear"


def _lora_scale(
    handle,
    layer: LoraLayer,
    context: TrainingContext,
) -> tuple[float, str]:
    if layer.alpha_key is not None:
        alpha = float(handle.get_tensor(layer.alpha_key).float().item())
        return alpha / layer.rank, f"tensor:{layer.alpha_key}"

    is_conv = len(layer.a_shape) > 2
    alpha = context.conv_alpha if is_conv else context.linear_alpha
    if alpha is not None:
        return alpha / layer.rank, f"training_config:{context.config_path}"
    return 1.0, "implicit_peft_scale"


def _low_rank_delta_norm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> tuple[float, float, int]:
    rank = a.shape[0]
    a_flat = a.float().reshape(rank, -1)
    if b.numel() != b.shape[0] * rank:
        raise ValueError(f"unsupported LoRA B shape {tuple(b.shape)}")
    b_flat = b.float().reshape(b.shape[0], rank)
    gram_a = a_flat @ a_flat.T
    gram_b = b_flat.T @ b_flat
    norm_sq = float(torch.sum(gram_a * gram_b).clamp_min(0).item())
    fro_norm = math.sqrt(norm_sq) * abs(scale)
    elements = b.shape[0] * a_flat.shape[1]
    rms = fro_norm / math.sqrt(elements) if elements else 0.0
    return fro_norm, rms, elements


def _slice_scale(handle, key: str, start: int, end: int, rows: int) -> torch.Tensor:
    scale_slice = handle.get_slice(key)
    shape = tuple(scale_slice.get_shape())
    if not shape or math.prod(shape) == 1:
        return handle.get_tensor(key).float()
    if shape[0] == rows:
        return scale_slice[start:end].float()
    return handle.get_tensor(key).float()


def _base_and_dora_stats(
    ref: BaseWeightRef,
    a: torch.Tensor,
    b: torch.Tensor,
    lora_scale: float,
    magnitude: Optional[torch.Tensor],
    chunk_rows: int = 256,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    base_norm_sq = 0.0
    dora_delta_norm_sq = 0.0
    rows = ref.shape[0]
    rank = a.shape[0]
    a_flat = a.float().reshape(rank, -1)
    b_flat = b.float().reshape(rows, rank)
    magnitude_flat = magnitude.float().reshape(-1) if magnitude is not None else None
    if magnitude_flat is not None and magnitude_flat.numel() != rows:
        return None, None, "DoRA magnitude shape does not match base output rows"

    try:
        with safe_open(str(ref.path), framework="pt", device="cpu") as handle:
            weight_slice = handle.get_slice(ref.key)
            stored_dtype = weight_slice.get_dtype()
            is_integer_weight = stored_dtype.startswith(("I", "U"))
            if is_integer_weight and ref.scale_key is None:
                return None, None, "integer base weight has no recognized scale tensor"
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                weight = weight_slice[start:end].float()
                if ref.scale_key is not None:
                    scale = _slice_scale(handle, ref.scale_key, start, end, rows)
                    while scale.ndim < weight.ndim:
                        scale = scale.unsqueeze(-1)
                    weight = weight * scale

                weight_flat = weight.reshape(end - start, -1)
                base_norm_sq += float(torch.sum(weight_flat * weight_flat).item())
                if magnitude_flat is not None:
                    delta = (b_flat[start:end] @ a_flat) * lora_scale
                    adapted = weight_flat + delta
                    adapted_norm = torch.linalg.vector_norm(adapted, dim=1).clamp_min(1e-12)
                    row_scale = magnitude_flat[start:end] / adapted_norm
                    effective = adapted * row_scale[:, None]
                    actual_delta = effective - weight_flat
                    dora_delta_norm_sq += float(torch.sum(actual_delta * actual_delta).item())
    except Exception as exc:
        return None, None, str(exc)

    return (
        math.sqrt(max(0.0, base_norm_sq)),
        math.sqrt(max(0.0, dora_delta_norm_sq)) if magnitude_flat is not None else None,
        None,
    )


def analyze_lora(
    lora_path: str | Path,
    *,
    base: Optional[str] = None,
    use_base: bool = True,
    search_roots: Iterable[str | Path] = (),
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    path = Path(lora_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix != ".safetensors":
        raise ValueError("AI Toolkit LoRA analysis currently requires a .safetensors file")

    metadata = load_lora_metadata(path)
    context = load_training_context(path)
    layers, ignored_keys = discover_lora_layers(path)
    if not layers:
        raise ValueError(f"No AI Toolkit LoRA A/B tensor pairs found in {path}")
    if progress:
        progress(f"Found {len(layers)} LoRA layers")

    base_reference, base_reference_source = infer_base_model_reference(metadata, context)
    selected_candidate = None
    base_matches: dict[str, BaseWeightRef] = {}
    if use_base:
        candidates = build_base_candidates(
            path,
            base_reference,
            explicit_base=base,
            search_roots=search_roots,
        )
        selected_candidate, base_matches = select_base_weights(layers, candidates)
        if progress:
            if selected_candidate is None:
                progress("No matching local base weights found; using LoRA-only metrics")
            else:
                progress(
                    f"Matched {len(base_matches)}/{len(layers)} layers against "
                    f"{selected_candidate.label}"
                )

    records: list[dict[str, Any]] = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for index, layer in enumerate(layers, start=1):
            a = handle.get_tensor(layer.a_key)
            b = handle.get_tensor(layer.b_key)
            scale, scale_source = _lora_scale(handle, layer, context)
            delta_fro, delta_rms, elements = _low_rank_delta_norm(a, b, scale)
            magnitude = (
                handle.get_tensor(layer.magnitude_key)
                if layer.magnitude_key is not None
                else None
            )

            base_ref = base_matches.get(layer.name)
            base_norm = None
            dora_delta_norm = None
            base_error = None
            zero_standard_delta = delta_fro == 0.0 and magnitude is None
            if base_ref is not None and not zero_standard_delta:
                base_norm, dora_delta_norm, base_error = _base_and_dora_stats(
                    base_ref,
                    a,
                    b,
                    scale,
                    magnitude,
                )

            effective_delta = dora_delta_norm if dora_delta_norm is not None else delta_fro
            effective_rms = effective_delta / math.sqrt(elements) if elements else 0.0
            relative_change = (
                effective_delta / base_norm
                if base_norm is not None and base_norm > 0
                else 0.0 if base_ref is not None and zero_standard_delta else None
            )
            records.append({
                "layer": layer.name,
                "layer_type": classify_layer(layer.name, layer.a_shape),
                "rank": layer.rank,
                "scale": scale,
                "scale_source": scale_source,
                "a_shape": list(layer.a_shape),
                "b_shape": list(layer.b_shape),
                "effective_shape": list(layer.output_shape),
                "parameter_count": elements,
                "delta_fro_norm": delta_fro,
                "delta_rms": delta_rms,
                "effective_delta_fro_norm": effective_delta,
                "effective_delta_rms": effective_rms,
                "base_fro_norm": base_norm,
                "relative_change": relative_change,
                "is_dora": magnitude is not None,
                "base_weight": (
                    {"path": str(base_ref.path), "key": base_ref.key}
                    if base_ref is not None
                    else None
                ),
                "base_error": base_error,
            })
            if progress and base_ref is not None and (index % 25 == 0 or index == len(layers)):
                progress(f"Compared {index}/{len(layers)} layers")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["layer_type"]].append(record)
    for group_records in groups.values():
        group_records.sort(
            key=lambda item: (
                item["relative_change"] is not None,
                (
                    item["relative_change"]
                    if item["relative_change"] is not None
                    else item["effective_delta_rms"]
                ),
            ),
            reverse=True,
        )

    selected_files = sorted(
        {str(ref.path) for ref in base_matches.values()}
    )
    return {
        "schema_version": 1,
        "lora_path": str(path),
        "metadata": metadata,
        "training_context": {
            "config_path": str(context.config_path) if context.config_path else None,
            "architecture": context.architecture,
            "network_type": context.network_type,
        },
        "base_model": {
            "reference": base or base_reference,
            "reference_source": "command_line" if base else base_reference_source,
            "candidate": selected_candidate.label if selected_candidate else None,
            "files": selected_files,
            "matched_layers": len(base_matches),
            "total_layers": len(layers),
        },
        "ranking": {
            "with_base": "effective_delta_fro_norm / base_fro_norm",
            "without_base": "effective_delta_rms",
            "note": (
                "DoRA effective deltas include the learned magnitude vector only when "
                "matching base weights are available."
            ),
        },
        "dora_layers": sum(record["is_dora"] for record in records),
        "groups": dict(sorted(groups.items())),
        "ignored_tensor_keys": ignored_keys,
    }


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.6f}"


def format_analysis_text(analysis: dict[str, Any], top: int = 20) -> str:
    base = analysis["base_model"]
    lines = [
        f"LoRA: {analysis['lora_path']}",
        f"Layers: {base['total_layers']}",
    ]
    if analysis["dora_layers"]:
        lines.append(f"DoRA layers: {analysis['dora_layers']}")
    if base["reference"]:
        lines.append(
            f"Base model reference: {base['reference']} ({base['reference_source']})"
        )
    if base["candidate"]:
        lines.append(
            f"Base comparison: {base['matched_layers']}/{base['total_layers']} layers "
            f"using {base['candidate']}"
        )
        for path in base["files"]:
            lines.append(f"  {path}")
    else:
        lines.append("Base comparison: unavailable; rankings use LoRA update RMS")
        if analysis["dora_layers"]:
            lines.append(
                "DoRA note: magnitude-vector changes require matching base weights "
                "and are not included in this LoRA-only ranking"
            )

    for group_name, records in analysis["groups"].items():
        lines.extend(("", f"[{group_name}] {len(records)} layer(s)"))
        for rank, record in enumerate(records[:top], start=1):
            relative = (
                f"{record['relative_change'] * 100:.6f}%"
                if record["relative_change"] is not None
                else "n/a"
            )
            lines.append(
                f"{rank:>3}. relative={relative:>13}  "
                f"delta_rms={_format_number(record['effective_delta_rms']):>11}  "
                f"delta_fro={_format_number(record['effective_delta_fro_norm']):>11}  "
                f"rank={record['rank']:<4} {record['layer']}"
            )
            if record["base_error"]:
                lines.append(f"     base warning: {record['base_error']}")
    return "\n".join(lines)
