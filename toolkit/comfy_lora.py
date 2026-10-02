import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from toolkit.metadata import add_model_hash_to_meta


_LEGACY_DORA_SUFFIXES = (
    ".lora_magnitude_vector.default.weight",
    ".lora_magnitude_vector.weight",
    ".lora_magnitude_vector",
    ".magnitude",
)
_DORA_CACHE_VERSION = "qwen21-comfy-dora-v2"


def _comfy_lora_search_paths() -> list[Path]:
    roots = []
    for env_name in ("AI_TOOLKIT_COMFYUI_ROOT", "COMFYUI_ROOT", "COMFYUI_PATH"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())
    roots.append(Path.home() / "ComfyUI")

    search_paths: list[Path] = []
    for root in roots:
        search_paths.append(root / "models" / "loras")
        config_path = root / "extra_model_paths.yaml"
        if not config_path.is_file():
            continue
        try:
            config = yaml.safe_load(config_path.read_text()) or {}
        except Exception:
            continue
        for section in config.values():
            if not isinstance(section, dict) or not section.get("loras"):
                continue
            base_path = Path(section.get("base_path", root)).expanduser()
            values = section["loras"]
            if isinstance(values, str):
                values = values.splitlines()
            for value in values:
                value = str(value).strip()
                if not value or value.startswith("#"):
                    continue
                path = Path(value).expanduser()
                search_paths.append(path if path.is_absolute() else base_path / path)

    # Common central-model layout, also used by the local Comfy installation.
    search_paths.append(Path.home() / "models" / "lora")
    return list(dict.fromkeys(path.resolve() for path in search_paths))


def resolve_comfy_lora_path(
    lora_name: str,
    search_paths: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    if not lora_name:
        return None
    requested = Path(os.path.expanduser(lora_name))
    if requested.is_absolute():
        return requested.resolve() if requested.is_file() else None

    for root in search_paths or _comfy_lora_search_paths():
        root = Path(root).expanduser().resolve()
        candidate = (root / requested).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _is_dora_key(key: str) -> bool:
    return key.endswith(".dora_scale") or key.endswith(_LEGACY_DORA_SUFFIXES)


def _qwen_comfy_key(key: str) -> str:
    # Very old ai-toolkit state dicts could retain their internal separator.
    key = key.replace("$$", ".")
    if key.startswith("transformer."):
        key = "diffusion_model." + key[len("transformer."):]
    elif key.startswith("transformer_blocks."):
        key = "diffusion_model." + key
    for suffix in _LEGACY_DORA_SUFFIXES:
        if key.endswith(suffix):
            return key[:-len(suffix)] + ".dora_scale"
    return key


def prepare_qwen_image_2_inference_dora(
    lora_name: str,
    cache_dir: str,
    search_paths: Optional[Iterable[Path]] = None,
) -> Optional[str]:
    """Return an absolute Comfy-compatible DoRA path, or None for a plain LoRA.

    The source checkpoint is never modified. Already compatible DoRAs are used
    directly; legacy ai-toolkit keys and one-dimensional magnitude tensors are
    written to a content-addressed cache beside the training output.
    """
    source = resolve_comfy_lora_path(lora_name, search_paths=search_paths)
    if source is None:
        return None

    with safe_open(str(source), framework="pt") as handle:
        keys = list(handle.keys())
        dora_keys = [key for key in keys if _is_dora_key(key)]
        if not dora_keys:
            return None
        needs_conversion = (
            any(_qwen_comfy_key(key) != key for key in keys)
            or any(
                len(handle.get_slice(key).get_shape()) == 1
                for key in dora_keys
            )
        )
        metadata = OrderedDict(handle.metadata() or {})

    if not needs_conversion:
        return str(source)

    stat = source.stat()
    fingerprint = hashlib.sha256(
        f"{_DORA_CACHE_VERSION}\0{source}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    destination_dir = Path(cache_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{source.stem}-{fingerprint}-comfy-dora.safetensors"
    if destination.is_file():
        return str(destination.resolve())

    source_state = load_file(str(source), device="cpu")
    state = OrderedDict()
    for key, value in source_state.items():
        new_key = _qwen_comfy_key(key)
        if new_key.endswith(".dora_scale") and value.ndim == 1:
            value = value.unsqueeze(1)
        if new_key in state:
            raise ValueError(
                f"DoRA key conversion collision in {source}: {new_key}"
            )
        state[new_key] = value

    metadata.pop("sshs_model_hash", None)
    metadata.pop("sshs_legacy_hash", None)
    metadata = add_model_hash_to_meta(state, metadata)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    try:
        save_file(state, str(temp_path), metadata)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(destination.resolve())
