#!/usr/bin/env python3
"""Repair ai-toolkit DoRA checkpoints for ComfyUI's magnitude broadcasting.

Older ai-toolkit DoRA saves used ``magnitude`` keys and one-dimensional vectors.
ComfyUI's vanilla LoRA loader expects linear-layer ``dora_scale`` tensors to
have shape ``(out_features, 1)``. This tool converts affected files atomically.
"""

import argparse
import gc
import os
import sys
from collections import OrderedDict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from toolkit.metadata import add_model_hash_to_meta


def checkpoint_paths(targets: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for target in targets:
        if target.is_dir():
            paths.update(target.glob("*.safetensors"))
        elif target.suffix == ".safetensors":
            paths.add(target)
        else:
            raise ValueError(f"Not a safetensors checkpoint or directory: {target}")
    return sorted(paths)


MAGNITUDE_SUFFIXES = (
    ".lora_magnitude_vector.default.weight",
    ".lora_magnitude_vector.weight",
    ".lora_magnitude_vector",
    ".magnitude",
)


def comfy_key(key: str) -> str:
    for suffix in MAGNITUDE_SUFFIXES:
        if key.endswith(suffix):
            return key[:-len(suffix)] + ".dora_scale"
    return key


def repair_checkpoint(
    path: Path,
    dry_run: bool = False,
    output_path: Path | None = None,
) -> int:
    with safe_open(str(path), framework="pt") as handle:
        metadata = OrderedDict(handle.metadata() or {})
        affected = [
            key
            for key in handle.keys()
            if any(key.endswith(suffix) for suffix in MAGNITUDE_SUFFIXES)
            or (
                key.endswith(".dora_scale")
                and len(handle.get_slice(key).get_shape()) == 1
            )
        ]

    if not affected or dry_run:
        return len(affected)

    old_state_dict = load_file(str(path), device="cpu")
    state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        new_key = comfy_key(key)
        if new_key.endswith(".dora_scale") and value.ndim == 1:
            value = value.unsqueeze(1)
        if new_key in state_dict:
            raise ValueError(f"DoRA key conversion collision in {path}: {new_key}")
        state_dict[new_key] = value
    del old_state_dict

    # Match ai-toolkit's normal save path so checkpoint metadata remains valid.
    metadata.pop("sshs_model_hash", None)
    metadata.pop("sshs_legacy_hash", None)
    metadata = add_model_hash_to_meta(state_dict, metadata)

    destination = output_path or path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.dora-repair.tmp")
    try:
        save_file(state_dict, str(temp_path), metadata)
        os.chmod(temp_path, path.stat().st_mode)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    del state_dict
    gc.collect()
    return len(affected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reshape old 1-D DoRA scales for ComfyUI compatibility."
    )
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write one input checkpoint to a new path instead of replacing it.",
    )
    args = parser.parse_args()

    paths = checkpoint_paths(args.targets)
    if args.output is not None and len(paths) != 1:
        parser.error("--output requires exactly one input checkpoint")
    if args.output is not None and args.output.exists() and args.output != paths[0]:
        parser.error(f"output already exists: {args.output}")

    repaired_files = 0
    repaired_vectors = 0
    for path in paths:
        count = repair_checkpoint(
            path,
            dry_run=args.dry_run,
            output_path=args.output,
        )
        if count:
            action = "would repair" if args.dry_run else "repaired"
            destination = args.output or path
            print(
                f"{action}: {path} -> {destination} ({count} DoRA vectors)",
                flush=True,
            )
            repaired_files += 1
            repaired_vectors += count

    action = "found" if args.dry_run else "repaired"
    print(
        f"{action} {repaired_vectors} DoRA vectors in {repaired_files} checkpoints",
        flush=True,
    )


if __name__ == "__main__":
    main()
