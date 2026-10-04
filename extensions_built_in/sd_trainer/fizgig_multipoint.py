"""Validation/expansion for the opt-in slider format (no model allocation)."""

import math
import os

from PIL import Image, ImageOps


def parse_points(config):
    points = config.get("points") if isinstance(config, dict) else None
    if not isinstance(points, list) or not points:
        raise ValueError("Multi-point slider requires points")
    ids, strengths = set(), set()
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("Multi-point targets must be objects")
        ident, strength = point.get("id"), point.get("strength")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("Multi-point IDs must be unique nonempty strings")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise ValueError("Multi-point strengths must be finite numbers")
        if strength in strengths:
            raise ValueError("Multi-point strengths must be unique (including zero)")
        for key in ("prefix", "negative_prefix"):
            if not isinstance(point.get(key, ""), str):
                raise ValueError("Multi-point prefixes must be text")
        ids.add(ident)
        strengths.add(strength)
    if not any(strengths):
        raise ValueError("Multi-point slider needs at least one nonzero training target")
    return sorted(points, key=lambda point: point["strength"])


def _text(value, label, required=False):
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"Multi-point {label} requires {'nonempty ' if required else ''}text")
    return value.strip()


def _assemble(prefix, base, negative=False):
    return f"{prefix}\n\n{base}" if prefix else ("" if negative else base)


def parse_multipoint_prompts(config, points):
    entries = config.get("prompt_entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Multi-point prompt slider requires prompt entries")
    zero = next((point for point in points if point["strength"] == 0), None)
    parsed = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Multi-point prompt entries must be objects")
        if entry.get("kind") == "simple":
            base = _text(entry.get("prompt"), "base prompt", True)
            targets = {}
            for point in points:
                prefix = _text(point.get("prefix", ""), "positive prefix", point["strength"] != 0)
                negative = _text(point.get("negative_prefix", ""), "negative prefix")
                targets[point["id"]] = (_assemble(prefix, base), _assemble(negative, base, True))
            neutral = targets[zero["id"]] if zero else (
                base, _assemble(_text(config.get("neutral_negative_prefix", ""), "neutral negative prefix"), base, True)
            )
        elif entry.get("kind") == "specific":
            raw = entry.get("targets")
            if not isinstance(raw, list) or len(raw) != len(points):
                raise ValueError("Multi-point specific entry must contain every point exactly once")
            targets = {}
            for target in raw:
                if not isinstance(target, dict) or target.get("point_id") in targets or target.get("point_id") not in {p["id"] for p in points}:
                    raise ValueError("Multi-point specific entry has duplicate/unknown point IDs")
                targets[target["point_id"]] = (
                    _text(target.get("prompt"), "target prompt", True),
                    _text(target.get("negative_prompt", ""), "target negative prompt"),
                )
            neutral = targets[zero["id"]] if zero else (
                _text(entry.get("neutral_prompt"), "neutral prompt", True),
                _text(entry.get("neutral_negative_prompt", ""), "neutral negative prompt"),
            )
        else:
            raise ValueError("Multi-point prompt entry requires simple or specific kind")
        parsed.append({"neutral": neutral, "targets": targets})
    return parsed


def _images(folder):
    if not isinstance(folder, str) or not os.path.isdir(folder):
        raise ValueError(f"Multi-point image folder does not exist: {folder}")
    images = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if os.path.basename(root) == "_controls":
            continue  # Match the normal dataloader's control-folder exclusion.
        for name in files:
            stem, ext = os.path.splitext(name)
            if name.startswith(".") or ext.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".jxl"}:
                continue  # Includes *.png.disabled, as in the ordinary loader.
            if ext.lower() == ".jxl":
                raise ValueError("Multi-point paired images do not support JXL")
            if stem in images:
                raise ValueError(f"Multi-point folder has ambiguous filename stem: {stem}")
            images[stem] = os.path.join(root, name)
    if not images:
        raise ValueError(f"Multi-point folder has no usable images: {folder}")
    return images


def validate_multipoint_images(datasets, points):
    if not datasets:
        raise ValueError("Multi-point image slider requires datasets")
    ids = {p["id"] for p in points}
    # Neutral captions: explicit zero, then +1, then lowest numerical strength.
    reference = next((p for p in points if p["strength"] == 0), None)
    reference = reference or next((p for p in points if p["strength"] == 1), points[0])
    groups = []
    for dataset in datasets:
        if dataset.get("is_reg") or not dataset.get("buckets", True) or dataset.get("num_frames", 1) != 1:
            raise ValueError("Multi-point image slider requires non-regularization bucketed still images")
        if not (dataset.get("cache_latents_to_disk") or dataset.get("cache_latents")):
            raise ValueError("Multi-point image slider requires Cache Latents")
        if dataset.get("augmentations") or dataset.get("augments") or dataset.get("controls") or dataset.get("control_path") or dataset.get("control_path_2") or dataset.get("control_path_3") or dataset.get("control_from_same_folder"):
            raise ValueError("Multi-point image slider does not support augmentations or edit controls")
        mappings = dataset.get("multipoint_images")
        if not isinstance(mappings, list) or len(mappings) != len(points):
            raise ValueError("Each multi-point dataset requires one folder per point")
        folders, images = {}, {}
        for mapping in mappings:
            if not isinstance(mapping, dict) or mapping.get("point_id") not in ids or mapping["point_id"] in folders:
                raise ValueError("Multi-point image folders contain duplicate/unknown point IDs")
            ident = mapping["point_id"]
            folders[ident] = mapping.get("folder_path")
            images[ident] = _images(folders[ident])
        stems = set(images[reference["id"]])
        if any(set(bank) != stems for bank in images.values()):
            raise ValueError("Multi-point image folders must have matching filename stems")
        for stem in stems:
            sizes = []
            for bank in images.values():
                with Image.open(bank[stem]) as image:
                    sizes.append(ImageOps.exif_transpose(image).size)
            if len(set(sizes)) != 1:
                raise ValueError(f"Multi-point pair {stem} has different dimensions")
        groups.append({"folders": folders, "images": images, "reference_id": reference["id"]})
    return groups
