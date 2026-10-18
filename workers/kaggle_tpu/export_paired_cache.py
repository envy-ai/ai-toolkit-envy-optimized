"""Export cached Qwen 2.1 Fizgig pairs without loading a transformer or encoder.

The optional preparation mode loads only the VAE. It deliberately does not
regenerate missing text conditioning or select an approximately matching cache.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_JOB = "qwen2.1-anime-34_copy-diffguidance"
PRESENTATION_KEYS = ("scale_to_width", "scale_to_height", "crop_x", "crop_y",
                     "crop_width", "crop_height", "flip_x", "flip_y")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_job(database: Path, name: str):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT job_config FROM Job WHERE name = ?", (name,)).fetchall()
    if len(row) != 1:
        raise ValueError(f"Expected one job named {name!r}, found {len(row)}")
    raw = json.loads(row[0][0])
    processes = raw["config"]["process"]
    if len(processes) != 1:
        raise ValueError("Paired export requires exactly one process")
    return processes[0]


def validate_profile(process):
    if process.get("type") != "fizgig_image_slider":
        raise ValueError("Export supports fizgig_image_slider only")
    model, train = process.get("model", {}), process.get("train", {})
    if model.get("arch") != "qwen_image_2":
        raise ValueError("Export supports qwen_image_2 only")
    if process.get("network", {}).get("type") not in ("lora", "dora"):
        raise ValueError("Export requires LoRA or signed DoRA")
    if not train.get("cache_text_embeddings") or train.get("train_text_encoder"):
        raise ValueError("Export requires frozen cached text conditioning")
    if process.get("fizgig_slider", {}).get("multipoint"):
        raise ValueError("Multipoint sliders are outside this export profile")
    if process.get("fizgig_slider", {}).get("anchor_prompts"):
        raise ValueError("Anchor prompts are outside this export profile")
    if process.get("trigger_word"):
        raise ValueError("Trigger-word processing is outside this export profile")
    if model.get("model_kwargs", {}).get("rgba"):
        raise ValueError("RGBA pairs are outside this RGB paired export profile")
    datasets = process.get("datasets", [])
    if not datasets:
        raise ValueError("Export requires paired datasets")
    for dataset in datasets:
        if not dataset.get("cache_latents_to_disk"):
            raise ValueError("Export requires disk latent caches")
        for key in ("random_crop", "random_scale", "shuffle_tokens", "token_dropout_rate",
                    "augmentations", "augments", "random_triggers", "trigger_word",
                    "anchor_path", "mask_path", "alpha_mask", "standardize_images",
                    "cache_tensors_to_disk", "dataset_path"):
            if dataset.get(key):
                raise ValueError(f"Unsupported paired export setting: dataset.{key}")
        if dataset.get("num_frames", 1) != 1 or not dataset.get("buckets", True):
            raise ValueError("Export requires bucketed still images")
        if dataset.get("network_weight", 1) != 1:
            raise ValueError("Export requires network_weight=1")
        if isinstance(dataset.get("resolution"), list) and len(dataset["resolution"]) != 1:
            raise ValueError("This paired export profile requires one resolution per dataset")
        # Fizgig forces caption dropout to zero; the export records that policy.


def make_model(process, device="cpu"):
    from toolkit.config_modules import ModelConfig
    QwenImage2Model = qwen_module("qwen_image_2").QwenImage2Model
    settings = dict(process["model"], dtype=process["train"].get("dtype", "bf16"))
    return QwenImage2Model(device, ModelConfig(**settings), dtype=settings["dtype"])


def qwen_module(name):
    """Load just Qwen; the general model registry eagerly imports CUDA kernels."""
    namespace = "_aitk_kaggle_qwen21"
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(ROOT / "extensions_built_in/diffusion_models/qwen_image_2")]
        sys.modules[namespace] = package
    return importlib.import_module(f"{namespace}.{name}")


def build_items(process):
    """Use the trainer's DTO, bucket layout and deterministic flip expansion."""
    from torchvision import transforms
    from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
    from toolkit.data_loader import RescaleTransform, image_extensions
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.dataloader_mixins import BucketsMixin
    from extensions_built_in.sd_trainer.FizgigSliderTrainer import validate_image_slider_pairs
    validate_profile(process)
    validate_image_slider_pairs(process["datasets"])
    sd = make_model(process)
    results = []
    for dataset_index, original in enumerate(preprocess_dataset_raw_config(process["datasets"])):
        raw = copy.deepcopy(original)
        raw["unconditional_path"] = raw.pop("control_path_1")
        raw["caption_dropout_rate"] = 0.0
        raw["fizgig_slider_pair"] = True
        raw["bucket_tolerance"] = sd.get_bucket_divisibility()
        config = DatasetConfig(**raw)
        folder = Path(config.folder_path)
        paths = []
        for directory, dirs, names in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            if Path(directory).name == "_controls":
                continue
            paths.extend(Path(directory) / n for n in sorted(names)
                         if not n.startswith(".") and n.lower().endswith(tuple(image_extensions)))
        items = [FileItemDTO(
            sd=sd, path=str(path), dataset_config=config, dataset_root=str(folder),
            dataloader_transforms=transforms.Compose([transforms.ToTensor(), RescaleTransform()]),
            size_database={}, latent_space_version=sd.get_latent_space_version(),
            text_embedding_space_version=sd.get_text_embedding_space_version(),
            encode_control_in_text_embeddings=True,
            cache_processed_control_text_embeddings=True,
            caption_dropout_keeps_control_images=True,
        ) for path in paths]
        items = items * config.num_repeats
        # The normal loader copies repeats before appending flipped copies.
        if config.flip_x:
            flipped = [copy.deepcopy(item) for item in items]
            for item in flipped:
                item.flip_x = True
            items += flipped
        if config.flip_y:
            flipped = [copy.deepcopy(item) for item in items]
            for item in flipped:
                item.flip_y = True
            items += flipped
        context = SimpleNamespace(file_list=items, dataset_config=config, epoch_num=0,
                                  buckets={}, is_video=False, dataset_path=str(folder),
                                  shuffle_buckets=lambda: None, build_batch_indices=lambda: None)
        BucketsMixin.setup_buckets(context, quiet=True)
        for item in items:
            item.load_caption()
            if not item.unconditional_path:
                raise ValueError(f"Missing negative pair for {item.path}")
            results.append(dict(dataset_index=dataset_index, positive_path=item.path,
                                negative_path=item.unconditional_path, item=item,
                                vae_recipe={key: getattr(sd.model_config, key) for key in
                                            ("extras_name_or_path", "vae_dtype")}))
    return results


def presentation(item):
    return {key: getattr(item, key) for key in PRESENTATION_KEYS}


def negative_identity(entry):
    return {"schema": 1, "negative_sha256": digest(entry["negative_path"]),
            "positive_presentation": presentation(entry["item"]),
            "latent_space_version": entry["item"].latent_space_version,
            "resize_filter": "mitchell_netravali_b_c_1_3_v1", "color_mode": "RGB",
            "vae_recipe": entry.get("vae_recipe", {})}


def validate_cache_metadata(path, expected):
    from safetensors import safe_open
    from toolkit.metadata import get_meta_for_safetensors
    with safe_open(path, framework="pt", device="cpu") as cache:
        metadata = cache.metadata() or {}
    for key, value in get_meta_for_safetensors(expected).items():
        if key not in ("software", "format") and metadata.get(key) != value:
            raise ValueError(f"{path}: incompatible cache metadata {key!r}")


def negative_path(entry, directory):
    identity = negative_identity(entry)
    return directory / (hashlib.sha256(canonical(identity).encode()).hexdigest() + ".safetensors")


def validate_latent(tensor, item, label):
    import torch
    expected = (64, item.crop_height // 16, item.crop_width // 16)
    if tuple(tensor.shape) != expected or not tensor.is_floating_point():
        raise ValueError(f"{label}: expected floating latent {expected}, got {tuple(tensor.shape)} {tensor.dtype}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label}: nonfinite cached latent")


def validate_conditioning(path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    data = load_file(path, device="cpu")
    if set(data) != {"text_embeds", "attention_mask", "image_slot_mask"}:
        raise ValueError(f"{path}: incompatible Qwen conditioning keys {sorted(data)}")
    with safe_open(path, framework="pt") as cache:
        if (cache.metadata() or {}).get("class_name") != "AdvancedPromptEmbeds":
            raise ValueError(f"{path}: expected AdvancedPromptEmbeds cache")
    text, valid, slots = (data[k] for k in ("text_embeds", "attention_mask", "image_slot_mask"))
    if text.ndim != 2 or text.shape[-1] != 4096 or valid.shape != text.shape[:1] or slots.shape != valid.shape:
        raise ValueError(f"{path}: incompatible Qwen conditioning shapes")
    if (valid.dtype not in (torch.bool, torch.int64, torch.int32)
            or not ((valid == 0) | (valid == 1)).all()
            or slots.dtype != torch.bool or slots.any()):
        raise ValueError(f"{path}: expected binary text mask and bool slots with no edit references")
    if not text.is_floating_point() or not torch.isfinite(text).all() or not valid.any():
        raise ValueError(f"{path}: invalid text conditioning")


def prepare_latents(process, entries, negative_cache_dir, device, seed):
    """Encode absent latents through the same DTO and Qwen VAE wrapper as Fizgig."""
    import torch
    from toolkit.metadata import get_meta_for_safetensors
    from toolkit.safetensors_cache import atomic_save_file
    BASE_REPO = qwen_module("qwen_image_2").BASE_REPO
    AutoencoderKLQwenImage21 = qwen_module("src.vae").AutoencoderKLQwenImage21
    sd = make_model(process, device)
    # Resolve the same extras source used by QwenImage2Model.load_model.
    base = sd.model_config.extras_name_or_path
    model = sd.model_config.name_or_path
    if base == model and not os.path.isdir(base):
        base = BASE_REPO
    elif os.path.isdir(model) and os.path.isdir(os.path.join(model, "text_encoder")):
        base = model
    torch.manual_seed(seed)
    sd.vae = AutoencoderKLQwenImage21.load(base, **sd.component_load_kwargs("vae"))
    sd.vae.requires_grad_(False).eval()
    try:
        with torch.no_grad():
            for entry in entries:
                item = entry["item"]
                for sign, path in (("positive", Path(item.get_latent_path())),
                                   ("negative", negative_path(entry, negative_cache_dir))):
                    if path.exists():
                        continue
                    print(f"Encoding {sign}: {item.path} flip_x={item.flip_x} flip_y={item.flip_y}", flush=True)
                    if sign == "positive":
                        item.load_image()
                        pixels = item.tensor
                    else:
                        item.load_unconditional_image()
                        pixels = item.unconditional_tensor
                    latent = sd.encode_images(pixels.unsqueeze(0).to(sd.vae_device_torch, dtype=sd.vae_torch_dtype))
                    latent = latent.squeeze(0).detach().cpu()
                    validate_latent(latent, item, str(path))
                    info = item.get_latent_info_dict() if sign == "positive" else {
                        "paired_negative_identity": canonical(negative_identity(entry)), "preparation_seed": seed}
                    atomic_save_file({"latent": latent}, path, metadata=get_meta_for_safetensors(info))
                    item.cleanup()
    finally:
        sd.vae.to("cpu")
        del sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def export(process, output: Path, *, negative_cache_dir=None, audit_only=False,
           prepare_missing=False, device="cuda", seed=42):
    from safetensors.torch import load_file, save_file
    validate_profile(process)
    entries = build_items(process)
    negative_cache_dir = Path(negative_cache_dir or ROOT / "tmp/kaggle_paired_negative_cache")
    missing = []
    for entry in entries:
        item = entry["item"]
        for label, path in (("positive", Path(item.get_latent_path())),
                            ("negative", negative_path(entry, negative_cache_dir)),
                            ("conditioning", Path(item.get_text_embedding_path()))):
            if not path.is_file():
                missing.append({"kind": label, "path": str(path)})
    missing = list({(x["kind"], x["path"]): x for x in missing}.values())
    if audit_only:
        return {"exposures": len(entries), "missing": missing,
                "ready": not missing, "preparation_required": bool(missing)}
    if any(x["kind"] == "conditioning" for x in missing):
        raise FileNotFoundError("Missing exact cached text conditioning; cache it in AI Toolkit first: "
                                + canonical([x for x in missing if x["kind"] == "conditioning"]))
    # Validate reusable inputs before spending GPU time generating the other side.
    checked = set()
    for entry in entries:
        item = entry["item"]
        positive = Path(item.get_latent_path())
        conditioning = Path(item.get_text_embedding_path())
        if positive.is_file() and str(positive) not in checked:
            validate_cache_metadata(positive, item.get_latent_info_dict())
            validate_latent(load_file(positive, device="cpu")["latent"], item, str(positive))
            checked.add(str(positive))
        if str(conditioning) not in checked:
            validate_conditioning(conditioning)
            checked.add(str(conditioning))
    if missing and not prepare_missing:
        raise FileNotFoundError("Missing paired latent caches; use --prepare-missing for local VAE encoding: " + canonical(missing))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Export destination already exists: {output}")
    if missing:
        prepare_latents(process, entries, negative_cache_dir, device, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    files, pairs, exposures, known = {}, [], [], {}
    source_hashes = {}

    def source_identity(path, folder):
        path = Path(path)
        source_hashes.setdefault(str(path), digest(path))
        return {"relative_path": str(path.relative_to(folder)), "sha256": source_hashes[str(path)]}

    def publish(path):
        sha = digest(path)
        relative = f"tensors/{sha}.safetensors"
        dest = stage / relative
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copyfile(path, dest)
        files[relative] = {"sha256": sha, "bytes": dest.stat().st_size}
        return {"path": relative, "sha256": sha}

    try:
        for entry in entries:
            item, dataset_index = entry["item"], entry["dataset_index"]
            identity = {"dataset_index": dataset_index, "positive": str(item.path),
                        "negative": entry["negative_path"], "presentation": presentation(item)}
            pair_id = hashlib.sha256(canonical(identity).encode()).hexdigest()
            exposures.append(pair_id)
            if pair_id in known:
                continue
            known[pair_id] = True
            positive_path = Path(item.get_latent_path())
            neg_path = negative_path(entry, negative_cache_dir)
            validate_cache_metadata(positive_path, item.get_latent_info_dict())
            validate_cache_metadata(neg_path, {"paired_negative_identity": canonical(negative_identity(entry))})
            positive, negative = (load_file(p, device="cpu")["latent"] for p in (positive_path, neg_path))
            validate_latent(positive, item, str(positive_path))
            validate_latent(negative, item, str(neg_path))
            conditioning = Path(item.get_text_embedding_path())
            validate_conditioning(conditioning)
            temporary = stage / "pair.safetensors"
            save_file({"positive": positive.contiguous(), "negative": negative.contiguous()}, temporary)
            tensors = publish(temporary)
            temporary.unlink()
            dataset = process["datasets"][dataset_index]
            pairs.append({"id": pair_id, "dataset_index": dataset_index,
                          "positive_source": source_identity(entry["positive_path"], dataset["folder_path"]),
                          "negative_source": source_identity(entry["negative_path"], dataset["control_path_1"]),
                          "presentation": presentation(item), "caption": item.caption,
                          "loss_multiplier": item.loss_multiplier, "latents": tensors,
                          "conditioning": publish(conditioning)})
        profile = {key: copy.deepcopy(process.get(key, {})) for key in
                   ("type", "model", "network", "train", "fizgig_slider")}
        components = {}
        for key in ("name_or_path", "text_encoder_path", "assistant_lora_path", "vae_path", "extras_name_or_path"):
            path = profile["model"].get(key)
            if path and Path(path).is_file():
                components[key] = {"filename": Path(path).name, "sha256": digest(path), "bytes": Path(path).stat().st_size}
                profile["model"][key] = f"component:{key}"
        # Explicit initialization policy prevents accidental local checkpoint resume.
        profile["network"].pop("pretrained_lora_path", None)
        profile["train"].pop("start_step", None)
        manifest = {"schema": 1, "kind": "qwen21_fizgig_paired_cache", "initialization": "fresh_adapter",
                    "source_config_sha256": hashlib.sha256(canonical(process).encode()).hexdigest(),
                    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "profile": profile, "components": components, "pairs": pairs,
                    "exposures": exposures, "files": files,
                    "preparation": {"seed": seed, "generated_missing_latents": bool(missing),
                                    "negative_cache_contract": 1,
                                    "legacy_positive_vae_provenance": "Cache identity includes architecture and presentation, not VAE weight hash"},
                    "source_files": {str(path.relative_to(ROOT)): digest(path) for path in (
                        Path(__file__), ROOT / "toolkit/dataloader_mixins.py",
                        ROOT / "toolkit/data_transfer_object/data_loader.py",
                        ROOT / "toolkit/image_resampling.py",
                        ROOT / "extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py")}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="JSON/YAML process or complete training job")
    source.add_argument("--database", type=Path, default=ROOT / "aitk_db.db")
    parser.add_argument("--job", default=DEFAULT_JOB)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/kaggle_qwen21_pairs")
    parser.add_argument("--negative-cache-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--prepare-missing", action="store_true", help="Encode missing latents locally with only the VAE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.config:
        import yaml
        raw = yaml.safe_load(args.config.read_text())
        process = raw["config"]["process"][0] if "config" in raw else raw
    else:
        process = load_job(args.database, args.job)
    result = export(process, args.output, negative_cache_dir=args.negative_cache_dir,
                    audit_only=args.audit_only, prepare_missing=args.prepare_missing,
                    device=args.device, seed=args.seed)
    if args.audit_only:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({"output": str(args.output), "pairs": len(result["pairs"]),
                          "exposures": len(result["exposures"]), "files": len(result["files"])}))


if __name__ == "__main__":
    main()
