"""Qwen Image 2.1 signed sliders, following Fizgig v6.7.0's two objectives.

The image slider's Control Dataset 1 is a *target*, not an edit reference.
It is routed through the dataloader's paired/unconditional image channel so
the normal Qwen prompt encoder never reserves image-reference tokens for it.
"""

import os
import random
import math
import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from toolkit.flow_training import (FlowTrainingProfile, trainer_flow_profile,
                                   guided_flow_prediction, render_flow_bank_image)
from toolkit.training_capabilities import validate_specialized_model
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.data_loader import get_dataloader_datasets, get_dataloader_from_datasets
from toolkit.train_tools import get_torch_dtype
from .fizgig_multipoint import parse_points, parse_multipoint_prompts, validate_multipoint_images


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
POSITIVE_EXTENSIONS = IMAGE_EXTENSIONS | {".jxl"}


def validate_image_slider_pairs(datasets, *, label="Image slider", allow_edit_controls=False):
    """Fail instead of silently training only the subset that has a negative."""
    if not datasets:
        raise ValueError(f"{label} requires at least one positive dataset")
    for dataset in datasets:
        positive_dir = dataset.get("folder_path")
        negative_dir = dataset.get("control_path_1")
        if not positive_dir or not os.path.isdir(positive_dir):
            raise ValueError(f"{label} positive dataset does not exist: {positive_dir}")
        if not negative_dir or not os.path.isdir(negative_dir):
            raise ValueError(f"{label} Control Dataset 1 does not exist: {negative_dir}")
        if os.path.realpath(positive_dir) == os.path.realpath(negative_dir):
            raise ValueError(f"{label} positive and negative datasets must be different folders")
        forbidden_controls = ("control_path", "control_from_same_folder", "controls")
        if not allow_edit_controls:
            forbidden_controls += ("control_path_2", "control_path_3")
        if any(dataset.get(key) for key in forbidden_controls):
            raise ValueError(f"{label} has unsupported control paths")
        if allow_edit_controls:
            if dataset.get("control_path_3") and not dataset.get("control_path_2"):
                raise ValueError(f"{label} requires Control Dataset 2 when Control Dataset 3 is used")
            for key in ("control_path_2", "control_path_3"):
                path = dataset.get(key)
                if path and not os.path.isdir(path):
                    raise ValueError(f"{label} edit source dataset does not exist: {path}")
        if dataset.get("is_reg"):
            raise ValueError(f"{label} does not support regularization datasets")
        if not dataset.get("buckets", True) or dataset.get("num_frames", 1) != 1:
            raise ValueError(f"{label} requires bucketed still-image datasets")

        negatives = {}
        for name in os.listdir(negative_dir):
            stem, ext = os.path.splitext(name)
            if ext in IMAGE_EXTENSIONS:
                negatives.setdefault(stem, []).append(os.path.join(negative_dir, name))
        positives = []
        for root, dirs, files in os.walk(positive_dir):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            positives.extend(
                os.path.join(root, name) for name in files
                if not name.startswith(".") and os.path.splitext(name)[1].lower() in POSITIVE_EXTENSIONS
                and os.path.basename(root) != "_controls"
            )
        if not positives:
            raise ValueError(f"{label} has no images in {positive_dir}")
        if not (dataset.get("cache_latents_to_disk") or dataset.get("cache_latents")):
            raise ValueError(f"{label} requires Cache Latents for its preferred images")
        if dataset.get("augmentations") or dataset.get("augments"):
            raise ValueError(f"{label} does not support stochastic image augmentations")
        for path in positives:
            name = os.path.basename(path)
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"{label} cannot pair unsupported image format: {path}")
            matches = negatives.get(os.path.splitext(name)[0], [])
            if len(matches) != 1:
                raise ValueError(f"{label} requires one rejected image matching {name}; found {len(matches)}")
            if allow_edit_controls:
                for key in ("control_path_2", "control_path_3"):
                    control_dir = dataset.get(key)
                    if control_dir:
                        stem = os.path.splitext(name)[0]
                        controls = [
                            os.path.join(control_dir, stem + ext) for ext in IMAGE_EXTENSIONS
                            if os.path.isfile(os.path.join(control_dir, stem + ext))
                        ]
                        if len(controls) != 1:
                            raise ValueError(
                                f"{label} requires one edit source in {key} matching {name}; found {len(controls)}"
                            )
            with Image.open(path) as pos, Image.open(matches[0]) as neg:
                if ImageOps.exif_transpose(pos).size != ImageOps.exif_transpose(neg).size:
                    raise ValueError(f"{label} pair {name} has different dimensions/crop at preferred and rejected")


def pair_difference_weights(positive, negative, strength):
    """Fizgig's per-latent-token difference weighting; mean weight remains 1."""
    difference = (positive.float() - negative.float()).abs().mean(dim=1)
    difference = difference.flatten(1)
    mean = difference.mean(dim=1, keepdim=True)
    relative = (difference / mean.clamp_min(1e-8)).clamp(max=8.0)
    weight = (1.0 - strength) + strength * relative
    weight = weight / weight.mean(dim=1, keepdim=True).clamp_min(1e-8)
    weight = torch.where(mean > 1e-6, weight, torch.ones_like(weight))
    return weight


def parse_prompt_triplets(slider):
    """Expand shared-prefix entries and accept both earlier triplet formats."""
    if "prompt_entries" in slider:
        entries = slider["prompt_entries"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("Prompt slider requires at least one prompt entry")
        triplets = []
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"Prompt slider entry {index} must be an object")
            kind = entry.get("kind")
            if kind == "simple":
                base = entry.get("prompt")
                positive_prefix = slider.get("positive_prefix")
                negative_prefix = slider.get("negative_prefix")
                if not isinstance(base, str) or not base.strip():
                    raise ValueError(f"Prompt slider simplified entry {index} requires a base prompt")
                if (not isinstance(positive_prefix, str) or not positive_prefix.strip()
                        or not isinstance(negative_prefix, str) or not negative_prefix.strip()):
                    raise ValueError("Prompt slider simplified entries require +1 and -1 prefixes")
                base = base.strip()
                triplets.append((
                    base,
                    f"{positive_prefix.strip()}\n\n{base}",
                    f"{negative_prefix.strip()}\n\n{base}",
                ))
            elif kind == "specific":
                triplets.append(_parse_specific_triplet(entry, index))
            else:
                raise ValueError(f"Prompt slider entry {index} has unsupported kind: {kind}")
        return triplets
    if "prompt_triplets" in slider:
        entries = slider["prompt_triplets"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("Prompt slider requires at least one prompt triplet")
    else:
        entries = [slider]
    triplets = []
    for index, entry in enumerate(entries, start=1):
        triplets.append(_parse_specific_triplet(entry, index))
    return triplets


def _parse_specific_triplet(entry, index):
    if not isinstance(entry, dict):
        raise ValueError(f"Prompt slider triplet {index} must be an object")
    prompts = tuple(entry.get(key) for key in (
        "neutral_prompt", "positive_prompt", "negative_prompt"
    ))
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError(f"Prompt slider triplet {index} requires neutral, +1 and -1 prompts")
    return tuple(prompt.strip() for prompt in prompts)


def parse_cfg_negative_prompts(slider, triplets):
    """Build neutral, +1 and -1 CFG negatives for each prompt entry."""
    if "prompt_entries" in slider:
        entries = slider["prompt_entries"]
    elif "prompt_triplets" in slider:
        entries = slider["prompt_triplets"]
    else:
        entries = [slider]
    prompts = []
    for index, (entry, triplet) in enumerate(zip(entries, triplets), start=1):
        if entry.get("kind") == "simple":
            legacy = slider.get("cfg_negative_prefix", "")
            prefixes = (
                legacy,
                slider.get("cfg_negative_prefix_positive", legacy),
                slider.get("cfg_negative_prefix_negative", legacy),
            )
            if any(not isinstance(prefix, str) for prefix in prefixes):
                raise ValueError("CFG negative prefixes must be text")
            prompts.append(tuple(
                f"{prefix.strip()}\n\n{triplet[0]}" if prefix.strip() else ""
                for prefix in prefixes
            ))
        else:
            legacy = entry.get("cfg_negative_prompt", "")
            cfg_prompts = (
                legacy,
                entry.get("cfg_negative_prompt_positive", legacy),
                entry.get("cfg_negative_prompt_negative", legacy),
            )
            if any(not isinstance(prompt, str) for prompt in cfg_prompts):
                raise ValueError(f"CFG negative prompts for entry {index} must be text")
            prompts.append(tuple(prompt.strip() for prompt in cfg_prompts))
    return prompts


def parse_anchor_prompts(slider):
    entries = slider.get("anchor_prompts", [])
    if not isinstance(entries, list):
        raise ValueError("Slider anchor prompts must be a list")
    prompts = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("prompt"), str) or not entry["prompt"].strip():
            raise ValueError(f"Anchor prompt {index} requires a positive prompt")
        negative = entry.get("negative_prompt", "")
        if not isinstance(negative, str):
            raise ValueError(f"Anchor prompt {index} negative prompt must be text")
        prompts.append((entry["prompt"].strip(), negative.strip()))
    return prompts


class FizgigSliderTrainer(DiffusionTrainer):
    """Common process for `fizgig_image_slider` and `fizgig_prompt_slider`."""

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        if (config.get("fizgig_slider") or {}).get("multipoint") is True:
            # BaseTrainProcess saves job.raw_config. Keep the user's legacy
            # drafts there, and route folders/captions only in a working copy.
            config = copy.deepcopy(config)
        mode = config.get("type")
        if mode not in ("fizgig_image_slider", "fizgig_prompt_slider"):
            raise ValueError(f"Unknown Fizgig slider type: {mode}")
        validate_specialized_model(config, mode)
        self.flow_profile = FlowTrainingProfile.from_model_config(config['model'])
        from toolkit.training_capabilities import cfg_reference_mode
        self.cfg_reference = cfg_reference_mode(config['model'])
        network = config.get("network") or {}
        if network.get("type") not in ("lora", "dora"):
            raise ValueError("Fizgig sliders require a LoRA or DoRA network (not LoKr)")
        train = config.setdefault("train", {})
        if train.get("train_text_encoder"):
            raise ValueError("Fizgig sliders train the image transformer only")
        if train.get("noise_scheduler") != "flowmatch":
            raise ValueError("Fizgig sliders require the flowmatch scheduler")
        if train.get("do_cfg") or train.get("do_random_cfg"):
            raise ValueError("Fizgig sliders do not support CFG training")
        if train.get("diff_output_preservation") or train.get("blank_prompt_preservation"):
            raise ValueError("Fizgig sliders do not support output-preservation losses")
        if train.get("frequency_loss_type", "none") != "none":
            raise ValueError("Fizgig sliders do not support auxiliary frequency loss")
        if train.get("do_guidance_loss") or train.get("do_differential_guidance"):
            raise ValueError("Fizgig sliders do not support additional guidance losses")
        if (train.get("ema_config") or {}).get("use_ema"):
            raise ValueError("Fizgig sliders do not support EMA")
        slider = config.get("fizgig_slider") or {}
        self.slider_multipoint = slider.get("multipoint", False)
        if not isinstance(self.slider_multipoint, bool):
            raise ValueError("Multi-point must be a boolean")
        self.slider_points = parse_points(slider.get("multipoint_config")) if self.slider_multipoint else []
        self.slider_multipoint_latents = {}
        self.slider_mode = "image_pairs" if mode == "fizgig_image_slider" else "prompt_pairs"
        self.slider_diff_weight = float(slider.get("diff_weight", 1.0))
        self.slider_guidance = float(slider.get("guidance", 3.0))
        self.slider_bank_size = int(slider.get("bank_size", 16))
        self.slider_bank_resolution = int(slider.get("bank_resolution", 768))
        self.slider_bank_steps = int(slider.get("bank_steps", 25))
        self.slider_cfg = float(slider.get("cfg_scale", 1.0))
        self.slider_anchor_prompts = parse_anchor_prompts(slider)
        self.slider_preservation_weight = float(slider.get("preservation_weight", 1.0))
        if not math.isfinite(self.slider_preservation_weight) or self.slider_preservation_weight < 0:
            raise ValueError("Slider Preservation Weight must be finite and nonnegative")
        self.slider_anchor_image_configs = []
        if not 0 <= self.slider_diff_weight <= 1:
            raise ValueError("Image slider difference weight must be between 0 and 1")
        if (not self.slider_multipoint and self.slider_guidance <= 0) or self.slider_bank_size < 1 or self.slider_bank_steps < 1:
            raise ValueError("Prompt slider guidance, bank size and bank steps must be positive")
        if self.slider_bank_resolution < 64 or self.slider_bank_resolution % 32:
            raise ValueError("Prompt slider practice image resolution must be a multiple of 32 and at least 64")
        if not math.isfinite(self.slider_cfg) or self.slider_cfg < 0:
            raise ValueError("Prompt slider CFG must be finite and nonnegative")

        if self.slider_mode == "image_pairs":
            datasets = config.get("datasets") or []
            if self.slider_multipoint:
                self.slider_image_groups = validate_multipoint_images(datasets, self.slider_points)
            else:
                validate_image_slider_pairs(datasets)
            for group_index, dataset in enumerate(datasets):
                anchor_path = dataset.get("anchor_path")
                if anchor_path:
                    if not os.path.isdir(anchor_path):
                        raise ValueError(f"Slider anchor image folder does not exist: {anchor_path}")
                    # Independent preservation examples, not paired targets or
                    # edit references. Use their own captions and bucket caches.
                    self.slider_anchor_image_configs.append({
                        "folder_path": anchor_path, "resolution": dataset.get("resolution", [1024]),
                        "caption_ext": dataset.get("caption_ext", "txt"),
                        "default_caption": "", "caption_dropout_rate": 0,
                        "cache_latents_to_disk": True, "cache_text_embeddings": True,
                        "buckets": True, "num_frames": 1, "batch_size": 1,
                        "num_workers": 0, "num_repeats": 1,
                        "fizgig_slider_anchor": True,
                    })
                if self.slider_multipoint:
                    group = self.slider_image_groups[group_index]
                    dataset["folder_path"] = group["folders"][group["reference_id"]]
                    dataset["dataset_path"] = None
                    dataset["fizgig_multipoint_group"] = group_index
                    dataset["control_path_1"] = None
                    dataset["unconditional_path"] = None
                else:
                    dataset["unconditional_path"] = dataset.pop("control_path_1")
                dataset["caption_dropout_rate"] = 0.0
                dataset["fizgig_slider_pair"] = True
        else:
            if self.slider_multipoint:
                self.slider_multipoint_prompts = parse_multipoint_prompts(slider["multipoint_config"], self.slider_points)
                # Reuse the neutral-only practice renderer, with no target-count
                # multiplier. The actual target predictions use a separate bank.
                self.slider_prompt_triplets = [(entry["neutral"][0],) for entry in self.slider_multipoint_prompts]
                self.slider_cfg_negative_prompts = [(entry["neutral"][1],) for entry in self.slider_multipoint_prompts] if self.slider_cfg > 1 else None
            else:
                self.slider_prompt_triplets = parse_prompt_triplets(slider)
                self.slider_cfg_negative_prompts = (
                    parse_cfg_negative_prompts(slider, self.slider_prompt_triplets)
                    if self.slider_cfg > 1.0 else None
                )
            if self.slider_bank_size < len(self.slider_prompt_triplets):
                raise ValueError(
                    "Prompt slider Practice Images must be at least the number of prompt triplets"
                )
            config["datasets"] = []
            train["cache_text_embeddings"] = False
            train["unload_text_encoder"] = True

        super().__init__(process_id, job, config, **kwargs)
        self.retain_vae_after_caching = self.slider_mode == "prompt_pairs" or bool(
            self.slider_anchor_prompts and self.slider_preservation_weight
        )
        self.slider_bank = []
        self.slider_embeds = None
        self.slider_cfg_negative_embeds = None
        self.negative_latents = {}
        self.slider_anchor_embeds = []
        self.slider_anchor_bank = []

    def update_training_metadata(self):
        super().update_training_metadata()
        from toolkit.flow_training import flow_training_metadata
        metadata = {"ss_slider": self.slider_mode, **flow_training_metadata(self)}
        if self.slider_multipoint:
            import json
            metadata["ss_slider_multipoint"] = json.dumps(self.slider_points)
            if self.slider_mode == "prompt_pairs":
                metadata["ss_slider_multipoint_prompts"] = json.dumps(self.slider_multipoint_prompts)
        if self.slider_mode == "image_pairs":
            metadata["ss_slider_diff_weight"] = str(self.slider_diff_weight)
        else:
            import json
            metadata["ss_slider_prompts"] = json.dumps(self.slider_prompt_triplets)
            metadata["ss_slider_guidance"] = str(self.slider_guidance)
            metadata["ss_slider_cfg_scale"] = str(self.slider_cfg)
            if self.slider_cfg_negative_prompts is not None:
                metadata["ss_slider_cfg_negative_prompts"] = json.dumps(self.slider_cfg_negative_prompts)
        if self.slider_anchor_prompts or self.slider_anchor_image_configs:
            import json
            metadata["ss_slider_anchor_prompts"] = json.dumps(self.slider_anchor_prompts)
            metadata["ss_slider_preservation_weight"] = str(self.slider_preservation_weight)
            metadata["ss_slider_anchor_image_folders"] = json.dumps([
                config["folder_path"] for config in self.slider_anchor_image_configs
            ])
        self.add_meta(metadata)

    def hook_before_train_loop(self):
        from toolkit.flow_training import log_flow_training_profile
        log_flow_training_profile(self)
        if self.network_config.type == "dora":
            # The slider switches strength between +1 and -1 every step.
            # Ordinary DoRA re-normalizes W +/- LoRA independently, whereas
            # ComfyUI scales the complete +1 DoRA delta around the base.
            self.network.signed_dora_slider = True
        if self.slider_mode == "image_pairs":
            if getattr(self, "slider_multipoint", False):
                self._run_with_optimizer_state_offload(self._cache_multipoint_image_latents)
            else:
                self._cache_negative_latents()
        if getattr(self, "slider_anchor_image_configs", []) and self.slider_preservation_weight:
            self._run_with_optimizer_state_offload(self._cache_anchor_images)
        if self.slider_mode == "prompt_pairs" or (getattr(self, "slider_anchor_prompts", []) and self.slider_preservation_weight):
            # Encode every triplet with only the text encoder resident.
            self.sd.set_device_state_preset("cache_text_encoder")
            try:
                with torch.no_grad():
                    text_negatives = getattr(self, 'slider_cfg', 2) > 1 and getattr(self, 'cfg_reference', None) != 'image_only'
                    if self.slider_mode == "prompt_pairs":
                        self.slider_embeds = [
                            tuple(self.sd.encode_prompt([prompt]).detach().to("cpu") for prompt in triplet)
                            for triplet in self.slider_prompt_triplets
                        ]
                        if self.slider_cfg_negative_prompts is not None and text_negatives:
                            self.slider_cfg_negative_embeds = [
                                tuple(self.sd.encode_prompt([prompt]).detach().to("cpu") for prompt in triplet)
                                for triplet in self.slider_cfg_negative_prompts
                            ]
                        if getattr(self, "slider_multipoint", False):
                            self.slider_target_embeds = [{
                                point["id"]: (
                                    self.sd.encode_prompt([entry["targets"][point["id"]][0]]).detach().to("cpu"),
                                    self.sd.encode_prompt([entry["targets"][point["id"]][1]]).detach().to("cpu") if text_negatives else None,
                                ) for point in self.slider_points if point["strength"] != 0
                            } for entry in self.slider_multipoint_prompts]
                    self.slider_anchor_embeds = [
                        (self.sd.encode_prompt([positive]).detach().to("cpu"),
                         self.sd.encode_prompt([negative]).detach().to("cpu") if text_negatives else None)
                        for positive, negative in getattr(self, "slider_anchor_prompts", [])
                        if self.slider_preservation_weight
                    ]
            finally:
                self.sd.restore_device_state()
        super().hook_before_train_loop()
        if self.slider_mode == "prompt_pairs":
            self._build_prompt_bank()
        if getattr(self, "slider_anchor_embeds", []):
            self._build_anchor_prompt_bank()

    def _cache_anchor_images(self):
        try:
            loader = get_dataloader_from_datasets(self.slider_anchor_image_configs, 1, self.sd)
            with torch.no_grad():
                for dataset in get_dataloader_datasets(loader):
                    if not dataset.file_list:
                        raise ValueError(f"Anchor image folder has no usable images: {dataset.dataset_path}")
                    for item in tqdm(dataset.file_list, desc="Caching slider anchor images"):
                        try:
                            latent = item.get_latent().detach().unsqueeze(0).cpu()
                            item.load_prompt_embedding()
                            self.slider_anchor_bank.append((latent, item.prompt_embeds.detach().to("cpu"), None))
                            from toolkit.training_examples import recording_enabled, source_item
                            if recording_enabled(self):
                                if not hasattr(self, '_loss_report_bank_sources'):
                                    self._loss_report_bank_sources = {}
                                self._loss_report_bank_sources[id(latent)] = source_item(item.path, item.caption,
                                    getattr(item.dataset_config, 'folder_path', None), role='preservation_anchor')
                        finally:
                            item.cleanup_latent()
                            item.cleanup_text_embedding()
        finally:
            # Encoder must leave GPU before the optimizer-offload wrapper
            # restores optimizer state. Normal setup subsequently unloads TE.
            self.sd.text_encoder_to("cpu")
        self.print(f"Slider cached {len(self.slider_anchor_bank)} anchor image presentations in RAM")

    def _build_anchor_prompt_bank(self):
        from torchvision.transforms.functional import to_tensor

        self.sd.save_device_state()
        network = self.network
        previous_active = network.is_active
        try:
            self.sd.text_encoder_to("cpu")
            self.sd.unet.to(self.device_torch)
            self.sd.vae.to(self.sd.vae_device_torch)
            network.is_active = False
            # Keep the frozen helper active, as in the training reference passes.
            with torch.no_grad():
                pipeline = self.sd.get_generation_pipeline()
                for index, (positive, negative) in enumerate(tqdm(self.slider_anchor_embeds, desc="Rendering slider anchor practice images")):
                    image = render_flow_bank_image(
                        self.sd, pipeline, positive.to(self.device_torch, dtype=self.sd.torch_dtype),
                        negative.to(self.device_torch, dtype=self.sd.torch_dtype) if negative is not None else None,
                        height=self.slider_bank_resolution, width=self.slider_bank_resolution,
                        steps=self.slider_bank_steps, cfg=self.slider_cfg,
                        seed=int(self.sample_config.seed) + 100_000 + index)
                    pixels = to_tensor(image).mul(2).sub(1).to(self.device_torch, dtype=self.sd.vae_torch_dtype)
                    latent = self.sd.encode_images(pixels.unsqueeze(0)).detach().cpu()
                    self.slider_anchor_bank.append((latent, positive.to("cpu"), negative.to("cpu") if negative is not None else None))
                    from toolkit.training_examples import save_practice_source, recording_enabled
                    if recording_enabled(self):
                        save_practice_source(self, latent, image, f'anchor_{index:06d}', self.slider_anchor_prompts[index][0])
        finally:
            # Embeds.to may mutate its container; do not retain bank embeds on GPU.
            for positive, negative in self.slider_anchor_embeds:
                positive.to("cpu")
                if negative is not None:
                    negative.to("cpu")
            network.is_active = previous_active
            self.sd.restore_device_state()

    def _train_anchor_preservation(self, accum_scale):
        latent, positive, negative = random.choice(self.slider_anchor_bank)
        from toolkit.training_examples import remember_bank_source
        remember_bank_source(self, latent)
        dtype = get_torch_dtype(self.train_config.dtype)
        with torch.no_grad():
            noisy, timestep, _ = self._noised_state(latent)
            positive = positive.detach().to(self.device_torch, dtype=dtype)
            negative = negative.detach().to(self.device_torch, dtype=dtype) if negative is not None else None
        network = self.network
        previous_active, previous_multiplier = network.is_active, network.multiplier
        total = 0.0
        try:
            network.is_active = False
            with torch.no_grad():
                reference = self._predict(noisy, timestep, positive, negative).detach()
            network.is_active = True
            # Both endpoints plus a random intermediate, all against the same
            # disabled-slider reference. Backprop before changing strength so
            # checkpoint recomputation uses the correct LoRA/DoRA multiplier.
            if getattr(self, "slider_multipoint", False):
                endpoints = [p["strength"] for p in self.slider_points if p["strength"] != 0]
                intermediate = random.uniform(min(0, *endpoints), max(0, *endpoints))
                strengths = endpoints + ([intermediate] if intermediate != 0 else [])
            else:
                strengths = (-1.0, 1.0, random.uniform(-1.0, 1.0))
            for strength in strengths:
                network.multiplier = strength
                loss = F.mse_loss(self._predict(noisy, timestep, positive, negative), reference)
                weighted = loss * (self.slider_preservation_weight / len(strengths))
                self.accelerator.backward(weighted * accum_scale)
                total += weighted.detach()
        finally:
            network.is_active, network.multiplier = previous_active, previous_multiplier
        self.additional_logs["slider/anchor_loss"] = total.item()
        return total

    @staticmethod
    def _pair_key(item):
        return (
            item.path, item.unconditional_path, item.scale_to_width, item.scale_to_height,
            item.crop_x, item.crop_y, item.crop_width, item.crop_height,
            item.flip_x, item.flip_y,
        )

    def _cache_negative_latents(self):
        if self.sd.vae is None:
            raise ValueError("Image slider requires the VAE to cache its -1 pairs")
        self.sd.set_device_state_preset("cache_latents")
        try:
            with torch.no_grad():
                for dataset in get_dataloader_datasets(self.data_loader):
                    for item in tqdm(dataset.file_list, desc="Caching slider -1 latents"):
                        key = self._pair_key(item)
                        if key not in self.negative_latents:
                            item.load_unconditional_image()
                            try:
                                pixels = item.unconditional_tensor.unsqueeze(0).to(
                                    self.sd.vae_device_torch, dtype=self.sd.vae_torch_dtype
                                )
                                latent = self.sd.encode_images(pixels).detach().squeeze(0).cpu()
                                self.negative_latents[key] = latent
                            finally:
                                item.cleanup_unconditional()
                        # Workers start after this hook. They need only the +1
                        # cached latent; the -1 latent stays in the trainer.
                        item.has_unconditional = False
        finally:
            self.sd.restore_device_state()
        self.print(f"Image slider cached {len(self.negative_latents)} -1 latents in RAM")

    @staticmethod
    def _multipoint_key(item):
        group = getattr(getattr(item, "dataset_config", None), "fizgig_multipoint_group", None)
        return (group, item.path, item.scale_to_width, item.scale_to_height, item.crop_x,
                item.crop_y, item.crop_width, item.crop_height, item.flip_x, item.flip_y)

    @torch.no_grad()
    def _cache_multipoint_image_latents(self):
        # Clone only dataset/item metadata, NOT models or tensors. Reuse the
        # normal threaded disk-cache pipeline, but copy the reference item's
        # exact crop/flip so even random crops stay paired across every point.
        for dataset in get_dataloader_datasets(self.data_loader):
            group = self.slider_image_groups[dataset.dataset_config.fizgig_multipoint_group]
            for point in self.slider_points:
                ident = point["id"]
                cached_dataset = copy.copy(dataset)
                # Dataset.__getstate__ deliberately removes sd when copied or
                # pickled for workers. This cache runs in the main process.
                cached_dataset.sd = self.sd
                cached_dataset.dataset_config = copy.copy(dataset.dataset_config)
                cached_dataset.dataset_config.folder_path = group["folders"][ident]
                cached_dataset.dataset_config.dataset_path = None
                cached_dataset.dataset_path = group["folders"][ident]
                cached_dataset.latent_cache = {}
                cached_dataset.file_list = []
                keys = []
                for original in dataset.file_list:
                    item = copy.copy(original)
                    item.dataset_config = cached_dataset.dataset_config
                    item.path = group["images"][ident][os.path.splitext(os.path.basename(original.path))[0]]
                    item._latent_path = None
                    item.is_latent_cached = False
                    item.tensor = None
                    for name in ("_encoded_latent", "_cached_first_frame_latent", "_cached_tensor_uint8"):
                        setattr(item, name, None)
                    cached_dataset.file_list.append(item)
                    keys.append(self._multipoint_key(original))
                cached_dataset.cache_latents_all_latents()
                if len(cached_dataset.file_list) != len(keys):
                    raise ValueError("Multi-point cache failed to load a paired image; refusing partial training")
                bank = self.slider_multipoint_latents.setdefault(ident, {})
                for key, item in zip(keys, cached_dataset.file_list):
                    try:
                        bank[key] = item.get_latent().detach().cpu()
                    finally:
                        item.cleanup_latent()
        self.print("Multi-point image latents cached on CPU/disk with shared crops and flips")

    def _build_prompt_bank(self):
        from torchvision.transforms.functional import to_tensor

        self.sd.save_device_state()
        network = self.network
        previous_multiplier, previous_active = network.multiplier, network.is_active
        try:
            # The prompt embeddings are already cached. Keep the Qwen
            # text encoder on CPU rather than co-resident with DiT and VAE.
            self.sd.text_encoder_to("cpu")
            self.sd.unet.to(self.device_torch)
            self.sd.vae.to(self.sd.vae_device_torch)
            network.multiplier = 0.0
            network.is_active = False
            # Practice states come from the same assisted base used for training.
            with torch.no_grad():
                pipeline = self.sd.get_generation_pipeline()
                for index in tqdm(range(self.slider_bank_size), desc="Rendering slider practice images"):
                    triplet_index = index % len(self.slider_prompt_triplets)
                    neutral_cpu = self.slider_embeds[triplet_index][0]
                    neutral = (neutral_cpu.detach() if getattr(self, "slider_multipoint", False) else neutral_cpu).to(
                        self.device_torch, dtype=self.sd.torch_dtype
                    )
                    unconditional = (
                        (self.slider_cfg_negative_embeds[triplet_index][0].detach()
                         if getattr(self, "slider_multipoint", False) else self.slider_cfg_negative_embeds[triplet_index][0]).to(
                            self.device_torch, dtype=self.sd.torch_dtype
                        ) if self.slider_cfg_negative_embeds is not None else None
                    )
                    seed = int(self.sample_config.seed) + 1000 + index
                    image = render_flow_bank_image(
                        self.sd, pipeline, neutral, unconditional,
                        height=self.slider_bank_resolution,
                        width=self.slider_bank_resolution,
                        steps=self.slider_bank_steps, cfg=self.slider_cfg, seed=seed)
                    pixels = to_tensor(image).mul(2).sub(1).to(self.device_torch, dtype=self.sd.vae_torch_dtype)
                    latent = self.sd.encode_images(pixels.unsqueeze(0)).detach().cpu()
                    self.slider_bank.append((latent, triplet_index))
                    from toolkit.training_examples import save_practice_source
                    save_practice_source(self, latent, image, f'practice_{index:06d}', self.slider_prompt_triplets[triplet_index][0])
        finally:
            network.multiplier = previous_multiplier
            network.is_active = previous_active
            self.sd.restore_device_state()
        self.print(
            f"Prompt slider practice bank: {len(self.slider_bank)} images "
            f"across {len(self.slider_prompt_triplets)} triplets"
        )

    def _noised_state(self, clean):
        # Fizgig v6.7.0: sigmoid-normal time, Qwen's resolution shift, linear
        # interpolation, and velocity target noise - clean.
        clean = clean.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
        batch_size, _, height, width = clean.shape
        mu = trainer_flow_profile(self).shift(height, width)
        t = torch.sigmoid(torch.randn(batch_size, device=self.device_torch, dtype=torch.float32) + mu)
        min_t = self.train_config.min_denoising_steps / 1000.0
        max_t = self.train_config.max_denoising_steps / 1000.0
        t = min_t + (max_t - min_t) * t
        noise = torch.randn_like(clean, dtype=torch.float32)
        t_view = t.view(-1, 1, 1, 1)
        noisy = ((1 - t_view) * clean.float() + t_view * noise).to(clean.dtype)
        target = noise - clean.float()
        return noisy, (t * 1000).to(self.device_torch), target

    def _predict(self, noisy, timestep, embeds, unconditional_embeds=None):
        if getattr(self.sd, 'arch', 'qwen_image_2') != 'qwen_image_2':
            # Image-pair captions and image preservation examples have no
            # text-negative branch. Match Qwen's existing CFG-1 prediction
            # there, even when prompt practice images use a higher CFG. The
            # native Ideogram image-only reference does not need text embeds.
            cfg = self.slider_cfg if (
                unconditional_embeds is not None
                or getattr(self, 'cfg_reference', None) == 'image_only'
            ) else 1.0
            return guided_flow_prediction(self.sd, noisy, timestep, embeds, unconditional_embeds, cfg)
        return self.sd.predict_noise(
            latents=noisy,
            timestep=timestep,
            conditional_embeddings=embeds,
            unconditional_embeddings=unconditional_embeds,
            guidance_scale=self.slider_cfg if unconditional_embeds is not None else 1.0,
            guidance_embedding_scale=1.0,
            batch=None,
        ).float()

    def _image_loss(self, clean, other, embeds):
        noisy, timestep, target = self._noised_state(clean)
        prediction = self._predict(noisy, timestep, embeds)
        squared_error = (prediction - target).square().mean(dim=1).flatten(1)
        if self.slider_diff_weight:
            weights = pair_difference_weights(clean, other, self.slider_diff_weight)
            squared_error = squared_error * weights
        return squared_error.mean()

    def _train_multipoint_prompt(self, accum_scale):
        latent, entry_index = random.choice(self.slider_bank)
        from toolkit.training_examples import remember_bank_source
        remember_bank_source(self, latent)
        noisy, timestep, _ = self._noised_state(latent)
        neutral = self.slider_embeds[entry_index][0].detach().to(self.device_torch, dtype=self.sd.torch_dtype)
        negative = self.slider_cfg_negative_embeds[entry_index][0].detach().to(self.device_torch, dtype=self.sd.torch_dtype) if self.slider_cfg_negative_embeds is not None else None
        points = [p for p in self.slider_points if p["strength"] != 0]
        total = 0.0
        for point in points:
            positive_cpu, negative_cpu = self.slider_target_embeds[entry_index][point["id"]]
            self.network.is_active = False
            with torch.no_grad():
                positive = positive_cpu.detach().to(self.device_torch, dtype=self.sd.torch_dtype)
                cfg_negative = negative_cpu.detach().to(self.device_torch, dtype=self.sd.torch_dtype) if negative_cpu is not None else None
                target = self._predict(noisy, timestep, positive, cfg_negative).detach()
                del positive, cfg_negative
            self.network.is_active = True
            self.network.multiplier = point["strength"]
            loss = F.mse_loss(self._predict(noisy, timestep, neutral, negative), target)
            self.accelerator.backward(loss * (accum_scale / len(points)))
            total += loss.detach() / len(points)
            del loss, target
        return total

    def _train_multipoint_image(self, batch, accum_scale):
        if batch is None or batch.latents is None:
            raise ValueError("Multi-point image slider requires cached image batches")
        points = [p for p in self.slider_points if p["strength"] != 0]
        from toolkit.training_examples import remember_source, source_item, recording_enabled
        if recording_enabled(self):
            for item in batch.file_items:
                group = self.slider_image_groups[item.dataset_config.fizgig_multipoint_group]
                stem = os.path.splitext(os.path.basename(item.path))[0]
                for point in self.slider_points:
                    remember_source(self, source_item(group['images'][point['id']][stem], item.caption,
                        group['folders'][point['id']], role=f"slider_strength_{point['strength']}"))
        dtype = get_torch_dtype(self.train_config.dtype)
        def pixels(point):
            return torch.stack([self.slider_multipoint_latents[point["id"]][self._multipoint_key(item)] for item in batch.file_items])
        # Spatial weights are computed on CPU; only the current point's target
        # goes to GPU. More points must not multiply persistent GPU storage.
        with torch.no_grad():
            zero = next((p for p in self.slider_points if p["strength"] == 0), None)
            reference = pixels(zero) if zero else None
            low = high = None
            if zero is None:
                for point in points:
                    latent = pixels(point).float()
                    low = latent if low is None else torch.minimum(low, latent)
                    high = latent if high is None else torch.maximum(high, latent)
            shared_weights = pair_difference_weights(high, low, self.slider_diff_weight) if zero is None else None
            embeds = batch.prompt_embeds if batch.prompt_embeds is not None else self.sd.encode_prompt(batch.get_caption_list())
            embeds = embeds.detach().to(self.device_torch, dtype=dtype)
        total = 0.0
        for point in points:
            with torch.no_grad():
                clean_cpu = pixels(point)
                weights = pair_difference_weights(clean_cpu, reference, self.slider_diff_weight) if zero else shared_weights
                clean = clean_cpu.to(self.device_torch, dtype=dtype)
                noisy, timestep, target = self._noised_state(clean)
            self.network.multiplier = point["strength"]
            prediction = self._predict(noisy, timestep, embeds)
            loss = ((prediction - target).square().mean(dim=1).flatten(1) * weights.to(self.device_torch)).mean()
            self.accelerator.backward(loss * (accum_scale / len(points)))
            total += loss.detach() / len(points)
            del prediction, loss, clean, noisy, target
        return total

    def train_single_accumulation(self, batch, accum_scale=1.0):
        network = self.network
        previous_multiplier, previous_active = network.multiplier, network.is_active
        total = 0.0
        try:
            network.is_active = True
            if getattr(self, "slider_multipoint", False):
                total = (self._train_multipoint_image(batch, accum_scale) if self.slider_mode == "image_pairs"
                         else self._train_multipoint_prompt(accum_scale))
            elif self.slider_mode == "image_pairs":
                if batch is None:
                    raise ValueError("Image slider requires a paired batch")
                with torch.no_grad():
                    dtype = get_torch_dtype(self.train_config.dtype)
                    if batch.latents is None:
                        raise ValueError("Image slider +1 latents were not cached")
                    positive = batch.latents.to(self.device_torch, dtype=dtype)
                    negative = torch.stack([
                        self.negative_latents[self._pair_key(item)] for item in batch.file_items
                    ]).to(self.device_torch, dtype=dtype)
                    if positive.shape != negative.shape:
                        raise ValueError("Image slider pair latents have different shapes")
                    embeds = (batch.prompt_embeds if batch.prompt_embeds is not None
                              else self.sd.encode_prompt(batch.get_caption_list()))
                    embeds = embeds.to(self.device_torch, dtype=dtype).detach()
                for multiplier, clean, other in ((1.0, positive, negative), (-1.0, negative, positive)):
                    network.multiplier = multiplier
                    loss = self._image_loss(clean, other, embeds)
                    self.accelerator.backward(loss * (0.5 * accum_scale))
                    total += 0.5 * loss.detach()
            else:
                bank_latent, triplet_index = random.choice(self.slider_bank)
                from toolkit.training_examples import remember_bank_source
                remember_bank_source(self, bank_latent)
                clean = bank_latent.to(self.device_torch)
                noisy, timestep, _ = self._noised_state(clean)
                neutral, positive, negative = [embed.to(self.device_torch, dtype=self.sd.torch_dtype)
                                               for embed in self.slider_embeds[triplet_index]]
                cfg_negatives = (
                    tuple(embed.to(self.device_torch, dtype=self.sd.torch_dtype)
                          for embed in self.slider_cfg_negative_embeds[triplet_index])
                    if self.slider_cfg_negative_embeds is not None else (None, None, None)
                )
                network.multiplier = 0.0
                with torch.no_grad():
                    frozen_neutral = self._predict(noisy, timestep, neutral, cfg_negatives[0])
                    delta = self.slider_guidance * (
                        self._predict(noisy, timestep, positive, cfg_negatives[1])
                        - self._predict(noisy, timestep, negative, cfg_negatives[2])
                    )
                for multiplier in (1.0, -1.0):
                    network.multiplier = multiplier
                    loss = F.mse_loss(self._predict(noisy, timestep, neutral, cfg_negatives[0]),
                                      frozen_neutral + multiplier * delta)
                    self.accelerator.backward(loss * (0.5 * accum_scale))
                    total += 0.5 * loss.detach()
            if getattr(self, "slider_anchor_bank", []) and self.slider_preservation_weight:
                total += self._train_anchor_preservation(accum_scale)
        finally:
            network.multiplier = previous_multiplier
            network.is_active = previous_active
        return total
