"""Distill a frozen Qwen Image 2.1 teacher's CFG into a CFG-1 LoRA.

The teacher and student share the frozen transformer. Only the student's
LoRA is trainable; teacher passes run sequentially with that adapter disabled.
Dataset images/captions supply latent states and positive conditioning.
"""

import json
import math
import os
from collections import OrderedDict

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm

from toolkit.flow_training import (FlowTrainingProfile, trainer_flow_profile, image_only_prompt_embeds,
                                   scoped_training_references)
from toolkit.training_capabilities import validate_specialized_model, validate_edit_references, cfg_reference_mode
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.basic import flush
from toolkit.data_loader import get_dataloader_datasets
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype


def guidance_distillation_target(positive, negative, cfg_scale, *, blank=None):
    """Full CFG, or only the contribution of replacing blank with negative."""
    if blank is None:
        return negative + cfg_scale * (positive - negative)
    return positive + (cfg_scale - 1.0) * (blank - negative)


class QwenGuidanceDistillationTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        validate_specialized_model(config, 'guidance_distillation')
        self.flow_profile = FlowTrainingProfile.from_model_config(config['model'])
        self.cfg_reference = cfg_reference_mode(config['model'])
        if (config.get("network") or {}).get("type") != "lora":
            raise ValueError("Qwen guidance distillation currently supports LoRA only")
        train = config.setdefault("train", {})
        if train.get("train_text_encoder") or train.get("train_unet") is False:
            raise ValueError("Guidance distillation trains only a transformer LoRA")
        if train.get("noise_scheduler") != "flowmatch" or train.get("timestep_type", "shift") != "shift":
            raise ValueError("Guidance distillation requires flowmatch with shifted timesteps")
        if not train.get("cache_text_embeddings"):
            raise ValueError("Guidance distillation requires Cache Text Embeddings")
        if train.get("unload_text_encoder"):
            raise ValueError("Use Cache Text Embeddings, not the blank-prompt Unload TE mode")
        if train.get("loss_type", "mse") != "mse":
            raise ValueError("Guidance distillation uses mean squared prediction error")
        conflicting = (
            "do_cfg", "do_random_cfg", "do_guidance_loss", "do_differential_guidance",
            "diff_output_preservation", "blank_prompt_preservation", "do_prior_divergence",
            "train_turbo", "inverted_mask_prior", "do_fft_loss", "correct_pred_norm",
            "learnable_snr_gos",
        )
        if any(train.get(key) for key in conflicting) or train.get("frequency_loss_type", "none") != "none":
            raise ValueError("Guidance distillation cannot be combined with other training objectives")
        if train.get("diffusion_feature_extractor_path") or any(config.get(key) for key in ("adapter", "embedding", "decorator")):
            raise ValueError("Guidance distillation does not support additional trainable adapters or feature losses")
        if config["model"].get("unconditional_lora_path"):
            raise ValueError("Guidance distillation requires the base teacher without unconditional adapters")
        if (train.get("ema_config") or {}).get("use_ema"):
            raise ValueError("Guidance distillation does not support EMA")
        minimum = float(train.get("min_denoising_steps", 0))
        maximum = float(train.get("max_denoising_steps", 999))
        if not 0 <= minimum < maximum <= 1000:
            raise ValueError("Guidance distillation requires 0 <= min timestep < max timestep <= 1000")

        settings = config.get("guidance_distillation") or {}
        self.teacher_cfg_scale = float(settings.get("teacher_cfg_scale", 4.0))
        self.teacher_negative_prompt = settings.get("negative_prompt", "")
        self.distillation_objective = settings.get("objective", "full_guidance")
        if not math.isfinite(self.teacher_cfg_scale) or self.teacher_cfg_scale < 1:
            raise ValueError("Teacher CFG must be finite and at least 1")
        if not isinstance(self.teacher_negative_prompt, str):
            raise ValueError("Teacher negative prompt must be text")
        if self.distillation_objective not in ("full_guidance", "negative_only"):
            raise ValueError("Guidance distillation objective must be full_guidance or negative_only")
        if config['model']['arch'] == 'ideogram4' and self.distillation_objective == 'negative_only':
            if self.cfg_reference != 'negative_prompt' or not self.teacher_negative_prompt.strip():
                raise ValueError('Ideogram negative-only distillation requires text-negative CFG and a nonempty negative prompt')

        logging = config.get('logging') or {}
        self.record_training_examples = logging.get('record_training_examples', True)
        self.record_training_rng = logging.get('record_training_rng', False)
        for key in ('record_training_examples', 'record_training_rng'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'logging.{key} must be true or false')
        if self.record_training_rng and not self.record_training_examples:
            raise ValueError('Record training examples before enabling training RNG snapshots')

        datasets = config.get("datasets") or []
        validate_edit_references(config, datasets)
        if not datasets:
            raise ValueError("Guidance distillation requires image/caption datasets")
        for dataset in datasets:
            if not dataset.get("folder_path") or not os.path.isdir(dataset["folder_path"]):
                raise ValueError(f"Guidance distillation dataset does not exist: {dataset.get('folder_path')}")
            if not dataset.get("buckets", True) or dataset.get("num_frames", 1) != 1:
                raise ValueError("Guidance distillation requires bucketed still images")
            if not (dataset.get("cache_latents_to_disk") or dataset.get("cache_latents")):
                raise ValueError("Guidance distillation requires Cache Latents for each dataset")
            if dataset.get("is_reg") or dataset.get("unconditional_path") or dataset.get("controls"):
                raise ValueError("Guidance distillation uses ordinary target images and optional edit references")
            if dataset.get("mask_path") or dataset.get("alpha_mask"):
                raise ValueError("Guidance distillation does not support masked losses")
            if dataset.get("augmentations") or dataset.get("augments") or dataset.get("random_crop") or dataset.get("random_scale"):
                raise ValueError("Guidance distillation requires deterministic cached image presentation")
            if dataset.get("control_from_same_folder") or dataset.get("inpaint_path"):
                raise ValueError("Guidance distillation requires fixed edit references, not random or inpaint controls")
            if dataset.get("shuffle_tokens") or dataset.get("token_dropout_rate", 0):
                raise ValueError("Guidance distillation requires deterministic cached captions")
            if float(dataset.get("network_weight", 1.0)) != 1.0:
                raise ValueError("Guidance distillation requires dataset LoRA Weight = 1")
            dataset["caption_dropout_rate"] = 0.0

        super().__init__(process_id, job, config, **kwargs)
        self.teacher_prompt_paths = {}
        # Edit predictions still need to VAE-encode their reference images.
        self.needs_vae_at_train_time = any(
            dataset.get(key) for dataset in datasets
            for key in ("control_path", "control_path_1", "control_path_2", "control_path_3", "control_from_same_folder")
        )

    def update_training_metadata(self):
        super().update_training_metadata()
        from toolkit.flow_training import flow_training_metadata
        self.add_meta({
            **flow_training_metadata(self),
            "ss_guidance_distillation": self.distillation_objective,
            "ss_teacher_cfg_scale": str(self.teacher_cfg_scale),
            "ss_teacher_negative_prompt": self.teacher_negative_prompt,
            "ss_student_cfg_scale": "1.0",
        })

    def _cache_teacher_prompts(self):
        if self.teacher_cfg_scale <= 1:
            return
        if getattr(self, 'cfg_reference', None) == 'image_only':
            return # Native Ideogram full-CFG needs no negative text encoder cache.
        captions = [self.teacher_negative_prompt]
        if self.distillation_objective == "negative_only" and getattr(self.sd, 'arch', None) != 'ideogram4':
            captions.append("")
        # Share text-only embeddings (and identical edit presentations) across
        # items. Existing keys include encoder identity, reference mtimes and
        # resize/crop/flip settings. Teacher CFG does not affect embeddings.
        paths_by_identity = {}
        self.sd.unet.to("cpu")
        flush()
        try:
            self.sd.text_encoder_to(self.device_torch)
            with torch.no_grad():
                for dataset in get_dataloader_datasets(self.data_loader):
                    for item in tqdm(dataset.file_list, desc="Caching guidance teacher prompts"):
                        prompt_paths = []
                        for caption in captions:
                            identity = json.dumps(item.get_text_embedding_info_dict(caption_override=caption), sort_keys=True)
                            path = paths_by_identity.setdefault(identity, item._build_text_embedding_path(caption_override=caption))
                            prompt_paths.append(path)
                        missing = [(path, caption) for path, caption in zip(prompt_paths, captions) if not os.path.exists(path)]
                        if missing:
                            controls = None
                            loaded_controls = False
                            try:
                                if item.encode_control_in_text_embeddings and item.control_path is not None:
                                    if getattr(item, "cache_processed_control_text_embeddings", False):
                                        loaded_controls = True
                                        item.load_control_image()
                                        controls = item.control_tensor_list
                                        if controls is None and item.control_tensor is not None:
                                            tensor = item.control_tensor
                                            controls = list(tensor) if tensor.dim() == 4 else [tensor]
                                        if getattr(self.sd, 'arch', None) == 'krea2' and controls is not None:
                                            # Match the positive disk-cache path's
                                            # device/dtype BEFORE VLM resizing. In
                                            # bf16 even a different pixel rounding
                                            # here changes the teacher conditioning.
                                            controls = [
                                                (tensor.unsqueeze(0) if tensor.dim() == 3 else tensor).to(
                                                    self.sd.device_torch, dtype=self.sd.torch_dtype
                                                ) for tensor in controls
                                            ]
                                    else:
                                        # Match the existing raw-reference text-cache path.
                                        paths = item.control_path if isinstance(item.control_path, list) else [item.control_path]
                                        controls = []
                                        for control_path in paths:
                                            with Image.open(control_path) as image:
                                                controls.append(to_tensor(ImageOps.exif_transpose(image).convert("RGB")))
                                for path, caption in missing:
                                    embeds = self.sd.encode_prompt(
                                        caption, control_images=controls,
                                        target_size=(item.crop_width, item.crop_height),
                                    ).detach().to("cpu")
                                    embeds.save(path)
                                    del embeds
                            finally:
                                if loaded_controls:
                                    item.cleanup_control()
                        self.teacher_prompt_paths[item.get_text_embedding_path()] = tuple(prompt_paths)
        finally:
            # Never restore the transformer onto GPU while the encoder still
            # occupies it. The normal trainer hook unloads TE after sample caches.
            self.sd.text_encoder_to("cpu")
            flush()

    def hook_before_train_loop(self):
        from toolkit.flow_training import log_flow_training_profile
        log_flow_training_profile(self)
        self._run_with_optimizer_state_offload(self._cache_teacher_prompts)
        super().hook_before_train_loop()

    def _teacher_embeds(self, batch):
        if getattr(self, 'cfg_reference', None) == 'image_only':
            return image_only_prompt_embeds(self.sd, batch.prompt_embeds), None
        paths = [self.teacher_prompt_paths[item.get_text_embedding_path()] for item in batch.file_items]
        dtype = get_torch_dtype(self.train_config.dtype)
        negative = concat_prompt_embeds([PromptEmbeds.load(item_paths[0]) for item_paths in paths]).to(
            self.device_torch, dtype=dtype,
        ).detach()
        blank = None
        if self.distillation_objective == "negative_only":
            if getattr(self.sd, 'arch', None) == 'ideogram4':
                blank = image_only_prompt_embeds(self.sd, batch.prompt_embeds).to(self.device_torch, dtype=dtype)
            else:
                blank = concat_prompt_embeds([PromptEmbeds.load(item_paths[1]) for item_paths in paths]).to(
                    self.device_torch, dtype=dtype,
                ).detach()
        return negative, blank

    def _predict(self, noisy, timesteps, embeds, batch):
        # Sequential teacher passes and the student all use a single branch.
        return self.sd.predict_noise(
            latents=noisy, timestep=timesteps, conditional_embeddings=embeds,
            unconditional_embeddings=None, guidance_scale=1.0,
            guidance_embedding_scale=1.0, batch=batch,
        ).float()

    @scoped_training_references
    def train_single_accumulation(self, batch, accum_scale=1.0):
        if batch is None or batch.latents is None or batch.prompt_embeds is None:
            raise ValueError("Guidance distillation requires cached latents and positive text embeddings")
        dtype = get_torch_dtype(self.train_config.dtype)
        record_examples = getattr(self, 'record_training_examples', False) and self.accelerator.is_main_process
        rng_state = None
        if record_examples and getattr(self, 'record_training_rng', False):
            from toolkit.training_examples import capture_training_rng
            rng_state = capture_training_rng(self.device_torch)
        with torch.no_grad():
            clean = batch.latents.to(self.device_torch, dtype=dtype)
            n, _, height, width = clean.shape
            t = torch.sigmoid(torch.randn(n, device=self.device_torch, dtype=torch.float32)
                              + trainer_flow_profile(self).shift(height, width))
            t_min = self.train_config.min_denoising_steps / 1000.0
            t_max = self.train_config.max_denoising_steps / 1000.0
            t = t_min + (t_max - t_min) * t
            timesteps = t * 1000.0
            noise = torch.randn(clean.shape, device=self.device_torch, dtype=torch.float32)
            if record_examples:
                noise_mean = noise.flatten(1).mean(1)
                noise_std = noise.flatten(1).std(1, correction=0)
            t_view = t.view(n, 1, 1, 1)
            noisy = ((1 - t_view) * clean.float() + t_view * noise).to(dtype)
            del clean, noise
            positive_embeds = batch.prompt_embeds.to(self.device_torch, dtype=dtype).detach()
            negative_embeds, blank_embeds = self._teacher_embeds(batch) if self.teacher_cfg_scale > 1 else (None, None)

        network = self.network
        was_active, was_multiplier = network.is_active, network.multiplier
        was_training = self.sd.unet.training
        try:
            network.is_active = False
            self.sd.unet.eval()
            with torch.no_grad():
                teacher_positive = self._predict(noisy, timesteps, positive_embeds, batch)
                teacher_negative = self._predict(noisy, timesteps, negative_embeds, batch) if negative_embeds is not None else None
                teacher_blank = self._predict(noisy, timesteps, blank_embeds, batch) if blank_embeds is not None else None
                target = teacher_positive if teacher_negative is None else guidance_distillation_target(
                    teacher_positive, teacher_negative, self.teacher_cfg_scale, blank=teacher_blank,
                )
                correction = (target - teacher_positive).square().mean().sqrt().item()
                if record_examples:
                    per_item_correction = (target - teacher_positive).square().flatten(1).mean(1).sqrt()
                del teacher_positive, teacher_negative, teacher_blank, negative_embeds, blank_embeds
            network.multiplier = 1.0
            network.is_active = True
            self.sd.unet.train()
            prediction = self._predict(noisy, timesteps, positive_embeds, batch)
            per_item_loss = (prediction - target).square().flatten(1).mean(1)
            weights = torch.as_tensor(batch.loss_multiplier_list, device=per_item_loss.device, dtype=per_item_loss.dtype)
            loss = (per_item_loss * weights).mean()
            self.accelerator.backward(loss * accum_scale)
            if record_examples:
                from toolkit.training_examples import training_example_records
                values = torch.stack((per_item_loss.detach(), weights, timesteps.float(),
                                      per_item_correction, noise_mean, noise_std), dim=1).detach().cpu().tolist()
                self.logger.log_training_examples(training_example_records(self, batch, values), rng_state)
            self.additional_logs.update({
                "distill/teacher_correction_rms": correction,
                "distill/prediction_mse": per_item_loss.mean().detach().item(),
            })
            return loss.detach()
        finally:
            network.is_active, network.multiplier = was_active, was_multiplier
            self.sd.unet.train(was_training)
