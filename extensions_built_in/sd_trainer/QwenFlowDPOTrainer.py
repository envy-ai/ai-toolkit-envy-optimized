"""Offline, paired Flow-DPO for Qwen Image 2.1 LoRAs.

Target Dataset is the preferred output. Control Dataset 1 is the rejected
output, routed through the image-slider pair loader rather than Qwen's edit
references. Control Datasets 2/3, when present, remain edit inputs.
"""

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from toolkit.flow_training import FlowTrainingProfile, trainer_flow_profile, scoped_training_references
from toolkit.training_capabilities import validate_specialized_model, validate_edit_references
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.FizgigSliderTrainer import validate_image_slider_pairs
from toolkit.data_loader import get_dataloader_datasets
from toolkit.train_tools import get_torch_dtype


def flow_dpo_terms(policy_win, policy_lose, reference_win, reference_lose, beta, sft_weight):
    """Return per-example loss, margin and exact error-gradient coefficient."""
    margin = (reference_win - policy_win) - (reference_lose - policy_lose)
    logit = (beta / 2.0) * margin
    dpo_loss = F.softplus(-logit)
    loss = dpo_loss + sft_weight * policy_win
    coefficient = (beta / 2.0) * torch.sigmoid(-logit)
    return loss, margin, coefficient


class QwenFlowDPOTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        validate_specialized_model(config, 'flow_dpo')
        self.flow_profile = FlowTrainingProfile.from_model_config(config['model'])
        from toolkit.training_capabilities import cfg_reference_mode
        self.cfg_reference = cfg_reference_mode(config['model'])
        if (config.get("network") or {}).get("type") != "lora":
            raise ValueError("Qwen Flow-DPO currently supports LoRA only")
        network_config = config["network"]
        if network_config.get("pretrained_lora_path"):
            raise ValueError("Qwen Flow-DPO cannot use a pretrained LoRA: its frozen reference would differ")
        if network_config.get("dropout") not in (None, 0, 0.0):
            raise ValueError("Qwen Flow-DPO requires LoRA dropout 0 for exact recomputed gradients")
        train = config.setdefault("train", {})
        if train.get("noise_scheduler") != "flowmatch":
            raise ValueError("Qwen Flow-DPO requires the flowmatch scheduler")
        if train.get("timestep_type", "shift") != "shift":
            raise ValueError("Qwen Flow-DPO uses Qwen's shifted timestep sampling")
        if train.get("train_text_encoder") or train.get("train_unet") is False:
            raise ValueError("Qwen Flow-DPO trains only a transformer LoRA")
        if not train.get("cache_text_embeddings"):
            raise ValueError("Qwen Flow-DPO requires cached text embeddings")
        if train.get("do_cfg") or train.get("do_random_cfg"):
            raise ValueError("Qwen Flow-DPO does not support CFG training")
        if train.get("diff_output_preservation") or train.get("blank_prompt_preservation"):
            raise ValueError("Qwen Flow-DPO does not support output-preservation losses")
        if train.get("frequency_loss_type", "none") != "none":
            raise ValueError("Qwen Flow-DPO does not support auxiliary frequency loss")
        if train.get("do_guidance_loss") or train.get("do_differential_guidance"):
            raise ValueError("Qwen Flow-DPO does not support auxiliary guidance losses")
        if (train.get("ema_config") or {}).get("use_ema"):
            raise ValueError("Qwen Flow-DPO does not support EMA")

        settings = config.get("flow_dpo") or {}
        self.dpo_beta = float(settings.get("beta", 1.0))
        self.dpo_sft_weight = float(settings.get("sft_weight", 0.0))
        if not math.isfinite(self.dpo_beta) or self.dpo_beta <= 0:
            raise ValueError("Qwen Flow-DPO beta must be finite and positive")
        if not math.isfinite(self.dpo_sft_weight) or self.dpo_sft_weight < 0:
            raise ValueError("Qwen Flow-DPO preferred-image loss weight must be finite and nonnegative")

        datasets = config.get("datasets") or []
        validate_edit_references(config, datasets, paired=True)
        validate_image_slider_pairs(datasets, label="Qwen Flow-DPO", allow_edit_controls=True)
        for dataset in datasets:
            loss_weight = float(dataset.get("loss_multiplier", 1.0))
            if not math.isfinite(loss_weight) or loss_weight < 0:
                raise ValueError("Flow-DPO Dataset Loss Weight must be finite and nonnegative")
            if dataset.get("unconditional_path"):
                raise ValueError("Qwen Flow-DPO reserves unconditional_path for Control Dataset 1")
            if dataset.get("random_crop") or dataset.get("random_scale"):
                raise ValueError("Qwen Flow-DPO pairs require deterministic crops")
            if float(dataset.get("network_weight", 1.0)) != 1.0:
                raise ValueError("Qwen Flow-DPO requires LoRA Weight = 1 for paired reference consistency")
            dataset["unconditional_path"] = dataset.pop("control_path_1")
            dataset["caption_dropout_rate"] = 0.0
            dataset["flow_dpo_pair"] = True
        super().__init__(process_id, job, config, **kwargs)
        self.rejected_latents = {}
        # Rejected target latents are cached up front; only real edit references
        # need the VAE during prediction. Do not let Unload VAE delete it then.
        self.needs_vae_at_train_time = any(
            dataset.get(key) for dataset in datasets
            for key in ('control_path', 'control_path_2', 'control_path_3')
        )

    def update_training_metadata(self):
        super().update_training_metadata()
        from toolkit.flow_training import flow_training_metadata
        self.add_meta({**flow_training_metadata(self), "ss_flow_dpo_beta": str(self.dpo_beta),
                       "ss_flow_dpo_sft_weight": str(self.dpo_sft_weight)})

    @staticmethod
    def _pair_key(item):
        return (
            item.path, item.unconditional_path, item.scale_to_width, item.scale_to_height,
            item.crop_x, item.crop_y, item.crop_width, item.crop_height,
            item.flip_x, item.flip_y,
        )

    def hook_before_train_loop(self):
        from toolkit.flow_training import log_flow_training_profile
        log_flow_training_profile(self)
        if self.sd.vae is None:
            raise ValueError("Qwen Flow-DPO requires the VAE to cache rejected images")
        self.sd.set_device_state_preset("cache_latents")
        try:
            with torch.no_grad():
                for dataset in get_dataloader_datasets(self.data_loader):
                    for item in tqdm(dataset.file_list, desc="Caching DPO rejected latents"):
                        key = self._pair_key(item)
                        if key not in self.rejected_latents:
                            item.load_unconditional_image()
                            try:
                                pixels = item.unconditional_tensor.unsqueeze(0).to(
                                    self.sd.vae_device_torch, dtype=self.sd.vae_torch_dtype
                                )
                                self.rejected_latents[key] = self.sd.encode_images(pixels).detach().squeeze(0).cpu()
                            finally:
                                item.cleanup_unconditional()
                        # Workers need the preferred cached latent only.
                        item.has_unconditional = False
        finally:
            self.sd.restore_device_state()
        self.print(f"Flow-DPO cached {len(self.rejected_latents)} rejected latents in RAM")
        super().hook_before_train_loop()

    def _predict(self, noisy, timesteps, embeds, batch):
        return self.sd.predict_noise(
            latents=noisy,
            timestep=timesteps,
            conditional_embeddings=embeds,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
        ).float()

    @scoped_training_references
    def train_single_accumulation(self, batch, accum_scale=1.0):
        if batch is None or batch.latents is None or batch.prompt_embeds is None:
            raise ValueError("Qwen Flow-DPO requires cached preferred latents and text embeddings")
        dtype = get_torch_dtype(self.train_config.dtype)
        with torch.no_grad():
            preferred = batch.latents.to(self.device_torch, dtype=dtype)
            rejected = torch.stack([
                self.rejected_latents[self._pair_key(item)] for item in batch.file_items
            ]).to(self.device_torch, dtype=dtype)
            if preferred.shape != rejected.shape:
                raise ValueError("Qwen Flow-DPO pair latent shapes differ")
            embeds = batch.prompt_embeds.to(self.device_torch, dtype=dtype).detach()
            n, _, height, width = preferred.shape
            t = torch.sigmoid(torch.randn(n, device=self.device_torch, dtype=torch.float32)
                              + trainer_flow_profile(self).shift(height, width))
            t_min = self.train_config.min_denoising_steps / 1000.0
            t_max = self.train_config.max_denoising_steps / 1000.0
            t = t_min + (t_max - t_min) * t
            timesteps = t * 1000.0
            noise = torch.randn(preferred.shape, device=self.device_torch, dtype=torch.float32)
            t_view = t.view(n, 1, 1, 1)
            preferred_noisy = ((1 - t_view) * preferred.float() + t_view * noise).to(dtype)
            rejected_noisy = ((1 - t_view) * rejected.float() + t_view * noise).to(dtype)
            preferred_target = noise - preferred.float()
            rejected_target = noise - rejected.float()
            weights = torch.as_tensor(getattr(batch, 'loss_multiplier_list', [1.0] * n),
                                      device=self.device_torch, dtype=torch.float32)
            if weights.shape != (n,) or not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Flow-DPO requires one finite nonnegative Dataset Loss Weight per image")

        def error(noisy, target):
            return (self._predict(noisy, timesteps, embeds, batch) - target).square().flatten(1).mean(1)

        network = self.network
        was_active = network.is_active
        try:
            network.is_active = False
            with torch.no_grad():
                reference_win = error(preferred_noisy, preferred_target)
                reference_lose = error(rejected_noisy, rejected_target)
            network.is_active = True
            # Detached score passes avoid keeping two Qwen transformer graphs.
            with torch.no_grad():
                policy_win = error(preferred_noisy, preferred_target)
                policy_lose = error(rejected_noisy, rejected_target)
                loss, margin, coefficient = flow_dpo_terms(
                    policy_win, policy_lose, reference_win, reference_lose,
                    self.dpo_beta, self.dpo_sft_weight,
                )
            # These are the exact partial derivatives of the pairwise loss
            # with respect to each policy error, evaluated at the same weights.
            win_error = error(preferred_noisy, preferred_target)
            self.accelerator.backward(
                (weights * (coefficient + self.dpo_sft_weight) * win_error).mean() * accum_scale
            )
            del win_error
            lose_error = error(rejected_noisy, rejected_target)
            self.accelerator.backward((-weights * coefficient * lose_error).mean() * accum_scale)
            del lose_error
            self.additional_logs.update({
                "dpo/margin": margin.mean().item(),
                "dpo/preference_accuracy": (margin > 0).float().mean().item(),
                "dpo/preferred_error": policy_win.mean().item(),
                "dpo/rejected_error": policy_lose.mean().item(),
                "dpo/reference_preferred_error": reference_win.mean().item(),
                "dpo/reference_rejected_error": reference_lose.mean().item(),
            })
            return (weights * loss).mean().detach()
        finally:
            network.is_active = was_active
