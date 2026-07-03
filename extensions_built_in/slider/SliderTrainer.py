import copy
import random
from collections import OrderedDict
from typing import Optional


def prepare_slider_train_config_for_text_encoder_unload(process_config: dict) -> dict:
    train_config = process_config.setdefault("train", {})
    if train_config.get("train_text_encoder", False):
        raise ValueError("Slider LoRA does not support training the text encoder")

    train_config["unload_text_encoder"] = True
    train_config["cache_text_embeddings"] = False
    if process_config.get("model", {}).get("arch") == "krea2" and "max_denoising_steps" not in train_config:
        sample_steps = process_config.get("sample", {}).get("sample_steps", 12)
        train_config["max_denoising_steps"] = max(3, int(sample_steps or 12))
    return process_config


def expand_slider_timestep_for_prediction(timestep, batch_size: int):
    if len(timestep.shape) == 0:
        timestep = timestep.unsqueeze(0)
    if timestep.shape[0] == batch_size:
        return timestep
    if timestep.shape[0] != 1:
        raise ValueError(
            f"Slider timestep batch {timestep.shape[0]} does not match prediction batch {batch_size}"
        )
    repeat_shape = [batch_size] + [1] * (len(timestep.shape) - 1)
    return timestep.repeat(*repeat_shape)


def normalize_slider_targets(slider_config: dict) -> list[dict]:
    raw_targets = slider_config.get("targets") or []
    if len(raw_targets) == 0:
        raw_targets = [
            {
                "target_class": slider_config.get("target_class", ""),
                "positive": slider_config.get(
                    "positive_prompt", slider_config.get("positive", "")
                ),
                "negative": slider_config.get(
                    "negative_prompt", slider_config.get("negative", "")
                ),
                "weight": slider_config.get("weight", 1.0),
                "shuffle": slider_config.get("shuffle", False),
            }
        ]

    targets = []
    for target in raw_targets:
        normalized = {
            "target_class": target.get("target_class", ""),
            "positive": target.get("positive", target.get("positive_prompt", "")),
            "negative": target.get("negative", target.get("negative_prompt", "")),
            "weight": target.get("weight", 1.0),
            "shuffle": target.get("shuffle", False),
        }
        if "multiplier" in target:
            normalized["multiplier"] = target["multiplier"]
        if len(normalized["positive"].strip()) == 0 and len(normalized["negative"].strip()) == 0:
            raise ValueError("Slider targets require a positive prompt, negative prompt, or both")
        targets.append(normalized)

    return targets


def _get_prompt_batch_size(prompt_embeds) -> int:
    text_embeds = prompt_embeds.text_embeds
    if isinstance(text_embeds, (list, tuple)):
        if len(text_embeds) == 0:
            return 0
        if len(text_embeds[0].shape) == 2:
            return len(text_embeds)
        return text_embeds[0].shape[0]
    return text_embeds.shape[0]


def build_slider_latent_noise(
    sd,
    height: int,
    width: int,
    batch_size: int,
    device,
    dtype,
    noise_offset: float = 0.0,
):
    import torch
    from toolkit.train_tools import apply_noise_offset

    vae_scale_factor = int(getattr(sd, "vae_scale_factor", 8))
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor

    channels = None
    transformer = getattr(sd, "transformer", None)
    transformer_config = getattr(transformer, "config", None)
    if transformer_config is not None:
        channels = getattr(transformer_config, "channels", None)
        if channels is None:
            channels = getattr(transformer_config, "in_channels", None)

    if channels is None:
        unet_config = getattr(getattr(sd, "unet_unwrapped", None), "config", None)
        if unet_config is not None:
            try:
                channels = unet_config["in_channels"]
            except (KeyError, TypeError):
                channels = getattr(unet_config, "in_channels", None)

    if channels is None:
        channels = 16

    noise = torch.randn(
        (batch_size, int(channels), latent_height, latent_width),
        device=device,
        dtype=dtype,
    )
    return apply_noise_offset(noise, noise_offset)


_IMPORT_ERROR = None

try:
    import torch
    from tqdm import tqdm

    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
    from toolkit.config_modules import SliderConfig
    from toolkit.prompt_utils import (
        ACTION_TYPES_SLIDER,
        PromptEmbedsCache,
        build_prompt_pair_batch_from_cache,
        concat_prompt_embeds,
        concat_prompt_pairs,
        encode_prompts_to_cache,
    )
    from toolkit.train_tools import get_torch_dtype
except ModuleNotFoundError as e:
    _IMPORT_ERROR = e
    torch = None
    DiffusionTrainer = object
    SliderConfig = None


class SliderTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        if torch is None:
            raise ModuleNotFoundError("SliderTrainer requires torch") from _IMPORT_ERROR
        prepare_slider_train_config_for_text_encoder_unload(config)
        super().__init__(process_id, job, config, **kwargs)
        self.prompt_txt_list: Optional[list[str]] = None
        raw_slider_config = copy.deepcopy(self.config.get("slider", {}))
        raw_slider_config["targets"] = normalize_slider_targets(raw_slider_config)
        self.slider_config = SliderConfig(**raw_slider_config)
        self.prompt_cache = PromptEmbedsCache()
        self.prompt_cache.prompts = {}
        self.prompt_pairs = []
        self.prompt_chunk_size = 1

    def hook_before_train_loop(self):
        if self.slider_config.prompt_file:
            self.print(f"Loading prompt file from {self.slider_config.prompt_file}")
            with open(self.slider_config.prompt_file, "r", encoding="utf-8") as f:
                self.prompt_txt_list = [
                    line.strip() for line in f.readlines() if len(line.strip()) > 0
                ]
            self.print(f"Found {len(self.prompt_txt_list)} prompts.")
            if not self.slider_config.prompt_tensors:
                random.shuffle(self.prompt_txt_list)
                self.prompt_txt_list = self.prompt_txt_list[: self.train_config.steps]

        cache = PromptEmbedsCache()
        cache.prompts = {}
        self.print("Building slider prompt cache")
        neutral_list = self.prompt_txt_list if self.prompt_txt_list is not None else [""]

        with torch.no_grad():
            prompts_to_cache = []
            for neutral in neutral_list:
                for target in self.slider_config.targets:
                    prompts_to_cache += [
                        f"{target.target_class}",
                        f"{target.target_class} {neutral}",
                        f"{target.positive}",
                        f"{target.positive} {neutral}",
                        f"{target.negative}",
                        f"{target.negative} {neutral}",
                        f"{neutral}",
                        f"{target.positive} {target.negative}",
                        f"{target.negative} {target.positive}",
                    ]

            prompts_to_cache = list(dict.fromkeys(prompts_to_cache))
            cache = encode_prompts_to_cache(
                prompt_list=prompts_to_cache,
                sd=self.sd,
                cache=cache,
                prompt_tensor_file=self.slider_config.prompt_tensors,
            )

            prompt_pairs = []
            for neutral in tqdm(neutral_list, desc="Building slider prompt pairs", leave=False):
                for target in self.slider_config.targets:
                    prompt_pair_batch = build_prompt_pair_batch_from_cache(
                        cache=cache,
                        target=target,
                        neutral=neutral,
                    )
                    if self.slider_config.batch_full_slide:
                        self.prompt_chunk_size = len(prompt_pair_batch)
                        prompt_pairs.append(concat_prompt_pairs(prompt_pair_batch).to("cpu"))
                    else:
                        self.prompt_chunk_size = 1
                        prompt_pairs += [pair.to("cpu") for pair in prompt_pair_batch]

            self.prompt_cache = cache
            self.prompt_pairs = prompt_pairs

        super().hook_before_train_loop()

    def _set_slider_train_timesteps(self, latents):
        noise_scheduler = self.sd.noise_scheduler
        max_steps = self.train_config.max_denoising_steps
        if hasattr(noise_scheduler, "set_train_timesteps"):
            linear_timesteps = any(
                [
                    self.train_config.linear_timesteps,
                    self.train_config.linear_timesteps2,
                    self.train_config.timestep_type == "linear",
                ]
            )
            timestep_type = "linear" if linear_timesteps else self.train_config.timestep_type
            noise_scheduler.set_train_timesteps(
                max_steps,
                device=self.device_torch,
                timestep_type=timestep_type,
                latents=latents,
                patch_size=getattr(self.sd, "patch_size", 1),
            )
        else:
            noise_scheduler.set_timesteps(max_steps, device=self.device_torch)

    def _reset_slider_prediction_timesteps(self):
        noise_scheduler = self.sd.noise_scheduler
        if hasattr(noise_scheduler, "set_train_timesteps"):
            noise_scheduler.set_train_timesteps(
                1000,
                device=self.device_torch,
                timestep_type="linear",
            )
        else:
            noise_scheduler.set_timesteps(1000, device=self.device_torch)

    def hook_train_loop(self, batch):
        dtype = get_torch_dtype(self.train_config.dtype)
        prompt_pair = self.prompt_pairs[
            torch.randint(0, len(self.prompt_pairs), (1,)).item()
        ]
        prompt_pair.to(self.device_torch, dtype=dtype)

        width, height = self.slider_config.resolutions[
            torch.randint(0, len(self.slider_config.resolutions), (1,)).item()
        ]
        true_batch_size = _get_prompt_batch_size(prompt_pair.target_class) * self.train_config.batch_size
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        pred_kwargs = {}

        def predict(text_embeddings, latents, timestep):
            return self.sd.predict_noise(
                latents=latents,
                text_embeddings=text_embeddings,
                timestep=timestep,
                guidance_scale=1.0,
                **pred_kwargs,
            )

        with torch.no_grad():
            self.sd.unet.eval()
            timesteps_to = torch.randint(
                1, self.train_config.max_denoising_steps - 1, (1,)
            ).item()
            noise = build_slider_latent_noise(
                sd=self.sd,
                height=height,
                width=width,
                batch_size=true_batch_size,
                device=self.device_torch,
                dtype=dtype,
                noise_offset=self.train_config.noise_offset,
            )
            latents = noise * self.sd.noise_scheduler.init_noise_sigma
            self._set_slider_train_timesteps(latents)

            self.network.multiplier = prompt_pair.multiplier_list
            denoised_latents = self.sd.diffuse_some_steps(
                latents,
                prompt_pair.target_class,
                start_timesteps=0,
                total_timesteps=timesteps_to,
                guidance_scale=1.0,
            ).detach()

            self._reset_slider_prediction_timesteps()
            current_timestep_index = int(
                timesteps_to * 1000 / self.train_config.max_denoising_steps
            )
            current_timestep = self.sd.noise_scheduler.timesteps[current_timestep_index]
            if len(current_timestep.shape) == 0:
                current_timestep = current_timestep.unsqueeze(0)

            embeddings = concat_prompt_embeds(
                [
                    prompt_pair.positive_target,
                    prompt_pair.empty_prompt,
                    prompt_pair.negative_target,
                ]
            ).to(self.device_torch, dtype=dtype)
            all_pred = predict(
                embeddings,
                torch.cat([denoised_latents] * 3, dim=0),
                expand_slider_timestep_for_prediction(
                    current_timestep,
                    batch_size=denoised_latents.shape[0] * 3,
                ),
            ).detach()
            positive_pred, neutral_pred, unconditional_pred = torch.chunk(all_pred, 3, dim=0)
            positive_latents = unconditional_pred
            neutral_latents = neutral_pred
            unconditional_latents = positive_pred

        self.sd.unet.train()
        optimizer.zero_grad(set_to_none=True)

        with self.network:
            self.network.multiplier = prompt_pair.multiplier_list
            target_latents = predict(
                prompt_pair.target_class,
                denoised_latents.detach(),
                expand_slider_timestep_for_prediction(
                    current_timestep,
                    batch_size=denoised_latents.shape[0],
                ),
            )

            offset = positive_latents - unconditional_latents
            offset_multiplier_list = []
            for action in prompt_pair.action_list:
                if action == ACTION_TYPES_SLIDER.ERASE_NEGATIVE:
                    offset_multiplier_list.append(-1.0)
                elif action == ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE:
                    offset_multiplier_list.append(1.0)
                else:
                    offset_multiplier_list.append(1.0)
            offset_multiplier = torch.tensor(
                offset_multiplier_list,
                device=offset.device,
                dtype=offset.dtype,
            ).view(offset.shape[0], 1, 1, 1)
            offset_neutral = (neutral_latents + offset * offset_multiplier).detach()

            loss = torch.nn.functional.mse_loss(
                target_latents.float(),
                offset_neutral.float(),
                reduction="none",
            )
            loss = loss.mean([1, 2, 3]).mean() * prompt_pair.weight
            loss.backward()

        optimizer.step()
        lr_scheduler.step()
        self.network.multiplier = 1.0
        prompt_pair.to("cpu")

        return OrderedDict({"loss": loss.detach().item()})
