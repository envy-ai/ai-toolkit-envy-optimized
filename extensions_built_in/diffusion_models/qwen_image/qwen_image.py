import os
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.vae.qwen_image import QwenImageVAE, QwenImageVAEHolderMixin
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import get_accelerator, unwrap_model
import torch.nn.functional as F
from toolkit.memory_management import MemoryManager
from toolkit.metadata import get_meta_for_safetensors
from safetensors.torch import load_file

from diffusers import (
    QwenImagePipeline,
    AutoencoderKLQwenImage,
)
from transformers import Qwen2VLProcessor
from toolkit.models.v2.diffusion_models.qwen_image import QwenImageTransformer2DModel
from toolkit.models.v2.text_encoders.qwen25_vl import Qwen25VLTextEncoder
from tqdm import tqdm
from toolkit.util.qwen_vae_gradient_checkpointing import patch_qwen_vae_gradient_checkpointing

patch_qwen_vae_gradient_checkpointing()

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

class _PipelineLoraController:
    def __init__(
        self,
        pipeline,
        adapter_name: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        load_immediately: bool = False,
    ):
        self.pipeline = pipeline
        self.adapter_name = adapter_name
        self.lora_path = lora_path
        self.device = device
        self.dtype = dtype
        self._is_loaded = False
        self._is_active = False
        self._adapter_offload_device = None
        self._adapter_offload_dtype = None
        if load_immediately:
            self._load()

    def _patch_transformer_to_for_adapter_offload(self):
        if not hasattr(self.pipeline, "transformer"):
            return
        transformer = self.pipeline.transformer
        if getattr(transformer, "_aitk_qwen_lora_to_is_patched", False):
            return

        original_to = transformer.to
        controller_ref = weakref.ref(self)

        def wrapped_to(*args, **kwargs):
            result = original_to(*args, **kwargs)
            controller = controller_ref()
            if controller is not None:
                controller._offload_adapter_if_needed()
            return result

        transformer.to = wrapped_to
        transformer._aitk_qwen_lora_to_is_patched = True

    def _move_transformer_to_target(self):
        if self.device is None or not hasattr(self.pipeline, "transformer"):
            return
        kwargs = {"device": self.device}
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype
        self.pipeline.transformer.to(**kwargs)

    @staticmethod
    def _move_parameter_to(parameter, device, dtype):
        kwargs = {"device": device}
        if dtype is not None and (
            parameter.data.is_floating_point() or parameter.data.is_complex()
        ):
            kwargs["dtype"] = dtype
        with torch.no_grad():
            parameter.data = parameter.data.to(**kwargs)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(**kwargs)

    @staticmethod
    def _move_module_or_parameter_to(value, device, dtype):
        if isinstance(value, torch.nn.Parameter):
            _PipelineLoraController._move_parameter_to(value, device, dtype)
            return True
        if hasattr(value, "to"):
            kwargs = {"device": device}
            if dtype is not None:
                kwargs["dtype"] = dtype
            value.to(**kwargs)
            return True
        return False

    def _move_adapter_to(self, device, dtype):
        if self.adapter_name is None or not hasattr(self.pipeline, "transformer"):
            return False
        if not hasattr(self.pipeline.transformer, "modules"):
            return False

        moved = False
        adapter_layer_attrs = (
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
            "lora_magnitude_vector",
            "lora_dropout",
        )
        for module in self.pipeline.transformer.modules():
            for attr_name in adapter_layer_attrs:
                adapter_layers = getattr(module, attr_name, None)
                if adapter_layers is None:
                    continue
                try:
                    adapter_layer = adapter_layers[self.adapter_name]
                except (KeyError, TypeError, AttributeError):
                    continue
                moved = (
                    self._move_module_or_parameter_to(adapter_layer, device, dtype)
                    or moved
                )
        return moved

    def _offload_adapter_if_needed(self):
        if (
            self._is_loaded
            and not self._is_active
            and self._adapter_offload_device is not None
        ):
            self._move_adapter_to(
                self._adapter_offload_device,
                self._adapter_offload_dtype,
            )

    def _load(self):
        if self._is_loaded or self.lora_path is None:
            return
        self._move_transformer_to_target()
        with (
            _peft_skip_torchao_if_missing_apply_subclass(),
            _quanto_skip_missing_base_weight_load(),
        ):
            self.pipeline.load_lora_weights(
                self.lora_path,
                adapter_name=self.adapter_name,
                low_cpu_mem_usage=False,
            )
        if hasattr(self.pipeline, "disable_lora"):
            self.pipeline.disable_lora()
        self._is_loaded = True
        self._patch_transformer_to_for_adapter_offload()

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool):
        value = bool(value)
        if value == self._is_active:
            return
        self._is_active = value
        if value:
            self._adapter_offload_device = None
            self._adapter_offload_dtype = None
            self._load()
            if self._is_loaded and self.device is not None:
                self._move_adapter_to(self.device, self.dtype)
            if self.adapter_name and hasattr(self.pipeline, "set_adapters"):
                self.pipeline.set_adapters(self.adapter_name)
            if hasattr(self.pipeline, "enable_lora"):
                self.pipeline.enable_lora()
        else:
            if hasattr(self.pipeline, "disable_lora"):
                self.pipeline.disable_lora()

    def force_to(self, device, dtype):
        target_device = torch.device(device)
        if self._is_loaded and not self._is_active and target_device.type == "cpu":
            self._adapter_offload_device = target_device
            self._adapter_offload_dtype = dtype
            if self._move_adapter_to(target_device, dtype):
                return

        self.device = target_device
        self.dtype = dtype
        if self._is_loaded:
            self._move_transformer_to_target()
            if self._is_active:
                self._move_adapter_to(self.device, self.dtype)
        return


@contextmanager
def _peft_skip_torchao_if_missing_apply_subclass():
    try:
        import peft.tuners.lora.model as peft_lora_model
        import peft.tuners.lora.torchao as peft_lora_torchao
    except Exception:
        yield
        return

    original_dispatch_model = peft_lora_model.dispatch_torchao
    original_dispatch_torchao = peft_lora_torchao.dispatch_torchao

    def _patched_dispatch_torchao(target, adapter_name, lora_config, **kwargs):
        if "get_apply_tensor_subclass" not in kwargs:
            return None
        return original_dispatch_torchao(target, adapter_name, lora_config, **kwargs)

    peft_lora_model.dispatch_torchao = _patched_dispatch_torchao
    peft_lora_torchao.dispatch_torchao = _patched_dispatch_torchao
    try:
        yield
    finally:
        peft_lora_model.dispatch_torchao = original_dispatch_model
        peft_lora_torchao.dispatch_torchao = original_dispatch_torchao


@contextmanager
def _quanto_skip_missing_base_weight_load():
    try:
        from optimum.quanto.nn.qmodule import QModuleMixin
    except Exception:
        yield
        return

    original_load_from_state_dict = QModuleMixin._load_from_state_dict

    def _patched_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_name = prefix + "weight"
        weight_prefix = weight_name + "."
        if getattr(self, "weight_qtype", None) is not None and weight_name not in state_dict:
            has_serialized_weight = any(
                key.startswith(weight_prefix) for key in state_dict.keys()
            )
            if not has_serialized_weight:
                return super(QModuleMixin, self)._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    False,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
        return original_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    QModuleMixin._load_from_state_dict = _patched_load_from_state_dict
    try:
        yield
    finally:
        QModuleMixin._load_from_state_dict = original_load_from_state_dict


class QwenImageModel(QwenImageVAEHolderMixin, BaseModel):
    arch = "qwen_image"
    _qwen_image_keep_visual = False
    _qwen_pipeline = QwenImagePipeline

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["QwenImageTransformer2DModel"]

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # 16 for the VAE, 2 for patch size

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen Image model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path
        model_dtype = dtype

        if base_model_path.endswith(".safetensors"):
            # use the repo for extras
            base_model_path = "Qwen/Qwen-Image"

        self.print_and_status_update("Loading transformer")
        transformer_cache_path = None
        transformer = None
        transformer_loaded_from_cache = False

        if not model_path.endswith(".safetensors") and os.path.exists(model_path):
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, "text_encoder")
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        # load + quantize + offload + placement, all driven by model_config
        transformer = QwenImageTransformer2DModel.load(
            model_path, **self.component_load_kwargs("transformer")
        )
        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = Qwen25VLTextEncoder.load_tokenizer(base_model_path, use_fast=False)
        text_encoder = Qwen25VLTextEncoder.load_model(base_model_path, dtype=dtype)

        # remove the visual model as it is not needed for image generation
        # (before quantization, so the dead tower is never quantized)
        self.processor = None
        if not self._qwen_image_keep_visual:
            text_encoder.drop_vision_tower()

        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))

        self.print_and_status_update("Loading VAE")
        vae = QwenImageVAE.load_model(base_model_path, dtype=dtype)

        self.noise_scheduler = QwenImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        if self._qwen_image_keep_visual:
            try:
                self.processor = Qwen2VLProcessor.from_pretrained(
                    model_path, subfolder="processor"
                )
            except OSError:
                self.processor = Qwen2VLProcessor.from_pretrained(
                    base_model_path, subfolder="processor"
                )
            kwargs["processor"] = self.processor

        pipe: QwenImagePipeline = self._qwen_pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # low_vram: the text encoder stays on cpu; get_prompt_embeds moves it
        # to the gpu on demand
        if not self.low_vram:
            text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        if self.model_config.inference_lora_path is not None:
            self.print_and_status_update("Registering inference lora")
            self._load_inference_lora()
        self.print_and_status_update("Model Loaded")

    def _load_inference_lora(self):
        self.assistant_lora = _PipelineLoraController(
            self.pipeline,
            adapter_name="inference",
            lora_path=self.model_config.inference_lora_path,
            device=self.device_torch,
            dtype=self.torch_dtype,
            load_immediately=False,
        )

    def get_generation_pipeline(self):
        scheduler = QwenImageModel.get_train_scheduler()

        pipeline: QwenImagePipeline = QwenImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: QwenImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        control_img = None
        if gen_config.ctrl_img is not None:
            raise NotImplementedError(
                "Control image generation is not supported in Qwen Image model... yet"
            )
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            # resize to width and height
            if control_img.size != (gen_config.width, gen_config.height):
                control_img = control_img.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )
        self.model.to(self.device_torch)

        # flush for low vram if we are doing that
        flush_between_steps = self.model_config.low_vram

        # Fix a bug in diffusers/torch
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if flush_between_steps:
                flush()
            latents = callback_kwargs["latents"]

            return {"latents": latents}

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        if self.model_config.low_vram:
            # set vae to tile decode
            pipeline.vae.enable_tiling()

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_embeds_mask=conditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.int64
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.int64
            ),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            true_cfg_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            **extra,
        ).images[0]

        if self.model_config.low_vram:
            # restore no tiling
            pipeline.vae.disable_tiling()

        return img

    def _get_qwen_prompt_text(self, prompt: List[str], image=None):
        template = self.pipeline.prompt_template_encode
        return (
            [template.format(prompt_item) for prompt_item in prompt],
            self.pipeline.prompt_template_encode_start_idx,
        )

    def _get_qwen_prompt_embeds_without_lm_head(
        self,
        prompt: str | List[str],
        image=None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device_torch
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        txt, drop_idx = self._get_qwen_prompt_text(prompt, image=image)

        processor = getattr(self, "processor", None)
        if processor is None:
            model_inputs = self.pipeline.tokenizer(
                txt,
                max_length=self.pipeline.tokenizer_max_length + drop_idx,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
        else:
            model_inputs = processor(
                text=txt,
                images=image,
                padding=True,
                return_tensors="pt",
            ).to(device)

        text_encoder_model = getattr(
            self.pipeline.text_encoder, "model", self.pipeline.text_encoder
        )
        text_encoder_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
            "second_per_grid_ts",
        ):
            value = getattr(model_inputs, key, None)
            if value is not None:
                text_encoder_kwargs[key] = value

        encoder_hidden_states = text_encoder_model(**text_encoder_kwargs)
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self.pipeline._extract_masked_hidden(
            hidden_states, model_inputs.attention_mask
        )
        split_hidden_states = [
            hidden_state[drop_idx:] for hidden_state in split_hidden_states
        ]
        attn_mask_list = [
            torch.ones(
                hidden_state.size(0), dtype=torch.long, device=hidden_state.device
            )
            for hidden_state in split_hidden_states
        ]
        max_seq_len = max(hidden_state.size(0) for hidden_state in split_hidden_states)
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [
                        hidden_state,
                        hidden_state.new_zeros(
                            max_seq_len - hidden_state.size(0), hidden_state.size(1)
                        ),
                    ]
                )
                for hidden_state in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([mask, mask.new_zeros(max_seq_len - mask.size(0))])
                for mask in attn_mask_list
            ]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    def _encode_qwen_prompt_without_lm_head(
        self,
        prompt: str | List[str],
        image=None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds_without_lm_head(
            prompt,
            image=image,
            device=device,
        )
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask.repeat(
                1, num_images_per_prompt, 1
            )
            prompt_embeds_mask = prompt_embeds_mask.view(
                batch_size * num_images_per_prompt, seq_len
            )
            if prompt_embeds_mask.all():
                prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)
        batch_size, num_channels_latents, height, width = latent_model_input.shape

        ps = self.transformer.config.patch_size

        # pack image tokens
        latent_model_input = latent_model_input.view(
            batch_size, num_channels_latents, height // ps, ps, width // ps, ps
        )
        latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
        latent_model_input = latent_model_input.reshape(
            batch_size, (height // ps) * (width // ps), num_channels_latents * (ps * ps)
        )

        # img_shapes passed to the model
        img_h2, img_w2 = height // ps, width // ps
        img_shapes = [[(1, img_h2, img_w2)]] * batch_size

        enc_hs = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        prompt_embeds_mask = text_embeddings.attention_mask.to(
            self.device_torch, dtype=torch.int64
        )

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(
                self.device_torch, self.torch_dtype
            ).detach(),
            timestep=(timestep / 1000).detach(),
            guidance=None,
            encoder_hidden_states=enc_hs.detach(),
            encoder_hidden_states_mask=prompt_embeds_mask.detach(),
            img_shapes=img_shapes,
            return_dict=False,
            **kwargs,
        )[0]

        # unpack
        noise_pred = noise_pred.view(
            batch_size, height // ps, width // ps, num_channels_latents, ps, ps
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, prompt_embeds_mask = self._encode_qwen_prompt_without_lm_head(
            prompt,
            device=self.device_torch,
            num_images_per_prompt=1,
        )
        # diffusers >=0.37 returns None when all tokens are valid (no padding)
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.int64
            )
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_embeds_mask
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        # comfy-format single-file save (diffusers keys ARE the comfy layout
        # for qwen image); prequantized layers keep their quantized storage
        transformer: QwenImageTransformer2DModel = unwrap_model(self.model)
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
        )

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "qwen_image"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    lora_keys_use_comfy_prefix = True
    lora_accept_bare_transformer_keys = True
