import os

from .flux2_model import Flux2Model
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False

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
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False

    def load_te(self):
        if self.flux2_klein_te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen3")

        text_encoder_cache_path = None
        text_encoder = None
        if self.model_config.quantize_te:
            text_encoder_cache_path = self.get_quantized_module_cache_path(
                component_name="text_encoder",
                qtype=self.model_config.qtype_te,
                source_ref=self.flux2_klein_te_path,
                extra_cache_key={"text_encoder_type": self.flux2_te_type},
            )
            text_encoder = self.load_quantized_module_cache(
                text_encoder_cache_path, "text encoder"
            )
        text_encoder_loaded_from_cache = text_encoder is not None

        if text_encoder is None:
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                self.flux2_klein_te_path,
                torch_dtype=dtype,
            )

        if self.model_config.quantize_te:
            if not text_encoder_loaded_from_cache:
                self.print_and_status_update("Quantizing Qwen3")
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                freeze(text_encoder)
                self.save_quantized_module_cache(
                    text_encoder, text_encoder_cache_path, "text encoder"
                )
                flush()
        elif not self.model_config.low_vram:
            text_encoder.to(self.device_torch, dtype=dtype)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(self.flux2_klein_te_path)
        return text_encoder, tokenizer


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"
