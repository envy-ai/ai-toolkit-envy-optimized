"""Iris 3B LoRA training in RGB pixel space."""
from pathlib import Path
import torch
from transformers import AutoTokenizer

from toolkit.basic import flush
from toolkit.models.base_model import BaseModel
from toolkit.models.FakeVAE import FakeVAE
from .scheduler import IrisFlowScheduler
from .transformer import IrisTransformer, architecture_config
from .text_encoder import IrisTextEncoder, TEXT_ENCODER_REPO, encode_iris_prompts
from .pipeline import IrisPipeline


class IrisModel(BaseModel):
    arch = "iris"
    lora_keys_use_comfy_prefix = True
    lora_accept_bare_transformer_keys = True
    supports_quantized_text_encoder_cache = True

    def __init__(self, device, model_config, dtype="bf16", **kwargs):
        super().__init__(device, model_config, dtype, **kwargs)
        self.is_flow_matching = self.is_transformer = True
        self.target_lora_modules = ["IrisTransformer"]
        self.vae_scale_factor = 1
        self.generation_cpu_offload_modules = {"text_encoder"}

    @staticmethod
    def get_train_scheduler():
        return IrisFlowScheduler()

    def get_bucket_divisibility(self):
        return 16

    def get_transformer_block_names(self):
        return IrisTransformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        return IrisTransformer.get_quantization_exclude_modules()

    def _load_component(self, cls, source, role, **source_kwargs):
        policy = self.component_load_kwargs(role)
        from toolkit.models.v2.pool import ComponentPool
        if ComponentPool.current is not None:
            return cls.load(source, **source_kwargs, **policy)
        qtype = policy["qtype"]
        component = "text_encoder" if role == "te" else "transformer"
        files = sorted(str(file) for file in Path(source).iterdir()
            if file.suffix in (".safetensors", ".json", ".yaml")) if Path(source).is_dir() else []
        cache = self.get_quantized_module_cache_path(component, qtype,
            source_ref={"source": source, "source_files": files, "options": source_kwargs},
            extra_cache_key={"quantize_kwargs": self.model_config.quantize_kwargs,
                "dtype": str(policy["dtype"]), "exclude": cls.get_quantization_exclude_modules()}) if qtype and "|" not in qtype else None
        if cache is None:
            return cls.load(source, **source_kwargs, **policy)
        model = self.load_quantized_module_cache(cache, component)
        if model is None:
            # Save before installing device-manager hooks, whose dynamic classes
            # and pinned GPU staging state must never enter a persistent cache.
            model = cls.load(source, **source_kwargs, **{**policy, "offload": 0.0})
            self.save_quantized_module_cache(model, cache, component)
        model._aitk_policy = None  # Saving parks weights on CPU; reapply placement.
        return model.aitk_post_load(**policy)

    def load_model(self):
        mc = self.model_config
        options = mc.model_kwargs
        self.print_and_status_update("Loading Iris 3B transformer")
        config = architecture_config(options["transformer_config"]) if "transformer_config" in options else None
        self.model = self._load_component(IrisTransformer, mc.name_or_path, "transformer", config=config,
            config_path=options.get("config_path"), checkpoint_filename=options.get("checkpoint_filename", "model.safetensors"))
        self.model.checkpoint_policy = options.get("activation_checkpointing", "full")
        from .src.models.ac import validate_policy
        self.model.ac_selective_every = int(options.get("ac_selective_every", 2))
        validate_policy(self.model.checkpoint_policy, self.model.ac_selective_every)
        self.model.activation_offload = bool(options.get("activation_offload", False))
        self.model.eval().requires_grad_(False)

        te_path = mc.text_encoder_path or options.get("text_encoder_path", TEXT_ENCODER_REPO)
        tokenizer_path = options.get("tokenizer_path", TEXT_ENCODER_REPO if str(te_path).endswith(".safetensors") else te_path)
        self.print_and_status_update("Loading Iris Qwen3-VL text tower")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.text_encoder = self._load_component(IrisTextEncoder, te_path, "te", subfolder="")
        self.text_encoder.eval().requires_grad_(False)
        self.vae = FakeVAE(scaling_factor=1.0).to(device=self.vae_device_torch, dtype=self.vae_torch_dtype)
        self.noise_scheduler = self.get_train_scheduler()
        self.pipeline = IrisPipeline(self)
        if mc.inference_lora_path:
            from toolkit.assistant_lora import load_assistant_lora_from_path
            self.inference_lora_network = load_assistant_lora_from_path(mc.inference_lora_path, self, strict=True)
            self.inference_lora_network.is_active = False
            self.inference_lora_network.force_to("cpu", self.torch_dtype)
        self.print_and_status_update("Iris model loaded")

    def get_prompt_embeds(self, prompt):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        if self.model_config.low_vram and not hasattr(self.model, "_memory_manager"):
            self.model.to("cpu")
            flush()
        if not hasattr(self.text_encoder, "_memory_manager"):
            self.text_encoder.to(self.te_device_torch)
        try:
            result = encode_iris_prompts(self.text_encoder, self.tokenizer, prompts, self.torch_dtype)
            if self.model_config.low_vram:
                result.to("cpu")
            return result
        finally:
            if self.model_config.low_vram:
                self.text_encoder.to("cpu")
                flush()

    def get_noise_prediction(self, latent_model_input, timestep, text_embeddings, **kwargs):
        if not hasattr(self.model, "_memory_manager") and self.model.device != self.device_torch:
            self.model.to(self.device_torch)
        features = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        cfg = self.model.cfg
        features = features.reshape(features.shape[0], features.shape[1], cfg.text_lap_num_layers, cfg.text_dim)
        mask = text_embeddings.attention_mask.to(self.device_torch).bool()
        return self.model(x=latent_model_input.to(self.device_torch, self.torch_dtype),
                          t=timestep.to(self.device_torch, torch.float32), y=features, y_mask=mask).x

    def get_loss_target(self, *args, **kwargs):
        return (kwargs["noise"] - kwargs["batch"].latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def get_generation_pipeline(self):
        return IrisPipeline(self)

    def generate_single_image(self, pipeline, gen_config, conditional_embeds, unconditional_embeds, generator, extra):
        width = max(16, gen_config.width // 16 * 16)
        height = max(16, gen_config.height // 16 * 16)
        return pipeline(conditional_embeds=conditional_embeds, unconditional_embeds=unconditional_embeds,
            height=height, width=width, num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale, generator=generator, latents=gen_config.latents)[0]
