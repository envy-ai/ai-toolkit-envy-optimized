"""Native Iris weights with the toolkit's single-file loading and memory policy."""
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download

from toolkit.models.v2._mixin import OstrisModelMixin
from .src.config import ModelConfig, PixelStageConfig
from .src.models.dit import IrisDiT as NativeIrisDiT


def architecture_config(raw=None):
    raw = dict(raw or {})
    pixel = raw.pop("pixel", {})
    return ModelConfig(**raw, pixel=PixelStageConfig(**pixel))


class IrisTransformer(NativeIrisDiT, OstrisModelMixin):
    aitk_subfolder = ""
    aitk_cast_quantized_load = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg
        self.activation_offload = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def get_transformer_block_names(cls):
        return ["blocks", "pixel_blocks", "y_embedder.layer_blocks", "y_embedder.refiner.blocks"]

    @classmethod
    def get_quantization_exclude_modules(cls):
        return ["s_embedder*", "t_embedder*", "modulation_cores*", "pixel_embedder*",
                "final_layer*", "y_embedder.layer_pool*"]

    def get_offload_ignore_modules(self):
        # These cores are called by every block through unregistered references.
        return list(self.modulation_cores.modules())

    @classmethod
    def _load_single_file_config(cls, config_path, subfolder):
        if config_path is None:
            return ModelConfig()
        return cls.aitk_load_config(config_path)

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        path = Path(path)
        if path.is_dir():
            path = path / "config.yaml"
        if path.is_file():
            raw = yaml.safe_load(path.read_text())
        else:
            raw = yaml.safe_load(Path(hf_hub_download(str(path), "config.yaml")).read_text())
        return architecture_config(raw.get("model", raw))

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        filename = kwargs.pop("checkpoint_filename", "model.safetensors")
        config = kwargs.pop("config", None) or cls.aitk_load_config(path)
        file = str(Path(path) / filename) if Path(path).is_dir() else hf_hub_download(path, filename)
        return cls._load_single_file(file, dtype=dtype, config=config)

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        # Also accept checkpoints wrapped in Comfy's MODEL namespace.
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if any(key.startswith(prefix) for key in state_dict):
                return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        return state_dict

    def enable_gradient_checkpointing(self):
        self.activation_checkpointing = getattr(self, "checkpoint_policy", "full")

    def disable_gradient_checkpointing(self):
        self.activation_checkpointing = "none"

    def forward(self, *args, **kwargs):
        from contextlib import nullcontext
        offload = self.activation_offload and torch.is_grad_enabled() and self.device.type == "cuda"
        with torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext():
            return super().forward(*args, **kwargs)
