"""Native MiniMax-H3-Image training and sampling wrapper.

This is a separate image checkpoint, not the H3 FL2VA video photo recipe.
The shared H3 loader supplies Qwen3-VL caching, quantization and layer
streaming; the image transformer, VAE, packing and six-step PDD sampler replace
their audio/video counterparts.
"""

import hashlib
import json
import math
import os

import torch
from PIL import Image
from safetensors import safe_open

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.models.v2.resolver import repo_id_from_name_or_path, resolve_comfy_file

from ..minimax_h3.minimax_h3 import COMFY_FILES as VIDEO_FILES
from ..minimax_h3.minimax_h3 import COMFY_REPO as VIDEO_REPO
from ..minimax_h3.minimax_h3 import MinimaxH3Model
from ..minimax_h3.src.packing import (
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from ..minimax_h3.src.text_encoder import trim_caption_tokens
from .src.transformer import MiniMaxH3ImageTransformer
from .src.vae import MiniMaxH3ImageVAE
from .src.pipeline import MiniMaxH3ImagePipeline, PDD_SIGMAS, reference_size


IMAGE_REPO = "Comfy-Org/MiniMax-H3-Image"
IMAGE_FILES = {
    "dit_image": "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
    "image_vae": "vae/minimax_h3_image_vae_fp16.safetensors",
}


def _component_identity(path: str):
    """Cheap cache identity for a resolved component, without reading weights."""
    info = os.stat(path)
    return (os.path.realpath(path), info.st_size, info.st_mtime_ns)


def _validate_pdd_grid(sigmas):
    if not isinstance(sigmas, (list, tuple)) or len(sigmas) != len(PDD_SIGMAS):
        raise ValueError("MiniMax H3 Image PDD sigma grid differs from the published six-step grid")
    try:
        actual = [float(value) for value in sigmas]
    except (TypeError, ValueError) as error:
        raise ValueError("MiniMax H3 Image PDD sigma grid contains a nonnumeric value") from error
    if any(not math.isfinite(value) for value in actual):
        raise ValueError("MiniMax H3 Image PDD sigma grid contains a non-finite value")
    if any(abs(value - expected) > 1e-6 for value, expected in zip(actual, PDD_SIGMAS)):
        raise ValueError("MiniMax H3 Image PDD sigma grid differs from the published six-step grid")


def _read_pdd_config(path):
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if config.get("num_inference_steps") != 6:
        raise ValueError("MiniMax H3 Image PDD config must declare six inference steps")
    _validate_pdd_grid(config.get("sigmas"))


class MinimaxH3ImageModel(MinimaxH3Model):
    arch = "minimax_h3_image"
    comfy_repo = IMAGE_REPO

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_lora_modules = ["MiniMaxH3ImageTransformer"]
        self.latent_space_version = "minimax_h3_image_v1"
        self.has_multiple_control_images = True
        self.image_reference_label = "Picture {index}: "
        self.max_text_length = int(self.model_config.model_kwargs.get("max_text_length", 4096))
        self._resolved_image_vae_path = None
        self._resolved_te_source = None

    def get_latent_space_version(self):
        source = self._resolved_image_vae_path
        if source is None:
            return self.latent_space_version
        payload = (
            "minimax_h3_image", "image_vae", _component_identity(source),
            "posterior_random_v1", self.model_config.latent_space_version,
        )
        digest = hashlib.sha256(repr(payload).encode()).hexdigest()[:16]
        return f"minimax_h3_image_latents_{digest}"

    def get_text_embedding_space_version(self):
        source = self._resolved_te_source
        if source is None:
            return "minimax_h3_image_text_v1"
        payload = (
            "minimax_h3_image", "qwen3vl_layer50", _component_identity(source),
            self.max_text_length, "plain_picture_1024_short_edge_v1",
        )
        digest = hashlib.sha256(repr(payload).encode()).hexdigest()[:16]
        return f"minimax_h3_image_text_{digest}"

    def _text_encoder_source(self):
        source = super()._text_encoder_source()
        self._resolved_te_source = source
        return source

    def _dit_component(self) -> str:
        partition = self.model_config.model_kwargs.get("partition", "image")
        if partition != "image":
            raise ValueError("minimax_h3_image only supports the image partition")
        return "dit_image"

    def _resolve_comfy_file(self, component: str) -> str:
        name_or_path = self.model_config.name_or_path
        if component == "text_encoder":
            # The released image checkpoint deliberately shares H3's TE.
            # Comfy-Org's image repo also publishes a copy, but use the
            # already-resident video source before fetching another.
            return resolve_comfy_file(
                VIDEO_FILES[component],
                repo_id=VIDEO_REPO,
                override_path=(self.model_config.model_kwargs.get("text_encoder_path")
                               or self.model_config.text_encoder_path),
                extra_roots=[name_or_path] if name_or_path and os.path.isdir(name_or_path) else [],
                status_fn=self.print_and_status_update,
            )
        if component not in IMAGE_FILES:
            raise ValueError(f"MiniMax H3 Image has no {component} component")
        override = self.model_config.model_kwargs.get(f"{component}_path")
        if component == "image_vae" and self.model_config.vae_path:
            override = self.model_config.vae_path
        return resolve_comfy_file(
            IMAGE_FILES[component],
            repo_id=repo_id_from_name_or_path(name_or_path, IMAGE_REPO),
            override_path=override,
            extra_roots=[name_or_path] if name_or_path and os.path.isdir(name_or_path) else [],
            status_fn=self.print_and_status_update,
        )

    def _load_transformer(self, dit_path=None) -> MiniMaxH3ImageTransformer:
        if dit_path is None:
            dit_path = self._resolve_comfy_file("dit_image")
        with safe_open(dit_path, framework="pt", device="cpu") as header:
            keys = set(header.keys())
            if "audio_patch_proj.weight" in keys or "final_layer.audio_out.weight" in keys:
                raise ValueError("MiniMax H3 Image requires an audio-free checkpoint")
            output_key = "final_layer.video_out.weight"
            if output_key not in keys:
                raise ValueError("MiniMax H3 Image checkpoint lacks the video output head")
            output_rows = header.get_slice(output_key).get_shape()[0]
            if output_rows not in (96, 576):
                raise ValueError(f"MiniMax H3 Image output has {output_rows} rows; expected 96 or 576")
            raw_config = (header.metadata() or {}).get("config")
        transformer_config = json.loads(raw_config).get("transformer", {}) if raw_config else {}
        if output_rows == 576:
            if transformer_config.get("pdd_steps", 6) != 6:
                raise ValueError("Six-head MiniMax H3 Image checkpoint has conflicting PDD metadata")
            if transformer_config.get("pdd_sigmas") is not None:
                _validate_pdd_grid(transformer_config["pdd_sigmas"])
            if not (transformer_config.get("pdd_steps") == 6 and transformer_config.get("pdd_sigmas") is not None):
                config_path = self.model_config.model_kwargs.get("pdd_config_path")
                if not config_path:
                    raise ValueError(
                        "Six-head MiniMax H3 Image checkpoint needs PDD metadata "
                        "or an explicit pdd_config_path"
                    )
                _read_pdd_config(config_path)
        else:
            # The unmodified release has one ordinary head. Its six fused
            # preview heads live in a separate file with a separate grid.
            if transformer_config.get("pdd_steps", 1) != 1:
                raise ValueError("Single-head MiniMax H3 Image checkpoint has conflicting PDD metadata")
            head_path = self.model_config.model_kwargs.get("pdd_fused_heads_path")
            if not head_path or not os.path.isfile(head_path):
                raise ValueError("Single-head H3 Image checkpoint needs pdd_fused_heads_path")
            config_path = self.model_config.model_kwargs.get("pdd_config_path")
            if config_path is None:
                config_path = os.path.join(os.path.dirname(head_path), "pdd_config.json")
            _read_pdd_config(config_path)
        self.print_and_status_update(f"Loading H3 Image transformer from {dit_path}")
        return MiniMaxH3ImageTransformer.load_model(dit_path, dtype=self.torch_dtype)

    def _attach_pdd_heads(self):
        # Comfy's six-head repack embeds the PDD bank directly. A single-head
        # original-file conversion needs the official fused-head companion.
        transformer = self.transformer
        if transformer.pdd_steps == 1:
            head_path = self.model_config.model_kwargs.get("pdd_fused_heads_path")
            if not head_path:
                raise ValueError(
                    "A single-head H3 Image transformer needs model_kwargs.pdd_fused_heads_path "
                    "for six-step native previews"
                )
            from safetensors.torch import load_file

            heads = load_file(head_path)
            if set(heads) != {"video_out.weight", "video_out.bias"}:
                raise ValueError("H3 Image PDD file must contain only video_out weight and bias")
            transformer.attach_pdd_fused_heads(heads["video_out.weight"], heads["video_out.bias"])

    def _load_vaes(self) -> MiniMaxH3ImageVAE:
        path = self._resolve_comfy_file("image_vae")
        self._resolved_image_vae_path = path
        vae = MiniMaxH3ImageVAE.load_model(path)
        vae.eval().requires_grad_(False)
        return vae

    @property
    def image_vae(self) -> MiniMaxH3ImageVAE:
        return self.vae

    def load_model(self):
        super().load_model()
        # Quantized-module caches do not persist the companion fused-head
        # buffers, so attach them after either a cache hit or a cold load.
        self._attach_pdd_heads()
        self.pipeline = MiniMaxH3ImagePipeline(self)

    def get_sample_control_image_paths(self, gen_config):
        return [path for path in (
            gen_config.ctrl_img_1 or gen_config.ctrl_img,
            gen_config.ctrl_img_2,
            gen_config.ctrl_img_3,
        ) if path is not None]

    def _present_image_control(self, image: Image.Image):
        width, height = reference_size(*image.size, 1024, 1024)
        return image if image.size == (width, height) else image.resize(
            (width, height), Image.Resampling.LANCZOS
        )

    def encode_images(self, image_list, device=None, dtype=None):
        if self.model_config.low_vram:
            self._offload_transformer_for_frozen_component()
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)
        items = []
        for image in image_list:
            if image.ndim != 3:
                raise ValueError("MiniMax H3 Image datasets must contain single RGB images")
            items.append(image.unsqueeze(1))
        batch = torch.stack(items).to(self.vae_device_torch, self.image_vae.dtype)
        return self.image_vae.encode(batch, sample=True).to(
            device or self.vae_device_torch, dtype or self.vae_torch_dtype
        )

    @torch.no_grad()
    def encode_keyframe_latents(self, frames):
        if self.model_config.low_vram:
            self._offload_transformer_for_frozen_component()
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)
        return self.image_vae.encode_reference(
            frames.to(self.vae_device_torch, self.image_vae.dtype)
        ).float()

    def decode_latents(self, latents, device=None, dtype=None):
        if self.model_config.low_vram:
            self._offload_transformer_for_frozen_component()
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)
        image = self.image_vae.decode(latents.to(self.vae.device, self.image_vae.dtype))
        return image.to(device, dtype=dtype) if device is not None else image

    def get_noise_prediction(
        self, latent_model_input, timestep, text_embeddings: AdvancedPromptEmbeds,
        batch=None, **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        b, _, t_lat, h_lat, w_lat = latent_model_input.shape
        if b != 1 or t_lat != 1:
            raise ValueError("MiniMax H3 Image training requires batch size 1 and one latent frame")
        if batch is not None:
            references = getattr(batch, "control_video_latents_list", None)
            has_references = references is not None and (
                not isinstance(references, (list, tuple)) or len(references) > 0
            )
            if has_references or getattr(batch, "first_frame_latents", None) is not None:
                raise ValueError("MiniMax H3 Image reference training is not supported")
        text_embeds, tags = trim_caption_tokens(
            text_embeddings.text_embeds[0], text_embeddings.text_token_tags[0], self.max_text_length
        )
        layout = build_packed_sequence(tags.cpu(), 1, h_lat, w_lat, 0)
        sigma = timestep.to(self.device_torch, torch.float32).reshape(-1) / 1000.0
        t = 1.0 - sigma
        row_t = build_row_timesteps(layout, float(t[0]), float(t[0]))[None]
        pred = self.model(
            hidden_states=patchify_video_latents(latent_model_input.to(self.device_torch, torch.float32)),
            encoder_hidden_states=text_embeds[None].to(self.device_torch, self.torch_dtype),
            row_timesteps=row_t.to(self.device_torch),
            token_tags=layout.token_tags[None].to(self.device_torch),
            position_ids=layout.position_ids[None].to(self.device_torch),
            video_indices=layout.video_indices.to(self.device_torch),
            text_indices=layout.text_indices.to(self.device_torch),
        )
        return -unpatchify_video_tokens(pred, 1, h_lat, w_lat)

    def get_generation_pipeline(self):
        return MiniMaxH3ImagePipeline(self)

    def generate_single_image(
        self, pipeline, gen_config, conditional_embeds, unconditional_embeds,
        generator, extra,
    ):
        if gen_config.num_frames != 1:
            raise ValueError("MiniMax H3 Image sampling only supports one image frame")
        if gen_config.guidance_scale != 1:
            raise ValueError("MiniMax H3 Image sampling requires guidance_scale: 1")
        if gen_config.num_inference_steps != 6:
            raise ValueError("MiniMax H3 Image sampling requires exactly six PDD steps")
        return pipeline(
            conditional_embeds=conditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            generator=generator,
            latents=gen_config.latents,
            ctrl_img=self.get_sample_control_image_paths(gen_config) or None,
        )[0]

    def get_base_model_version(self):
        return "minimax_h3_image"
