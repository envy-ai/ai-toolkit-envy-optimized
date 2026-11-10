"""Six-step native MiniMax-H3-Image PDD preview sampler."""

import os

import numpy as np
import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from toolkit.sample_progress import SampleProgressMixin
from ...minimax_h3.src.packing import (
    KEYFRAME_NOISE_AUG_T,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from ...minimax_h3.src.text_encoder import trim_caption_tokens


PDD_SIGMAS = (
    1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
    0.5327739119529724, 0.23200887441635132, 0.0,
)


def reference_size(width: int, height: int, low: int, high: int):
    """Comfy's short-edge reference scaling, rounded to the H3 32px canvas."""
    short = min(width, height)
    scale = min(max(short, low), high) / short
    return tuple(max(32, round(side * scale / 32) * 32) for side in (width, height))


class MiniMaxH3ImagePipeline(SampleProgressMixin):
    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    @torch.no_grad()
    def __call__(
        self, conditional_embeds, height=2048, width=2048,
        num_inference_steps=6, guidance_scale=1.0, generator=None,
        latents=None, ctrl_img=None, **kwargs,
    ):
        if num_inference_steps != 6 or guidance_scale != 1:
            raise ValueError("MiniMax H3 Image requires six PDD steps and guidance 1")
        if height % 32 or width % 32:
            raise ValueError("MiniMax H3 Image width and height must be multiples of 32")
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        h_lat, w_lat = height // 16, width // 16
        text_embeds, tags = trim_caption_tokens(
            conditional_embeds.text_embeds[0], conditional_embeds.text_token_tags[0], model.max_text_length
        )
        text_embeds = text_embeds[None].to(device, dtype)
        tags = tags.cpu().long()

        refs = []
        if ctrl_img is not None:
            sources = ctrl_img if isinstance(ctrl_img, (list, tuple)) else [ctrl_img]
            for source in sources:
                image = Image.open(source).convert("RGB") if isinstance(source, (str, os.PathLike)) else source.convert("RGB")
                ref_w, ref_h = reference_size(*image.size, 1024, 2048)
                if image.size != (ref_w, ref_h):
                    image = image.resize((ref_w, ref_h), Image.Resampling.LANCZOS)
                pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
                pixels = (pixels.float() / 255.0 * 2.0 - 1.0)[None, :, None]
                refs.append(model.encode_keyframe_latents(pixels).to(device))
            if not bool((tags == 0).any()):
                raise ValueError("H3 Image reference sampling needs matching Qwen vision tokens")
        ref_blocks = tuple((1, z.shape[-2], z.shape[-1]) for z in refs)
        layout = build_packed_sequence(tags, 1, h_lat, w_lat, 0, ref_blocks=ref_blocks)
        position_ids = layout.position_ids[None].to(device)
        token_tags = layout.token_tags[None].to(device)
        video_indices = layout.video_indices.to(device)
        text_indices = layout.text_indices.to(device)

        cond_parts = []
        reference_seed = int(generator.initial_seed()) if generator is not None else 0
        for z in refs:
            # Reference rows are pinned near clean; the small augmentation
            # follows the released H3 reference semantics. The official
            # implementation draws noise after patchification, so the CPU RNG
            # stream is in packed-row order rather than latent-channel order.
            packed = patchify_video_latents(z.float())
            ref_generator = torch.Generator(device="cpu").manual_seed(reference_seed)
            noise = torch.randn(packed.shape, generator=ref_generator, dtype=torch.float32).to(device)
            cond_parts.append(KEYFRAME_NOISE_AUG_T * packed + (1.0 - KEYFRAME_NOISE_AUG_T) * noise)
        cond_rows = torch.cat(cond_parts, dim=1) if cond_parts else None
        if getattr(getattr(model, "model_config", None), "low_vram", False) and model.vae.device != torch.device("cpu"):
            model.vae.to("cpu")
            from toolkit.basic import flush

            flush()
        if getattr(model.transformer, "device", None) == torch.device("cpu"):
            model.transformer.to(device)

        if latents is None:
            latents = randn_tensor(
                (1, 24, 1, h_lat, w_lat), generator=generator, dtype=torch.float32
            )
        elif tuple(latents.shape) != (1, 24, 1, h_lat, w_lat):
            raise ValueError("MiniMax H3 Image provided latents have the wrong shape")
        rows = patchify_video_latents(latents.to(device, torch.float32))
        sigmas = torch.tensor(PDD_SIGMAS, device=device, dtype=torch.float32)
        for i in self.progress_bar(range(6)):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            time = 1.0 - float(sigma)
            row_t = build_row_timesteps(layout, time, time)[None].to(device)
            video_in = torch.cat([cond_rows, rows], dim=1) if cond_rows is not None else rows
            pred = model.transformer(
                hidden_states=video_in,
                encoder_hidden_states=text_embeds,
                row_timesteps=row_t,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                text_indices=text_indices,
                pdd_step=i,
            )[:, layout.num_condition_video_rows:].float()
            denoised = rows + sigma * pred
            emit = getattr(model, "_emit_sample_step", None)
            if emit is not None and getattr(model, "sample_step_hook", None) is not None:
                emit(unpatchify_video_tokens(denoised, 1, h_lat, w_lat), i, 6)
            rows = rows + (sigma - sigma_next) * pred

        image = model.decode_latents(unpatchify_video_tokens(rows, 1, h_lat, w_lat))
        pixels = ((image.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        return [Image.fromarray(pixels[0, :, 0].permute(1, 2, 0).cpu().numpy())]
