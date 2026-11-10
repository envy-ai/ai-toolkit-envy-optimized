"""Official MiniMax-H3-Image VAE using the shared H3 encoder and decoder.

The image checkpoint has the video encoder's key layout but a 24-layer,
16-head decoder with 2D rotary positions, one decoded frame per latent,
and a 2D post-quantization convolution.
"""

from contextlib import nullcontext

import torch

from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3VideoVAE


class MiniMaxH3ImageVAE(MiniMaxH3VideoVAE):
    """Single-frame H3 VAE with independent training and reference sampling."""

    REFERENCE_POSTERIOR_SEED = 42

    def __init__(self, **kwargs):
        super().__init__(
            decoder_num_layers=24,
            decoder_heads=16,
            decoder_head_dim=64,
            rope_dim_ratio=0.5,
            decoder_rope_axes=2,
            decoder_patch_size_t=1,
            post_quant_conv_2d=True,
            token_drop=0,
            **kwargs,
        )

    @classmethod
    def load_from_state_dict(cls, state_dict, dtype=None, **kwargs):
        """Strict Comfy load without allocating a second 2.34 GB decoder.

        The published image config supplies latent statistics, while Comfy
        single files may omit them. The shared VAE defaults are those official
        statistics; if a file includes the buffers, use its values instead.
        """
        from accelerate import init_empty_weights

        state_dict = dict(state_dict)
        stats = {
            key: state_dict.pop(key).float()
            for key in ("latents_mean", "latents_std")
            if key in state_dict
        }
        with init_empty_weights(include_buffers=False):
            model = cls()
        model.load_state_dict(state_dict, strict=True, assign=True)
        for key, value in stats.items():
            target = getattr(model, key)
            if value.shape != target.shape:
                raise ValueError(
                    f"MiniMax-H3-Image {key} shape {tuple(value.shape)} "
                    f"does not match {tuple(target.shape)}"
                )
            target.copy_(value)
        model.eval().requires_grad_(False)
        return model

    def encode(self, pixels, sample=True, generator=None, fp16_round=False):
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(2)
        if pixels.ndim != 5 or pixels.shape[1] != 3 or pixels.shape[2] != 1:
            raise ValueError(
                "MiniMax-H3-Image encode expects RGB images shaped "
                f"[batch, 3, 1, height, width], got {tuple(pixels.shape)}"
            )
        return super().encode(
            pixels, sample=sample, generator=generator, fp16_round=fp16_round
        )

    def encode_reference(self, pixels, *, fp16_round=False):
        """Match the official image encoder's fixed CPU posterior seed 42.

        Training calls ``encode(..., sample=True)`` without this fixed seed so
        its posterior sample follows the trainer's RNG. Caches must record
        which path supplied their latent values.
        """
        generator = torch.Generator(device="cpu").manual_seed(
            self.REFERENCE_POSTERIOR_SEED
        )
        return self.encode(pixels, sample=True, generator=generator, fp16_round=fp16_round)

    def _decode_pixels(self, z):
        return self.decoder(self.post_quant_conv(z.squeeze(2)).unsqueeze(2))

    def decode(self, latents, autocast_fp16=True):
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels or latents.shape[2] != 1:
            raise ValueError(
                "MiniMax-H3-Image decode expects latents shaped "
                f"[batch, {self.latent_channels}, 1, height, width], got {tuple(latents.shape)}"
            )
        shape = (1, -1, 1, 1, 1)
        z = latents.float() * self.latents_std.view(shape) + self.latents_mean.view(shape)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_fp16 and z.is_cuda
            else nullcontext()
        )
        with ctx:
            pixels = self._decode_clip(z)
        pixels = pixels.float() * self.pixel_std + self.pixel_mean
        return pixels.clamp(0.0, 1.0) * 2.0 - 1.0
