"""Native sequential CFG Euler sampling using the same conditioning as training."""
import torch
from tqdm.auto import tqdm


class HunyuanImage3Pipeline:
    show_sample_step_progress = True

    def __init__(self, holder):
        self.holder = holder
        self.transformer = holder.model
        self.vae = holder.vae
        self.text_encoder = holder.text_encoder[0]
        self.scheduler = holder.noise_scheduler
        self._progress_bar_config = dict(desc='Sampling', leave=True, position=0,
                                         unit='step', dynamic_ncols=True, miniters=1,
                                         mininterval=0)

    def to(self, *args, **kwargs):
        # Holder/device presets own component placement; never upload the 80B backbone here.
        return self

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config.update(kwargs)

    @torch.no_grad()
    def __call__(self, conditional_embeds, unconditional_embeds, width, height,
                 num_inference_steps=50, guidance_scale=2.5, generator=None, latents=None):
        holder = self.holder
        if latents is None:
            noise_device = generator.device if generator is not None else holder.device_torch
            latents = torch.randn((1, 32, height // 16, width // 16), generator=generator,
                                   device=noise_device, dtype=torch.float32).to(holder.device_torch)
        else:
            latents = latents.to(holder.device_torch, torch.float32)
        sigmas = torch.linspace(1, 0, num_inference_steps + 1, device=latents.device)
        shift = holder.variant.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        with tqdm(total=num_inference_steps, **self._progress_bar_config) as progress:
            for i in range(num_inference_steps):
                t = (sigmas[i] * 1000).reshape(1)
                positive = holder.get_noise_prediction(latents, t, conditional_embeds)
                if guidance_scale != 1:
                    if unconditional_embeds is None:
                        raise ValueError('CFG requires unconditional Hunyuan conditioning')
                    negative = holder.get_noise_prediction(latents, t, unconditional_embeds)
                    positive = negative + guidance_scale * (positive - negative)
                latents = latents + positive.float() * (sigmas[i + 1] - sigmas[i])
                holder._emit_sample_step(latents, i, num_inference_steps)
                progress.update(1)
        return holder.decode_to_images(latents)
