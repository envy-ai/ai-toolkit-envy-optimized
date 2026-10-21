"""Native FIBO2 Euler/CFG previews; toolkit packed latents, reference flow schedule."""
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from toolkit.basic import flush

from .conditioning import flow_mu


class Fibo2Pipeline:
    def __init__(self, model):
        self.model = model

    @property
    def transformer(self):
        return self.model.transformer

    @property
    def text_encoder(self):
        return self.model.text_encoder

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(self, conditional_embeds, unconditional_embeds=None, height=1024, width=1024,
                 num_inference_steps=30, guidance_scale=5., generator=None, latents=None,
                 ref_latents=None, **kwargs):
        for name, value in (('height', height), ('width', width)):
            if type(value) is not int or value <= 0 or value % 16:
                raise ValueError(f'FIBO2 {name} must be a positive integer multiple of 16; got {value!r}')
        if type(num_inference_steps) is not int or num_inference_steps <= 0:
            raise ValueError('FIBO2 num_inference_steps must be a positive integer')
        model = self.model
        # Conditioning and detached reference latents are complete. Do not keep
        # the full VLM/VAE beside the DiT during uncached native previews/banks.
        if getattr(model, 'text_encoder', None) is not None:
            model.text_encoder_to('cpu')
        if getattr(model, 'vae', None) is not None:
            model.vae.to('cpu')
        flush()
        shape = (len(conditional_embeds.text_embeds), model.transformer.params.in_channels, height//16, width//16)
        state = randn_tensor(shape, generator=generator, device=self.device, dtype=torch.float32) if latents is None else latents.to(self.device, torch.float32)
        sigmas = torch.linspace(1., 1/num_inference_steps, num_inference_steps, device=self.device)
        mu = flow_mu(height//16, width//16)
        exp_mu = torch.exp(torch.tensor(mu, device=self.device))
        sigmas = exp_mu * sigmas / (1 + (exp_mu-1)*sigmas)
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        for index, (sigma, next_sigma) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            pred = model.predict_embeddings(state, sigma.expand(shape[0])*1000, conditional_embeds, ref_latents).float()
            if guidance_scale != 1 and unconditional_embeds is not None:
                negative = model.predict_embeddings(state, sigma.expand(shape[0])*1000, unconditional_embeds, ref_latents).float()
                pred = negative + guidance_scale*(pred-negative)
            if getattr(model, 'sample_step_hook', None) is not None:
                model._emit_sample_step(state-sigma*pred, index, num_inference_steps)
            state = state + (next_sigma-sigma)*pred
        images = model.decode_latents(state).float().add(1).div(2).clamp(0, 1)
        return [Image.fromarray((image.permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')) for image in images]
