"""Native DPM-Solver++ preview sampling, with sequential CFG to limit memory."""
import torch
from PIL import Image
from toolkit.sample_progress import SampleProgressMixin
from diffusers.utils.torch_utils import randn_tensor
from .src.flow.solver import FlowDPMSolver


class IrisPipeline(SampleProgressMixin):
    def __init__(self, model):
        self.model = model

    @property
    def text_encoder(self):
        # Do not retain an encoder that the trainer has unloaded after caching.
        return self.model.text_encoder

    @property
    def transformer(self):
        return self.model.model

    @property
    def vae(self):
        return self.model.vae

    def to(self, *args, **kwargs):
        return self

    @torch.no_grad()
    def __call__(self, conditional_embeds, unconditional_embeds, height, width,
                 num_inference_steps, guidance_scale, generator, latents=None):
        if num_inference_steps < 1:
            raise ValueError("Iris sampling requires at least one step")
        model = self.model
        x = latents if latents is not None else randn_tensor(
            (1, 3, height, width), generator=generator, device=model.device_torch, dtype=torch.float32)
        x = x.to(model.device_torch, torch.float32)
        grid = FlowDPMSolver.time_grid(num_inference_steps, 4.0)
        history = []
        for i in self.progress_bar(range(1, num_inference_steps + 1)):
            s, t = grid[i - 1], grid[i]
            time = torch.full((x.shape[0],), s * 1000, device=x.device)
            velocity = model.get_noise_prediction(x, time, conditional_embeds).float()
            if unconditional_embeds is not None and guidance_scale != 1.0:
                negative = model.get_noise_prediction(x, time, unconditional_embeds).float()
                velocity = negative + guidance_scale * (velocity - negative)
            clean = x - s * velocity
            history.append((s, clean))
            if min(i, 2, num_inference_steps + 1 - i) == 1:
                x = FlowDPMSolver._first_order(x, s, t, clean)
            else:
                x = FlowDPMSolver._second_order(x, history[-2], history[-1], t)
            history = history[-2:]
            if model.sample_step_hook is not None:
                model.sample_step_hook(i - 1, num_inference_steps, x)
        pixels = ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return [Image.fromarray(p) for p in pixels.permute(0, 2, 3, 1).cpu().numpy()]
