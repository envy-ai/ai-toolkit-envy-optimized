"""Shared Fizgig image-slider objective, independent of trainer and model registries.

Functions retain the trainer's original arithmetic and method-compatible context.
"""
import torch
import torch.nn.functional as F
from toolkit.flow_training import trainer_flow_profile
from toolkit.train_tools import get_torch_dtype

def pair_difference_weights(positive, negative, strength):
    """Fizgig's per-latent-token difference weighting; mean weight remains 1."""
    difference = (positive.float() - negative.float()).abs().mean(dim=1)
    difference = difference.flatten(1)
    mean = difference.mean(dim=1, keepdim=True)
    relative = (difference / mean.clamp_min(1e-8)).clamp(max=8.0)
    weight = (1.0 - strength) + strength * relative
    weight = weight / weight.mean(dim=1, keepdim=True).clamp_min(1e-8)
    weight = torch.where(mean > 1e-6, weight, torch.ones_like(weight))
    return weight

def slider_squared_error(trainer, prediction, target):
    """Match ordinary differential guidance with a detached extrapolated target."""
    config = getattr(trainer, "train_config", None)
    if getattr(config, "do_differential_guidance", False):
        with torch.no_grad():
            scale = getattr(config, "differential_guidance_scale", 3.0)
            target = prediction + scale * (target - prediction)
    return F.mse_loss(prediction, target, reduction="none")

def slider_loss_mean(trainer, loss, timesteps):
    """Weight each example by its noise level before reducing the batch."""
    gamma = getattr(getattr(trainer, "train_config", None), "min_snr_gamma", None)
    if gamma is None or gamma <= 0.000001:
        return loss.mean()
    per_image = loss.flatten(1).mean(dim=1)
    # _noised_state samples continuous sigma = timestep / 1000 directly,
    # without initializing the scheduler's discrete training schedule.
    # For x_t = (1-sigma)*clean + sigma*noise, SNR = (1-sigma)^2/sigma^2.
    # Compute gamma/SNR directly, including finite weights at sigma 0 and 1.
    sigma = timesteps.detach().to(device=loss.device, dtype=torch.float32).flatten() / 1000.0
    signal_sq = (1.0 - sigma).square()
    weights = (gamma * sigma.square() / signal_sq.clamp_min(torch.finfo(torch.float32).tiny)).clamp(max=1.0)
    return (per_image * weights).mean()

def _noised_state(self, clean):
        # Fizgig v6.7.0: sigmoid-normal time, Qwen's resolution shift, linear
        # interpolation, and velocity target noise - clean.
        clean = clean.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
        batch_size, _, height, width = clean.shape
        mu = trainer_flow_profile(self).shift(height, width)
        t = torch.sigmoid(torch.randn(batch_size, device=self.device_torch, dtype=torch.float32) + mu)
        min_t = self.train_config.min_denoising_steps / 1000.0
        max_t = self.train_config.max_denoising_steps / 1000.0
        t = min_t + (max_t - min_t) * t
        noise = torch.randn_like(clean, dtype=torch.float32)
        t_view = t.view(-1, 1, 1, 1)
        noisy = ((1 - t_view) * clean.float() + t_view * noise).to(clean.dtype)
        target = noise - clean.float()
        return noisy, (t * 1000).to(self.device_torch), target

def _image_loss(self, clean, other, embeds):
        noisy, timestep, target = self._noised_state(clean)
        prediction = self._predict(noisy, timestep, embeds)
        squared_error = slider_squared_error(self, prediction, target).mean(dim=1).flatten(1)
        if self.slider_diff_weight:
            weights = pair_difference_weights(clean, other, self.slider_diff_weight)
            squared_error = squared_error * weights
        return slider_loss_mean(self, squared_error, timestep)
