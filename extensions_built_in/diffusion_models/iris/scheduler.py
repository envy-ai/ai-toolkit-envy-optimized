"""Reuse toolkit flow training, sampling noise levels with Iris's shift of four."""
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


class IrisFlowScheduler(CustomFlowMatchEulerDiscreteScheduler):
    def __init__(self):
        super().__init__(num_train_timesteps=1000, shift=4.0, use_dynamic_shifting=False)

    def set_train_timesteps(self, num_timesteps, device, timestep_type="sigmoid", latents=None, patch_size=16):
        times = super().set_train_timesteps(num_timesteps, device, timestep_type, latents, patch_size)
        if timestep_type not in ("shift", "flux_shift", "lumina2_shift"):
            sigma = times / 1000
            sigma = 4 * sigma / (1 + 3 * sigma)
            self.timesteps = sigma * 1000
            self._set_training_schedule_values(sigma, device=device)
        return self.timesteps
