"""Independent ordinary LoRAs sharing a single frozen diffusion transformer."""

import os
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from toolkit.lora_special import LoRASpecialNetwork


class SliderSpaceNetwork(nn.Module):
    def __init__(self, *args, num_directions, **kwargs):
        super().__init__()
        self.network_config = kwargs.get('network_config')
        self.network_type = 'lora'
        self.can_merge_in = False
        self.is_merged_in = False
        self.is_lorm = False
        self.is_assistant_adapter = False
        self.did_change_weights = False
        self.is_checkpointing = False
        self.active_direction = 0
        self._active = False
        self._multiplier = kwargs.get('multiplier', 1.0)
        # Each child patches the same modules. Inactive children transparently
        # call their original forward, so precisely one adapter contributes.
        # There is no second base model and no dense base-weight copy.
        self.directions = nn.ModuleList([
            LoRASpecialNetwork(*args, **{key: list(value) if isinstance(value, list) else value
                                       for key, value in kwargs.items()}) for _ in range(num_directions)
        ])
        self._sync_activity()

    def _sync_activity(self):
        for index, direction in enumerate(self.directions):
            direction.is_active = self._active and index == self.active_direction

    def select_direction(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self.directions):
            raise ValueError('SliderSpace direction is out of range')
        self.active_direction = index
        self._sync_activity()

    @property
    def is_active(self):
        return self._active

    @is_active.setter
    def is_active(self, value):
        self._active = bool(value)
        self._sync_activity()

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value):
        self._multiplier = value
        for direction in self.directions:
            direction.multiplier = value

    def _update_torch_multiplier(self):
        for direction in self.directions:
            direction._update_torch_multiplier()

    def force_to(self, device, dtype):
        for direction in self.directions:
            direction.force_to(device, dtype)

    def apply_to(self, *args, **kwargs):
        for direction in self.directions:
            direction.apply_to(*args, **kwargs)

    def prepare_grad_etc(self, *args, **kwargs):
        for direction in self.directions:
            direction.prepare_grad_etc(*args, **kwargs)

    def prepare_optimizer_params(self, text_encoder_lr=None, unet_lr=None, default_lr=None):
        return [group for direction in self.directions
                for group in direction.prepare_optimizer_params(text_encoder_lr, unet_lr, default_lr)]

    def get_all_modules(self):
        return [module for direction in self.directions for module in direction.get_all_modules()]

    def enable_gradient_checkpointing(self):
        self.is_checkpointing = True
        for direction in self.directions:
            direction.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.is_checkpointing = False
        for direction in self.directions:
            direction.disable_gradient_checkpointing()

    def get_state_dict(self, *args, **kwargs):
        return self.directions[self.active_direction].get_state_dict(*args, **kwargs)

    def save_weights(self, filename, *args, **kwargs):
        # Used by regular internal/Comfy sampling: always a standard, single
        # direction LoRA. Complete training state is saved separately below.
        destination = Path(filename)
        staging = destination.parent / '.sliderspace_exports'
        staging.mkdir(parents=True, exist_ok=True)
        temporary = staging / (destination.name + '.tmp.safetensors')
        try:
            self.directions[self.active_direction].save_weights(str(temporary), *args, **kwargs)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def save_bank(self, filename, metadata):
        state = {key: value.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous()
                 for key, value in self.state_dict().items()}
        temporary = filename + '.tmp'
        save_file(state, temporary, metadata=metadata)
        os.replace(temporary, filename)

    def load_weights(self, filename):
        state = load_file(filename) if isinstance(filename, str) else filename
        if not state or not all(key.startswith('directions.') for key in state):
            raise ValueError('Resume SliderSpace from its complete bank checkpoint, not a single direction LoRA')
        self.load_state_dict(state, strict=True)
        for module in self.get_all_modules():
            module._set_runtime_scale(float(module.alpha.detach().float().item()) / module.lora_dim)
        return None

    def merge_in(self, *args, **kwargs):
        return

    def merge_out(self, *args, **kwargs):
        return

    def __enter__(self):
        self.is_active = True
        return self

    def __exit__(self, *args):
        self.is_active = False
