"""Non-dynamic Comfy patcher retaining immutable CPU masters through clones."""
import weakref
from collections import namedtuple

import torch
import comfy.model_patcher

from aitk_shared_models.ownership import restore_masters, verify_masters
from aitk_shared_models.protocol import SharedModelError

PATCHERS = weakref.WeakSet()
WeightBackup = namedtuple('WeightBackup', ('weight', 'inplace_update'))


class SharedModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, model, load_device, offload_device, *args, **kwargs):
        if torch.device(offload_device).type != 'cpu' or kwargs.get('weight_inplace_update', False) or kwargs.get('fast_disk', False):
            raise SharedModelError('Shared patcher requires CPU masters, no in-place patch or fast-disk packing')
        super().__init__(model, load_device, offload_device, *args, **kwargs)
        if hasattr(model, '_aitk_load_device') and torch.device(load_device) != model._aitk_load_device:
            raise SharedModelError('Shared model cannot be retargeted to another GPU')
        model._aitk_load_device = torch.device(load_device)
        self.aitk_store = getattr(model, '_aitk_store', None)
        PATCHERS.add(self)

    def load(self, device_to=None, **kwargs):
        if device_to is not None and torch.device(device_to) != self.model._aitk_load_device:
            raise SharedModelError('Shared model supports only its negotiated GPU')
        return super().load(device_to=device_to, **kwargs)

    def clone(self, disable_dynamic=False, model_override=None, force_deepcopy=False):
        if force_deepcopy or model_override is not None:
            raise SharedModelError('Deep/multi-device shared model clones are unsupported')
        verify_masters(self.model)
        cloned = super().clone(disable_dynamic=True)
        cloned.aitk_store = self.aitk_store
        return cloned

    def deepclone_multigpu(self, *args, **kwargs):
        raise SharedModelError('Shared release v1 supports one GPU; multi-GPU cloning is unsupported')

    def pin_weight_to_device(self, key):
        # All shared transfers use the arena's bounded stager. Calling Comfy's
        # pinning or flattening routines here would allocate another backbone.
        return False

    def unpin_weight(self, key):
        self.pinned.discard(key)

    @torch.inference_mode(False)
    @torch.no_grad()
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if inplace_update or self.weight_inplace_update:
            raise SharedModelError('Shared base in-place patches are forbidden')
        weight, set_func, convert_func = comfy.model_patcher.get_key_weight(self.model, key)
        if key not in self.patches and not force_cast:
            return weight
        if device_to is None or torch.device(device_to).type != 'cuda':
            raise SharedModelError('Shared LoRA merges require bounded private GPU temporaries')
        temp_dtype = comfy.model_management.lora_compute_dtype(device_to) if key in self.patches else weight.dtype
        if weight.numel() * torch.empty((), dtype=temp_dtype).element_size() > 512 * 1024**2:
            raise SharedModelError('LoRA patch temporary exceeds shared-mode bound')
        if key not in self.backup and not return_weight:
            self.backup[key] = WeightBackup(weight.to(device=self.offload_device), False)
        # Comfy cast_to_device(copy=True) creates empty_like(QuantizedTensor),
        # which clones its CPU scale Params before the move. Move descriptor
        # leaves directly through bounded staging, then convert/patch on GPU.
        temp_weight = weight.to(device=device_to, dtype=temp_dtype, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)
        out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key) if key in self.patches else temp_weight
        if set_func is not None:
            return set_func(out_weight, inplace_update=False,
                seed=comfy.utils.string_to_seed(key), return_weight=return_weight)
        if key in self.patches:
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype,
                seed=comfy.utils.string_to_seed(key))
        if return_weight:
            return out_weight
        comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        result = super().unpatch_model(device_to, unpatch_weights)
        if unpatch_weights and device_to is not None and torch.device(device_to).type == 'cpu':
            restore_masters(self.model)
            verify_masters(self.model)
        return result

    def add_patches(self, patches, strength_patch=1., strength_model=1.):
        if strength_model != 1.:
            raise SharedModelError('Shared base scaling/mutation is unsupported')
        for key in patches:
            if '.experts_' in key or '.gate.wg.' in key:
                raise SharedModelError('Shared adapters target attention/shared MLP only')
        return super().add_patches(patches, strength_patch, strength_model)


def release_all_shared():
    seen = set()
    for patcher in list(PATCHERS):
        if id(patcher.model) in seen:
            continue
        seen.add(id(patcher.model))
        patcher.unpatch_model(torch.device('cpu'))
        for module in patcher.model.modules():
            if hasattr(module, '_resident_bank'):
                module._resident_bank = None
            if hasattr(module, '_prefetch'):
                module._prefetch = None
        arena = patcher.aitk_store['arena']
        owner = arena._tensor_owner
        if owner is not None and hasattr(owner, 'stager'):
            owner.stager.close()
