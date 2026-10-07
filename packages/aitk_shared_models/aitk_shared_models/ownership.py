"""Module sidecars retain CPU masters through Parameter/quantized wrapping."""
import types

import torch

from .protocol import SharedModelError
from .tensors import owner_of


def storage_leaves(value):
    """Use tensor-flatten contracts; quantization parameter tensors are storage too."""
    seen = set()
    def walk(tensor):
        if id(tensor) in seen:
            return
        seen.add(id(tensor))
        try:
            names, _ = tensor.__tensor_flatten__()
        except (AttributeError, TypeError):
            yield tensor
            return
        for name in names:
            inner = getattr(tensor, name)
            if isinstance(inner, torch.Tensor):
                yield from walk(inner)
    yield from walk(value)


def capture_masters(module, strict=False):
    masters = {}
    for child in module.modules():
        for kind, slots in (('parameter', child._parameters), ('buffer', child._buffers)):
            for name, value in slots.items():
                if value is None or value.is_meta:
                    continue
                leaves = list(storage_leaves(value))
                shared = [owner_of(leaf) is not None for leaf in leaves]
                if any(shared):
                    if not all(shared) or value.requires_grad:
                        raise SharedModelError(f'{name}: shared weight has private storage or requires gradients')
                    masters[(child, kind, name)] = value.detach()
                elif strict and value.numel() * value.element_size() > 1024**2:
                    raise SharedModelError(f'{name}: model-sized private CPU base allocation')
    module._shared_cpu_masters = masters
    install_shared_apply(module, masters)
    return masters


def install_shared_apply(module, masters):
    def shared_apply(child, fn, recurse=True):
        if recurse:
            for descendant in child.children():
                descendant._apply(fn)
        for kind, slots in (('parameter', child._parameters), ('buffer', child._buffers)):
            for name, value in list(slots.items()):
                if value is None:
                    continue
                master = masters.get((child, kind, name))
                if master is None:
                    moved = fn(value)
                else:
                    probe = fn(torch.empty(0, dtype=value.dtype, device=value.device))
                    if probe.device.type == 'cpu':
                        if probe.dtype != master.dtype:
                            raise SharedModelError('Shared CPU base dtype conversions are unsupported')
                        moved = master
                    elif probe.device.type == 'meta':
                        moved = torch.empty(value.shape, dtype=value.dtype, device='meta')
                    else:
                        moved = fn(value)
                if kind == 'parameter':
                    if value.requires_grad:
                        value.data = moved
                        if value.grad is not None:
                            value.grad = fn(value.grad)
                    else:
                        with torch.inference_mode(False):
                            slots[name] = torch.nn.Parameter(moved, requires_grad=False)
                else:
                    slots[name] = moved
        return child
    for child in module.modules():
        if not hasattr(child, '_shared_original_apply'):
            child._shared_original_apply = child._apply
            child._apply = types.MethodType(shared_apply, child)


def restore_masters(module):
    for (child, kind, name), master in module._shared_cpu_masters.items():
        if kind == 'parameter':
            child._parameters[name] = torch.nn.Parameter(master, requires_grad=False)
        else:
            child._buffers[name] = master


def verify_masters(module):
    for (child, kind, name), master in module._shared_cpu_masters.items():
        current = (child._parameters if kind == 'parameter' else child._buffers)[name]
        for original, attached in zip(storage_leaves(master), storage_leaves(current)):
            if attached.device.type == 'cpu':
                if owner_of(original) is None or owner_of(attached) is not owner_of(original) or original.data_ptr() != attached.data_ptr():
                    raise SharedModelError(f'{name}: shared master changed during cloning/parking')
