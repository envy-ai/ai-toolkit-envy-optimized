"""Guarded tensor aliases of CPU-read-only pages.

PyTorch has no read-only Tensor flag. A raw frombuffer tensor is therefore kept
private behind SharedTensor's write-schema guard; OS read-only pages are a second
line of defence. No writable Python buffer is fabricated. Dtype copies on CPU,
pinning and clones are rejected, rather than silently duplicating the backbone.
This bridge remains experimental until qualified with each quantized runtime.
"""
import weakref
import warnings

import torch
from torch.utils._pytree import tree_map

from .protocol import SharedModelError

_OWNERS = []
_TORCH_DTYPES = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
                 "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
                 "F16": torch.float16, "BF16": torch.bfloat16,
                 "F32": torch.float32, "F64": torch.float64}


class TensorOwner:
    def __init__(self, arena, address):
        self.arena = arena
        self.address = address
        self.size = arena.manifest["arena_bytes"]
        self.views = {}
        self.cpu_temporary_peak_bytes = 0
        _OWNERS.append(weakref.ref(self))

    def track(self, value):
        key = id(value)
        self.views[key] = weakref.ref(value, lambda _: self.views.pop(key, None))

    def has_views(self):
        return bool(self.views)


def owner_of(tensor):
    if isinstance(tensor, SharedTensor):
        return tensor._shared_owner
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.is_meta:
        return None
    try:
        pointer = tensor.data_ptr()
    except RuntimeError:
        return None
    for ref in list(_OWNERS):
        owner = ref()
        if owner is None:
            _OWNERS.remove(ref)
        elif owner.address <= pointer < owner.address + owner.size:
            return owner
    return None


def require_alias(source, result, label="shared tensor"):
    owner = owner_of(source)
    if owner is None or owner_of(result) is not owner or source.data_ptr() != result.data_ptr():
        raise SharedModelError(f"{label}: attachment copied or changed the shared byte offset")
    return result


class ReadonlyStorageInfo:
    """Read-only accounting/Comfy metadata interface, never a writable Storage."""
    __slots__ = ('_owner', '_pointer', '_bytes', '__weakref__')

    def __init__(self, tensor):
        self._owner = tensor._shared_owner
        raw = tensor._shared_base
        storage = raw.untyped_storage()
        self._pointer, self._bytes = storage.data_ptr(), storage.nbytes()
        self._owner.track(self)

    def data_ptr(self):
        return self._pointer

    def nbytes(self):
        return self._bytes

    def __setitem__(self, key, value):
        raise SharedModelError('Immutable shared storage has no write interface')


def bounded_cpu_cast(tensor, dtype, limit_bytes=64 * 1024**2):
    owner = owner_of(tensor)
    if owner is None:
        return tensor.to(dtype=dtype)
    needed = tensor.numel() * torch.empty((), dtype=dtype).element_size()
    if needed > limit_bytes:
        raise SharedModelError('CPU conditioning temporary exceeds explicit shared-mode cap')
    raw = tensor._shared_base if isinstance(tensor, SharedTensor) else tensor
    owner.cpu_temporary_peak_bytes = max(owner.cpu_temporary_peak_bytes, needed)
    return raw.to(dtype=dtype)


class SharedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, base, owner):
        # Cached immutable masters cross Comfy's inference-mode execution and
        # ordinary finally cleanup. Keep alias metadata/version counters normal
        # in every context; this changes no backing bytes or gradient policy.
        with torch.inference_mode(False), torch._C._DisableTorchDispatch():
            obj = torch.Tensor._make_subclass(cls, base, require_grad=False)
        obj._shared_base = base
        obj._shared_owner = owner
        owner.track(obj)
        return obj

    def __repr__(self):
        return f"SharedTensor(shape={tuple(self.shape)}, dtype={self.dtype}, arena={self._shared_owner.arena.manifest['arena_uuid']})"

    def numpy(self, *args, **kwargs):
        raise SharedModelError("Raw NumPy aliases of immutable weights are unsupported")

    def tolist(self):
        if self.numel() > 65536:
            raise SharedModelError("Only small shared metadata can be materialized as Python values")
        return self._shared_base.tolist()

    def untyped_storage(self):
        return ReadonlyStorageInfo(self)

    def storage(self):
        raise SharedModelError("Exporting writable storage interfaces is unsupported")

    def pin_memory(self, *args, **kwargs):
        raise SharedModelError("Shared weights cannot be privately pinned; use bounded staging")

    def as_subclass(self, cls):
        if cls is not SharedTensor:
            raise SharedModelError("Removing shared write protection is unsupported")
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = func._schema
        for argument, value in zip(schema.arguments, args):
            if argument.alias_info is not None and argument.alias_info.is_write:
                found = []
                tree_map(lambda x: found.append(x) if isinstance(x, torch.Tensor) and owner_of(x) is not None else x, value)
                if found:
                    raise SharedModelError(f"In-place write to immutable arena: {func}")
        for argument in schema.arguments:
            if argument.name in kwargs and argument.alias_info is not None and argument.alias_info.is_write:
                value = kwargs[argument.name]
                if isinstance(value, torch.Tensor) and owner_of(value) is not None:
                    raise SharedModelError(f"In-place write to immutable arena: {func}")
        name = schema.name
        if name in ("aten::clone", "aten::_pin_memory", "aten::cat", "aten::stack"):
            raise SharedModelError(f"Private shared-weight copy is forbidden: {name}")
        if name == 'aten::copy_' and len(args) >= 2 and owner_of(args[1]) is not None:
            destination, source = args[0], args[1]
            if destination.device.type != 'cuda':
                raise SharedModelError('Private CPU shared-weight copy requires an explicitly bounded staging/cast API')
            owner = owner_of(source)
            if not hasattr(owner, 'stager'):
                from .registration import BoundedStager
                owner.stager = BoundedStager()
            return owner.stager.copy_into(source, destination)
        if name == "aten::_to_copy":
            destination = torch.device(kwargs.get("device", "cpu"))
            if destination.type == "cpu":
                raise SharedModelError("Shared CPU weights cannot be copied/cast; retain original storage dtype")
            value = args[0]
            owner = owner_of(value)
            if owner is None:
                raise SharedModelError("Shared transfer owner is missing")
            if not hasattr(owner, "stager"):
                from .registration import BoundedStager
                owner.stager = BoundedStager()
            moved = owner.stager.to_device(value, destination, non_blocking=kwargs.get("non_blocking", False))
            if kwargs.get("dtype") is not None and moved.dtype != kwargs["dtype"]:
                moved = moved.to(dtype=kwargs["dtype"])
            return moved
        def unwrap(value):
            return value._shared_base if isinstance(value, SharedTensor) else value
        with torch._C._DisableTorchDispatch():
            result = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        def wrap(value):
            owner = owner_of(value) if isinstance(value, torch.Tensor) else None
            return SharedTensor(value, owner) if owner is not None else value
        return tree_map(wrap, result)


@torch.inference_mode(False)
def tensor_view(arena, name, slice_spec=None):
    desc = arena.manifest["tensors"][name]
    if desc["length"] == 0:
        return torch.empty(desc["shape"], dtype=_TORCH_DTYPES[desc["dtype"]])
    # The warning describes the raw Tensor's lack of a read-only flag. That raw
    # object never leaves this bridge; SharedTensor checks mutating operators.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        base = torch.frombuffer(arena.mapping, dtype=_TORCH_DTYPES[desc["dtype"]],
                                count=desc["length"] // torch.empty((), dtype=_TORCH_DTYPES[desc["dtype"]]).element_size(),
                                offset=desc["offset"])
    base = base.as_strided(desc["shape"], desc["strides"])
    if arena._tensor_owner is None:
        arena._tensor_owner = TensorOwner(arena, base.data_ptr() - desc["offset"])
    value = SharedTensor(base, arena._tensor_owner)
    if slice_spec is not None:
        if isinstance(slice_spec, int):
            if not desc["shape"] or not 0 <= slice_spec < desc["shape"][0]:
                raise SharedModelError("Expert index outside bank")
            value = value[slice_spec]
        else:
            value = value[slice_spec]
    return value
