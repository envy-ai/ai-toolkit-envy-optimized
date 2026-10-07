"""Explicit registration probe and bounded staging. Imported only by active owners.

Registration is opt-in after qualification. Default staging never privately pins
an entire weight: two buffers share a 512 MiB budget. No CUDA work occurs on import.
"""
import contextlib
import ctypes
import threading

from .protocol import SharedModelError


class HostRegistration:
    def __init__(self, runtime, budget_bytes=512 * 1024**2):
        self.runtime = runtime
        self.budget_bytes = budget_bytes
        self.ranges = {}
        self.pending = []
        self.owners = []
        self.capability_verified = False

    @staticmethod
    def _check(result):
        code = result[0] if isinstance(result, tuple) else result
        if code != 0:
            raise SharedModelError(f"CUDA host registration failed ({code}); use bounded staging")

    def probe_readonly(self, tensor):
        from .tensors import owner_of
        if owner_of(tensor) is None or tensor.device.type != "cpu":
            raise SharedModelError("Probe requires an immutable mapped tensor")
        pointer = tensor.data_ptr() // 4096 * 4096
        length = ((tensor.data_ptr() + tensor.numel() * tensor.element_size() + 4095) // 4096 * 4096) - pointer
        # cudaHostRegisterReadOnly=8. Actual RO mapping is the capability probe;
        # success must still be qualified with asynchronous H2D in both runtimes.
        self._check(self.runtime.cudaHostRegister(pointer, length, 8))
        self._check(self.runtime.cudaHostUnregister(pointer))
        self.capability_verified = True

    def register(self, tensor):
        from .tensors import owner_of
        owner = owner_of(tensor)
        if not self.capability_verified or owner is None:
            raise SharedModelError("Read-only host registration has not passed the capability probe")
        begin = tensor.data_ptr() // 4096 * 4096
        end = (tensor.data_ptr() + tensor.numel() * tensor.element_size() + 4095) // 4096 * 4096
        for (a, b) in self.ranges:
            if a <= begin and end <= b:
                return
            if begin < b and a < end:
                raise SharedModelError("Overlapping registration must use the existing bank coverage")
        if sum(b - a for a, b in self.ranges) + end - begin > self.budget_bytes:
            raise SharedModelError("Host registration budget exceeded")
        self._check(self.runtime.cudaHostRegister(begin, end - begin, 8))
        self.ranges[(begin, end)] = owner

    def record_transfer(self, event):
        self.pending.append(event)

    def close(self):
        for event in self.pending:
            event.synchronize()
        self.pending.clear()
        for begin, end in list(self.ranges):
            self._check(self.runtime.cudaHostUnregister(begin))
            del self.ranges[(begin, end)]


class BoundedStager:
    def __init__(self, buffer_bytes=256 * 1024**2):
        if not 0 < buffer_bytes <= 256 * 1024**2:
            raise SharedModelError("Each staging buffer is capped at 256 MiB")
        self.buffer_bytes = buffer_bytes
        self.buffers = []
        self.events = [None, None]
        self.index = 0
        self.lock = threading.RLock()
        self.bytes_copied = 0

    def to_device(self, tensor, device, non_blocking=True):
        import torch
        from .tensors import owner_of, SharedTensor
        if owner_of(tensor) is None:
            return tensor.to(device, non_blocking=non_blocking)
        device = torch.device(device)
        if device.type != "cuda":
            if device.type == "cpu":
                return tensor
            raise SharedModelError("Shared staging supports one CUDA device")
        if not tensor.is_contiguous():
            raise SharedModelError("Shared transfer requires a contiguous view")
        raw = tensor._shared_base if isinstance(tensor, SharedTensor) else tensor
        destination = torch.empty_like(raw, device=device)
        self.copy_into(tensor, destination, non_blocking=non_blocking)
        return destination

    def copy_into(self, tensor, destination, non_blocking=True):
        import torch
        from .tensors import SharedTensor
        raw = tensor._shared_base if isinstance(tensor, SharedTensor) else tensor
        device = destination.device
        if not raw.is_contiguous() or not destination.is_contiguous() or raw.shape != destination.shape:
            raise SharedModelError('Shared bounded staging requires matching contiguous extents')
        if raw.dtype != destination.dtype:
            moved = self.to_device(tensor, device, non_blocking=non_blocking)
            destination.copy_(moved)
            return destination
        with self.lock, torch.cuda.device(device):
            if not self.buffers:
                self.buffers = [torch.empty(self.buffer_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
            source_bytes = raw.view(torch.uint8).reshape(-1)
            dest_bytes = destination.view(torch.uint8).reshape(-1)
            for offset in range(0, source_bytes.numel(), self.buffer_bytes):
                slot = self.index % 2
                self.index += 1
                event = self.events[slot]
                if event is not None:
                    event.synchronize()
                length = min(self.buffer_bytes, source_bytes.numel() - offset)
                host = self.buffers[slot][:length]
                host.copy_(source_bytes[offset:offset + length])
                dest_bytes[offset:offset + length].copy_(host, non_blocking=True)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                self.events[slot] = event
                self.bytes_copied += length
            if not non_blocking:
                torch.cuda.current_stream(device).synchronize()
            return destination

    def close(self):
        with self.lock:
            for event in self.events:
                if event is not None:
                    event.synchronize()
            self.events = [None, None]
            self.buffers.clear()
