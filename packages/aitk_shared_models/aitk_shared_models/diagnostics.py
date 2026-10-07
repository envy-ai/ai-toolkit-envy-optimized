"""Physical-memory accounting and phase timing, without CUDA imports."""
import contextlib
import gc
import time
import warnings
from pathlib import Path


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def process_memory(pid="self"):
    wanted = {"Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty", "Rss"}
    result = {}
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in wanted:
            result[key] = int(line.split()[1]) * 1024
    return result


def live_cuda_tensor_summary(limit=16):
    """Failure-only diagnostics, never tensor contents or a quiescence exemption."""
    import torch
    result = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for value in gc.get_objects():
            try:
                if not isinstance(value, torch.Tensor) or value.device.type != 'cuda':
                    continue
                names = set()
                for container in gc.get_referrers(value):
                    if isinstance(container, dict):
                        for key, item in list(container.items()):
                            if item is value and isinstance(key, str) and key != 'value':
                                names.add(key)
                result.append({'type': type(value).__name__, 'shape': list(value.shape),
                    'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size(),
                    'reference_keys': sorted(names)[:8]})
                if len(result) >= limit:
                    break
            except (RuntimeError, ReferenceError, TypeError):
                continue
    return result


@contextlib.contextmanager
def timed(phases, name):
    start = time.monotonic()
    try:
        yield
    finally:
        phases[name] = phases.get(name, 0) + time.monotonic() - start
