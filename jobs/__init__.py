"""Job exports, loaded when a config selects the corresponding job type."""

from importlib import import_module
from types import ModuleType
import sys

from .BaseJob import BaseJob

_LAZY_JOBS = {
    "ExtractJob", "TrainJob", "MergeJob", "ModJob", "GenerateJob", "ExtensionJob",
}
__all__ = ["BaseJob", *sorted(_LAZY_JOBS)]


def __getattr__(name):
    if name not in _LAZY_JOBS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    job_class = getattr(import_module(f".{name}", __name__), name)
    globals()[name] = job_class
    return job_class


class _JobExports(ModuleType):
    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name in _LAZY_JOBS and isinstance(value, ModuleType):
            return getattr(value, name)
        return value


sys.modules[__name__].__class__ = _JobExports
