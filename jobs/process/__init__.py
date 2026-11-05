"""Process exports, loaded when a job needs the corresponding process class."""

from importlib import import_module
from types import ModuleType
import sys

from .BaseProcess import BaseProcess
from .BaseExtensionProcess import BaseExtensionProcess

_LAZY_PROCESSES = {
    "BaseExtractProcess", "ExtractLoconProcess", "ExtractLoraProcess",
    "BaseTrainProcess", "TrainVAEProcess", "BaseMergeProcess",
    "TrainSliderProcess", "TrainSliderProcessOld", "TrainSDRescaleProcess",
    "ModRescaleLoraProcess", "GenerateProcess", "TrainESRGANProcess",
    "BaseSDTrainProcess",
}
__all__ = ["BaseProcess", "BaseExtensionProcess", *sorted(_LAZY_PROCESSES)]


def __getattr__(name):
    if name not in _LAZY_PROCESSES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    process_class = getattr(import_module(f".{name}", __name__), name)
    globals()[name] = process_class
    return process_class


class _ProcessExports(ModuleType):
    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name in _LAZY_PROCESSES and isinstance(value, ModuleType):
            return getattr(value, name)
        return value


sys.modules[__name__].__class__ = _ProcessExports
