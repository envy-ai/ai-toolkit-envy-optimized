import ctypes
import gc
import sys
import torch
from toolkit.basic import flush
from toolkit.memory_management import MemoryManager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        # register a dummy parameter to avoid errors in some cases
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a fake text encoder and should not be used for inference."
        )
        return None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def to(self, *args, **kwargs):
        return self


def _detach_and_cpu(te: torch.nn.Module):
    MemoryManager.detach(te)
    # bypass any nopped-out .to() override and force an actual CPU move
    torch.nn.Module.to(te, 'cpu')


def unload_text_encoder(model: "BaseModel"):
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.

    is_minimax_h3 = getattr(model, "arch", None) == "minimax_h3"

    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            text_encoder_list = []
            pipe = model.pipeline

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
            if hasattr(pipe, "text_encoder"):
                _detach_and_cpu(pipe.text_encoder)
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                pipe.text_encoder = te

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                real_te = getattr(pipe, f"text_encoder_{i}")
                _detach_and_cpu(real_te)
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                setattr(pipe, f"text_encoder_{i}", te)
                i += 1
            model.text_encoder = text_encoder_list
        else:
            # only has a single text encoder
            text_encoder = model.text_encoder
            _detach_and_cpu(text_encoder)
            # H3's Qwen3-VL conditioner is exceptionally large.  During
            # cached-embedding training it is never needed again, but a stale
            # reference (for example one held while a cache operation unwinds)
            # can otherwise keep its CPU tensors resident.  Move its storage
            # to meta before replacing the public handle so such references
            # cannot retain tens of GB of weights.
            if is_minimax_h3:
                try:
                    text_encoder.to_empty(device="meta")
                except (AttributeError, RuntimeError):
                    # Keep the existing safe CPU unload as a fallback for an
                    # unusual module that cannot be moved to meta.
                    pass
            model.text_encoder = FakeTextEncoder(
                device=model.device_torch,
                dtype=model.torch_dtype
            )

    torch.cuda.empty_cache()
    gc.collect()
    # glibc may otherwise retain the freed H3 encoder allocations in its heap,
    # even though the model is no longer reachable.  This is Linux-only and
    # intentionally limited to the exceptionally large H3 encoder path.
    if is_minimax_h3 and sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    flush()
