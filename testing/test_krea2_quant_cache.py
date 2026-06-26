import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.krea2.krea2 import (  # noqa: E402
    Krea2Model,
    SingleStreamDiT,
)


class FakeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def eval(self):
        return self

    def requires_grad_(self, requires_grad):
        self.requires_grad = requires_grad
        return self


def make_krea_model():
    model = Krea2Model.__new__(Krea2Model)
    model.torch_dtype = torch.float32
    model.vae_torch_dtype = torch.float32
    model.device_torch = torch.device("cpu")
    model.vae_device_torch = torch.device("cpu")
    model.target_lora_modules = [SingleStreamDiT.__name__]
    model.model_config = SimpleNamespace(
        name_or_path="/tmp/krea2/raw.safetensors",
        model_kwargs={
            "checkpoint_filename": "raw.safetensors",
            "mmdit_config": {"layers": 1},
        },
        quantize=True,
        qtype="qfloat8",
        qtype_te="qfloat8",
        quantize_te=False,
        quantize_kwargs={"exclude": ["skip.me"]},
        layer_offloading=False,
        layer_offloading_transformer_percent=0,
        layer_offloading_text_encoder_percent=0,
        low_vram=False,
        inference_lora_path=None,
    )
    model.print_and_status_update = mock.Mock()
    model._load_text_encoder = mock.Mock(
        return_value=("tokenizer", "processor", FakeModule())
    )
    model._load_vae = mock.Mock(return_value=FakeModule())
    model.get_quantized_module_cache_path = mock.Mock(return_value="/tmp/krea-cache.pt")
    model.save_quantized_module_cache = mock.Mock()
    return model


class Krea2QuantizedCacheTests(unittest.TestCase):
    def test_quantized_transformer_is_saved_to_cache_after_cache_miss(self):
        model = make_krea_model()
        transformer = FakeModule()
        model._load_transformer = mock.Mock(return_value=transformer)
        model.load_quantized_module_cache = mock.Mock(return_value=None)

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.krea2.krea2.quantize_model"
            ) as quantize_model,
            mock.patch(
                "extensions_built_in.diffusion_models.krea2.krea2.Krea2Pipeline",
                return_value="pipeline",
            ),
            mock.patch.object(Krea2Model, "get_train_scheduler", return_value="scheduler"),
        ):
            model.load_model()

        model.get_quantized_module_cache_path.assert_called_once()
        cache_kwargs = model.get_quantized_module_cache_path.call_args.kwargs
        self.assertEqual(cache_kwargs["component_name"], "transformer")
        self.assertEqual(cache_kwargs["qtype"], "qfloat8")
        self.assertEqual(
            cache_kwargs["source_ref"]["name_or_path"], "/tmp/krea2/raw.safetensors"
        )
        self.assertEqual(
            cache_kwargs["source_ref"]["checkpoint_filename"], "raw.safetensors"
        )
        self.assertEqual(cache_kwargs["source_ref"]["mmdit_config"]["layers"], 1)
        self.assertEqual(
            cache_kwargs["extra_cache_key"],
            {
                "quantize_kwargs": {"exclude": ["skip.me"]},
                "target_lora_modules": [SingleStreamDiT.__name__],
            },
        )
        model.load_quantized_module_cache.assert_called_once_with(
            "/tmp/krea-cache.pt", "transformer"
        )
        quantize_model.assert_called_once_with(model, transformer)
        model.save_quantized_module_cache.assert_called_once_with(
            transformer, "/tmp/krea-cache.pt", "transformer"
        )
        self.assertIs(model.model, transformer)

    def test_quantized_transformer_cache_hit_skips_loading_and_quantization(self):
        model = make_krea_model()
        cached_transformer = FakeModule()
        model._load_transformer = mock.Mock()
        model.load_quantized_module_cache = mock.Mock(return_value=cached_transformer)

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.krea2.krea2.quantize_model"
            ) as quantize_model,
            mock.patch(
                "extensions_built_in.diffusion_models.krea2.krea2.Krea2Pipeline",
                return_value="pipeline",
            ),
            mock.patch.object(Krea2Model, "get_train_scheduler", return_value="scheduler"),
        ):
            model.load_model()

        model._load_transformer.assert_not_called()
        quantize_model.assert_not_called()
        model.save_quantized_module_cache.assert_not_called()
        self.assertIs(model.model, cached_transformer)


if __name__ == "__main__":
    unittest.main()
