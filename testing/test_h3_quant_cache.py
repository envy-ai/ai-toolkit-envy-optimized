import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.output_capturing import (
    _CAN_RECORD_REGISTRY,
    capture_outputs,
)

from toolkit.util.ostris_quant import OstrisLinear


def _load_h3_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    package_path = repo_root / "extensions_built_in/diffusion_models/minimax_h3"
    package_name = "minimax_h3_quant_cache_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package
    module_name = f"{package_name}.minimax_h3"
    spec = importlib.util.spec_from_file_location(
        module_name, package_path / "minimax_h3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


H3_MODULE = _load_h3_module()
MinimaxH3Model = H3_MODULE.MinimaxH3Model


class FakeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._fake_device = torch.device("cpu")

    @property
    def device(self):
        return self._fake_device

    def to(self, device=None, *args, **kwargs):
        if device is not None and not isinstance(device, torch.dtype):
            self._fake_device = torch.device(device)
        return self


class FakeOstrisLinear(OstrisLinear):
    def __init__(self):
        torch.nn.Module.__init__(self)


class FakeQuantizedTextEncoder(FakeModule):
    def __init__(self):
        super().__init__()
        self.quantized_layer = FakeOstrisLinear()


class TinyHiddenStateModel(PreTrainedModel):
    """Small stand-in for Transformers' hook-based Qwen hidden-state API."""

    config_class = PretrainedConfig
    _can_record_outputs = {"hidden_states": torch.nn.Linear}

    def __init__(self, config):
        super().__init__(config)
        self.layer = torch.nn.Linear(2, 2)
        self.post_init()

    @capture_outputs
    def forward(self, values, **kwargs):
        return BaseModelOutput(last_hidden_state=self.layer(values))


def make_h3_model():
    model = MinimaxH3Model.__new__(MinimaxH3Model)
    model.torch_dtype = torch.float32
    model.te_torch_dtype = torch.float32
    model.device_torch = torch.device("cpu")
    model.dtype = "bf16"
    model.target_lora_modules = ["MiniMaxH3Transformer"]
    model.model_config = SimpleNamespace(
        name_or_path="Comfy-Org/MiniMax-H3",
        te_name_or_path=None,
        model_kwargs={"partition": "fl2va_pruned"},
        quantize=True,
        qtype="convrotint6",
        quantize_kwargs={},
        quantize_te=True,
        qtype_te="nvfp4",
        assistant_lora_path="/tmp/h3-training-adapter.safetensors",
        low_vram=False,
        low_vram_layer_streaming=False,
        layer_offloading=False,
        layer_offloading_transformer_percent=0,
        layer_offloading_text_encoder_percent=0,
    )
    model.print_and_status_update = mock.Mock()
    model._resolve_comfy_file = mock.Mock(
        side_effect=lambda component: f"/tmp/{component}.safetensors"
    )
    model._load_vaes = mock.Mock(return_value=FakeModule())
    model.get_quantized_module_cache_path = mock.Mock(
        side_effect=lambda component_name, **kwargs: f"/tmp/{component_name}-cache.pt"
    )
    model.save_quantized_module_cache = mock.Mock()
    model.load_training_adapter = mock.Mock()
    return model


class H3QuantizedCacheTests(unittest.TestCase):
    def _load_model(self, model):
        with (
            mock.patch.object(H3_MODULE, "quantize_model") as quantize_model,
            mock.patch.object(H3_MODULE, "quantize") as quantize_text_encoder,
            mock.patch.object(H3_MODULE, "MiniMaxH3Pipeline", return_value="pipeline"),
            mock.patch.object(
                MinimaxH3Model, "get_train_scheduler", return_value="scheduler"
            ),
        ):
            model.load_model()
        return quantize_model, quantize_text_encoder

    def test_cache_miss_quantizes_and_caches_both_modules(self):
        model = make_h3_model()
        transformer = FakeModule()
        text_encoder = FakeQuantizedTextEncoder()
        model.load_quantized_module_cache = mock.Mock(side_effect=[None, None])
        model._load_transformer = mock.Mock(return_value=transformer)
        model._load_text_encoder = mock.Mock(
            return_value=("tokenizer", "processor", text_encoder)
        )
        model._load_tokenizer_processor = mock.Mock()

        call_order = mock.Mock()
        call_order.attach_mock(model.save_quantized_module_cache, "save")
        call_order.attach_mock(model.load_training_adapter, "adapter")

        quantize_model, quantize_text_encoder = self._load_model(model)

        model._load_transformer.assert_called_once_with(
            "/tmp/dit_fl2va_pruned.safetensors"
        )
        quantize_model.assert_called_once_with(model, transformer)
        model._load_text_encoder.assert_called_once_with(
            "/tmp/text_encoder.safetensors"
        )
        quantize_text_encoder.assert_not_called()
        self.assertEqual(
            model.save_quantized_module_cache.call_args_list,
            [
                mock.call(transformer, "/tmp/transformer-cache.pt", "transformer"),
                mock.call(
                    text_encoder, "/tmp/text_encoder-cache.pt", "text encoder"
                ),
            ],
        )
        self.assertLess(
            call_order.mock_calls.index(
                mock.call.save(
                    transformer, "/tmp/transformer-cache.pt", "transformer"
                )
            ),
            call_order.mock_calls.index(mock.call.adapter(transformer)),
        )

        cache_calls = model.get_quantized_module_cache_path.call_args_list
        self.assertEqual(cache_calls[0].kwargs["qtype"], "convrotint6")
        self.assertEqual(cache_calls[1].kwargs["qtype"], "nvfp4")
        self.assertEqual(
            cache_calls[1].kwargs["source_ref"]["checkpoint"],
            "/tmp/text_encoder.safetensors",
        )

    def test_cache_hits_skip_source_weight_loading_and_quantization(self):
        model = make_h3_model()
        transformer = FakeModule()
        text_encoder = FakeQuantizedTextEncoder()
        model.load_quantized_module_cache = mock.Mock(
            side_effect=[transformer, text_encoder]
        )
        model._load_transformer = mock.Mock()
        model._load_text_encoder = mock.Mock()
        model._load_tokenizer_processor = mock.Mock(
            return_value=("tokenizer", "processor")
        )

        quantize_model, quantize_text_encoder = self._load_model(model)

        model._load_transformer.assert_not_called()
        model._load_text_encoder.assert_not_called()
        model._load_tokenizer_processor.assert_called_once_with()
        quantize_model.assert_not_called()
        quantize_text_encoder.assert_not_called()
        model.save_quantized_module_cache.assert_not_called()
        model.load_training_adapter.assert_called_once_with(transformer)
        self.assertIs(model.model, transformer)
        self.assertIs(model.text_encoder, text_encoder)

    def test_h3_opts_into_text_encoder_cache_without_enabling_it_globally(self):
        model = MinimaxH3Model.__new__(MinimaxH3Model)
        model.dtype = "bf16"
        with tempfile.TemporaryDirectory() as cache_dir:
            model.model_config = SimpleNamespace(
                cache_quantized_models=True,
                quantized_model_cache_dir=cache_dir,
            )
            cache_path = model.get_quantized_module_cache_path(
                component_name="text_encoder",
                qtype="nvfp4",
                source_ref={"checkpoint": "/tmp/h3-text-encoder.safetensors"},
            )

        self.assertIsNotNone(cache_path)
        self.assertTrue(cache_path.endswith(".pt"))

    def test_cache_load_restores_transformers_hidden_state_capture(self):
        model = make_h3_model()
        cached_model = TinyHiddenStateModel(PretrainedConfig())

        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = str(pathlib.Path(cache_dir) / "text_encoder.pt")
            torch.save(cached_model, cache_path)
            registry_key = str(TinyHiddenStateModel)
            _CAN_RECORD_REGISTRY.pop(registry_key, None)

            restored = model.load_quantized_module_cache(
                cache_path, "text encoder"
            )

        self.assertIs(
            _CAN_RECORD_REGISTRY[registry_key],
            TinyHiddenStateModel._can_record_outputs,
        )
        outputs = restored(torch.ones(1, 2), output_hidden_states=True)
        self.assertIsNotNone(outputs.hidden_states)
        self.assertEqual(len(outputs.hidden_states), 2)


if __name__ == "__main__":
    unittest.main()
