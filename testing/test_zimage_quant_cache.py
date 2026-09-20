import importlib.util
import pathlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


def _load_zimage_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "extensions_built_in/diffusion_models/z_image/z_image.py"
    )
    spec = importlib.util.spec_from_file_location(
        "zimage_quant_cache_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ZIMAGE_MODULE = _load_zimage_module()
ZImageModel = ZIMAGE_MODULE.ZImageModel


class FakeModule(torch.nn.Module):
    def __init__(self, quantized=False):
        super().__init__()
        self.aitk_is_quantized = quantized

    def to(self, *args, **kwargs):
        return self


class FakeTextEncoder(FakeModule):
    def requires_grad_(self, requires_grad):
        self.requires_grad = requires_grad
        return self

    def eval(self):
        return self


class FakePipeline:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.text_encoder = None
        self.transformer = None


def make_zimage_model():
    model = ZImageModel.__new__(ZImageModel)
    model.torch_dtype = torch.float32
    model.device_torch = torch.device("cpu")
    model.low_vram = False
    model.target_lora_modules = ["ZImageTransformer2DModel"]
    model.model_config = SimpleNamespace(
        name_or_path="/tmp/zimage/model.safetensors",
        extras_name_or_path="Tongyi-MAI/Z-Image-Turbo",
        quantize=True,
        quantize_kwargs={},
        qtype="qfloat8",
        assistant_lora_path=None,
        accuracy_recovery_adapter=None,
        layer_offloading=False,
        layer_offloading_transformer_percent=0,
        quantize_te=False,
        qtype_te="qfloat8",
        layer_offloading_text_encoder_percent=0,
        low_vram=False,
    )
    model.print_and_status_update = mock.Mock()
    model.component_load_kwargs = mock.Mock(return_value={})
    model.get_quantized_module_cache_path = mock.Mock(
        return_value="/tmp/zimage-cache.pt"
    )
    model.save_quantized_module_cache = mock.Mock()
    return model


class ZImageQuantizedCacheTests(unittest.TestCase):
    def _load_model(self, model, transformer):
        with (
            mock.patch.object(
                ZIMAGE_MODULE.Qwen3TextEncoder,
                "load_tokenizer",
                return_value="tokenizer",
            ),
            mock.patch.object(
                ZIMAGE_MODULE.Qwen3TextEncoder,
                "load",
                return_value=FakeTextEncoder(),
            ),
            mock.patch.object(
                ZIMAGE_MODULE.KLVAE,
                "load_model",
                return_value=FakeModule(),
            ),
            mock.patch.object(ZIMAGE_MODULE, "ZImagePipeline", FakePipeline),
            mock.patch.object(ZIMAGE_MODULE, "quantize_model") as quantize_model,
            mock.patch.object(
                ZImageModel, "get_train_scheduler", return_value="scheduler"
            ),
        ):
            model.load_model()
        return quantize_model

    def test_universal_loader_quantization_is_cached_without_double_quantizing(self):
        model = make_zimage_model()
        transformer = FakeModule(quantized=True)
        model.load_quantized_module_cache = mock.Mock(return_value=None)
        model.load_transformer = mock.Mock(
            return_value=(transformer, "Tongyi-MAI/Z-Image-Turbo")
        )

        quantize_model = self._load_model(model, transformer)

        quantize_model.assert_not_called()
        model.save_quantized_module_cache.assert_called_once_with(
            transformer, "/tmp/zimage-cache.pt", "transformer"
        )
        self.assertIs(model.model, transformer)

    def test_quantized_cache_hit_skips_universal_loader_and_quantization(self):
        model = make_zimage_model()
        transformer = FakeModule()
        model.load_quantized_module_cache = mock.Mock(return_value=transformer)
        model.load_transformer = mock.Mock()

        quantize_model = self._load_model(model, transformer)

        model.load_transformer.assert_not_called()
        quantize_model.assert_not_called()
        model.save_quantized_module_cache.assert_not_called()
        self.assertTrue(transformer.aitk_is_quantized)

    def test_assistant_lora_path_quantizes_without_writing_an_empty_cache_path(self):
        model = make_zimage_model()
        model.model_config.assistant_lora_path = "/tmp/assistant.safetensors"
        transformer = FakeModule()
        model.load_quantized_module_cache = mock.Mock()
        model.load_transformer = mock.Mock(
            return_value=(transformer, "Tongyi-MAI/Z-Image-Turbo")
        )
        model.load_training_adapter = mock.Mock()

        quantize_model = self._load_model(model, transformer)

        model.get_quantized_module_cache_path.assert_not_called()
        model.load_quantized_module_cache.assert_not_called()
        model.load_training_adapter.assert_called_once_with(transformer)
        quantize_model.assert_called_once_with(model, transformer)
        model.save_quantized_module_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
