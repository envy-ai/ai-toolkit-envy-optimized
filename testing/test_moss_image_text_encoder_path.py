"""MOSS text encoder path selection and form exposure."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from extensions_built_in.diffusion_models.moss_image.moss_image import (
    MossImageModel,
    QWEN35_9B_REPO,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class MossTextEncoderPathTests(unittest.TestCase):
    def setUp(self):
        self.model = MossImageModel.__new__(MossImageModel)
        self.model.model_config = SimpleNamespace(
            te_name_or_path=None,
            text_encoder_path=None,
            model_kwargs={},
        )
        self.model.is_edit = False
        self.model.print_and_status_update = lambda _message: None

    def load_text_encoder(self, **load_kwargs):
        captured = {}
        loaded_encoder = MagicMock()

        def load_model(path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return loaded_encoder

        self.model.component_load_kwargs = lambda _role: {
            "dtype": "bf16",
            "qtype": "convrot8",
            "use_comfy_weights": True,
        }
        with patch(
            "extensions_built_in.diffusion_models.moss_image.moss_image.Qwen35TextOnlyEncoder.load_model",
            side_effect=load_model,
        ) as loader, patch(
            "extensions_built_in.diffusion_models.moss_image.moss_image.comfy_text_encoder_file",
            return_value="/comfy/text_encoder.safetensors",
        ) as comfy_file, patch(
            "extensions_built_in.diffusion_models.moss_image.moss_image.resolve_hub_file",
            side_effect=lambda path, **_kwargs: f"resolved:{path}",
        ) as resolve_file:
            self.model.model_config.model_kwargs.update(load_kwargs.pop("model_kwargs", {}))
            for key, value in load_kwargs.items():
                setattr(self.model.model_config, key, value)
            result = self.model._load_text_encoder()
        self.assertIs(result, loaded_encoder)
        loader.assert_called_once()
        return captured, comfy_file, resolve_file

    def test_precedence_is_te_name_then_direct_path_then_model_kwargs_then_default(self):
        cases = (
            (
                {"te_name_or_path": "/explicit/te-name"},
                "/explicit/te-name",
                None,
            ),
            (
                {
                    "te_name_or_path": None,
                    "text_encoder_path": "/explicit/text-encoder",
                },
                "/explicit/text-encoder",
                None,
            ),
            (
                {
                    "te_name_or_path": None,
                    "text_encoder_path": None,
                    "model_kwargs": {"text_encoder_path": "/kwargs/text-encoder"},
                },
                "/kwargs/text-encoder",
                None,
            ),
            (
                {
                    "te_name_or_path": None,
                    "text_encoder_path": None,
                    "model_kwargs": {},
                },
                "/comfy/text_encoder.safetensors",
                QWEN35_9B_REPO,
            ),
        )
        for config, expected_path, expected_config in cases:
            with self.subTest(config=config):
                self.setUp()
                captured, comfy_file, _resolve_file = self.load_text_encoder(**config)
                self.assertEqual(captured["path"], expected_path)
                self.assertEqual(captured["kwargs"]["config_path"], expected_config)
                self.assertEqual(
                    comfy_file.call_count,
                    1 if expected_path.startswith("/comfy/") else 0,
                )

    def test_comfy_safetensors_default_is_resolved_with_qwen_config(self):
        captured, comfy_file, resolve_file = self.load_text_encoder(
            te_name_or_path=None,
            text_encoder_path=None,
            model_kwargs={},
        )
        comfy_file.assert_called_once_with(
            qtype="convrot8", hf_token=unittest.mock.ANY, status_fn=unittest.mock.ANY
        )
        resolve_file.assert_not_called()
        self.assertEqual(captured["path"], "/comfy/text_encoder.safetensors")
        self.assertEqual(captured["kwargs"]["config_path"], QWEN35_9B_REPO)
        self.assertEqual(captured["kwargs"]["subfolder"], "")
        self.assertEqual(captured["kwargs"]["qtype"], "convrot8")

    def test_explicit_safetensors_path_is_resolved_and_uses_qwen_config(self):
        captured, comfy_file, resolve_file = self.load_text_encoder(
            text_encoder_path="/models/qwen35.safetensors"
        )
        comfy_file.assert_not_called()
        resolve_file.assert_called_once_with(
            "/models/qwen35.safetensors",
            hf_token=unittest.mock.ANY,
            status_fn=unittest.mock.ANY,
        )
        self.assertEqual(captured["path"], "resolved:/models/qwen35.safetensors")
        self.assertEqual(captured["kwargs"]["config_path"], QWEN35_9B_REPO)

    def test_all_moss_forms_expose_existing_text_encoder_path_field(self):
        ui = (REPO_ROOT / "extensions_built_in/diffusion_models/ui.tsx").read_text()
        variants = ("moss_image", "moss_image:base", "moss_image:edit", "moss_image:base_edit")
        for variant in variants:
            start = ui.index(f'name: "{variant}"')
            next_start = ui.find('\n  {\n    name:', start + 1)
            block = ui[start:next_start if next_start != -1 else len(ui)]
            with self.subTest(variant=variant):
                self.assertIn('"model.text_encoder_path"', block)


if __name__ == "__main__":
    unittest.main()
