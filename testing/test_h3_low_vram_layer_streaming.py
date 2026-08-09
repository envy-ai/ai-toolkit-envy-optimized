import pathlib
import unittest
from unittest.mock import Mock

import torch

from toolkit.comfy_sample import ComfyApiClient
from toolkit.config_modules import ModelConfig
from toolkit.data_loader import get_safe_dataloader_num_workers
from toolkit.unloader import FakeTextEncoder, unload_text_encoder


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class H3LowVramLayerStreamingTests(unittest.TestCase):
    def test_config_defaults_to_current_streaming_behavior_and_can_disable_it(self):
        default = ModelConfig(name_or_path="Comfy-Org/MiniMax-H3", arch="minimax_h3")
        disabled = ModelConfig(
            name_or_path="Comfy-Org/MiniMax-H3",
            arch="minimax_h3",
            low_vram_layer_streaming=False,
        )

        self.assertTrue(default.low_vram_layer_streaming)
        self.assertFalse(disabled.low_vram_layer_streaming)

    def test_h3_loader_only_automatically_streams_when_option_is_enabled(self):
        source = (
            REPO_ROOT
            / "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py"
        ).read_text()

        self.assertIn("self.model_config.low_vram_layer_streaming", source)
        self.assertIn("auto_layer_offload", source)
        self.assertIn("if transformer_offload_percent > 0:", source)
        self.assertIn("transformer.to(self.device_torch)", source)

    def test_ui_exposes_h3_low_vram_streaming_control(self):
        options = (REPO_ROOT / "ui/src/app/jobs/new/options.tsx").read_text()
        form = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        types = (REPO_ROOT / "ui/src/types.ts").read_text()

        self.assertIn("'model.low_vram_layer_streaming'", options)
        self.assertIn("low_vram_layer_streaming?: boolean", types)
        self.assertIn('label="Stream H3 Layers Through CPU RAM"', form)
        self.assertIn("config.process[0].model.low_vram_layer_streaming", form)

    def test_h3_disables_forked_dataloader_workers(self):
        class DatasetConfig:
            num_workers = 2

        class H3Model:
            arch = "minimax_h3"

        class OtherModel:
            arch = "flux"

        self.assertEqual(get_safe_dataloader_num_workers([DatasetConfig()], H3Model()), 0)
        self.assertEqual(get_safe_dataloader_num_workers([DatasetConfig()], OtherModel()), 2)

    def test_h3_unload_releases_encoder_storage_even_if_a_reference_lingers(self):
        class H3Model:
            arch = "minimax_h3"
            device_torch = torch.device("cpu")
            torch_dtype = torch.bfloat16

            def __init__(self):
                self.text_encoder = torch.nn.Linear(8, 8)

        model = H3Model()
        lingering_reference = model.text_encoder

        unload_text_encoder(model)

        self.assertIsInstance(model.text_encoder, FakeTextEncoder)
        self.assertEqual(lingering_reference.weight.device.type, "meta")

    def test_comfy_vram_release_wait_accepts_an_unloaded_gpu(self):
        client = object.__new__(ComfyApiClient)
        client._request_json = Mock(return_value={
            "devices": [{"vram_total": 24 * 1024 ** 3, "vram_free": 23.5 * 1024 ** 3}]
        })

        self.assertTrue(client.wait_for_vram_release(timeout=0.01))


if __name__ == "__main__":
    unittest.main()
