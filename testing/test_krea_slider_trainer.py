import types
import unittest
from pathlib import Path


class KreaSliderTrainerTests(unittest.TestCase):
    def test_legacy_prompt_fields_materialize_single_slider_target(self):
        from extensions_built_in.slider.SliderTrainer import normalize_slider_targets

        slider_config = {
            "target_class": "person",
            "positive_prompt": "person smiling",
            "negative_prompt": "person frowning",
        }

        targets = normalize_slider_targets(slider_config)

        self.assertEqual(
            targets,
            [
                {
                    "target_class": "person",
                    "positive": "person smiling",
                    "negative": "person frowning",
                    "weight": 1.0,
                    "shuffle": False,
                }
            ],
        )

    def test_slider_train_config_forces_text_encoder_unload_without_datasets(self):
        from extensions_built_in.slider.SliderTrainer import (
            prepare_slider_train_config_for_text_encoder_unload,
        )

        process_config = {
            "datasets": [],
            "train": {
                "unload_text_encoder": False,
                "cache_text_embeddings": True,
                "train_text_encoder": False,
            },
        }

        prepare_slider_train_config_for_text_encoder_unload(process_config)

        self.assertTrue(process_config["train"]["unload_text_encoder"])
        self.assertFalse(process_config["train"]["cache_text_embeddings"])

    def test_krea_slider_uses_sample_steps_as_default_denoising_cap(self):
        from extensions_built_in.slider.SliderTrainer import (
            prepare_slider_train_config_for_text_encoder_unload,
        )

        process_config = {
            "model": {"arch": "krea2"},
            "sample": {"sample_steps": 12},
            "train": {
                "unload_text_encoder": False,
                "cache_text_embeddings": True,
                "train_text_encoder": False,
            },
        }

        prepare_slider_train_config_for_text_encoder_unload(process_config)

        self.assertEqual(process_config["train"]["max_denoising_steps"], 12)

    def test_slider_prediction_timestep_expands_to_full_batch(self):
        from extensions_built_in.slider.SliderTrainer import (
            expand_slider_timestep_for_prediction,
        )

        class FakeTimestep:
            shape = (1,)

            def repeat(self, *args):
                return args

        expanded = expand_slider_timestep_for_prediction(FakeTimestep(), batch_size=12)

        self.assertEqual(expanded, (12,))

    def test_slider_prediction_timestep_expands_to_full_torch_batch(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this Python environment")

        from extensions_built_in.slider.SliderTrainer import (
            expand_slider_timestep_for_prediction,
        )

        timestep = torch.tensor([157])

        expanded = expand_slider_timestep_for_prediction(timestep, batch_size=12)

        self.assertEqual(expanded.shape, (12,))
        self.assertTrue(torch.equal(expanded, torch.tensor([157] * 12)))

    def test_slider_config_defaults_to_single_action_batches(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_source = (repo_root / "toolkit/config_modules.py").read_text()

        self.assertIn("kwargs.get('batch_full_slide', False)", config_source)

    def test_slider_latent_noise_uses_krea_latent_shape(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this Python environment")

        from extensions_built_in.slider.SliderTrainer import build_slider_latent_noise

        sd = types.SimpleNamespace(
            vae_scale_factor=8,
            transformer=types.SimpleNamespace(config=types.SimpleNamespace(channels=16)),
        )

        noise = build_slider_latent_noise(
            sd=sd,
            height=512,
            width=768,
            batch_size=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
            noise_offset=0.0,
        )

        self.assertEqual(noise.shape, (3, 16, 64, 96))


if __name__ == "__main__":
    unittest.main()
