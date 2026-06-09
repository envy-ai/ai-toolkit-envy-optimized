import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class FakeEmbeds:
    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self


class FakeModule:
    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


class SDTrainerPromptOffloadTests(unittest.TestCase):
    def test_unconditional_prompt_encoding_offloads_optimizer_state(self):
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
        from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

        optimizer_state_device = {"value": "cuda"}
        events = []

        def fake_move_optimizer_state_to_device(optimizer, device, non_blocking=False):
            optimizer_state_device["value"] = str(device)
            events.append(("move", str(device)))

        class FakeSD:
            encode_control_in_text_embeddings = False
            has_multiple_control_images = False
            device_torch = torch.device("cuda")
            torch_dtype = torch.float32
            unet = FakeModule()
            vae = FakeModule()
            noise_scheduler = object()

            def encode_prompt(self, *args, **kwargs):
                events.append(("encode", optimizer_state_device["value"]))
                return FakeEmbeds()

        trainer = SDTrainer.__new__(SDTrainer)
        trainer.optimizer = object()
        trainer.device_torch = torch.device("cuda")
        trainer.is_caching_text_embeddings = False
        trainer.sd = FakeSD()
        trainer.train_config = SimpleNamespace(
            unconditional_prompt="",
            do_prior_divergence=False,
            negative_prompt=None,
            unload_text_encoder=False,
            blank_prompt_preservation=False,
            diffusion_feature_extractor_path=None,
        )
        trainer.do_long_prompts = False
        trainer.is_latents_cached = True
        trainer.adapter = None
        trainer.cached_blank_embeds = None

        with patch.object(BaseSDTrainProcess, "hook_before_train_loop", lambda self: None), patch(
            "extensions_built_in.sd_trainer.SDTrainer.add_all_snr_to_noise_scheduler",
            lambda *args, **kwargs: None,
        ), patch(
            "extensions_built_in.sd_trainer.SDTrainer.move_optimizer_state_to_device",
            fake_move_optimizer_state_to_device,
            create=True,
        ):
            trainer.hook_before_train_loop()

        self.assertEqual(
            events[:3],
            [
                ("move", "cpu"),
                ("encode", "cpu"),
                ("move", "cuda"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
