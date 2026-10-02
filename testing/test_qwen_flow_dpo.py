from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.QwenFlowDPOTrainer import (
    QwenFlowDPOTrainer, flow_dpo_terms,
)
from toolkit.dataloader_mixins import LatentCachingFileItemDTOMixin


class QwenFlowDPOTests(unittest.TestCase):
    def test_loss_sign_and_split_backward_coefficients(self):
        win = torch.tensor([1.3, 0.8], requires_grad=True)
        lose = torch.tensor([1.1, 1.4], requires_grad=True)
        ref_win = torch.tensor([1.0, 1.0])
        ref_lose = torch.tensor([1.0, 1.0])
        loss, margin, coefficient = flow_dpo_terms(win, lose, ref_win, ref_lose, 2.0, 0.2)
        direct_win, direct_lose = torch.autograd.grad(loss.mean(), (win, lose))
        torch.testing.assert_close(direct_win, (coefficient.detach() + 0.2) / 2)
        torch.testing.assert_close(direct_lose, -coefficient.detach() / 2)
        self.assertGreater(margin[1].item(), margin[0].item())
        self.assertLess(loss[1].item(), loss[0].item())
        neutral, _, _ = flow_dpo_terms(ref_win, ref_lose, ref_win, ref_lose, 2.0, 0)
        torch.testing.assert_close(neutral, torch.full_like(neutral, torch.log(torch.tensor(2.0))))

    def test_slider_style_pairing_routes_only_dataset_one_as_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            win, lose, source = (root / name for name in ("win", "lose", "source"))
            for folder in (win, lose, source):
                folder.mkdir()
                Image.new("RGB", (64, 64)).save(folder / "a.png")
            dataset = {
                "folder_path": str(win),
                "control_path_1": str(lose),
                "control_path_2": str(source),
                "cache_latents_to_disk": True,
            }
            config = {
                "type": "qwen_flow_dpo", "model": {"arch": "qwen_image_2"},
                "network": {"type": "lora"},
                "train": {"noise_scheduler": "flowmatch", "cache_text_embeddings": True},
                "datasets": [dataset],
            }
            with patch.object(DiffusionTrainer, "__init__", return_value=None):
                trainer = QwenFlowDPOTrainer(0, None, config)
            self.assertEqual(trainer.dpo_beta, 1.0)
            self.assertEqual(dataset["unconditional_path"], str(lose))
            self.assertEqual(dataset["control_path_2"], str(source))
            self.assertNotIn("control_path_1", dataset)
            self.assertTrue(dataset["flow_dpo_pair"])
            self.assertEqual(dataset["caption_dropout_rate"], 0)

    def test_invalid_mode_and_pair_rejected_before_training(self):
        config = {
            "model": {"arch": "qwen_image_2"},
            "network": {"type": "dora"},
            "train": {"noise_scheduler": "flowmatch", "cache_text_embeddings": True},
        }
        with self.assertRaisesRegex(ValueError, "LoRA only"):
            QwenFlowDPOTrainer(0, None, config)
        config["network"]["type"] = "lora"
        with self.assertRaisesRegex(ValueError, "Control Dataset 1|positive dataset"):
            QwenFlowDPOTrainer(0, None, config)

    def test_edit_source_must_match_each_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            win, lose, source = (root / name for name in ("win", "lose", "source"))
            for folder in (win, lose, source):
                folder.mkdir()
            Image.new("RGB", (64, 64)).save(win / "a.png")
            Image.new("RGB", (64, 64)).save(lose / "a.png")
            config = {
                "model": {"arch": "qwen_image_2"}, "network": {"type": "lora"},
                "train": {"noise_scheduler": "flowmatch", "cache_text_embeddings": True},
                "datasets": [{"folder_path": str(win), "control_path_1": str(lose),
                              "control_path_2": str(source), "cache_latents_to_disk": True}],
            }
            with self.assertRaisesRegex(ValueError, "one edit source"):
                QwenFlowDPOTrainer(0, None, config)

    def test_preferred_latent_cache_tracks_file_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.png"
            path.write_bytes(b"original")
            item = SimpleNamespace(
                path=str(path), scale_to_width=64, scale_to_height=64,
                crop_x=0, crop_y=0, crop_width=64, crop_height=64,
                latent_space_version="qwen_image_2", latent_version=1,
                dataset_config=SimpleNamespace(flow_dpo_pair=True, buckets=True,
                                               cache_tensors_to_disk=False),
                flip_x=False, flip_y=False, is_video=False, is_audio_model=False,
            )
            first = LatentCachingFileItemDTOMixin.get_latent_info_dict(item)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            second = LatentCachingFileItemDTOMixin.get_latent_info_dict(item)
            self.assertNotEqual(first["source_file_identity_v1"], second["source_file_identity_v1"])

    def test_rejected_latents_cached_before_workers_load_pairs(self):
        class Item:
            path = "preferred/a.png"
            unconditional_path = "rejected/a.png"
            scale_to_width = scale_to_height = crop_width = crop_height = 32
            crop_x = crop_y = 0
            flip_x = flip_y = False
            has_unconditional = True

            def load_unconditional_image(self):
                self.unconditional_tensor = torch.zeros(3, 32, 32)

            def cleanup_unconditional(self):
                self.unconditional_tensor = None

        item = Item()
        device_states = []
        sd = SimpleNamespace(
            vae=object(), vae_device_torch=torch.device("cpu"), vae_torch_dtype=torch.float32,
            set_device_state_preset=lambda name: device_states.append(name),
            restore_device_state=lambda: device_states.append("restored"),
            encode_images=lambda pixels: torch.ones(1, 2, 2, 2),
        )
        trainer = object.__new__(QwenFlowDPOTrainer)
        trainer.sd = sd
        trainer.rejected_latents = {}
        trainer.data_loader = object()
        with patch.object(DiffusionTrainer, "hook_before_train_loop", return_value=None), patch(
            "extensions_built_in.sd_trainer.QwenFlowDPOTrainer.get_dataloader_datasets",
            return_value=[SimpleNamespace(file_list=[item])],
        ), patch.object(QwenFlowDPOTrainer, "print"):
            trainer.hook_before_train_loop()
        self.assertEqual(device_states, ["cache_latents", "restored"])
        self.assertIn(QwenFlowDPOTrainer._pair_key(item), trainer.rejected_latents)
        self.assertFalse(item.has_unconditional)
        self.assertIsNone(item.unconditional_tensor)

    def test_reference_disabled_and_condition_batch_shared(self):
        parameter = torch.nn.Parameter(torch.tensor(0.25))
        network = SimpleNamespace(is_active=False)
        seen = []
        backprops = []

        class Embeds:
            def to(self, *args, **kwargs):
                return self

            def detach(self):
                return self

        class Accelerator:
            def backward(self, loss):
                backprops.append(loss.item())
                loss.backward()

        item = SimpleNamespace(
            path="preferred/a.png", unconditional_path="rejected/a.png",
            scale_to_width=32, scale_to_height=32,
            crop_x=0, crop_y=0, crop_width=32, crop_height=32,
            flip_x=False, flip_y=False,
        )
        fake = SimpleNamespace(
            network=network, device_torch=torch.device("cpu"),
            train_config=SimpleNamespace(dtype="fp32", min_denoising_steps=0,
                                         max_denoising_steps=999),
            rejected_latents={QwenFlowDPOTrainer._pair_key(item): torch.ones(1, 2, 2)},
            dpo_beta=1.0, dpo_sft_weight=0.1, accelerator=Accelerator(),
            additional_logs={}, _pair_key=QwenFlowDPOTrainer._pair_key,
        )
        batch = SimpleNamespace(latents=torch.zeros(1, 1, 2, 2),
                                prompt_embeds=Embeds(), file_items=[item])

        def predict(noisy, timestep, embeds, actual_batch):
            seen.append((network.is_active, actual_batch is batch, torch.is_grad_enabled()))
            return noisy.float() * 0.1 + parameter * float(network.is_active)

        fake._predict = predict
        torch.manual_seed(42)
        result = QwenFlowDPOTrainer.train_single_accumulation(fake, batch)
        self.assertEqual([entry[0] for entry in seen], [False, False, True, True, True, True])
        self.assertTrue(all(entry[1] for entry in seen))
        self.assertEqual(len(backprops), 2)
        self.assertFalse(network.is_active)
        self.assertTrue(torch.isfinite(result))
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad))
        self.assertIn("dpo/margin", fake.additional_logs)


if __name__ == "__main__":
    unittest.main()
