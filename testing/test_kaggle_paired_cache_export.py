"""Focused CPU contracts for the scoped Kaggle paired-cache exporter."""

import builtins
import hashlib
import importlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from workers.kaggle_tpu import export_paired_cache as exporter


def valid_profile():
    return {
        "type": "fizgig_image_slider",
        "model": {"arch": "qwen_image_2"},
        "network": {"type": "dora"},
        "train": {"cache_text_embeddings": True},
        "datasets": [{"cache_latents_to_disk": True}],
    }


class FakeItem:
    def __init__(self, positive, negative, latent, conditioning):
        self.path = str(positive)
        self.negative_path = str(negative)
        self.latent_path = str(latent)
        self.conditioning_path = str(conditioning)
        self.caption = "a test caption"
        self.loss_multiplier = 1.0
        self.latent_space_version = "qwen21-test-v1"
        self.scale_to_width = self.scale_to_height = 32
        self.crop_x = self.crop_y = 0
        self.crop_width = self.crop_height = 32
        self.flip_x = self.flip_y = False

    def get_latent_path(self):
        return self.latent_path

    def get_text_embedding_path(self):
        return self.conditioning_path

    def get_latent_info_dict(self):
        return {"latent_space_version": self.latent_space_version}


def write_cache_files(root, item, positive_value=1.0, negative_value=2.0):
    import torch
    from safetensors.torch import save_file

    positive_latent = torch.full((64, 2, 2), positive_value, dtype=torch.float32)
    negative_latent = torch.full((64, 2, 2), negative_value, dtype=torch.float32)
    save_file({"latent": positive_latent}, item.latent_path,
              metadata=item.get_latent_info_dict())

    from toolkit.safetensors_cache import atomic_save_file
    atomic_save_file(
        {"latent": negative_latent},
        item.negative_path,
        metadata={"paired_negative_identity": exporter.canonical(item.test_negative_identity)},
    )
    save_file(
        {
            "text_embeds": torch.ones((2, 4096), dtype=torch.float32),
            "attention_mask": torch.tensor([1, 1], dtype=torch.int64),
            "image_slot_mask": torch.tensor([False, False]),
        },
        item.conditioning_path,
        metadata={"class_name": "AdvancedPromptEmbeds"},
    )


class PairedCacheExporterTests(unittest.TestCase):
    def test_negative_preparation_uses_paired_pixels_and_records_identity(self):
        import torch
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positive, negative = root / "after.png", root / "before.png"
            positive.write_bytes(b"after")
            negative.write_bytes(b"before")
            item = FakeItem(positive, negative, root / "positive.safetensors", root / "text.safetensors")
            Path(item.latent_path).write_bytes(b"existing positive is reused")
            pixels = torch.full((3, 32, 32), 0.25)
            item.load_unconditional_image = lambda: setattr(item, "unconditional_tensor", pixels)
            item.cleanup = lambda: setattr(item, "unconditional_tensor", None)
            captured = []

            class FakeVAE(torch.nn.Module):
                @classmethod
                def load(cls, *args, **kwargs):
                    return cls()

            sd = types.SimpleNamespace(
                model_config=types.SimpleNamespace(extras_name_or_path="test-base", name_or_path="test-base"),
                vae_device_torch=torch.device("cpu"), vae_torch_dtype=torch.float32,
                component_load_kwargs=lambda role: {},
                encode_images=lambda value: (captured.append(value.clone()) or torch.ones(1, 64, 2, 2)),
            )
            entry = {"dataset_index": 0, "positive_path": str(positive),
                     "negative_path": str(negative), "item": item}
            modules = types.SimpleNamespace(BASE_REPO="test-repo", AutoencoderKLQwenImage21=FakeVAE)
            with mock.patch.object(exporter, "make_model", return_value=sd), mock.patch.object(
                exporter, "qwen_module", return_value=modules
            ):
                exporter.prepare_latents(valid_profile(), [entry], root / "negative-cache", "cpu", 42)
            torch.testing.assert_close(captured[0], pixels.unsqueeze(0))
            cached = exporter.negative_path(entry, root / "negative-cache")
            torch.testing.assert_close(load_file(cached)["latent"], torch.ones(64, 2, 2))
            exporter.validate_cache_metadata(cached, {
                "paired_negative_identity": exporter.canonical(exporter.negative_identity(entry))})
            with self.assertRaisesRegex(ValueError, "incompatible cache metadata"):
                exporter.validate_cache_metadata(cached, {"paired_negative_identity": "wrong"})

    def test_import_stays_stdlib_only(self):
        real_import = builtins.__import__
        forbidden = ("torch", "torchvision", "safetensors", "toolkit", "extensions_built_in")

        def reject_heavy_import(name, *args, **kwargs):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
                raise AssertionError(f"heavy import during exporter import: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_heavy_import):
            importlib.reload(exporter)

    def test_profile_accepts_minimal_qwen_image_slider_and_rejects_unsupported_modes(self):
        exporter.validate_profile(valid_profile())

        invalid_profiles = (
            ({**valid_profile(), "type": "fizgig_prompt_slider"}, "fizgig_image_slider only"),
            ({**valid_profile(), "model": {"arch": "sdxl"}}, "qwen_image_2 only"),
            ({**valid_profile(), "network": {"type": "loha"}}, "LoRA or signed DoRA"),
            ({**valid_profile(), "train": {"cache_text_embeddings": False}}, "cached text conditioning"),
            ({**valid_profile(), "fizgig_slider": {"multipoint": True}}, "Multipoint sliders"),
            ({**valid_profile(), "trigger_word": "tag"}, "Trigger-word processing"),
            ({**valid_profile(), "datasets": [{"cache_latents_to_disk": False}]}, "disk latent caches"),
            ({**valid_profile(), "datasets": [{"cache_latents_to_disk": True, "random_crop": True}]}, "random_crop"),
        )
        for process, message in invalid_profiles:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                exporter.validate_profile(process)

    def test_build_items_preserves_real_pairs_and_repeat_flip_exposures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positives, negatives = root / "positive", root / "negative"
            positives.mkdir()
            negatives.mkdir()
            Image.new("RGB", (32, 32), (100, 20, 30)).save(positives / "sample.png")
            Image.new("RGB", (32, 32), (20, 100, 30)).save(negatives / "sample.png")
            # Hidden thumbnail folders must not create training exposures.
            (positives / ".thumbs").mkdir()
            Image.new("RGB", (32, 32)).save(positives / ".thumbs" / "hidden.png")
            profile = valid_profile()
            profile["datasets"] = [{
                "folder_path": str(positives),
                "control_path_1": str(negatives),
                "cache_latents_to_disk": True,
                "resolution": 32,
                "num_repeats": 2,
                "flip_x": True,
                "default_caption": "caption",
            }]

            class FakeModel:
                load_rgba = False
                use_raw_control_images = False
                model_config = types.SimpleNamespace(extras_name_or_path="test-vae", vae_dtype="bf16")

                def get_bucket_divisibility(self):
                    return 16

                def get_latent_space_version(self):
                    return "qwen21-test-v1"

                def get_text_embedding_space_version(self):
                    return "qwen21-text-test-v1"

            with mock.patch.object(exporter, "make_model", return_value=FakeModel()):
                entries = exporter.build_items(profile)

            self.assertEqual(len(entries), 4)
            self.assertEqual({entry["positive_path"] for entry in entries}, {str(positives / "sample.png")})
            self.assertEqual({entry["negative_path"] for entry in entries}, {str(negatives / "sample.png")})
            self.assertEqual([entry["item"].flip_x for entry in entries], [False, False, True, True])
            self.assertEqual([entry["dataset_index"] for entry in entries], [0, 0, 0, 0])

    def test_export_requires_caches_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positives, negatives = root / "positive", root / "negative"
            positives.mkdir()
            negatives.mkdir()
            (positives / "p.png").write_bytes(b"positive source")
            (negatives / "p.png").write_bytes(b"negative source")
            item = FakeItem(positives / "p.png", negatives / "p.png", root / "missing-latent.safetensors",
                            root / "missing-conditioning.safetensors")
            profile = valid_profile()
            profile["datasets"] = [{"folder_path": str(positives), "control_path_1": str(negatives),
                                    "cache_latents_to_disk": True}]
            with mock.patch.object(exporter, "build_items", return_value=[{
                "dataset_index": 0, "positive_path": item.path,
                "negative_path": item.negative_path, "item": item,
            }]):
                with self.assertRaisesRegex(FileNotFoundError, "Missing exact cached text conditioning"):
                    exporter.export(profile, root / "out", negative_cache_dir=root / "negative-cache")

    def test_export_copies_content_addressed_tensors_and_hashes_manifest(self):
        import torch
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positives, negatives, negative_cache = root / "positive", root / "negative", root / "negative-cache"
            positives.mkdir()
            negatives.mkdir()
            positive_source, negative_source = positives / "p.png", negatives / "p.png"
            positive_source.write_bytes(b"positive source")
            negative_source.write_bytes(b"negative source")
            item = FakeItem(positive_source, negative_source, root / "positive-latent.safetensors",
                            root / "conditioning.safetensors")
            item.negative_path = str(negative_source)
            item.latent_path = str(root / "positive-latent.safetensors")
            item.conditioning_path = str(root / "conditioning.safetensors")
            # The negative cache's location is content addressed from the source and presentation identity.
            item.negative_path = str(negative_source)
            profile = valid_profile()
            profile["datasets"] = [{"folder_path": str(positives), "control_path_1": str(negatives),
                                    "cache_latents_to_disk": True}]
            entry = {"dataset_index": 0, "positive_path": str(positive_source),
                     "negative_path": str(negative_source), "item": item}
            negative_cache.mkdir()
            item.negative_path = str(exporter.negative_path(entry, negative_cache))
            item.test_negative_identity = exporter.negative_identity(entry)
            write_cache_files(root, item)
            output = root / "export"

            with (
                mock.patch.object(exporter, "build_items", return_value=[entry]),
                mock.patch.object(exporter.subprocess, "check_output", return_value="test-commit\n"),
            ):
                manifest = exporter.export(profile, output, negative_cache_dir=negative_cache)

            self.assertEqual(manifest["kind"], "qwen21_fizgig_paired_cache")
            self.assertEqual(len(manifest["pairs"]), 1)
            self.assertEqual(len(manifest["exposures"]), 1)
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
            self.assertEqual(len(manifest["files"]), 2)
            for relative, details in manifest["files"].items():
                copied = output / relative
                self.assertTrue(copied.is_file())
                self.assertEqual(copied.stat().st_size, details["bytes"])
                actual_hash = hashlib.sha256(copied.read_bytes()).hexdigest()
                self.assertEqual(actual_hash, details["sha256"])
                self.assertEqual(Path(relative).name, f"{actual_hash}.safetensors")

            pair_ref = manifest["pairs"][0]["latents"]
            pair_file = load_file(output / pair_ref["path"])
            torch.testing.assert_close(pair_file["positive"], torch.ones((64, 2, 2)))
            torch.testing.assert_close(pair_file["negative"], torch.full((64, 2, 2), 2.0))
            conditioning_ref = manifest["pairs"][0]["conditioning"]
            self.assertEqual(conditioning_ref["sha256"], manifest["files"][conditioning_ref["path"]]["sha256"])
            self.assertEqual(manifest["pairs"][0]["positive_source"]["sha256"],
                             hashlib.sha256(positive_source.read_bytes()).hexdigest())
            self.assertEqual(manifest["pairs"][0]["negative_source"]["sha256"],
                             hashlib.sha256(negative_source.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
