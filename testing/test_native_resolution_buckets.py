import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor

from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
from toolkit.data_loader import AiToolkitDataset


def fake_sd(divisibility=16):
    return SimpleNamespace(
        arch="test",
        vae=None,
        unet=None,
        load_rgba=False,
        use_raw_control_images=False,
        adapter=None,
        is_audio_model=False,
        is_multimodal_llm=False,
        get_bucket_divisibility=lambda: divisibility,
        get_latent_space_version=lambda: "native-resolution-test",
        get_text_embedding_space_version=lambda: "native-resolution-test",
        encode_control_in_text_embeddings=False,
        te_padding_side="right",
    )


def patterned_image(width, height, offset=0):
    pixels = torch.arange(width * height * 3, dtype=torch.int64).reshape(height, width, 3)
    pixels = (pixels + offset).remainder(256).to(torch.uint8).numpy()
    return Image.fromarray(pixels, mode="RGB")


class NativeResolutionBucketTests(unittest.TestCase):
    def make_dataset(self, root, *, native=True, source_configs=None, **options):
        options.setdefault("resolution", 128)
        config = DatasetConfig(
            dataset_path=str(root),
            native_resolution=native,
            **options,
        )
        with patch.object(AiToolkitDataset, "cache_latents_all_latents"):
            return AiToolkitDataset(
                config,
                batch_size=1,
                sd=fake_sd(),
                source_configs=source_configs,
            )

    def test_mixed_native_sizes_preserve_exact_pixels_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "wide.png": patterned_image(96, 64, 11),
                "tall.png": patterned_image(64, 128, 73),
            }
            for name, image in sources.items():
                image.save(root / name)

            dataset = self.make_dataset(root)
            self.assertEqual(set(dataset.buckets), {"96x64", "64x128"})

            for item in dataset.file_list:
                item.load_and_process_image(dataset.transform)
                source = ToTensor()(sources[Path(item.path).name]) * 2 - 1
                self.assertEqual(tuple(item.tensor.shape[-2:]), (sources[Path(item.path).name].height,
                                                                  sources[Path(item.path).name].width))
                self.assertTrue(torch.equal(item.tensor, source), Path(item.path).name)
                self.assertEqual((item.scale_to_width, item.scale_to_height),
                                 (sources[Path(item.path).name].width, sources[Path(item.path).name].height))
                self.assertEqual((item.crop_x, item.crop_y), (0, 0))

    def test_pooled_native_sources_bucket_by_exact_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a, source_b = root / "a", root / "b"
            source_a.mkdir()
            source_b.mkdir()
            patterned_image(64, 96).save(source_a / "one.png")
            patterned_image(64, 96, 31).save(source_b / "two.png")
            patterned_image(80, 96).save(source_b / "three.png")

            config_a = DatasetConfig(dataset_path=str(source_a), native_resolution=True, resolution=128)
            config_b = DatasetConfig(dataset_path=str(source_b), native_resolution=True, resolution=128)
            with patch.object(AiToolkitDataset, "cache_latents_all_latents"):
                dataset = AiToolkitDataset(config_a, batch_size=1, sd=fake_sd(),
                                           source_configs=[config_a, config_b])

            self.assertEqual(set(dataset.buckets), {"64x96#s0", "80x96#s0"})
            self.assertEqual(len(dataset.file_list), 3)
            for item in dataset.file_list:
                item.load_and_process_image(dataset.transform)
                self.assertEqual(tuple(item.tensor.shape[-2:]), (item.height, item.width))

    def test_incompatible_configs_are_rejected(self):
        incompatible = [
            {"buckets": False},
            {"scale": 0.5},
            {"random_scale": True},
            {"random_crop": True},
            {"square_crop": True},
            {"flip_x": True},
            {"flip_y": True},
            {"augments": ["ColorJitter"]},
            {"augmentations": [{"method": "ColorJitter"}]},
            {"standardize_images": True},
            {"type": "video"},
            {"num_frames": 2},
            {"auto_frame_count": True},
            {"fast_image_size": True},
            {"fizgig_slider_pair": True},
            {"flow_dpo_pair": True},
            {"fizgig_multipoint_group": "group"},
        ]
        for options in incompatible:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "native_resolution"):
                DatasetConfig(native_resolution=True, **options)

    def test_non_divisible_source_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patterned_image(70, 64).save(root / "not-aligned.png")
            config = DatasetConfig(dataset_path=str(root), native_resolution=True)
            with patch.object(AiToolkitDataset, "cache_latents_all_latents"):
                with self.assertRaisesRegex(ValueError, "native_resolution requires both target dimensions"):
                    AiToolkitDataset(config, batch_size=1, sd=fake_sd(divisibility=16))

    def test_source_size_change_after_bucket_setup_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "changed.png"
            patterned_image(64, 64).save(path)
            dataset = self.make_dataset(root)
            patterned_image(80, 64).save(path)

            with self.assertRaisesRegex(ValueError, "native_resolution target dimensions changed"):
                dataset.file_list[0].load_and_process_image(dataset.transform)

    def test_exif_oriented_dimensions_define_the_native_bucket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = patterned_image(64, 32)
            exif = source.getexif()
            exif[274] = 6
            path = root / "rotated.jpg"
            source.save(path, quality=100, subsampling=0, exif=exif)
            expected = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

            dataset = self.make_dataset(root)
            item = dataset.file_list[0]
            self.assertEqual((item.width, item.height), (32, 64))
            self.assertEqual(set(dataset.buckets), {"32x64"})
            item.load_and_process_image(dataset.transform)
            self.assertTrue(torch.equal(item.tensor, ToTensor()(expected) * 2 - 1))

    def test_native_resolution_list_does_not_duplicate_dataset(self):
        raw = [{"dataset_path": "/images", "native_resolution": True,
                "resolution": [256, 512, 768]}]
        configs = preprocess_dataset_raw_config(raw)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["dataset_path"], "/images")
        self.assertEqual(configs[0]["resolution"], 512)

    def test_native_latent_cache_identity_is_separate_from_resized_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patterned_image(64, 64).save(root / "same.png")
            native = self.make_dataset(root, native=True)
            resized = self.make_dataset(root, native=False, resolution=64)
            native_item, resized_item = native.file_list[0], resized.file_list[0]

            native_info = native_item.get_latent_info_dict()
            resized_info = resized_item.get_latent_info_dict()
            self.assertTrue(native_info["native_resolution"])
            self.assertIn("source_file_identity_v1", native_info)
            self.assertNotIn("native_resolution", resized_info)
            self.assertNotEqual(native_item.get_latent_path(), resized_item.get_latent_path())


if __name__ == "__main__":
    unittest.main()
