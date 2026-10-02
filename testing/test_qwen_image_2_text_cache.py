import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from toolkit.dataloader_mixins import TextEmbeddingCachingMixin


class FakePromptEmbeds:
    def save(self, path):
        with open(path, "w") as handle:
            handle.write("cached")


class FakeProcessedControlFileItem:
    encode_control_in_text_embeddings = True
    caption_dropout_keeps_control_images = True
    cache_processed_control_text_embeddings = True
    caption = "turn this into a painting"
    caption_dop = caption
    control_path = "/dataset/before/example.png"
    control_video_paths = None
    dopsd_self_ref = False
    is_video = False

    def __init__(self, cache_dir):
        self.normal_path = os.path.join(cache_dir, "normal.safetensors")
        self.blank_path = os.path.join(cache_dir, "blank.safetensors")
        self.control_tensor = None
        self.control_tensor_list = None
        self.is_text_embedding_cached = False
        self.latent_load_device = None
        self.cleaned_control = False

    def get_text_embedding_path(self, recalculate=False):
        return self.normal_path

    def get_blank_text_embedding_path(self, recalculate=False):
        return self.blank_path

    def get_dropout_caption(self):
        return ""

    def load_control_image(self):
        # The exact bucket-resized/cropped tensor used by the training batch.
        self.control_tensor = torch.zeros(3, 768, 512)

    def cleanup_control(self):
        self.cleaned_control = True
        self.control_tensor = None
        self.control_tensor_list = None


class FakeProcessedControlSD:
    device = "cpu"
    device_torch = torch.device("cpu")
    torch_dtype = torch.float32
    has_multiple_control_images = True
    encode_first_frame_in_text_embeddings = False

    def __init__(self):
        self.encode_calls = []
        self.device_state_preset = None

    def set_device_state_preset(self, preset):
        self.device_state_preset = preset

    def encode_prompt(self, caption, control_images=None):
        self.encode_calls.append((caption, control_images))
        return FakePromptEmbeds()


class QwenImage2TextCacheTests(unittest.TestCase):
    def test_joint_reference_cache_uses_processed_control_for_caption_and_dropout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = TextEmbeddingCachingMixin.__new__(TextEmbeddingCachingMixin)
            dataset.dataset_path = tmp_dir
            dataset.sd = FakeProcessedControlSD()
            dataset.dataset_config = SimpleNamespace(
                caption_dropout_rate=0.5,
                diff_output_preservation=False,
                do_i2v=False,
            )
            dataset.transform = None
            file_item = FakeProcessedControlFileItem(tmp_dir)
            dataset.file_list = [file_item]

            TextEmbeddingCachingMixin.cache_text_embeddings(dataset)

        self.assertEqual(
            [call[0] for call in dataset.sd.encode_calls],
            ["turn this into a painting", ""],
        )
        for _, control_images in dataset.sd.encode_calls:
            self.assertIsInstance(control_images, list)
            self.assertEqual(len(control_images), 1)
            self.assertEqual(control_images[0].shape, (1, 3, 768, 512))
        self.assertTrue(file_item.cleaned_control)
        self.assertTrue(file_item.is_text_embedding_cached)


if __name__ == "__main__":
    unittest.main()
