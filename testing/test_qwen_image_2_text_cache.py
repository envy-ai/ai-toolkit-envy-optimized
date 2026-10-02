import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from toolkit.dataloader_mixins import TextEmbeddingCachingMixin
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model


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
    crop_width = 512
    crop_height = 768

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

    def encode_prompt(self, caption, control_images=None, target_size=None):
        self.encode_calls.append((caption, control_images, target_size))
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
        for _, control_images, target_size in dataset.sd.encode_calls:
            self.assertIsInstance(control_images, list)
            self.assertEqual(len(control_images), 1)
            self.assertEqual(control_images[0].shape, (1, 3, 768, 512))
            self.assertEqual(target_size, (512, 768))
        self.assertTrue(file_item.cleaned_control)
        self.assertTrue(file_item.is_text_embedding_cached)

    def test_qwen_reference_modes_keep_bucketed_low_memory_default(self):
        def model(**model_kwargs):
            return QwenImage2Model(
                'cpu', ModelConfig(
                    arch='qwen_image_2', name_or_path='Comfy-Org/Qwen-Image-2.1',
                    model_kwargs=model_kwargs,
                ), dtype='float32',
            )

        default = model()
        self.assertFalse(default.match_target_res)
        self.assertFalse(default.use_raw_control_images)
        self.assertTrue(default.cache_processed_control_text_embeddings)
        self.assertTrue(default.caption_dropout_keeps_control_images)
        self.assertFalse(default.text_embedding_uses_target_size)

        matched = model(match_target_res=True)
        self.assertTrue(matched.use_raw_control_images)
        self.assertFalse(matched.cache_processed_control_text_embeddings)
        self.assertTrue(matched.text_embedding_uses_target_size)
        self.assertNotEqual(
            default.get_text_embedding_space_version(),
            matched.get_text_embedding_space_version(),
        )


if __name__ == "__main__":
    unittest.main()
