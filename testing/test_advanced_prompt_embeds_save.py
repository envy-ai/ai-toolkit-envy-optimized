"""Advanced prompt cache writes tolerate shared CPU tensor storage."""

import tempfile
import unittest
from pathlib import Path

import torch

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


class AdvancedPromptEmbedsSaveTests(unittest.TestCase):
    def round_trip(self, embeds, directory, filename='embeds.safetensors'):
        path = Path(directory) / filename
        embeds.save(str(path))
        return AdvancedPromptEmbeds.load(str(path))

    def test_exact_alias_preserves_values_dtype_and_frozen_metadata(self):
        shared = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        embeds = AdvancedPromptEmbeds(
            text_embeds=[shared],
            base_text_embeds=[shared],
            token_mask=[mask],
        )
        embeds.frozen_dtype_keys = ['base_text_embeds', 'token_mask']
        source = {
            key: (value[0].clone(), value[0].data_ptr(), value[0].stride(), value[0]._version)
            for key, value in embeds._store.items()
        }

        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            loaded = self.round_trip(embeds, directory)

        self.assertCountEqual(loaded.keys(), embeds.keys())
        self.assertEqual(loaded.frozen_dtype_keys, ['base_text_embeds', 'token_mask'])
        self.assertEqual(loaded.text_embeds[0].dtype, torch.bfloat16)
        self.assertEqual(loaded.base_text_embeds[0].dtype, torch.bfloat16)
        self.assertEqual(loaded.token_mask[0].dtype, torch.bool)
        torch.testing.assert_close(loaded.text_embeds[0], shared)
        torch.testing.assert_close(loaded.base_text_embeds[0], shared)
        torch.testing.assert_close(loaded.token_mask[0], mask)
        self.assertNotEqual(
            loaded.text_embeds[0].data_ptr(), loaded.base_text_embeds[0].data_ptr()
        )

        converted = loaded.to(dtype=torch.float32)
        self.assertEqual(converted.text_embeds[0].dtype, torch.float32)
        self.assertEqual(converted.base_text_embeds[0].dtype, torch.bfloat16)
        self.assertEqual(converted.token_mask[0].dtype, torch.bool)

        for key, (value, pointer, stride, version) in source.items():
            actual = embeds[key][0]
            torch.testing.assert_close(actual, value)
            self.assertEqual(actual.data_ptr(), pointer)
            self.assertEqual(actual.stride(), stride)
            self.assertEqual(actual._version, version)

    def test_detached_and_overlapping_noncontiguous_views_round_trip_independently(self):
        backing = torch.arange(48, dtype=torch.float32).reshape(6, 8)
        view = backing[:, ::2]
        detached_alias = view.detach()
        overlapping_view = backing[:, 2:6:2]
        self.assertFalse(view.is_contiguous())
        self.assertEqual(view.untyped_storage().data_ptr(), detached_alias.untyped_storage().data_ptr())
        self.assertEqual(view.untyped_storage().data_ptr(), overlapping_view.untyped_storage().data_ptr())

        embeds = AdvancedPromptEmbeds(
            text_embeds=[view],
            base_text_embeds=[detached_alias],
            glyph_text_embeds=[overlapping_view],
        )
        embeds.frozen_dtype_keys = ['glyph_text_embeds']
        expected = {
            key: embeds[key][0].clone()
            for key in embeds.keys()
        }
        original = {
            key: (embeds[key][0].data_ptr(), embeds[key][0].stride(), embeds[key][0]._version)
            for key in embeds.keys()
        }

        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            loaded = self.round_trip(embeds, directory)

        self.assertEqual(loaded.frozen_dtype_keys, ['glyph_text_embeds'])
        for key, value in expected.items():
            torch.testing.assert_close(loaded[key][0], value)
            self.assertTrue(loaded[key][0].is_contiguous())
            pointer, stride, version = original[key]
            self.assertEqual(embeds[key][0].data_ptr(), pointer)
            self.assertEqual(embeds[key][0].stride(), stride)
            self.assertEqual(embeds[key][0]._version, version)


if __name__ == '__main__':
    unittest.main()
