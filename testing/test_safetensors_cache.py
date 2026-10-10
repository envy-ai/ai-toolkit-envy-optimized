"""CPU regressions for interrupted caches and dataset startup recovery."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file

from toolkit.dataloader_mixins import LatentCachingMixin, LatentCachingFileItemDTOMixin, TextEmbeddingCachingMixin
from toolkit.prompt_utils import PromptEmbeds
from toolkit.safetensors_cache import CachedTensorError, atomic_save_file, is_valid_cache


class SafetensorsCacheTests(unittest.TestCase):
    def test_headers_are_validated_without_loading_tensor_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.safetensors'
            self.assertFalse(is_valid_cache(str(path)))
            path.write_bytes(bytes(129504))
            with self.assertLogs('toolkit.safetensors_cache', level='WARNING') as logs:
                self.assertFalse(is_valid_cache(str(path), ('latent',)))
            self.assertIn(str(path), logs.output[0])
            self.assertTrue(path.exists())  # Validation does not remove active caches.
            atomic_save_file({'latent': torch.ones(2)}, path)
            with patch('toolkit.safetensors_cache.load_file', side_effect=AssertionError('payload read')):
                self.assertTrue(is_valid_cache(str(path), ('latent',)))
                with self.assertLogs('toolkit.safetensors_cache', level='WARNING'):
                    self.assertFalse(is_valid_cache(str(path), ('missing_tensor',)))

    def test_interrupted_write_preserves_previous_cache_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.safetensors'
            atomic_save_file({'latent': torch.ones(2)}, path)
            previous = path.read_bytes()
            def interrupted(tensors, temporary, metadata=None):
                Path(temporary).write_bytes(bytes(16))
                raise OSError('interrupted write')
            with patch('toolkit.safetensors_cache.save_file', interrupted), self.assertRaises(OSError):
                atomic_save_file({'latent': torch.zeros(2)}, path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_completed_write_is_published_only_after_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.safetensors'
            atomic_save_file({'latent': torch.ones(2)}, path)
            def serialize(tensors, temporary, metadata=None):
                self.assertEqual(Path(temporary).parent, path.parent)
                save_file(tensors, temporary, metadata=metadata)
                torch.testing.assert_close(load_file(str(path))['latent'], torch.ones(2))
            with patch('toolkit.safetensors_cache.save_file', serialize):
                atomic_save_file({'latent': torch.zeros(2)}, path, metadata={'source': 'test'})
            torch.testing.assert_close(load_file(str(path))['latent'], torch.zeros(2))
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_latent_startup_recaches_corrupt_files_without_dropping_images(self):
        for to_memory in (False, True):
            with self.subTest(to_memory=to_memory), tempfile.TemporaryDirectory() as directory:
                valid = Path(directory) / 'valid.safetensors'
                broken = Path(directory) / 'broken.safetensors'
                atomic_save_file({'latent': torch.ones(2)}, valid)
                broken.write_bytes(bytes(129504))
                items = []
                for cache in (valid, broken):
                    item = SimpleNamespace(path=str(cache.with_suffix('.png')), is_latent_cached=False)
                    item.get_latent_path = lambda recalculate=False, cache=cache: str(cache)
                    item.load_and_process_image = Mock()
                    items.append(item)
                dataset = LatentCachingMixin.__new__(LatentCachingMixin)
                dataset.dataset_path = directory
                dataset.dataset_config = SimpleNamespace(cache_latents_num_workers=1)
                dataset.is_caching_latents_to_disk = True
                dataset.is_caching_latents_to_memory = to_memory
                dataset.file_list = items
                dataset.transform = None
                dataset.sd = SimpleNamespace(device='cpu', set_device_state_preset=Mock(), restore_device_state=Mock())
                encoded = []
                def encode(item, path, state, needs_encode, disk, memory):
                    if needs_encode:
                        encoded.append(path)
                        atomic_save_file({'latent': torch.zeros(2)}, path)
                    elif memory:
                        torch.testing.assert_close(state['latent'], torch.ones(2))
                dataset._cache_one_latent = encode
                dataset._remove_file_items = Mock()
                with self.assertLogs('toolkit.safetensors_cache', level='WARNING'):
                    dataset.cache_latents_all_latents()
                self.assertEqual(encoded, [str(broken)])
                items[0].load_and_process_image.assert_not_called()
                items[1].load_and_process_image.assert_called_once()
                dataset._remove_file_items.assert_not_called()
                self.assertTrue(all(item.is_latent_cached for item in items))
                self.assertTrue(is_valid_cache(str(broken), ('latent',)))

    def test_text_startup_recaches_corrupt_caption_and_dropout_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = TextEmbeddingCachingMixin.__new__(TextEmbeddingCachingMixin)
            dataset.dataset_path = directory
            dataset.dataset_config = SimpleNamespace(caption_dropout_rate=.5, diff_output_preservation=False)
            encode = Mock(side_effect=lambda caption: PromptEmbeds(torch.ones(1, 2, 3)))
            dataset.sd = SimpleNamespace(device='cpu', set_device_state_preset=Mock(), encode_prompt=encode,
                                         encode_first_frame_in_text_embeddings=False)
            dataset.file_list = []
            for index in range(2):
                normal = Path(directory) / f'{index}-caption.safetensors'
                blank = Path(directory) / f'{index}-blank.safetensors'
                for path in (normal, blank):
                    if index == 0:
                        path.write_bytes(bytes(64))
                    else:
                        PromptEmbeds(torch.zeros(1, 2, 3)).save(str(path))
                item = SimpleNamespace(caption=f'caption {index}', encode_control_in_text_embeddings=False,
                                       control_path=None, is_video=False, is_text_embedding_cached=False)
                item.get_text_embedding_path = lambda recalculate=False, path=normal: str(path)
                item.get_blank_text_embedding_path = lambda recalculate=False, path=blank: str(path)
                item.get_dropout_caption = lambda: ''
                dataset.file_list.append(item)
            with self.assertLogs('toolkit.safetensors_cache', level='WARNING'):
                dataset.cache_text_embeddings()
            self.assertEqual([call.args[0] for call in encode.call_args_list], ['caption 0', ''])
            self.assertTrue(all(item.is_text_embedding_cached for item in dataset.file_list))
            for path in Path(directory).glob('*.safetensors'):
                self.assertTrue(is_valid_cache(str(path)))

    def test_runtime_cache_error_identifies_tensor_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'broken.safetensors'
            path.write_bytes(bytes(64))
            with self.assertRaises(CachedTensorError) as raised:
                PromptEmbeds.load(str(path))
            self.assertEqual(raised.exception.path, str(path))
            self.assertIn('cached tensors', str(raised.exception))
            item = SimpleNamespace(is_latent_cached=True, _encoded_latent=None,
                                   get_latent_path=lambda: str(path))
            with self.assertRaises(CachedTensorError) as raised:
                LatentCachingFileItemDTOMixin.get_latent(item)
            self.assertEqual(raised.exception.path, str(path))


if __name__ == '__main__':
    unittest.main()
