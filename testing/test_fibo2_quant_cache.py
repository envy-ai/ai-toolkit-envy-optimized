"""Strict FIBO2 quantized-module cache failures remain visible to the loader."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from extensions_built_in.diffusion_models.fibo2.fibo2 import Fibo2Model
from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.util.ostris_quant import OstrisLinear


class TinyQuantizedModel(torch.nn.Module, OstrisModelMixin):
    def __init__(self, *, checkpoint_quantized=False):
        super().__init__()
        self.proj = OstrisLinear(2, 2)
        self.proj.ostris_quantizer = SimpleNamespace(qtype='convrot8')
        self.aitk_is_quantized = checkpoint_quantized
        self.aitk_qtype = 'convrot8' if checkpoint_quantized else None
        self.aitk_quantization_from_checkpoint = checkpoint_quantized

    def get_transformer_block_names(self):
        return []

    def get_quantization_exclude_modules(self):
        return []


class Fibo2QuantCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(Fibo2Model)
        self.model.print_and_status_update = lambda _message: None

    def assert_cache_is_written(self, module):
        cache_owner = object.__new__(Fibo2Model)
        cache_owner.print_and_status_update = lambda _message: None
        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            cache_path = str(Path(directory) / 'cache.pt')

            def write_fixture(_module, path):
                Path(path).write_bytes(b'fixture cache')

            with patch('torch.save', side_effect=write_fixture) as save:
                cache_owner.save_quantized_module_cache(
                    module, cache_path, 'transformer', strict=True
                )
            save.assert_called_once()
            self.assertEqual(Path(cache_path).read_bytes(), b'fixture cache')

    def test_corrupt_quantized_cache_propagates_in_strict_mode(self):
        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            cache_path = Path(directory) / 'broken.pt'
            cache_path.write_bytes(b'not a torch module')
            with self.assertRaisesRegex(RuntimeError, 'Failed to load requested quantized transformer cache'):
                self.model.load_quantized_module_cache(str(cache_path), 'transformer', strict=True)
            self.assertIsNone(self.model.load_quantized_module_cache(str(cache_path), 'transformer'))

    def test_quantized_cache_write_failure_propagates_in_strict_mode(self):
        module = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            cache_path = str(Path(directory) / 'cache.pt')
            with patch('torch.save', side_effect=OSError('fixture write failure')):
                with self.assertRaisesRegex(RuntimeError, 'Failed to save requested quantized transformer cache'):
                    self.model.save_quantized_module_cache(module, cache_path, 'transformer', strict=True)
                # The ordinary-model default stays permissive; only FIBO2 opts into strict mode.
                self.model.save_quantized_module_cache(module, cache_path, 'transformer')
            self.assertFalse(Path(cache_path).exists())
            self.assertFalse(Path(cache_path + '.tmp').exists())

    def test_matching_checkpoint_quantization_is_retained_without_cache_write(self):
        module = TinyQuantizedModel(checkpoint_quantized=True)
        messages = []
        base_model = SimpleNamespace(print_and_status_update=messages.append)
        module.aitk_post_load(qtype='convrot8', base_model=base_model)
        self.assertTrue(module.aitk_checkpoint_quantization_retained)
        self.assertTrue(any('keeping it' in message for message in messages))

        cache_owner = object.__new__(Fibo2Model)
        cache_owner.print_and_status_update = messages.append
        module.aitk_checkpoint_quantization_retained = True
        original_state_dict = lambda: {'fixture': torch.tensor(1)}
        module.orig_state_dict = original_state_dict
        module.state_dict = lambda: {'patched': torch.tensor(2)}
        original_bound_state_dict = module.state_dict

        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            cache_path = Path(directory) / 'not-created' / 'transformer.pt'
            with (
                patch('torch.save') as save,
                patch('toolkit.models.base_model.os.makedirs') as makedirs,
                patch.object(module, 'to', wraps=module.to) as move,
            ):
                cache_owner.save_quantized_module_cache(
                    module, str(cache_path), 'transformer', strict=True
                )

            save.assert_not_called()
            makedirs.assert_not_called()
            move.assert_not_called()
            self.assertFalse(cache_path.parent.exists())
            self.assertIs(module.orig_state_dict, original_state_dict)
            self.assertIs(module.state_dict, original_bound_state_dict)

    def test_fresh_quantization_does_not_become_checkpoint_retention(self):
        module = TinyQuantizedModel(checkpoint_quantized=False)
        self.assertFalse(module.aitk_quantization_from_checkpoint)

        with patch('toolkit.util.quantize.quantize_module') as quantize:
            module.aitk_post_load(qtype='convrot8')
            quantize.assert_called_once()
            self.assertTrue(module.aitk_is_quantized)
            self.assertFalse(module.aitk_quantization_from_checkpoint)

            # A subsequent matching post-load sees the same quantized layer,
            # but must not attribute its provenance to an input checkpoint.
            module.aitk_post_load(qtype='convrot8')

        self.assertFalse(module.aitk_quantization_from_checkpoint)
        self.assertFalse(module.aitk_checkpoint_quantization_retained)

    def test_checkpoint_conversion_and_ara_remain_cacheable(self):
        module = TinyQuantizedModel(checkpoint_quantized=True)
        with patch('toolkit.util.quantize.dequantize_ostris_to_linear') as dequantize:
            module.aitk_post_load(qtype=None)
            dequantize.assert_called_once_with(module)
        self.assertFalse(module.aitk_quantization_from_checkpoint)
        self.assertFalse(module.aitk_checkpoint_quantization_retained)
        self.assert_cache_is_written(module)

        # A different backend request converts the checkpoint representation,
        # then quantizes the resulting module under the selected policy.
        module = TinyQuantizedModel(checkpoint_quantized=True)
        with (
            patch('toolkit.util.quantize.get_qtype', return_value=object()),
            patch('toolkit.util.quantize.dequantize_ostris_to_linear') as dequantize,
            patch('toolkit.util.quantize.quantize_module') as quantize,
        ):
            module.aitk_post_load(qtype='other-backend')
            dequantize.assert_called_once_with(module)
            quantize.assert_called_once()
        self.assertFalse(module.aitk_quantization_from_checkpoint)
        self.assertFalse(module.aitk_checkpoint_quantization_retained)
        self.assert_cache_is_written(module)

        module = TinyQuantizedModel(checkpoint_quantized=True)
        base_model = SimpleNamespace(print_and_status_update=lambda _message: None)
        with (
            patch('toolkit.util.quantize.dequantize_ostris_to_linear'),
            patch('toolkit.util.quantize.attach_ara_and_quantize') as attach_ara,
        ):
            def assert_checkpoint_origin_cleared(_base, converted, *_args, **_kwargs):
                self.assertFalse(converted.aitk_quantization_from_checkpoint)
                self.assertFalse(converted.aitk_checkpoint_quantization_retained)

            attach_ara.side_effect = assert_checkpoint_origin_cleared
            module.aitk_post_load(
                qtype='convrot8|fixture-ara.safetensors', base_model=base_model
            )
            attach_ara.assert_called_once()
        self.assertFalse(module.aitk_quantization_from_checkpoint)
        self.assertFalse(module.aitk_checkpoint_quantization_retained)
        self.assert_cache_is_written(module)


if __name__ == '__main__':
    unittest.main()
