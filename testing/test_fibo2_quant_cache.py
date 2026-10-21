"""Strict FIBO2 quantized-module cache failures remain visible to the loader."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from extensions_built_in.diffusion_models.fibo2.fibo2 import Fibo2Model


class Fibo2QuantCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(Fibo2Model)
        self.model.print_and_status_update = lambda _message: None

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


if __name__ == '__main__':
    unittest.main()
