"""Qwen 2.1 processor startup with its real config-less tokenizer layout."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extensions_built_in.diffusion_models.qwen_image_2.src.text_encoder import QwenImage21TextEncoder
from toolkit.models.v2._mixin import OstrisModelMixin


class QwenImage21ProcessorTests(unittest.TestCase):
    def test_declared_tokenizer_type_and_explicit_override_are_forwarded(self):
        with patch.object(OstrisModelMixin, 'load_processor', return_value='processor') as parent:
            self.assertEqual(QwenImage21TextEncoder.load_processor('checkpoint'), 'processor')
            parent.assert_called_once_with('checkpoint', None, tokenizer_type='qwen2')
            QwenImage21TextEncoder.load_processor('checkpoint', 'processor', tokenizer_type='qwen2')
            self.assertEqual(parent.call_args.kwargs['tokenizer_type'], 'qwen2')

    def test_real_offline_processor_without_config_json(self):
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache('Qwen/Qwen-Image-2.1', 'processor/tokenizer.json')
        if not isinstance(cached, str):
            self.skipTest('Optional real processor fixture is not locally cached')
        source = Path(cached).parent
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'processor'
            target.mkdir()
            for file in source.iterdir():
                if file.is_file() and file.name != 'config.json':
                    (target / file.name).symlink_to(file.resolve())
            self.assertFalse((target / 'config.json').exists())
            processor = QwenImage21TextEncoder.load_processor(directory, local_files_only=True)
            self.assertEqual(type(processor).__name__, 'Qwen3VLProcessor')
            self.assertIn('Qwen2Tokenizer', type(processor.tokenizer).__name__)
            self.assertEqual(processor.tokenizer.encode('A red cat'), [32, 2518, 8251])


if __name__ == '__main__':
    unittest.main()
