import importlib.util
import pathlib
import unittest

import torch


def _load_krea_mmdit():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "extensions_built_in/diffusion_models/krea2/src/mmdit.py"
    spec = importlib.util.spec_from_file_location("krea2_mmdit_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Krea2AttentionMaskTest(unittest.TestCase):
    def test_mask_stays_key_padding_shaped(self):
        mmdit = _load_krea_mmdit()
        padding_mask = torch.tensor([[True, True, False, True]])

        attention_mask = mmdit._mask(padding_mask)

        self.assertEqual(attention_mask.shape, (1, 1, 1, 4))
        self.assertTrue(
            torch.equal(attention_mask[0, 0, 0], padding_mask[0])
        )


if __name__ == "__main__":
    unittest.main()
