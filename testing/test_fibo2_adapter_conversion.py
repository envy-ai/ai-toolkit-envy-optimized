"""Numerical checks for split-to-fused FIBO adapter projection conversion."""

import unittest

import torch

from extensions_built_in.diffusion_models.fibo2.src.adapters import canonical_adapter


class Fibo2AdapterConversionTests(unittest.TestCase):
    def test_unequal_rank_split_qkv_fusion_preserves_each_scaled_projection(self):
        torch.manual_seed(57)
        prefix = 'transformer.layers.0.attention.'
        split = {}
        expected = []
        for projection, rank, alpha in (('to_q', 2, 1.0), ('to_k', 3, 6.0), ('to_v', 1, 0.5)):
            down = torch.randn(rank, 5)
            up = torch.randn(4, rank)
            split[prefix + projection + '.lora_A.weight'] = down
            split[prefix + projection + '.lora_B.weight'] = up
            split[prefix + projection + '.alpha'] = torch.tensor(alpha)
            expected.append(up @ down * (alpha / rank))

        converted = canonical_adapter(split)
        fused_down = converted[prefix + 'qkv.lora_A.weight']
        fused_up = converted[prefix + 'qkv.lora_B.weight']
        actual = fused_up @ fused_down
        self.assertEqual(tuple(fused_down.shape), (6, 5))
        self.assertEqual(tuple(fused_up.shape), (12, 6))
        self.assertTrue(torch.allclose(actual, torch.cat(expected, dim=0), atol=1e-6, rtol=1e-6))
        self.assertEqual(converted[prefix + 'qkv.alpha'].item(), 6)

    def test_mixed_split_and_fused_projection_layout_is_rejected(self):
        prefix = 'transformer.layers.0.attention.'
        state = {
            prefix + 'to_q.lora_A.weight': torch.ones(1, 2),
            prefix + 'to_q.lora_B.weight': torch.ones(2, 1),
            prefix + 'to_k.lora_A.weight': torch.ones(1, 2),
            prefix + 'to_k.lora_B.weight': torch.ones(2, 1),
            prefix + 'to_v.lora_A.weight': torch.ones(1, 2),
            prefix + 'to_v.lora_B.weight': torch.ones(2, 1),
            prefix + 'qkv.lora_A.weight': torch.ones(3, 2),
            prefix + 'qkv.lora_B.weight': torch.ones(6, 3),
        }
        with self.assertRaisesRegex(ValueError, 'both split and fused weights'):
            canonical_adapter(state)


if __name__ == '__main__':
    unittest.main()
