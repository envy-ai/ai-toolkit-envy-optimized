"""Small CPU contracts for FIBO2 latent and text-conditioning boundaries."""

import unittest
from unittest.mock import patch

import torch

from extensions_built_in.diffusion_models.fibo2.src import conditioning
from extensions_built_in.diffusion_models.fibo2.src.conditioning import (
    BUNDLE_LAYERS,
    flow_mu,
    hidden_state_bundles,
    pack_latents,
    unpack_latents,
)


class Fibo2ConditioningTests(unittest.TestCase):
    def test_cfg_negative_preserves_explicit_chat_text_and_omits_images(self):
        raw = ' { "short_description" : " negative ", "text_render" : [{"text":"glyph"}] } '
        reference_sentinel = object()
        with patch.object(conditioning, 'draw_text_image', side_effect=AssertionError('negative glyph rendering')):
            chat, images = conditioning.prompt_inputs(raw, [reference_sentinel], is_unconditional=True)
            blank_chat, blank_images = conditioning.prompt_inputs('  \n ', [reference_sentinel], is_unconditional=True)

        self.assertEqual(chat, conditioning.TEMPLATE.format(raw))
        self.assertEqual(images, [])
        self.assertEqual(blank_chat, conditioning.TEMPLATE.format(conditioning.NULL_CAPTION))
        self.assertEqual(blank_images, [])

        glyph = object()
        with patch.object(conditioning, 'draw_text_image', return_value=glyph) as draw:
            positive_chat, positive_images = conditioning.prompt_inputs(
                '{"short_description":"positive","text_render":[{"text":"glyph"}]}')
        self.assertEqual(draw.call_count, 1)
        self.assertEqual(positive_images, [glyph])
        self.assertIn(conditioning.VISION_MARKER, positive_chat)

    def test_packed_latent_layout_roundtrips_with_gradient(self):
        raw = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6).requires_grad_()

        packed = pack_latents(raw)

        self.assertEqual(tuple(packed.shape), (2, 12, 2, 3))
        torch.testing.assert_close(unpack_latents(packed), raw, rtol=0, atol=0)
        packed.square().mean().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(raw.grad.abs().sum().item(), 0)

    def test_text_bundles_select_reference_layers_and_normalize_in_fp32(self):
        hidden = [torch.full((2, 4, 3), float(index + 1), dtype=torch.bfloat16) for index in range(36)]

        bundles = hidden_state_bundles(hidden)

        self.assertEqual(tuple(bundles.shape), (2, 6, 4, 9))
        self.assertEqual(bundles.dtype, torch.bfloat16)
        for bundle_index, layer_triplet in enumerate(BUNDLE_LAYERS):
            pieces = []
            for layer in layer_triplet:
                value = hidden[layer].float()
                pieces.append(value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6))
            expected = torch.cat(pieces, dim=-1).to(torch.bfloat16)
            torch.testing.assert_close(bundles[:, bundle_index], expected, rtol=0, atol=0)

    def test_flow_shift_uses_packed_image_tokens_once(self):
        self.assertEqual(flow_mu(16, 16), 0.5)
        self.assertAlmostEqual(flow_mu(64, 64), 1.15)
        self.assertAlmostEqual(flow_mu(32, 32), 0.63)


if __name__ == "__main__":
    unittest.main()
