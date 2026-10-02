import builtins
import unittest
from unittest import mock

import torch

from toolkit.util.losses import wavelet_loss


class WaveletLossTests(unittest.TestCase):
    def test_does_not_import_pytorch_wavelets(self):
        original_import = builtins.__import__

        def reject_pytorch_wavelets(name, *args, **kwargs):
            if name.startswith("pytorch_wavelets"):
                raise AssertionError("wavelet_loss imported pytorch_wavelets")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_pytorch_wavelets):
            result = wavelet_loss(
                torch.zeros((1, 2, 8, 8)),
                torch.zeros((1, 2, 8, 8)),
                torch.zeros((1, 2, 8, 8)),
            )

        self.assertEqual(tuple(result.shape), (1, 8, 4, 4))

    def test_exact_prediction_has_zero_loss(self):
        latents = torch.randn((2, 3, 8, 10))
        noise = torch.randn_like(latents)
        model_pred = noise - latents

        loss = wavelet_loss(model_pred, latents, noise)

        torch.testing.assert_close(loss, torch.zeros_like(loss))

    def test_loss_backpropagates_to_prediction(self):
        model_pred = torch.randn((2, 3, 8, 10), requires_grad=True)
        latents = torch.randn_like(model_pred)
        noise = torch.randn_like(model_pred)

        wavelet_loss(model_pred, latents, noise).mean().backward()

        self.assertIsNotNone(model_pred.grad)
        self.assertGreater(model_pred.grad.abs().sum().item(), 0.0)

    def test_odd_dimensions_are_zero_padded(self):
        shape = (2, 3, 7, 9)
        loss = wavelet_loss(
            torch.randn(shape),
            torch.randn(shape),
            torch.randn(shape),
        )

        self.assertEqual(tuple(loss.shape), (2, 12, 4, 5))


if __name__ == "__main__":
    unittest.main()
