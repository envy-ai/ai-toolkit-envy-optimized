"""CPU boundary checks for packed FLUX.2 VAE inference tiling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from extensions_built_in.diffusion_models.fibo2.fibo2 import Fibo2Model
from extensions_built_in.diffusion_models.fibo2.src.vae import Fibo2AutoEncoder
from toolkit.models.v2.vae.flux2_kl import AutoEncoder


class Fibo2VaeTilingTests(unittest.TestCase):
    def test_packed_latents_fitting_one_tile_call_untiled_decoder_once(self):
        vae = Fibo2AutoEncoder.__new__(Fibo2AutoEncoder)
        torch.nn.Module.__init__(vae)
        latents = torch.randn(1, 128, 64, 64)
        calls = []

        def fake_decode(_self, value):
            calls.append(tuple(value.shape))
            return F.interpolate(value[:, :3], scale_factor=16, mode="nearest")

        with patch.object(AutoEncoder, "decode", fake_decode):
            tiled = vae.decode_tiled(latents, tile=64, stride=48)
            self.assertEqual(calls, [(1, 128, 64, 64)])
            calls.clear()
            untiled = AutoEncoder.decode(vae, latents)

        self.assertEqual(calls, [(1, 128, 64, 64)])
        torch.testing.assert_close(tiled, untiled, rtol=0, atol=0)

    def test_decode_latents_keeps_grad_path_untiled_and_uses_tiled_inference(self):
        class FakeVae:
            dtype = torch.float32

            def __init__(self):
                self.calls = []

            def to(self, _device):
                return self

            def decode(self, value):
                self.calls.append("decode")
                return value * 2

            def decode_tiled(self, value):
                self.calls.append("decode_tiled")
                return value * 3

        model = object.__new__(Fibo2Model)
        model.vae = FakeVae()
        model.vae_device_torch = torch.device("cpu")
        model.vae_torch_dtype = torch.float32
        model.model_config = SimpleNamespace(low_vram=True)
        latents = torch.ones(1, 128, 2, 2, requires_grad=True)

        decoded = model.decode_latents(latents)
        self.assertEqual(model.vae.calls, ["decode"])
        decoded.sum().backward()
        torch.testing.assert_close(latents.grad, torch.full_like(latents, 2))

        model.vae.calls.clear()
        with torch.no_grad():
            preview = model.decode_latents(latents.detach())
        self.assertEqual(model.vae.calls, ["decode_tiled"])
        torch.testing.assert_close(preview, torch.full_like(latents, 3))


if __name__ == "__main__":
    unittest.main()
