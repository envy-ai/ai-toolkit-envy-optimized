import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.util.losses import (
    build_pixel_frequency_mask,
    frequency_filtered_pixel_loss,
)


def horizontal_sine(height=64, width=64, period=16.0):
    x = torch.arange(width, dtype=torch.float32)
    row = torch.sin(2.0 * torch.pi * x / period)
    return row.view(1, 1, 1, width).expand(1, 1, height, width).clone()


class PixelFrequencyLossTests(unittest.TestCase):
    def test_low_and_high_pass_split_at_an_adjustable_pixel_period(self):
        prediction = horizontal_sine(period=16.0)
        target = torch.zeros_like(prediction)

        high_loss = frequency_filtered_pixel_loss(
            prediction,
            target,
            filter_type="high_pass",
            cutoff_period=18.0,
            transition=0.0,
        )
        low_loss = frequency_filtered_pixel_loss(
            prediction,
            target,
            filter_type="low_pass",
            cutoff_period=18.0,
            transition=0.0,
        )

        self.assertGreater(high_loss.item(), 0.1)
        self.assertLess(low_loss.item(), 1e-10)

    def test_band_pass_selects_and_notch_excludes_the_same_period_range(self):
        prediction = horizontal_sine(period=16.0)
        target = torch.zeros_like(prediction)

        band_loss = frequency_filtered_pixel_loss(
            prediction,
            target,
            filter_type="band_pass",
            min_period=12.0,
            max_period=20.0,
            transition=0.0,
        )
        notch_loss = frequency_filtered_pixel_loss(
            prediction,
            target,
            filter_type="notch",
            min_period=12.0,
            max_period=20.0,
            transition=0.0,
        )

        self.assertGreater(band_loss.item(), 0.1)
        self.assertLess(notch_loss.item(), 1e-10)

    def test_low_pass_keeps_dc_and_high_pass_rejects_it(self):
        prediction = torch.ones((1, 1, 32, 32), dtype=torch.float32)
        target = torch.zeros_like(prediction)

        low_loss = frequency_filtered_pixel_loss(
            prediction,
            target,
            filter_type="low_pass",
            cutoff_period=18.0,
            transition=4.0,
        )
        high_loss = frequency_filtered_pixel_loss(
            prediction,
            target,
            filter_type="high_pass",
            cutoff_period=18.0,
            transition=4.0,
        )

        self.assertGreater(low_loss.item(), 0.1)
        self.assertLess(high_loss.item(), 1e-10)

    def test_filtered_loss_backpropagates_to_prediction(self):
        prediction = horizontal_sine(period=16.0).requires_grad_(True)
        target = torch.zeros_like(prediction)

        loss = frequency_filtered_pixel_loss(
            prediction,
            target,
            filter_type="band_pass",
            min_period=12.0,
            max_period=20.0,
            transition=2.0,
        )
        loss.backward()

        self.assertIsNotNone(prediction.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)

    def test_invalid_band_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "min_period"):
            build_pixel_frequency_mask(
                64,
                64,
                filter_type="band_pass",
                min_period=20.0,
                max_period=12.0,
            )

    def test_trainer_auxiliary_loss_keeps_gradient_through_decoder(self):
        clean_latents = horizontal_sine(period=16.0).requires_grad_(True)
        trainer = SimpleNamespace(
            sd=SimpleNamespace(decode_latents=lambda latents: latents),
            train_config=SimpleNamespace(
                frequency_loss_type="band_pass",
                frequency_loss_cutoff=18.0,
                frequency_loss_min_period=12.0,
                frequency_loss_max_period=20.0,
                frequency_loss_transition=2.0,
            ),
        )
        batch = SimpleNamespace(
            tensor=torch.zeros_like(clean_latents),
            latents=torch.zeros_like(clean_latents),
        )

        loss = SDTrainer._calculate_frequency_pixel_loss(
            trainer,
            clean_latents=clean_latents,
            batch=batch,
            loss_multiplier=torch.ones(1),
        )
        loss.backward()

        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(clean_latents.grad.abs().sum().item(), 0.0)

    def test_trainer_decodes_only_the_configured_frequency_patch(self):
        decoded_latent_shapes = []

        def decode_latents(latents):
            decoded_latent_shapes.append(tuple(latents.shape[-2:]))
            return F.interpolate(latents, scale_factor=2, mode="bilinear", align_corners=False)

        clean_latents = horizontal_sine(height=64, width=64, period=16.0).requires_grad_(True)
        trainer = SimpleNamespace(
            sd=SimpleNamespace(decode_latents=decode_latents, vae_scale_factor=2),
            train_config=SimpleNamespace(
                frequency_loss_type="band_pass",
                frequency_loss_cutoff=18.0,
                frequency_loss_min_period=12.0,
                frequency_loss_max_period=20.0,
                frequency_loss_transition=2.0,
                frequency_loss_patch_size=32,
                frequency_loss_activation_offload=True,
            ),
        )
        batch = SimpleNamespace(
            tensor=torch.zeros((1, 1, 128, 128)),
            latents=torch.zeros_like(clean_latents),
        )

        loss = SDTrainer._calculate_frequency_pixel_loss(
            trainer,
            clean_latents=clean_latents,
            batch=batch,
            loss_multiplier=torch.ones(1),
        )
        loss.backward()

        self.assertEqual(len(decoded_latent_shapes), 1)
        self.assertLessEqual(decoded_latent_shapes[0][0], 20)  # 16-cell core + context
        self.assertLessEqual(decoded_latent_shapes[0][1], 20)
        self.assertGreater(clean_latents.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
