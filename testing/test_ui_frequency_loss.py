import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class UIFrequencyLossTests(unittest.TestCase):
    def test_form_exposes_all_pixel_frequency_filters_and_period_controls(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()

        self.assertIn('label="Pixel Frequency Loss"', source)
        for filter_type in ("low_pass", "high_pass", "band_pass", "notch"):
            self.assertIn(f"value: '{filter_type}'", source)
        self.assertIn('label="Cutoff Period (pixels)"', source)
        self.assertIn('label="Shortest Period (pixels)"', source)
        self.assertIn('label="Longest Period (pixels)"', source)
        self.assertIn('label="Transition Width (pixels)"', source)
        self.assertIn('label="Frequency Loss Weight"', source)
        self.assertIn('label="Frequency Patch Size (pixels)"', source)
        self.assertIn('label="Offload Frequency Activations"', source)

    def test_frequency_loss_defaults_and_types_are_serializable(self):
        defaults = (REPO_ROOT / "ui/src/app/jobs/new/jobConfig.ts").read_text()
        types = (REPO_ROOT / "ui/src/types.ts").read_text()

        self.assertIn("frequency_loss_type: 'none'", defaults)
        self.assertIn("frequency_loss_cutoff: 18", defaults)
        self.assertIn("frequency_loss_min_period: 14", defaults)
        self.assertIn("frequency_loss_max_period: 28", defaults)
        self.assertIn("frequency_loss_patch_size: 384", defaults)
        self.assertIn("frequency_loss_activation_offload: true", defaults)
        self.assertIn("'low_pass' | 'high_pass' | 'band_pass' | 'notch'", types)

    def test_trainer_keeps_content_loss_and_adds_frequency_loss(self):
        source = (REPO_ROOT / "extensions_built_in/sd_trainer/SDTrainer.py").read_text()

        self.assertIn("frequency_loss = self._calculate_frequency_pixel_loss", source)
        self.assertIn("frequency_loss * self.train_config.frequency_loss_weight", source)
        self.assertIn("loss = loss + additional_loss", source)


if __name__ == "__main__":
    unittest.main()
