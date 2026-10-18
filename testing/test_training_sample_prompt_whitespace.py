import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config_modules import GenerateImageConfig, SampleConfig


class TrainingSamplePromptWhitespaceTests(unittest.TestCase):
    def test_legacy_generate_config_keeps_non_newline_option_parsing(self):
        config = GenerateImageConfig(
            prompt="portrait --w 640 --seed 17",
            output_path="/tmp/legacy-sample.png",
        )

        self.assertEqual(config.prompt, "portrait")
        self.assertEqual(config.width, 640)
        self.assertEqual(config.seed, 17)

    def test_opted_in_config_preserves_boundary_newlines_without_flags(self):
        prompt = "\n  first line\n\nlast line  \n"
        config = GenerateImageConfig(
            prompt=prompt,
            output_path="/tmp/multiline-sample.png",
            preserve_prompt_whitespace=True,
        )

        self.assertEqual(config.prompt, prompt)

    def test_sample_passes_exact_text_and_legacy_flags_to_generate_images(self):
        prompt = "\n  portrait setup\n\nat dusk  \n--n \n  haze\n\nextra limbs  \n--w 640 --h 512 --seed 17"
        sample_config = SampleConfig(
            width=512,
            height=512,
            seed=23,
            samples=[{"prompt": prompt, "neg": "sample-level negative"}],
        )
        captured = []

        class FakeModel:
            def validate_sample_config(self, _sample_config):
                pass

            def generate_images(self, configs, **kwargs):
                captured.extend(configs)

        trainer = object.__new__(BaseSDTrainProcess)
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.sample_config = sample_config
        trainer.first_sample_config = sample_config
        trainer.save_root = tempfile.gettempdir()
        trainer.embedding = None
        trainer.adapter = None
        trainer.trigger_word = None
        trainer.adapter_config = None
        trainer.ema = None
        trainer.sd = FakeModel()
        trainer.logger = None
        trainer._refresh_live_sample_config = lambda: None
        trainer._get_krea_preview_control_settings = lambda: None
        trainer.post_process_generate_image_config_list = lambda configs: configs

        trainer.sample(step=7)

        self.assertEqual(len(captured), 1)
        config = captured[0]
        self.assertEqual(config.prompt, "\n  portrait setup\n\nat dusk  \n")
        self.assertEqual(config.negative_prompt, "\n  haze\n\nextra limbs  \n")
        self.assertEqual((config.width, config.height), (640, 512))
        self.assertEqual(config.seed, 17)

    def test_cached_training_sample_prompts_enable_whitespace_preservation(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "extensions_built_in/sd_trainer/SDTrainer.py"
        ).read_text()
        method_start = source.index("def cache_sample_prompts(self):")
        method_end = source.index("def before_dataset_load(self):", method_start)
        self.assertIn("preserve_prompt_whitespace=True", source[method_start:method_end])


if __name__ == "__main__":
    unittest.main()
