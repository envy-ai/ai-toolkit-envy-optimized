import ast
import os
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

    def test_prepend_trigger_word_is_opt_in_and_keeps_saved_prompt_unchanged(self):
        prompt = "\n  portrait setup\n\nat dusk  \n--n \n  haze\n\nextra limbs  \n--w 640 --h 512 --seed 17"
        sample_config = SampleConfig(
            width=512,
            height=512,
            seed=23,
            samples=[{"prompt": prompt, "neg": "sample-level negative"}],
            prepend_trigger_word=True,
        )
        captured = []

        class FakeModel:
            def validate_sample_config(self, _sample_config):
                pass

            def inject_trigger_into_prompt(self, prompt, _trigger_word, **_kwargs):
                return prompt

            def generate_images(self, configs, **kwargs):
                captured.extend(configs)

        trainer = object.__new__(BaseSDTrainProcess)
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.sample_config = sample_config
        trainer.first_sample_config = sample_config
        trainer.save_root = tempfile.gettempdir()
        trainer.embedding = None
        trainer.adapter = None
        trainer.trigger_word = "amber style"
        trainer.adapter_config = None
        trainer.ema = None
        trainer.sd = FakeModel()
        trainer.logger = None
        trainer._refresh_live_sample_config = lambda: None
        trainer._get_krea_preview_control_settings = lambda: None
        trainer.post_process_generate_image_config_list = lambda configs: configs

        trainer.sample(step=7)

        config = captured[0]
        self.assertEqual(config.prompt, "amber style\n\n\n  portrait setup\n\nat dusk  \n")
        self.assertEqual(config.negative_prompt, "\n  haze\n\nextra limbs  \n")
        self.assertEqual((config.width, config.height), (640, 512))
        self.assertEqual(config.seed, 17)
        self.assertEqual(sample_config.prompts[0], prompt)

    def test_prepend_trigger_word_default_disabled_and_missing_trigger(self):
        self.assertFalse(SampleConfig(samples=[{"prompt": "scene"}]).prepend_trigger_word)

        def rendered_prompt(*, enabled, trigger_word):
            sample_config = SampleConfig(
                samples=[{"prompt": "scene"}], prepend_trigger_word=enabled,
            )
            captured = []

            class FakeModel:
                def validate_sample_config(self, _sample_config):
                    pass

                def inject_trigger_into_prompt(self, prompt, _trigger_word, **_kwargs):
                    return prompt

                def generate_images(self, configs, **kwargs):
                    captured.extend(configs)

            trainer = object.__new__(BaseSDTrainProcess)
            trainer.accelerator = SimpleNamespace(is_main_process=True)
            trainer.sample_config = trainer.first_sample_config = sample_config
            trainer.save_root = tempfile.gettempdir()
            trainer.embedding = trainer.adapter = trainer.adapter_config = trainer.ema = None
            trainer.trigger_word = trigger_word
            trainer.sd = FakeModel()
            trainer.logger = None
            trainer._refresh_live_sample_config = lambda: None
            trainer._get_krea_preview_control_settings = lambda: None
            trainer.post_process_generate_image_config_list = lambda configs: configs
            trainer.sample(step=1)
            return captured[0].prompt

        self.assertEqual(rendered_prompt(enabled=False, trigger_word="amber"), "scene")
        self.assertEqual(rendered_prompt(enabled=True, trigger_word=None), "scene")
        self.assertEqual(rendered_prompt(enabled=True, trigger_word=""), "scene")

    def test_cached_prompt_encoding_matches_generated_prompt_with_trigger_prefix(self):
        prompt = "\n  portrait\n\n--w 640"
        sample_config = SampleConfig(
            samples=[{"prompt": prompt}], prepend_trigger_word=True,
        )
        encoded_prompts = []

        class FakeEmbeds:
            def to(self, *_args, **_kwargs):
                return self

        class FakeSD:
            encode_control_in_text_embeddings = False

            def inject_trigger_into_prompt(self, prompt, _trigger_word, **_kwargs):
                return prompt

            def encode_prompt(self, prompt, **_kwargs):
                encoded_prompts.append(prompt)
                return FakeEmbeds()

        source = (
            Path(__file__).resolve().parents[1]
            / "extensions_built_in/sd_trainer/SDTrainer.py"
        ).read_text()
        module = ast.parse(source)
        cache_method = next(
            node for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "SDTrainer"
            for node in node.body
            if isinstance(node, ast.FunctionDef) and node.name == "cache_sample_prompts"
        )
        extracted = ast.Module(body=[cache_method], type_ignores=[])
        namespace = {"os": os, "GenerateImageConfig": GenerateImageConfig}
        exec(compile(extracted, "SDTrainer.py", "exec"), namespace)
        cache = SimpleNamespace(
            train_config=SimpleNamespace(disable_sampling=False),
            sample_config=sample_config,
            save_root=tempfile.gettempdir(),
            trigger_word="amber",
            sd=FakeSD(),
        )
        namespace["cache_sample_prompts"](cache)

        self.assertEqual(encoded_prompts[0], "amber\n\n\n  portrait\n\n")
        self.assertFalse(encoded_prompts[1])
        self.assertEqual(sample_config.prompts[0], prompt)

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
