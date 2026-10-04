"""Example YAMLs must pass the real specialized constructors without loading models."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.FizgigSliderTrainer import FizgigSliderTrainer
from extensions_built_in.sd_trainer.FlowDPOTrainer import FlowDPOTrainer
from extensions_built_in.sd_trainer.GuidanceDistillationTrainer import GuidanceDistillationTrainer
from extensions_built_in.sd_trainer.SliderSpaceTrainer import SliderSpaceTrainer
from extensions_built_in.sd_trainer.DiffusionKTOTrainer import DiffusionKTOTrainer

ROOT = Path(__file__).resolve().parents[1]


class CrossModelExampleTests(unittest.TestCase):
    def test_all_fifteen_examples_validate_before_model_loading(self):
        trainers = {'fizgig_image_slider': FizgigSliderTrainer, 'fizgig_prompt_slider': FizgigSliderTrainer,
                    'flow_dpo': FlowDPOTrainer, 'guidance_distillation': GuidanceDistillationTrainer,
                    'sliderspace': SliderSpaceTrainer}
        with tempfile.TemporaryDirectory() as directory:
            positive, negative = Path(directory) / 'positive', Path(directory) / 'negative'
            for folder in (positive, negative):
                folder.mkdir()
                Image.new('RGB', (128, 128)).save(folder / 'a.png')
            count = 0
            for arch in ('krea2', 'anima', 'ideogram4'):
                for mode, trainer_type in trainers.items():
                    with self.subTest(arch=arch, mode=mode):
                        filename = ROOT / f'config/examples/train_{mode}_{arch}.yaml'
                        config = yaml.safe_load(filename.read_text())['config']['process'][0]
                        self.assertEqual(config['type'], mode)
                        self.assertEqual(config['model']['arch'], arch)
                        for dataset in config['datasets']:
                            dataset['folder_path'] = str(positive)
                            if dataset.get('control_path_1'):
                                dataset['control_path_1'] = str(negative)
                        if mode == 'sliderspace':
                            config['sliderspace']['discovery_datasets'][0]['folder_path'] = str(positive)
                        with patch.object(DiffusionTrainer, '__init__', lambda obj, *args, **kwargs:
                            setattr(obj, 'accelerator', SimpleNamespace(num_processes=1))):
                            trainer = trainer_type(0, None, copy.deepcopy(config))
                        self.assertEqual(trainer.flow_profile.arch, arch)
                        self.assertFalse(config['train']['train_text_encoder'])
                        count += 1
            self.assertEqual(count, 15)

    def test_four_kto_examples_with_independent_bucketed_labeled_images(self):
        with tempfile.TemporaryDirectory() as directory:
            liked, disliked = Path(directory) / 'liked', Path(directory) / 'disliked'
            liked.mkdir()
            disliked.mkdir()
            Image.new('RGB', (128, 64)).save(liked / 'one.png')
            Image.new('RGB', (64, 128)).save(disliked / 'different.png')
            Image.new('RGB', (96, 64)).save(disliked / 'extra.png')
            for arch, suffix in [('qwen_image_2', 'qwen_image_21'), ('krea2', 'krea2'),
                                 ('anima', 'anima'), ('ideogram4', 'ideogram4')]:
                with self.subTest(arch=arch):
                    filename = ROOT / f'config/examples/train_diffusion_kto_{suffix}.yaml'
                    config = yaml.safe_load(filename.read_text())['config']['process'][0]
                    self.assertEqual(config['type'], 'diffusion_kto')
                    self.assertEqual(config['model']['arch'], arch)
                    for dataset in config['datasets']:
                        dataset['folder_path'] = str(liked if dataset['kto_label'] == 'liked' else disliked)
                    with patch.object(DiffusionTrainer, '__init__', return_value=None), patch.object(DiffusionKTOTrainer, 'print'):
                        trainer = DiffusionKTOTrainer(0, None, copy.deepcopy(config))
                    self.assertEqual(trainer.flow_profile.arch, arch)
                    self.assertEqual(trainer.kto_source_counts, {'liked': 1, 'disliked': 2})
                    self.assertEqual(trainer.kto_settings.reference_estimator, 'score_window')
                    self.assertEqual(trainer.kto_settings.score_window_size, 4)


if __name__ == '__main__':
    unittest.main()
