from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.FizgigSliderTrainer import FizgigSliderTrainer, parse_anchor_prompts
from toolkit.config_modules import DatasetConfig
from toolkit.dataloader_mixins import LatentCachingFileItemDTOMixin
from toolkit.prompt_utils import PromptEmbeds
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model

MODULE = 'extensions_built_in.sd_trainer.FizgigSliderTrainer'


def embed(value=1):
    return PromptEmbeds(torch.full((1, 2, 3), float(value)))


class AnchorTests(unittest.TestCase):
    def test_anchor_parse_does_not_apply_direction_prefixes(self):
        self.assertEqual(parse_anchor_prompts({
            'positive_prefix': 'large', 'negative_prefix': 'small',
            'anchor_prompts': [{'prompt': ' dog ', 'negative_prompt': ' blur '}, {'prompt': ' landscape '}],
        }), [('dog', 'blur'), ('landscape', '')])
        for value in [None, {}, ['dog'], [{'prompt': ''}], [{'prompt': 'dog', 'negative_prompt': None}]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_anchor_prompts({'anchor_prompts': value})

    def test_constructor_image_anchors_are_independent_not_pairs_or_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pos, neg, anchors = [root / name for name in ('pos', 'neg', 'anchors')]
            for path in (pos, neg, anchors):
                path.mkdir()
            Image.new('RGB', (64, 64)).save(pos / 'cat.png')
            Image.new('RGB', (64, 64)).save(neg / 'cat.png')
            Image.new('RGB', (128, 64)).save(anchors / 'dog.png')
            config = {
                'type': 'fizgig_image_slider', 'model': {'arch': 'qwen_image_2'},
                'network': {'type': 'dora'}, 'train': {'noise_scheduler': 'flowmatch'},
                'datasets': [{'folder_path': str(pos), 'control_path_1': str(neg), 'anchor_path': str(anchors),
                              'resolution': [1024], 'default_caption': 'cat', 'cache_latents_to_disk': True}],
                'fizgig_slider': {'anchor_prompts': [{'prompt': 'dog'}]},
            }
            with patch.object(DiffusionTrainer, '__init__', return_value=None):
                trainer = FizgigSliderTrainer(0, None, config)
            self.assertTrue(trainer.retain_vae_after_caching)
            self.assertEqual(config['datasets'][0]['unconditional_path'], str(neg))
            self.assertEqual(trainer.slider_anchor_image_configs[0]['folder_path'], str(anchors))
            self.assertEqual(trainer.slider_anchor_image_configs[0]['default_caption'], '')
            self.assertNotIn('control_path_1', trainer.slider_anchor_image_configs[0])
            self.assertTrue(trainer.slider_anchor_image_configs[0]['cache_text_embeddings'])
            self.assertTrue(trainer.slider_anchor_image_configs[0]['fizgig_slider_anchor'])
            config['datasets'][0]['control_path_1'] = str(neg)
            config['datasets'][0]['anchor_path'] = str(root / 'missing')
            with self.assertRaisesRegex(ValueError, 'anchor image folder does not exist'):
                FizgigSliderTrainer(0, None, config)

    def test_invalid_weight_rejected(self):
        config = {
            'type': 'fizgig_prompt_slider', 'model': {'arch': 'qwen_image_2'},
            'network': {'type': 'lora'}, 'train': {'noise_scheduler': 'flowmatch'},
        }
        for weight in (-1, float('inf'), float('nan')):
            config['fizgig_slider'] = {'preservation_weight': weight}
            with self.assertRaisesRegex(ValueError, 'Preservation Weight'):
                FizgigSliderTrainer(0, None, config)

    def test_hook_encodes_anchor_negative_only_above_cfg_one(self):
        for cfg in (1, 3):
            with self.subTest(cfg=cfg):
                config = {
                    'type': 'fizgig_prompt_slider', 'model': {'arch': 'qwen_image_2'},
                    'network': {'type': 'lora'}, 'train': {'noise_scheduler': 'flowmatch'},
                    'fizgig_slider': {'neutral_prompt': 'cat', 'positive_prompt': 'large cat', 'negative_prompt': 'small cat',
                                      'anchor_prompts': [{'prompt': 'dog', 'negative_prompt': 'blur'}], 'cfg_scale': cfg},
                }
                with patch.object(DiffusionTrainer, '__init__', return_value=None):
                    trainer = FizgigSliderTrainer(0, None, config)
                trainer.network_config = SimpleNamespace(type='lora')
                encoded = []

                def encode(prompt):
                    encoded.append(prompt[0])
                    return embed()

                trainer.sd = SimpleNamespace(set_device_state_preset=lambda preset: None, restore_device_state=lambda: None,
                                             encode_prompt=encode)
                with patch.object(DiffusionTrainer, 'hook_before_train_loop'), patch.object(trainer, '_build_prompt_bank'), patch.object(trainer, '_build_anchor_prompt_bank') as anchors:
                    trainer.hook_before_train_loop()
                anchors.assert_called_once()
                self.assertIn('dog', encoded)
                self.assertEqual('blur' in encoded, cfg > 1)
                self.assertEqual(trainer.slider_anchor_embeds[0][1] is not None, cfg > 1)

    def trainer(self):
        trainer = object.__new__(FizgigSliderTrainer)
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=0, max_denoising_steps=999)
        trainer.network = SimpleNamespace(multiplier=.7, is_active=False)
        trainer.slider_preservation_weight = 2
        trainer.slider_anchor_bank = [(torch.zeros(1, 1, 2, 2), embed(1), embed(0))]
        trainer.additional_logs = {}
        return trainer

    def test_reference_frozen_endpoint_and_intermediate_gradients_and_cfg_alignment(self):
        trainer = self.trainer()
        parameter = torch.nn.Parameter(torch.tensor(.5))
        predictions, backwards = [], []

        def predict(noisy, timestep, positive, negative):
            predictions.append((trainer.network.is_active, torch.is_grad_enabled(), noisy, timestep, positive, negative))
            return torch.ones_like(noisy) * (10 + parameter * trainer.network.multiplier * trainer.network.is_active)

        def backward(loss):
            backwards.append(trainer.network.multiplier)
            loss.backward()

        trainer._predict = predict
        trainer.accelerator = SimpleNamespace(backward=backward)
        with patch(MODULE + '.random.uniform', return_value=.5):
            loss = trainer._train_anchor_preservation(.25)
        self.assertEqual(backwards, [-1, 1, .5])
        self.assertAlmostEqual(loss.item(), .375, places=5)
        self.assertAlmostEqual(parameter.grad.item(), .375, places=5)
        self.assertEqual([p[:2] for p in predictions], [(False, False), (True, True), (True, True), (True, True)])
        for entry in predictions:
            self.assertIs(entry[2], predictions[0][2])
            self.assertIs(entry[3], predictions[0][3])
            self.assertIs(entry[4], predictions[0][4])
            self.assertIs(entry[5], predictions[0][5])
        self.assertFalse(trainer.network.is_active)
        self.assertEqual(trainer.network.multiplier, .7)
        self.assertAlmostEqual(trainer.additional_logs['slider/anchor_loss'], .375, places=5)

    def test_common_endpoint_drift_is_penalized(self):
        # Unlike matching -1 to +1, this catches both ends changing identically.
        trainer = self.trainer()
        drift = torch.nn.Parameter(torch.tensor(1.))
        trainer._predict = lambda noisy, *args: torch.ones_like(noisy) * (5 + drift * trainer.network.is_active)
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        loss = trainer._train_anchor_preservation(1)
        self.assertAlmostEqual(loss.item(), 2)
        self.assertAlmostEqual(drift.grad.item(), 4)

    def test_strength_restored_on_backward_failure(self):
        trainer = self.trainer()
        parameter = torch.nn.Parameter(torch.tensor(1.))
        trainer._predict = lambda noisy, *args: noisy + parameter * trainer.network.is_active
        trainer.accelerator = SimpleNamespace(backward=lambda loss: (_ for _ in ()).throw(RuntimeError('backward failed')))
        with self.assertRaisesRegex(RuntimeError, 'backward failed'):
            trainer._train_anchor_preservation(1)
        self.assertFalse(trainer.network.is_active)
        self.assertEqual(trainer.network.multiplier, .7)

    def test_anchor_loss_is_added_after_main_loss_for_both_slider_modes(self):
        for mode in ('image_pairs', 'prompt_pairs'):
            with self.subTest(mode=mode):
                trainer = self.trainer()
                trainer.slider_mode = mode
                main_parameter = torch.nn.Parameter(torch.tensor(0.))
                anchor_parameter = torch.nn.Parameter(torch.tensor(.5))
                trainer.slider_anchor_bank = [(torch.zeros(1, 1, 2, 2), embed(3), None)]
                backwards = []

                def backward(loss):
                    backwards.append(trainer.network.multiplier)
                    loss.backward()

                def predict(noisy, timestep, conditioning, negative=None):
                    value = conditioning.text_embeds[0, 0, 0]
                    parameter = anchor_parameter if value == 3 else main_parameter
                    return torch.ones_like(noisy) * (value + parameter * trainer.network.multiplier * trainer.network.is_active)

                trainer._predict = predict
                trainer.accelerator = SimpleNamespace(backward=backward)
                trainer.sd = SimpleNamespace(torch_dtype=torch.float32)
                if mode == 'image_pairs':
                    trainer._pair_key = lambda item: 'pair'
                    trainer.negative_latents = {'pair': torch.ones(1, 2, 2)}
                    trainer._image_loss = lambda clean, other, embeds: (main_parameter * trainer.network.multiplier - 1).square()
                    batch = SimpleNamespace(latents=torch.zeros(1, 1, 2, 2), file_items=[object()], prompt_embeds=embed())
                else:
                    trainer.slider_bank = [(torch.zeros(1, 1, 2, 2), 0)]
                    trainer.slider_embeds = [(embed(0), embed(1), embed(-1))]
                    trainer.slider_cfg_negative_embeds = None
                    trainer.slider_guidance = 1
                    batch = None
                with patch(MODULE + '.random.uniform', return_value=.5):
                    loss = trainer.train_single_accumulation(batch, accum_scale=.25)
                self.assertEqual(backwards, [1, -1, -1, 1, .5])
                self.assertAlmostEqual(loss.item(), (1 if mode == 'image_pairs' else 4) + .375, places=5)
                self.assertIsNotNone(main_parameter.grad)
                self.assertAlmostEqual(anchor_parameter.grad.item(), .375, places=5)
                self.assertFalse(trainer.network.is_active)
                self.assertEqual(trainer.network.multiplier, .7)

    def test_zero_weight_skips_anchor_cache_and_render_setup(self):
        trainer = self.trainer()
        trainer.network_config = SimpleNamespace(type='lora')
        trainer.slider_mode = 'image_pairs'
        trainer.slider_preservation_weight = 0
        trainer.slider_anchor_prompts = [('dog', 'blur')]
        trainer.slider_anchor_image_configs = [{'folder_path': '/anchors'}]
        trainer.slider_anchor_embeds = []
        with patch.object(trainer, '_cache_negative_latents'), patch.object(trainer, '_cache_anchor_images') as images, patch.object(trainer, '_build_anchor_prompt_bank') as prompts, patch.object(DiffusionTrainer, 'hook_before_train_loop'):
            trainer.hook_before_train_loop()
        images.assert_not_called()
        prompts.assert_not_called()

    def test_image_cache_uses_normal_loader_cpu_banks_and_cleanup(self):
        trainer = self.trainer()
        trainer.slider_anchor_image_configs = [{'folder_path': '/anchors'}]
        trainer.slider_anchor_bank = []
        events = []
        item = SimpleNamespace(
            get_latent=lambda: torch.ones(1, 2, 2), prompt_embeds=embed(),
            load_prompt_embedding=lambda: events.append('load embeds'),
            cleanup_latent=lambda: events.append('clean latents'),
            cleanup_text_embedding=lambda: events.append('clean embeds'),
        )
        trainer.sd = SimpleNamespace(text_encoder_to=lambda device: events.append(str(device)))
        trainer.print = lambda *args: None
        with patch(MODULE + '.get_dataloader_from_datasets', return_value='loader') as make_loader, patch(
            MODULE + '.get_dataloader_datasets', return_value=[SimpleNamespace(file_list=[item])],
        ):
            trainer._cache_anchor_images()
        make_loader.assert_called_once_with(trainer.slider_anchor_image_configs, 1, trainer.sd)
        self.assertEqual(events, ['load embeds', 'clean latents', 'clean embeds', 'cpu'])
        latent, positive, negative = trainer.slider_anchor_bank[0]
        self.assertEqual(latent.shape, (1, 1, 2, 2))
        self.assertEqual(positive.text_embeds.device.type, 'cpu')
        self.assertIsNone(negative)

    def test_real_image_loader_caches_anchor_captions_and_refreshes_modified_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (64, 64), 'black').save(root / 'dog.png')
            (root / 'dog.txt').write_text('a medium sized dog')
            trainer = self.trainer()
            trainer.slider_anchor_bank = []
            trainer.slider_anchor_image_configs = [{
                'folder_path': directory, 'resolution': [64], 'num_workers': 0,
                'cache_latents_to_disk': True, 'cache_text_embeddings': True,
                'fizgig_slider_anchor': True, 'caption_dropout_rate': 0,
            }]
            sd = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='unused', vae_dtype='fp32'), dtype='fp32')
            sd.vae = torch.nn.Identity()
            sd.set_device_state_preset = lambda preset: None
            sd.restore_device_state = lambda: None
            sd.text_encoder_to = lambda device: None
            encoded = []
            sd.encode_prompt = lambda caption, **kwargs: encoded.append(caption) or embed()
            sd.encode_images = lambda pixels: pixels.mean().expand(1, 2, 4, 4).clone()
            trainer.sd = sd
            trainer.print = lambda *args: None
            trainer._cache_anchor_images()
            self.assertEqual(encoded, ['a medium sized dog'])
            self.assertEqual(trainer.slider_anchor_bank[0][0].shape, (1, 2, 4, 4))
            self.assertAlmostEqual(trainer.slider_anchor_bank[0][0].mean().item(), -1)
            Image.new('RGB', (64, 64), 'white').save(root / 'dog.png')
            trainer.slider_anchor_bank = []
            trainer._cache_anchor_images()
            self.assertAlmostEqual(trainer.slider_anchor_bank[0][0].mean().item(), 1)
            self.assertEqual(encoded, ['a medium sized dog'])

    def test_prompt_bank_uses_cfg_negatives_disabled_adapter_and_separate_seed(self):
        trainer = self.trainer()
        trainer.slider_anchor_bank = []
        trainer.slider_anchor_embeds = [(embed(), embed(0))]
        trainer.slider_bank_resolution = 64
        trainer.slider_bank_steps = 1
        trainer.slider_cfg = 3
        trainer.sample_config = SimpleNamespace(seed=42)
        calls = []

        def pipeline(positive, **kwargs):
            self.assertTrue(trainer.sd.assistant_lora.is_active)
            calls.append((trainer.network.is_active, positive, kwargs))
            return [Image.new('RGB', (64, 64))]

        trainer.sd = SimpleNamespace(
            save_device_state=lambda: None, restore_device_state=lambda: None, assistant_lora=SimpleNamespace(is_active=True),
            text_encoder_to=lambda device: None, unet=SimpleNamespace(to=lambda device: None),
            vae=SimpleNamespace(to=lambda device: None), vae_device_torch=torch.device('cpu'),
            torch_dtype=torch.float32, vae_torch_dtype=torch.float32,
            get_generation_pipeline=lambda: pipeline, encode_images=lambda pixels: torch.ones(1, 1, 2, 2),
        )
        trainer._build_anchor_prompt_bank()
        self.assertFalse(calls[0][0])
        self.assertEqual(calls[0][2]['guidance_scale'], 3)
        self.assertIsNotNone(calls[0][2]['unconditional_embeds'])
        self.assertEqual(calls[0][2]['generator'].initial_seed(), 100042)
        self.assertEqual(len(trainer.slider_anchor_bank), 1)
        self.assertEqual(trainer.slider_anchor_bank[0][0].device.type, 'cpu')

    def test_anchor_image_latent_key_tracks_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dog.png'
            path.write_bytes(b'first')
            item = SimpleNamespace(
                path=str(path), scale_to_width=64, scale_to_height=64,
                crop_x=0, crop_y=0, crop_width=64, crop_height=64,
                latent_space_version='qwen_image_2', latent_version=1,
                dataset_config=DatasetConfig(fizgig_slider_anchor=True),
                flip_x=False, flip_y=False, is_video=False, is_audio_model=False,
            )
            first = LatentCachingFileItemDTOMixin.get_latent_info_dict(item)
            path.write_bytes(b'changed anchor')
            second = LatentCachingFileItemDTOMixin.get_latent_info_dict(item)
            self.assertNotEqual(first['source_file_identity_v1'], second['source_file_identity_v1'])


if __name__ == '__main__':
    unittest.main()
