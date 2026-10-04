import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.FizgigSliderTrainer import FizgigSliderTrainer, pair_difference_weights
from extensions_built_in.sd_trainer.fizgig_multipoint import (
    parse_points, parse_multipoint_prompts, validate_multipoint_images,
)
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model
from toolkit.config_modules import ModelConfig
from toolkit.data_loader import get_dataloader_from_datasets, get_dataloader_datasets
from toolkit.models.DoRA import DoRAModule
from toolkit.prompt_utils import PromptEmbeds

MODULE = 'extensions_built_in.sd_trainer.FizgigSliderTrainer'


def embed(value=0):
    return PromptEmbeds(torch.full((1, 2, 3), float(value)))


def multipoint(zero=False):
    points = [dict(id=ident, strength=strength, prefix=prefix, negative_prefix=negative)
              for ident, strength, prefix, negative in [('baby', -1, 'baby', 'adult'), ('adult', 1, 'adult', 'baby'), ('mega', 2, 'mega', 'small')]]
    if zero:
        points.append(dict(id='zero', strength=0, prefix='', negative_prefix='noise'))
    return dict(points=points, neutral_negative_prefix='blur', prompt_entries=[dict(kind='simple', prompt='cat')])


def process_config(mode='prompt', mp=None):
    return dict(type=f'fizgig_{mode}_slider', model=dict(arch='qwen_image_2'), network=dict(type='dora'),
                train=dict(noise_scheduler='flowmatch'), fizgig_slider=dict(multipoint=True, multipoint_config=mp or multipoint()))


class MultipointTests(unittest.TestCase):
    def test_points_numeric_sort_fractions_and_invalid_values(self):
        config = multipoint(True)
        config['points'][0]['strength'] = -.25
        self.assertEqual([p['strength'] for p in parse_points(config)], [-.25, 0, 1, 2])
        for strength in [float('nan'), float('inf'), '1', True, None]:
            invalid = copy.deepcopy(config)
            invalid['points'][0]['strength'] = strength
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                parse_points(invalid)
        for points in [[], [dict(id='zero', strength=0)], [dict(id='a', strength=1), dict(id='b', strength=1.0)],
                       [dict(id='a', strength=1), dict(id='a', strength=2)], [dict(id='a', strength=0), dict(id='b', strength=-0.0)]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                parse_points(dict(points=points))

    def test_simple_neutral_and_target_assembly_with_zero(self):
        config = multipoint()
        entry = parse_multipoint_prompts(config, parse_points(config))[0]
        self.assertEqual(entry['neutral'], ('cat', 'blur\n\ncat'))
        self.assertEqual(entry['targets']['baby'], ('baby\n\ncat', 'adult\n\ncat'))
        config = multipoint(True)
        entry = parse_multipoint_prompts(config, parse_points(config))[0]
        self.assertEqual(entry['neutral'], ('cat', 'noise\n\ncat'))
        config['points'][-1]['prefix'] = 'awkward'
        self.assertEqual(parse_multipoint_prompts(config, parse_points(config))[0]['neutral'][0], 'awkward\n\ncat')
        config['points'][0]['prefix'] = ''
        with self.assertRaisesRegex(ValueError, 'positive prefix'):
            parse_multipoint_prompts(config, parse_points(config))

    def test_specific_explicit_zero_overrides_implicit_neutral(self):
        config = multipoint(True)
        entry = dict(kind='specific', neutral_prompt='ignored', neutral_negative_prompt='ignored negative',
                     targets=[dict(point_id=p['id'], prompt=p['id'], negative_prompt=f"no {p['id']}") for p in config['points']])
        config['prompt_entries'] = [entry]
        parsed = parse_multipoint_prompts(config, parse_points(config))[0]
        self.assertEqual(parsed['neutral'], ('zero', 'no zero'))
        for targets in [entry['targets'][:-1], entry['targets'][:-1] + [entry['targets'][0]],
                        entry['targets'][:-1] + [dict(point_id='unknown', prompt='x')]]:
            invalid = {**config, 'prompt_entries': [{**entry, 'targets': targets}]}
            with self.assertRaises(ValueError):
                parse_multipoint_prompts(invalid, parse_points(config))

    def test_constructor_validates_before_model_initialization_and_retains_anchors(self):
        config = process_config(mp=multipoint(True))
        config['fizgig_slider']['anchor_prompts'] = [dict(prompt='dog')]
        with patch.object(DiffusionTrainer, '__init__', return_value=None) as parent:
            trainer = FizgigSliderTrainer(0, None, config)
        self.assertEqual(parent.call_count, 1)
        self.assertEqual(trainer.slider_prompt_triplets, [('cat',)])
        self.assertEqual(trainer.slider_anchor_prompts, [('dog', '')])
        working = parent.call_args.args[2]
        self.assertEqual(working['datasets'], [])
        self.assertTrue(working['train']['unload_text_encoder'])
        self.assertNotIn('datasets', config)
        invalid = process_config()
        invalid['fizgig_slider']['multipoint_config']['points'][2]['prefix'] = ''
        with patch.object(DiffusionTrainer, '__init__', return_value=None) as parent, self.assertRaises(ValueError):
            FizgigSliderTrainer(0, None, invalid)
        parent.assert_not_called()

    def test_legacy_ignores_invalid_multipoint_draft(self):
        config = process_config()
        config['fizgig_slider'].update(multipoint=False, multipoint_config={'broken': True},
                                      neutral_prompt='cat', positive_prompt='large cat', negative_prompt='small cat')
        with patch.object(DiffusionTrainer, '__init__', return_value=None):
            trainer = FizgigSliderTrainer(0, None, config)
        self.assertFalse(trainer.slider_multipoint)
        self.assertEqual(trainer.slider_prompt_triplets, [('cat', 'large cat', 'small cat')])

    @staticmethod
    def trainer(mode='prompt'):
        trainer = object.__new__(FizgigSliderTrainer)
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=0, max_denoising_steps=999)
        trainer.network = SimpleNamespace(multiplier=.7, is_active=False)
        trainer.sd = SimpleNamespace(torch_dtype=torch.float32)
        trainer.slider_multipoint = True
        trainer.slider_mode = 'prompt_pairs' if mode == 'prompt' else 'image_pairs'
        trainer.slider_points = parse_points(multipoint(True))
        trainer.slider_bank = [(torch.zeros(1, 1, 2, 2), 0)]
        trainer.slider_embeds = [(embed(5),)]
        trainer.slider_cfg_negative_embeds = [(embed(6),)]
        trainer.slider_target_embeds = [{ident: (embed(value), embed(-value)) for ident, value in [('baby', -3), ('adult', 2), ('mega', 7)]}]
        trainer.slider_anchor_bank = []
        trainer.slider_preservation_weight = 1
        trainer.additional_logs = {}
        return trainer

    def test_prompt_direct_targets_balanced_gradients_and_backward_strengths(self):
        trainer = self.trainer()
        parameter = torch.nn.Parameter(torch.tensor(.5))
        records, backwards = [], []
        noisy, timestep = torch.zeros(1, 1, 2, 2), torch.tensor([500.])
        trainer._noised_state = lambda latent: (noisy, timestep, None)
        def predict(x, t, positive, negative):
            records.append((trainer.network.is_active, torch.is_grad_enabled(), x, t,
                            positive.text_embeds.mean().item(), negative.text_embeds.mean().item()))
            return torch.ones_like(x) * (parameter * trainer.network.multiplier if trainer.network.is_active else positive.text_embeds.mean())
        def backward(loss):
            backwards.append(trainer.network.multiplier)
            loss.backward()
        trainer._predict = predict
        trainer.accelerator = SimpleNamespace(backward=backward)
        loss = trainer.train_single_accumulation(None, accum_scale=.25)
        expected = sum((.5 * strength - target) ** 2 for strength, target in [(-1, -3), (1, 2), (2, 7)]) / 3
        expected_grad = sum(2 * (.5 * strength - target) * strength for strength, target in [(-1, -3), (1, 2), (2, 7)]) / 3 * .25
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertAlmostEqual(parameter.grad.item(), expected_grad, places=5)
        self.assertEqual(backwards, [-1, 1, 2])
        self.assertEqual([r[:2] for r in records], [(False, False), (True, True)] * 3)
        self.assertEqual([r[4:] for r in records[::2]], [(-3, 3), (2, -2), (7, -7)])
        self.assertEqual([r[4:] for r in records[1::2]], [(5, 6)] * 3)
        self.assertTrue(all(r[2] is noisy and r[3] is timestep for r in records))
        self.assertEqual(trainer.network.multiplier, .7)
        self.assertFalse(trainer.network.is_active)

    def test_multipoint_network_restored_on_backward_exception(self):
        trainer = self.trainer()
        trainer._predict = lambda x, *args: x + torch.tensor(1., requires_grad=True)
        trainer.accelerator = SimpleNamespace(backward=lambda loss: (_ for _ in ()).throw(RuntimeError('failure')))
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            trainer.train_single_accumulation(None)
        self.assertEqual(trainer.network.multiplier, .7)
        self.assertFalse(trainer.network.is_active)

    def test_checkpoint_recomputation_keeps_each_point_strength_until_backward_finishes(self):
        from torch.utils.checkpoint import checkpoint
        trainer = self.trainer()
        parameter = torch.nn.Parameter(torch.tensor(.5))
        recomputations = []
        def predict(noisy, timestep, positive, negative):
            if not trainer.network.is_active:
                return torch.ones_like(noisy) * positive.text_embeds.mean()
            def forward(x, p):
                recomputations.append((trainer.network.multiplier, torch.is_grad_enabled()))
                return x + (p * trainer.network.multiplier).square()
            return checkpoint(forward, noisy, parameter, use_reentrant=True)
        trainer._predict = predict
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        trainer.train_single_accumulation(None)
        self.assertEqual(recomputations, [(-1, False), (-1, True), (1, False), (1, True), (2, False), (2, True)])

    def test_anchor_checks_full_range_averaged_and_skips_actual_zero(self):
        for intermediate, strengths in [(1.5, [-1, 1, 2, 1.5]), (0, [-1, 1, 2])]:
            trainer = self.trainer()
            trainer.slider_preservation_weight = 2
            trainer.slider_anchor_bank = [(torch.zeros(1, 1, 2, 2), embed(1), None)]
            parameter = torch.nn.Parameter(torch.tensor(.5))
            backwards = []
            trainer._predict = lambda x, *args: x + parameter * trainer.network.multiplier * trainer.network.is_active
            def backward(loss):
                backwards.append(trainer.network.multiplier)
                loss.backward()
            trainer.accelerator = SimpleNamespace(backward=backward)
            with patch(MODULE + '.random.uniform', return_value=intermediate) as uniform:
                loss = trainer._train_anchor_preservation(.25)
            uniform.assert_called_with(-1, 2)
            self.assertEqual(backwards, strengths)
            self.assertAlmostEqual(loss.item(), 2 * sum((.5 * s) ** 2 for s in strengths) / len(strengths), places=5)
            self.assertAlmostEqual(parameter.grad.item(), .5 * sum(s ** 2 for s in strengths) / len(strengths), places=5)

    def test_prompt_embeds_cached_before_parent_hook_and_cfg_gated(self):
        for cfg in (1, 3):
            trainer = self.trainer()
            trainer.slider_cfg = cfg
            trainer.slider_anchor_prompts = []
            trainer.slider_anchor_image_configs = []
            trainer.network_config = SimpleNamespace(type='dora')
            config = multipoint(True)
            trainer.slider_multipoint_prompts = parse_multipoint_prompts(config, trainer.slider_points)
            trainer.slider_prompt_triplets = [('cat',)]
            trainer.slider_cfg_negative_prompts = [('noise\n\ncat',)] if cfg > 1 else None
            trainer.slider_cfg_negative_embeds = None
            encoded = []
            trainer.sd.set_device_state_preset = lambda *args: None
            trainer.sd.restore_device_state = lambda: None
            trainer.sd.encode_prompt = lambda prompt: encoded.append(prompt[0]) or embed()
            trainer._build_prompt_bank = lambda: None
            def parent(instance):
                self.assertEqual(len(instance.slider_target_embeds[0]), 3)
                self.assertTrue(all(e.device.type == 'cpu' for pair in instance.slider_target_embeds[0].values() for e in (p.text_embeds for p in pair if p is not None)))
            with patch.object(DiffusionTrainer, 'hook_before_train_loop', parent):
                trainer.hook_before_train_loop()
            self.assertTrue(trainer.network.signed_dora_slider)
            self.assertIn('mega\n\ncat', encoded)
            self.assertEqual('small\n\ncat' in encoded, cfg > 1)
            self.assertNotIn('dog', encoded)

    def test_practice_count_is_entries_not_points_and_neutral_cfg(self):
        trainer = self.trainer()
        trainer.slider_prompt_triplets = [('cat',), ('awkward cat',)]
        trainer.slider_embeds.append((embed(9),))
        trainer.slider_cfg_negative_embeds.append((embed(10),))
        trainer.slider_bank = []
        trainer.slider_bank_size, trainer.slider_bank_resolution, trainer.slider_bank_steps = 4, 64, 1
        trainer.slider_cfg = 3
        trainer.sample_config = SimpleNamespace(seed=42)
        calls = []
        def pipeline(positive, **kwargs):
            calls.append((trainer.network.is_active, positive.text_embeds.mean().item(), kwargs['unconditional_embeds'].text_embeds.mean().item()))
            return [Image.new('RGB', (64, 64))]
        trainer.sd = SimpleNamespace(assistant_lora=None, save_device_state=lambda: None, restore_device_state=lambda: None,
            text_encoder_to=lambda *args: None, unet=SimpleNamespace(to=lambda *args: None), vae=SimpleNamespace(to=lambda *args: None),
            vae_device_torch=torch.device('cpu'), torch_dtype=torch.float32, vae_torch_dtype=torch.float32,
            get_generation_pipeline=lambda: pipeline, encode_images=lambda pixels: torch.zeros(1, 1, 2, 2))
        trainer.print = lambda *args: None
        trainer._build_prompt_bank()
        self.assertEqual(calls, [(False, 5, 6), (False, 9, 10)] * 2)
        self.assertEqual([index for latent, index in trainer.slider_bank], [0, 1, 0, 1])

    @staticmethod
    def image_dataset(root, config):
        mappings = []
        for point in config['points']:
            folder = root / point['id']
            folder.mkdir()
            Image.new('RGB', (128, 64), 'black').save(folder / 'cat.png')
            (folder / 'ignored.png.disabled').write_bytes(b'not an image')
            (folder / 'cat.txt').write_text(point['id'] + ' neutral caption')
            mappings.append(dict(point_id=point['id'], folder_path=str(folder)))
        return dict(folder_path=str(root / 'adult'), control_path_1=str(root / 'baby'), resolution=[64],
                    multipoint_images=mappings, cache_latents_to_disk=True, cache_text_embeddings=True,
                    flip_x=True, num_workers=0, caption_dropout_rate=0)

    def test_pair_validation_dimensions_missing_ambiguous_disabled_and_neutral_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            config = multipoint(True)
            dataset = self.image_dataset(Path(directory), config)
            points = parse_points(config)
            self.assertEqual(validate_multipoint_images([dataset], points)[0]['reference_id'], 'zero')
            without_zero = [p for p in points if p['strength'] != 0]
            dataset['multipoint_images'] = [m for m in dataset['multipoint_images'] if m['point_id'] != 'zero']
            self.assertEqual(validate_multipoint_images([dataset], without_zero)[0]['reference_id'], 'adult')
            without_adult = [p for p in without_zero if p['strength'] != 1]
            fallback = {**dataset, 'multipoint_images': [m for m in dataset['multipoint_images'] if m['point_id'] != 'adult']}
            self.assertEqual(validate_multipoint_images([fallback], without_adult)[0]['reference_id'], 'baby')
            path = Path(directory) / 'mega' / 'cat.png'
            Image.new('RGB', (64, 64)).save(path)
            with self.assertRaisesRegex(ValueError, 'different dimensions'):
                validate_multipoint_images([dataset], without_zero)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'no usable'):
                validate_multipoint_images([dataset], without_zero)
            Image.new('RGB', (128, 64)).save(path)
            Image.new('RGB', (128, 64)).save(path.with_suffix('.jpg'))
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                validate_multipoint_images([dataset], without_zero)

    def test_real_image_cache_preserves_crop_flips_and_mtime_and_uses_zero_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = multipoint(True)
            dataset = self.image_dataset(Path(directory), config)
            job = process_config('image', config)
            job['datasets'] = [dataset]
            with patch.object(DiffusionTrainer, '__init__', return_value=None) as parent:
                trainer = FizgigSliderTrainer(0, None, job)
            working_dataset = parent.call_args.args[2]['datasets'][0]
            self.assertEqual(working_dataset['folder_path'], str(Path(directory) / 'zero'))
            self.assertIsNone(working_dataset['control_path_1'])
            self.assertEqual(dataset['folder_path'], str(Path(directory) / 'adult'))
            self.assertEqual(dataset['control_path_1'], str(Path(directory) / 'baby'))
            sd = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='unused', vae_dtype='fp32'), dtype='fp32')
            sd.vae = torch.nn.Identity()
            sd.set_device_state_preset = lambda *args: None
            sd.restore_device_state = lambda: None
            sd.text_encoder_to = lambda *args: None
            captions = []
            sd.encode_prompt = lambda caption, **kwargs: captions.append(caption) or embed()
            sd.encode_images = lambda pixels: pixels.mean().expand(1, 2, 4, 4).clone()
            trainer.sd, trainer.print = sd, lambda *args: None
            trainer.data_loader = get_dataloader_from_datasets([working_dataset], 1, sd)
            trainer._cache_multipoint_image_latents()
            reference_items = get_dataloader_datasets(trainer.data_loader)[0].file_list
            expected_keys = {trainer._multipoint_key(item) for item in reference_items}
            for bank in trainer.slider_multipoint_latents.values():
                self.assertEqual(set(bank), expected_keys)
                self.assertTrue(all(latent.device.type == 'cpu' for latent in bank.values()))
                self.assertTrue(all(latent.mean().item() == -1 for latent in bank.values()))
            self.assertEqual(captions, ['zero neutral caption'])
            Image.new('RGB', (128, 64), 'white').save(Path(directory) / 'mega' / 'cat.png')
            trainer.slider_multipoint_latents = {}
            trainer._cache_multipoint_image_latents()
            self.assertTrue(all(latent.mean().item() == 1 for latent in trainer.slider_multipoint_latents['mega'].values()))
            self.assertTrue(all(latent.mean().item() == -1 for latent in trainer.slider_multipoint_latents['zero'].values()))
            self.assertEqual(captions, ['zero neutral caption'])

    def test_image_losses_balanced_and_zero_never_trains(self):
        for with_zero in (False, True):
            trainer = self.trainer('image')
            trainer.slider_diff_weight = 1
            item = SimpleNamespace(path='cat.png', scale_to_width=2, scale_to_height=2, crop_x=0, crop_y=0,
                                   crop_width=2, crop_height=2, flip_x=False, flip_y=False)
            key = trainer._multipoint_key(item)
            trainer.slider_multipoint_latents = {ident: {key: torch.full((1, 2, 2), float(value))}
                for ident, value in [('baby', -2), ('adult', 2), ('mega', 5), ('zero', 0)]}
            if not with_zero:
                trainer.slider_points = [p for p in trainer.slider_points if p['strength'] != 0]
            parameter = torch.nn.Parameter(torch.tensor(.5))
            backwards, conditioned = [], []
            trainer._noised_state = lambda clean: (clean, torch.tensor([500.]), -clean)
            def predict(noisy, timestep, positive):
                conditioned.append(positive.text_embeds.mean().item())
                return torch.ones_like(noisy) * parameter * trainer.network.multiplier
            def backward(loss):
                backwards.append(trainer.network.multiplier)
                loss.backward()
            trainer._predict = predict
            trainer.accelerator = SimpleNamespace(backward=backward)
            batch = SimpleNamespace(latents=torch.zeros(1, 1, 2, 2), file_items=[item], prompt_embeds=embed(6))
            loss = trainer.train_single_accumulation(batch, .25)
            expected = sum((.5 * s + clean) ** 2 for s, clean in [(-1, -2), (1, 2), (2, 5)]) / 3
            expected_grad = sum(2 * (.5 * s + clean) * s for s, clean in [(-1, -2), (1, 2), (2, 5)]) / 3 * .25
            self.assertEqual(backwards, [-1, 1, 2])
            self.assertEqual(conditioned, [6] * 3)
            self.assertAlmostEqual(loss.item(), expected, places=5)
            self.assertAlmostEqual(parameter.grad.item(), expected_grad, places=5)

    def test_same_reference_folder_in_two_groups_has_distinct_bank_keys(self):
        base = dict(path='cat.png', scale_to_width=2, scale_to_height=2, crop_x=0, crop_y=0,
                    crop_width=2, crop_height=2, flip_x=False, flip_y=False)
        first = SimpleNamespace(**base, dataset_config=SimpleNamespace(fizgig_multipoint_group=0))
        second = SimpleNamespace(**base, dataset_config=SimpleNamespace(fizgig_multipoint_group=1))
        self.assertNotEqual(FizgigSliderTrainer._multipoint_key(first), FizgigSliderTrainer._multipoint_key(second))

    def test_real_cache_multiresolution_random_crops_and_flips_are_identical_for_identical_images(self):
        with tempfile.TemporaryDirectory() as directory:
            config = multipoint()
            dataset = self.image_dataset(Path(directory), config)
            image = Image.new('RGB', (128, 64))
            image.putdata([(x * 2, y * 4, (x + y) % 256) for y in range(64) for x in range(128)])
            for mapping in dataset['multipoint_images']:
                image.save(Path(mapping['folder_path']) / 'cat.png')
            dataset.update(resolution=[64, 96], random_crop=True, cache_text_embeddings=False)
            job = process_config('image', config)
            job['datasets'] = [dataset]
            with patch.object(DiffusionTrainer, '__init__', return_value=None) as parent:
                trainer = FizgigSliderTrainer(0, None, job)
            working_dataset = parent.call_args.args[2]['datasets'][0]
            sd = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='unused', vae_dtype='fp32'), dtype='fp32')
            sd.vae = torch.nn.Identity()
            sd.set_device_state_preset = lambda *args: None
            sd.restore_device_state = lambda: None
            sd.encode_images = lambda pixels: pixels.mean().expand(1, 2, 4, 4).clone()
            trainer.sd, trainer.print = sd, lambda *args: None
            trainer.data_loader = get_dataloader_from_datasets([working_dataset], 1, sd)
            trainer._cache_multipoint_image_latents()
            originals = [item for actual in get_dataloader_datasets(trainer.data_loader) for item in actual.file_list]
            self.assertEqual(len(originals), 4)  # Two resolutions and x flips.
            for original in originals:
                key = trainer._multipoint_key(original)
                expected = original.get_latent()
                for bank in trainer.slider_multipoint_latents.values():
                    self.assertTrue(torch.equal(bank[key], expected))
                original.cleanup_latent()

    def test_signed_dora_fraction_and_outlying_strength_match_comfy_delta(self):
        class Network:
            signed_dora_slider = True
            is_active = True
            is_lorm = False
            is_merged_in = False
            _multiplier = 1.
            torch_multiplier = torch.tensor([1.])
        network = Network()
        layer = torch.nn.Linear(4, 3, bias=False)
        module = DoRAModule('slider', layer, network=network, lora_dim=2, alpha=2)
        with torch.no_grad():
            module.lora_up.weight.fill_(.2)
            module.lora_down.weight.fill_(.3)
        x = torch.randn(2, 4)
        # Keep the adapter's original module forwarding intact.
        module.apply_to()
        network._multiplier, network.torch_multiplier = 0, torch.tensor([0.])
        baseline = layer(x).detach()
        network._multiplier, network.torch_multiplier = 1, torch.tensor([1.])
        delta = layer(x).detach() - baseline
        for strength in [-.25, 0, 1.5, 2, 3]:
            network._multiplier, network.torch_multiplier = strength, torch.tensor([float(strength)])
            self.assertTrue(torch.allclose(layer(x), baseline + strength * delta, atol=1e-5))

    def test_spatial_weights_use_zero_or_shared_spread_and_identical_fallback(self):
        for with_zero in (False, True):
            trainer = self.trainer('image')
            trainer.slider_diff_weight = 1
            item = SimpleNamespace(path='cat.png', scale_to_width=2, scale_to_height=2, crop_x=0, crop_y=0,
                                   crop_width=2, crop_height=2, flip_x=False, flip_y=False)
            key = trainer._multipoint_key(item)
            clean = {ident: torch.zeros(1, 1, 2, 2) for ident in ['baby', 'adult', 'mega', 'zero']}
            clean['adult'][0, 0, 0, 0] = 2
            clean['mega'][0, 0, 1, 1] = 4
            trainer.slider_multipoint_latents = {ident: {key: pixels[0]} for ident, pixels in clean.items()}
            if not with_zero:
                trainer.slider_points = [p for p in trainer.slider_points if p['strength'] != 0]
            parameter = torch.nn.Parameter(torch.tensor(.5))
            trainer._noised_state = lambda pixels: (pixels, torch.tensor([500.]), -pixels)
            trainer._predict = lambda pixels, *args: torch.ones_like(pixels) * parameter * trainer.network.multiplier
            trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
            expected = 0
            spread = torch.maximum(clean['adult'], clean['mega'])
            for strength, ident in [(-1, 'baby'), (1, 'adult'), (2, 'mega')]:
                weights = pair_difference_weights(clean[ident], clean['zero'], 1) if with_zero else pair_difference_weights(spread, clean['baby'], 1)
                self.assertTrue(torch.isfinite(weights).all())
                self.assertAlmostEqual(weights.mean().item(), 1)
                expected += ((.5 * strength + clean[ident]).square().flatten(1) * weights).mean().item() / 3
            batch = SimpleNamespace(latents=clean['zero'], file_items=[item], prompt_embeds=embed())
            self.assertAlmostEqual(trainer.train_single_accumulation(batch).item(), expected, places=5)


if __name__ == '__main__':
    unittest.main()
