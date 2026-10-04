import tempfile
import hashlib
import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn

from toolkit.config_modules import NetworkConfig, GenerateImageConfig, SampleConfig
from toolkit.metadata import load_metadata_from_safetensors
from toolkit.models.sliderspace_network import SliderSpaceNetwork
from toolkit.sliderspace import (SliderSpaceConfig, SemanticEncoder, discovery_signature,
                                parse_auto_sample_strengths,
                                discover_directions, flow_clean_prediction, semantic_direction_loss,
                                scan_discovery_images, discovery_entries, prepare_discovery_image,
                                discovery_image_size, verify_discovery_sources)
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.SliderSpaceTrainer import SliderSpaceTrainer


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


def make_bank(num_directions=3):
    model = nn.Sequential(TinyBlock()).requires_grad_(False)
    config = NetworkConfig(type='lora', linear=2, linear_alpha=1)
    bank = SliderSpaceNetwork(text_encoder=[], unet=model, num_directions=num_directions,
                              lora_dim=2, alpha=1, network_type='lora', network_config=config,
                              train_unet=True, train_text_encoder=False, target_lin_modules=['TinyBlock'])
    bank.apply_to([], model, False, True)
    bank.force_to('cpu', torch.float32)
    bank._update_torch_multiplier()
    bank.is_active = True
    return bank, model


class SliderSpaceMathTests(unittest.TestCase):
    def test_config_defaults_and_strict_invalid_values(self):
        self.assertEqual(SliderSpaceConfig.parse({'concept_prompts': ['  cat ']}).concept_prompts, ['cat'])
        for override in ({'concept_prompts': ['']}, {'num_directions': True}, {'num_directions': 0},
                         {'discovery_samples': 4}, {'resolution': 129}, {'cfg_scale': 0.5},
                         {'loss_weight': 0}, {'preview_strength': float('nan')}, {'preview_direction': 5}, {'preview_auto': 1},
                         {'feature_device': 'mps'}, {'seed': -1}, {'extra': 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                SliderSpaceConfig.parse({'concept_prompts': ['cat'], **override})
        with self.assertRaises(ValueError):
            SliderSpaceConfig.parse({'concept_prompts': ['cat'], 'discovery_buckets': 1})

    def test_auto_strengths_parse_finite_csv_in_order_and_render_zero_once(self):
        config = SliderSpaceConfig.parse({'concept_prompts': ['cat']})
        self.assertEqual(parse_auto_sample_strengths(config.preview_auto_strengths), [-1, 1])
        self.assertEqual(parse_auto_sample_strengths(' -.5, 0, +.25, 1.5, .25, -0, 2e-2 '), [-.5, .25, 1.5, .02])
        self.assertEqual(parse_auto_sample_strengths('0'), [])
        for value in ('', ' ', '1,', ',1', '1,,2', 'cat', 'NaN', 'Infinity', '1e309', '0x10', None, [1], True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SliderSpaceConfig.parse({'concept_prompts': ['cat'], 'preview_auto_strengths': value})

    def test_flow_clean_prediction_is_exact(self):
        clean, noise = torch.randn(2, 4, 3, 3), torch.randn(2, 4, 3, 3)
        time = torch.tensor([0.3, 0.8])
        noisy = (1 - time[:, None, None, None]) * clean + time[:, None, None, None] * noise
        torch.testing.assert_close(flow_clean_prediction(noisy, noise - clean, time * 1000), clean)

    def test_pca_deterministic_orthonormal_and_rank_failure(self):
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(24, 8, generator=generator)
        directions, variance, scores = discover_directions(features, 4)
        torch.testing.assert_close(directions @ directions.T, torch.eye(4))
        torch.testing.assert_close(directions, discover_directions(features, 4)[0])
        self.assertEqual(scores.shape, (24, 4))
        self.assertTrue(torch.all(variance > 0))
        self.assertLessEqual(variance.sum().item(), 1)
        for features in (torch.ones(10, 8), torch.full((10, 8), float('nan'))):
            with self.assertRaises(ValueError):
                discover_directions(features, 4)

    def test_zero_delta_has_useful_gradient_and_teacher_is_detached(self):
        base = torch.randn(1, 8, requires_grad=True)
        student = base.detach().clone().requires_grad_()
        loss = semantic_direction_loss(student, base, torch.ones(8))
        self.assertEqual(loss.item(), 1)
        loss.backward()
        self.assertIsNone(base.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertAlmostEqual(semantic_direction_loss(base.detach() + 1, base, torch.ones(8)).item(), 0, places=6)

    def test_cache_identity_ignores_preview_but_tracks_nested_checkpoint_mtime(self):
        config = SliderSpaceConfig.parse({'concept_prompts': ['cat']})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'transformer' / 'model.safetensors'
            path.parent.mkdir()
            path.write_bytes(b'checkpoint')
            model = {'arch': 'qwen_image_2', 'name_or_path': directory}
            first = discovery_signature(config, model)
            config.preview_direction, config.preview_strength = 2, -0.5
            config.preview_auto = True
            config.preview_auto_strengths = '-0.5, 0.5, 2'
            self.assertEqual(first, discovery_signature(config, model))
            path.write_bytes(b'changed checkpoint')
            self.assertNotEqual(first, discovery_signature(config, model))
            config.concept_prompts = ['dog']
            self.assertNotEqual(first, discovery_signature(config, model))

    def test_helper_changes_invalidate_discovery_and_resume_signature(self):
        config = SliderSpaceConfig.parse({'concept_prompts': ['cat']})
        model = {'arch': 'qwen_image_2'}
        baseline = discovery_signature(config, model)
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / 'helper.safetensors'
            helper.write_bytes(b'helper')
            model['assistant_lora_path'] = str(helper)
            assisted = discovery_signature(config, model)
            self.assertNotEqual(baseline, assisted)
            helper.write_bytes(b'updated helper')
            self.assertNotEqual(assisted, discovery_signature(config, model))

    def test_semantic_encoder_frozen_and_differentiable_without_download(self):
        encoder = SemanticEncoder.__new__(SemanticEncoder)
        nn.Module.__init__(encoder)
        class Vision(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(3, 8)
            def forward(self, pixel_values):
                return SimpleNamespace(image_embeds=self.proj(pixel_values.mean((2, 3))))
        encoder.model = Vision().eval().requires_grad_(False)
        encoder.image_size = 8
        encoder.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        encoder.register_buffer('std', torch.ones(1, 3, 1, 1))
        image = torch.randn(1, 3, 16, 16, requires_grad=True)
        encoder(image).square().sum().add(encoder(image).sum()).backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))

    def test_bucketed_features_letterbox_consistently_and_keep_gradients(self):
        encoder = SemanticEncoder.__new__(SemanticEncoder)
        nn.Module.__init__(encoder)
        seen = []
        class Vision(nn.Module):
            def forward(self, pixel_values):
                seen.append(pixel_values.detach().clone())
                return SimpleNamespace(image_embeds=pixel_values.mean((2, 3)))
        encoder.model, encoder.image_size, encoder.preserve_aspect_ratio = Vision(), 8, True
        encoder.register_buffer('mean', torch.full((1, 3, 1, 1), .5))
        encoder.register_buffer('std', torch.ones(1, 3, 1, 1))
        pixels = torch.ones(1, 3, 8, 16, requires_grad=True)
        with torch.no_grad():
            reference = encoder(pixels)
        adapted = encoder(pixels)
        torch.testing.assert_close(reference, adapted)
        self.assertEqual(seen[0].shape, (1, 3, 8, 8))
        self.assertEqual(seen[0][:, :, :2].abs().sum().item(), 0)
        torch.testing.assert_close(seen[0][:, :, 2:6], torch.full((1, 3, 4, 8), .5))
        adapted[:, 0].sum().backward()
        self.assertGreater(pixels.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(pixels.grad).all())


class SliderSpaceSourceTests(unittest.TestCase):
    def config(self, directory, **settings):
        return SliderSpaceConfig.parse({'discovery_mode': 'provided', 'concept_prompts': ['fallback'],
            'discovery_datasets': [{'folder_path': directory, 'default_caption': 'folder caption'}],
            'num_directions': 1, 'resolution': 128, 'discovery_buckets': True, **settings})

    def test_recursive_scan_dedup_disabled_and_caption_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / 'nested'
            nested.mkdir()
            for path in (root / 'a.PNG', nested / 'b.jpg', nested / 'c.webp'):
                Image.new('RGB', (200, 100)).save(path)
            (root / 'ignored.png.disabled').write_bytes(b'disabled')
            (root / 'a.txt').write_text('specific caption')
            (nested / 'alias.png').symlink_to(root / 'a.PNG')
            config = self.config(directory)
            config.discovery_datasets.append({'folder_path': str(nested), 'default_caption': 'second folder'})
            entries = scan_discovery_images(config)
            self.assertEqual(len(entries), 3)
            self.assertEqual([entry['caption'] for entry in entries], ['specific caption', 'folder caption', 'folder caption'])
            self.assertTrue(all(entry['size'][0] != entry['size'][1] for entry in entries))
            config.discovery_datasets[0]['default_caption'] = ''
            self.assertEqual(scan_discovery_images(config)[1]['caption'], 'fallback')
            config.concept_prompts = []
            self.assertEqual(scan_discovery_images(config)[1]['caption'], '')

    def test_source_modes_counts_and_invalid_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'no supported images'):
                scan_discovery_images(self.config(directory))
            for i in range(3):
                Image.new('RGB', (128, 128)).save(Path(directory) / f'{i}.png')
            provided = self.config(directory, discovery_samples=0, concept_prompts=[])
            self.assertEqual(len(discovery_entries(provided)), 3)
            both = self.config(directory, discovery_mode='both', discovery_samples=1, num_directions=3)
            entries = discovery_entries(both)
            self.assertEqual(len(entries), 4)
            self.assertEqual(entries[-1], {'source': 'generated', 'caption': 'fallback', 'seed': 42})
            both.num_directions = 4
            with self.assertRaisesRegex(ValueError, '5 total discovery images'):
                discovery_entries(both)
            generated = self.config(directory, discovery_mode='generated', discovery_samples=2)
            generated.discovery_datasets = [{'folder_path': '/nonexistent'}]
            self.assertEqual([entry['source'] for entry in discovery_entries(generated)], ['generated'] * 2)
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                scan_discovery_images(self.config(str(Path(directory) / 'missing')))
        for mode in ('invalid', 'provided', 'both'):
            with self.assertRaises(ValueError):
                SliderSpaceConfig.parse({'discovery_mode': mode, 'concept_prompts': ['cat']})

    def test_source_identity_tracks_mtime_captions_additions_and_buckets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'a.png'
            Image.new('RGB', (200, 100)).save(image)
            config = self.config(directory)
            entries = scan_discovery_images(config)
            first = discovery_signature(config, {})
            verify_discovery_sources(entries)
            stat = image.stat()
            os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
            self.assertNotEqual(first, discovery_signature(config, {}))
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verify_discovery_sources(entries)
            entries = scan_discovery_images(config)
            first = discovery_signature(config, {})
            (root / 'a.txt').write_text('new caption')
            self.assertNotEqual(first, discovery_signature(config, {}))
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verify_discovery_sources(entries)
            first = discovery_signature(config, {})
            Image.new('RGB', (100, 200)).save(root / 'b.png')
            self.assertNotEqual(first, discovery_signature(config, {}))
            first = discovery_signature(config, {})
            config.discovery_buckets = False
            self.assertNotEqual(first, discovery_signature(config, {}))
            entries = scan_discovery_images(config)
            (root / 'a.txt').unlink()
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verify_discovery_sources(entries)

    def test_generated_legacy_signature_is_unchanged(self):
        config = SliderSpaceConfig.parse({'concept_prompts': ['cat']})
        settings = asdict(config)
        for key in ('discovery_mode', 'discovery_datasets', 'discovery_buckets', 'preview_direction', 'preview_strength', 'preview_auto', 'preview_auto_strengths', 'loss_weight'):
            settings.pop(key)
        payload = {'version': 1, 'sliderspace': settings, 'model': {}, 'local_identity': {'feature_encoder': None}}
        legacy = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        self.assertEqual(discovery_signature(config, {}), legacy)
        config.discovery_buckets = True
        self.assertEqual(discovery_signature(config, {}), legacy)

    def test_mitchell_buckets_exif_budget_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'wide.png'
            exif = Image.Exif()
            exif[274] = 6
            Image.new('RGB', (400, 200), (20, 50, 100)).save(image, exif=exif)
            original = image.read_bytes()
            size = discovery_image_size(image, 128, True)
            self.assertLess(size[0], size[1])
            self.assertLessEqual(size[0] * size[1], 128**2)
            self.assertTrue(all(value % 32 == 0 for value in size))
            self.assertEqual(prepare_discovery_image(image, 128, True).size, tuple(size))
            self.assertEqual(prepare_discovery_image(image, 128).size, (128, 128))
            self.assertEqual(image.read_bytes(), original)
            for shape in ((1, 10000), (10000, 1), (40, 50), (31, 31)):
                Image.new('RGB', shape).save(image)
                size = discovery_image_size(image, 128, True)
                self.assertTrue(all(value >= 32 and value % 32 == 0 for value in size))
                self.assertLessEqual(size[0] * size[1], 128**2)


class SliderSpaceBankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)

    def test_independent_updates_signed_strength_and_base_unchanged(self):
        bank, model = make_bank()
        x = torch.randn(2, 4)
        base_weights = model[0].linear.weight.detach().clone()
        bank.is_active = False
        baseline = model(x).detach()
        bank.is_active = True
        optimizer = torch.optim.AdamW(bank.prepare_optimizer_params(0.1, 0.1, 0.1), lr=0.1)
        bank.select_direction(1)
        model(x).sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in bank.directions[0].parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in bank.directions[2].parameters()))
        optimizer.step()
        positive = model(x).detach()
        self.assertFalse(torch.allclose(positive, baseline))
        bank.multiplier = -1
        torch.testing.assert_close(model(x), 2 * baseline - positive)
        bank.multiplier = 0
        torch.testing.assert_close(model(x), baseline)
        bank.multiplier = 1
        bank.select_direction(0)
        torch.testing.assert_close(model(x), baseline)
        torch.testing.assert_close(model[0].linear.weight, base_weights)
        self.assertFalse(any('linear.weight' in key for key in bank.state_dict()))

    def test_complete_bank_roundtrip_and_standard_single_direction_export(self):
        bank, model = make_bank()
        with torch.no_grad():
            for i, direction in enumerate(bank.directions):
                direction.get_all_modules()[0].lora_up.weight.fill_(i + 1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'bank.safetensors')
            bank.save_bank(path, {'test': 'value'})
            second, _ = make_bank()
            second.load_weights(path)
            for key, value in bank.state_dict().items():
                torch.testing.assert_close(second.state_dict()[key], value)
            bank.select_direction(2)
            export = str(Path(directory) / 'direction.safetensors')
            bank.save_weights(export)
            self.assertFalse(any(key.startswith('directions.') for key in load_file(export)))
            with self.assertRaisesRegex(ValueError, 'complete bank'):
                second.load_weights(export)

    def test_offloaded_quantized_base_direction_chain_has_gradients(self):
        from toolkit.util.convrot_quant import ConvRotIntNQuantizer
        from toolkit.util.ostris_quant import convert_linear_to_ostris
        from toolkit.memory_management.manager_modules import OstrisLinearLayerMemoryManager
        model = nn.Sequential(TinyBlock()).requires_grad_(False)
        # Rotated quantization requires aligned dimensions.
        model[0].linear = nn.Linear(64, 16).requires_grad_(False)
        convert_linear_to_ostris(model[0].linear, ConvRotIntNQuantizer(4, rot_size=16))
        OstrisLinearLayerMemoryManager(model[0].linear, SimpleNamespace(process_device=torch.device('cpu')))
        bank = SliderSpaceNetwork(text_encoder=[], unet=model, num_directions=2, lora_dim=2, alpha=1,
                                  network_type='lora', network_config=NetworkConfig(type='lora'),
                                  train_unet=True, train_text_encoder=False, target_lin_modules=['TinyBlock'])
        bank.apply_to([], model, False, True)
        bank.force_to('cpu', torch.float32)
        bank._update_torch_multiplier()
        bank.select_direction(0)
        bank.is_active = True
        model(torch.randn(2, 64, requires_grad=True)).sum().backward()
        self.assertGreater(bank.directions[0].get_all_modules()[0].lora_up.weight.grad.abs().sum().item(), 0)
        self.assertTrue(model[0].linear.is_ostris_quantized)

    def test_qwen_public_key_layout_and_explicit_alpha(self):
        if not torch.cuda.is_available():
            with patch('torch.cuda.current_device', return_value=0), patch(
                    'torch.cuda.get_device_properties', return_value=SimpleNamespace(warp_size=32, major=8)), patch(
                    'triton.autotune', side_effect=lambda *args, **kwargs: lambda function: function):
                from testing import test_qwen_image_2_training_adapter as fixture
        else:
            from testing import test_qwen_image_2_training_adapter as fixture
        model = fixture.QwenImage2TrainingAdapterTests().make_model()
        model.model = fixture.QwenImage21Transformer2DModel().requires_grad_(False)
        config = NetworkConfig(type='lora', linear=2, linear_alpha=0.5)
        bank = SliderSpaceNetwork([], model.model, num_directions=2, lora_dim=2, alpha=0.5,
                                  train_unet=True, train_text_encoder=False, network_type='lora',
                                  network_config=config, is_transformer=True, transformer_only=True,
                                  target_lin_modules=model.target_lora_modules, base_model=model)
        bank.apply_to([], model.model, False, True)
        bank._update_torch_multiplier()
        bank.is_active = True
        bank.select_direction(1)
        model.model(torch.randn(1, 2)).sum().backward()
        self.assertIsNotNone(bank.directions[1].get_all_modules()[0].lora_up.weight.grad)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'direction.safetensors')
            bank.save_weights(path)
            saved = load_file(path)
            prefix = 'diffusion_model.transformer_blocks.0.attn.to_q'
            self.assertIn(prefix + '.lora_A.weight', saved)
            self.assertIn(prefix + '.lora_B.weight', saved)
            self.assertEqual(saved[prefix + '.alpha'].item(), 0.5)


class Embed:
    def __init__(self, value=0):
        self.value = value
    def detach(self):
        return Embed(self.value)
    def to(self, *args, **kwargs):
        return self


class SliderSpaceTrainerTests(unittest.TestCase):
    def config(self):
        return {'type': 'sliderspace', 'model': {'arch': 'qwen_image_2'}, 'network': {'type': 'lora'},
                'train': {'steps': 6}, 'sliderspace': {'concept_prompts': ['cat'], 'num_directions': 2,
                                                      'discovery_samples': 6, 'resolution': 128}}

    def test_constructor_rejects_incompatible_modes_without_mutating_config(self):
        original = self.config()
        with patch.object(DiffusionTrainer, '__init__', lambda obj, *args, **kwargs: setattr(obj, 'accelerator', SimpleNamespace(num_processes=1))):
            trainer = SliderSpaceTrainer(0, None, original)
            self.assertEqual(trainer.sliderspace.num_directions, 2)
        self.assertEqual(original, self.config())
        for section, key, value in [('network', 'type', 'dora'), ('train', 'batch_size', 2),
                                    ('train', 'gradient_accumulation_steps', 2), ('train', 'steps', 1),
                                    ('train', 'do_cfg', True), ('model', 'inference_lora_path', 'adapter')]:
            config = self.config()
            config[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                SliderSpaceTrainer(0, None, config)

    def make_trainer(self, directory):
        trainer = SliderSpaceTrainer.__new__(SliderSpaceTrainer)
        trainer.sliderspace = SliderSpaceConfig.parse(self.config()['sliderspace'])
        trainer.discovery_entries = discovery_entries(trainer.sliderspace)
        trainer.discovery_id = 'test-signature'
        trainer.save_root = directory
        trainer.network, model = make_bank(2)
        trainer.accelerator = SimpleNamespace(is_main_process=True, backward=lambda loss: loss.backward())
        trainer.train_config = SimpleNamespace(min_denoising_steps=0, max_denoising_steps=999, start_step=None)
        trainer.device_torch = torch.device('cpu')
        trainer.concept_embeds, trainer.negative_embeds = [Embed(1)], Embed(-1)
        trainer.bank_embeds = [Embed(1) for _ in trainer.discovery_entries]
        trainer.step_num, trainer.epoch_num = 1, 0
        trainer.direction_vectors = torch.eye(3)[:2]
        trainer.variance_ratio = torch.tensor([0.6, 0.3])
        trainer.resume_bank_path = None
        trainer.optimizer = torch.optim.AdamW(trainer.network.prepare_optimizer_params(.01, .01, .01))
        trainer.save_config = SimpleNamespace(dtype='fp32', max_step_saves_to_keep=1)
        trainer.meta = {}
        trainer.network_config = trainer.network.network_config
        trainer.update_training_metadata = lambda: None
        trainer.job = SimpleNamespace(name='test', config_path=None)
        trainer._stage = lambda message: None
        trainer.post_save_hook = lambda path: None
        calls = []
        def predict(latents, conditional_embeddings, **kwargs):
            calls.append((torch.is_grad_enabled(), trainer.network.is_active, conditional_embeddings.value))
            output = model(latents.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return torch.cat((output, output[:, :1]), dim=1) * conditional_embeddings.value
        trainer.sd = SimpleNamespace(unet=model, torch_dtype=torch.float32, vae_torch_dtype=torch.float32,
                                     vae_device_torch=torch.device('cpu'), predict_noise=predict,
                                     decode_latents=lambda latent: latent[:, :3],
                                     text_encoder_to=lambda device: None,
                                     save_device_state=lambda: None, restore_device_state=lambda: None)
        trainer.semantic_encoder = lambda pixels: pixels.mean((2, 3))
        return trainer, calls

    def test_training_cfg_gradients_and_direction_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, calls = self.make_trainer(directory)
            trainer._discovery_root().mkdir(parents=True)
            for i in range(6):
                save_file({'latent': torch.randn(1, 4, 8, 8)}, str(trainer._bank_paths(i)[1]))
            previous = trainer.network.active_direction, trainer.network.multiplier, trainer.network.is_active
            loss = trainer.train_single_accumulation(None)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(calls, [(False, False, 1), (False, False, -1), (True, True, 1), (True, True, -1)])
            self.assertGreater(trainer.network.directions[1].get_all_modules()[0].lora_up.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(param.grad is None for param in trainer.network.directions[0].parameters()))
            self.assertEqual(previous, (trainer.network.active_direction, trainer.network.multiplier, trainer.network.is_active))

    def test_export_resume_signature_and_whole_generation_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.make_trainer(directory)
            for step in (2, 4):
                trainer.step_num = step
                trainer.save(step)
            root = Path(directory)
            self.assertFalse((root / 'test_direction_01_000000002.safetensors').exists())
            self.assertTrue((root / 'test_direction_01_000000004.safetensors').exists())
            self.assertEqual(len(load_file(str(root / 'test_direction_01_000000004.safetensors'))), 3)
            path = trainer.get_latest_save_path()
            self.assertEqual(Path(path).name, 'bank_000000004.safetensors')
            self.assertTrue(Path(trainer.get_optimizer_state_path()).is_file())
            trainer.load_weights(path)
            trainer.discovery_id = 'changed'
            with self.assertRaisesRegex(ValueError, 'changed on resume'):
                trainer.load_weights(path)
            trainer.discovery_id = 'test-signature'
            trainer.save()
            self.assertTrue((root / 'test_direction_02.safetensors').exists())
            self.assertEqual(load_metadata_from_safetensors(path)['training_info']['step'], 4)

    def test_sample_identity_uses_actual_generated_filename_and_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.make_trainer(directory)
            trainer.sliderspace.preview_direction, trainer.sliderspace.preview_strength = 2, -0.5
            configs = [GenerateImageConfig(prompt='cat', output_path=directory + '/[time]_000000020_[count].png')]
            trainer.post_process_generate_image_config_list(configs)
            self.assertIn('_direction_02_strength_minus0.5_0.png', configs[0].get_image_path(0))
            self.assertEqual(configs[0].network_multiplier, -0.5)
            self.assertIn('comfy_direction_02', trainer._get_comfy_lora_display_filename(20))

    def test_live_preview_and_sampling_restore_direction_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.make_trainer(directory)
            trainer.sliderspace.preview_direction = 2
            trainer.network.select_direction(0)
            trainer.network.is_active = False
            with patch.object(DiffusionTrainer, 'sample', side_effect=RuntimeError('sample failed')):
                with self.assertRaisesRegex(RuntimeError, 'sample failed'):
                    trainer.sample(20)
            self.assertEqual(trainer.network.active_direction, 0)
            self.assertFalse(trainer.network.is_active)
            trainer.job.config_path = 'fake.yaml'
            with patch.object(DiffusionTrainer, '_refresh_live_sample_config'), patch('toolkit.config.get_config', return_value={
                    'config': {'process': [{'sliderspace': {'preview_direction': 1, 'preview_strength': 0, 'preview_auto': True,
                                                          'preview_auto_strengths': '-0.5, 0.5, 2'}}]}}):
                trainer.process_id = 0
                trainer._refresh_live_sample_config()
            self.assertEqual(trainer.sliderspace.preview_strength, 0)
            self.assertTrue(trainer.sliderspace.preview_auto)
            self.assertEqual(trainer.sliderspace.preview_auto_strengths, '-0.5, 0.5, 2')

    def test_auto_sampling_native_and_comfy_use_one_prompt_seed_and_correct_direction(self):
        for comfy in (False, True):
            with self.subTest(comfy=comfy), tempfile.TemporaryDirectory() as directory:
                trainer, _ = self.make_trainer(directory)
                trainer.sliderspace.preview_auto = True
                trainer.sample_config = SampleConfig(samples=[{'prompt': 'cat'}, {'prompt': 'ignored'}], seed=-1,
                                                     walk_seed=True, comfy={'enabled': comfy})
                trainer.first_sample_config = trainer.sample_config
                trainer.adapter_config = trainer.embedding = trainer.adapter = trainer.trigger_word = trainer.ema = None
                trainer.logger = None
                trainer.maybe_stop = lambda: None
                trainer.update_status = lambda *args: None
                previous = trainer.network.active_direction, trainer.network.multiplier, trainer.network.is_active
                renders = []
                def render(configs, *args, **kwargs):
                    self.assertEqual(len(configs), 1)
                    config = configs[0]
                    renders.append((trainer.network.active_direction, config.network_multiplier, config.prompt,
                                    config.seed, config.get_image_path(0), trainer._get_comfy_lora_display_filename(20)))
                trainer.sd.generate_images = render
                trainer._render_comfy_samples = render
                trainer.sample(20, is_first=True)
                self.assertEqual([(row[0], row[1]) for row in renders], [(0, 0), (0, -1), (0, 1), (1, -1), (1, 1)])
                self.assertEqual({row[2] for row in renders}, {'cat'})
                self.assertEqual(len({row[3] for row in renders}), 1)
                self.assertEqual(len({row[4] for row in renders}), 5)
                for index, row in enumerate(renders):
                    self.assertTrue(row[4].endswith(f'_{index}.jpg'))
                    self.assertIn(f'comfy_direction_{row[0] + 1:02d}_000000020', row[5])
                self.assertIn('_auto_direction_00_strength_plus0_', renders[0][4])
                self.assertEqual(previous, (trainer.network.active_direction, trainer.network.multiplier, trainer.network.is_active))
                self.assertIsNone(trainer._auto_sample_preview)
                self.assertEqual((trainer.sliderspace.preview_direction, trainer.sliderspace.preview_strength), (1, 1))

    def test_auto_sampling_restores_training_state_when_a_direction_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.make_trainer(directory)
            trainer.sliderspace.preview_auto = True
            trainer.network.select_direction(1)
            trainer.network.multiplier, trainer.network.is_active = 0.25, False
            previous = trainer.network.active_direction, trainer.network.multiplier, trainer.network.is_active
            with patch.object(DiffusionTrainer, 'sample', side_effect=[None, None, RuntimeError('render failed')]):
                with self.assertRaisesRegex(RuntimeError, 'render failed'):
                    trainer.sample(20)
            self.assertEqual(previous, (trainer.network.active_direction, trainer.network.multiplier, trainer.network.is_active))
            self.assertIsNone(trainer._auto_sample_preview)
            self.assertIsNone(trainer._auto_sample_seed)

    def test_auto_sampling_uses_custom_strengths_in_order_after_one_base(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.make_trainer(directory)
            trainer.sliderspace.preview_auto = True
            trainer.sliderspace.preview_auto_strengths = '-0.5, 0, 0.25, 1.5, 0.25'
            renders = []
            def render(*args, **kwargs):
                configs = trainer.post_process_generate_image_config_list([
                    GenerateImageConfig(prompt='cat', output_path=directory + '/[time]_000000020_[count].png')])
                renders.append((trainer.network.active_direction, configs[0].network_multiplier))
            with patch.object(DiffusionTrainer, 'sample', side_effect=render):
                trainer.sample(20)
            self.assertEqual(renders, [(0, 0), (0, -.5), (0, .25), (0, 1.5), (1, -.5), (1, .25), (1, 1.5)])

    def test_completed_update_count_resumes_next_direction_and_missing_optimizer_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.make_trainer(directory)
            trainer.update_status = lambda *args: None
            trainer.train_config.steps = 20
            trainer.step_num = 3
            with patch.object(DiffusionTrainer, 'hook_train_loop', return_value={'loss': 1}):
                trainer.hook_train_loop(None)
            self.assertEqual(trainer.completed_updates, 4)
            trainer.save(3)
            self.assertTrue((Path(directory) / 'test_direction_01_000000004.safetensors').exists())
            trainer.resume_bank_path = trainer.get_latest_save_path()
            optimizer = Path(trainer.get_optimizer_state_path())
            optimizer.unlink()
            with self.assertRaisesRegex(ValueError, 'optimizer paired'):
                trainer.get_optimizer_state_path()

    def test_decode_checkpoint_always_clears_state_even_during_recomputation(self):
        from torch.utils.checkpoint import checkpoint
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.make_trainer(directory)
            state = SimpleNamespace(cache=None, clears=0)
            def clear():
                state.cache = None
                state.clears += 1
            state.clear_cache = clear
            trainer.sd.vae = state
            def decode(latent):
                state.cache = latent.clone()
                return latent.sin()
            trainer.sd.decode_latents = decode
            latent = torch.randn(1, 3, 8, 8, requires_grad=True)
            checkpoint(trainer._decode_training_prediction, latent, use_reentrant=False).sum().backward()
            self.assertIsNone(state.cache)
            self.assertGreaterEqual(state.clears, 2)
            self.assertTrue(torch.isfinite(latent.grad).all())

    def test_discovery_reuses_cache_rejects_changes_and_restores_network(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.make_trainer(directory)
            trainer.sd.encode_images = lambda pixels: torch.cat((pixels, pixels[:, :1]), dim=1)
            renders = []
            def pipeline(*args, generator, **kwargs):
                renders.append(generator.initial_seed())
                gen = torch.Generator().manual_seed(generator.initial_seed())
                color = tuple(torch.randint(0, 255, (3,), generator=gen).tolist())
                return [Image.new('RGB', (128, 128), color)]
            trainer.sd.get_generation_pipeline = lambda: pipeline
            trainer.sd.vae = nn.Identity()
            previous = trainer.network.is_active
            trainer._build_discovery_bank()
            self.assertEqual(len(renders), 6)
            torch.testing.assert_close(trainer.direction_vectors @ trainer.direction_vectors.T, torch.eye(2))
            self.assertEqual(trainer.network.is_active, previous)
            trainer._build_discovery_bank()
            self.assertEqual(len(renders), 6)
            trainer._bank_paths(0)[0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed or is incomplete'):
                trainer._build_discovery_bank()

    def test_provided_and_combined_banks_keep_bucket_shapes_and_caption_conditioning(self):
        for mode in ('provided', 'both'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'source'
                source.mkdir()
                for i, (shape, color) in enumerate([((200, 100), (200, 30, 50)),
                                                   ((100, 200), (30, 200, 50)),
                                                   ((128, 128), (30, 50, 200))]):
                    Image.new('RGB', shape, color).save(source / f'{i}.png')
                    (source / f'{i}.txt').write_text(f'caption {i}')
                trainer, calls = self.make_trainer(directory)
                trainer.sliderspace = SliderSpaceConfig.parse({**asdict(trainer.sliderspace),
                    'discovery_mode': mode, 'discovery_datasets': [{'folder_path': str(source)}],
                    'discovery_buckets': True, 'discovery_samples': 1})
                trainer.discovery_entries = discovery_entries(trainer.sliderspace)
                trainer.bank_embeds = [Embed(i + 2) for i in range(len(trainer.discovery_entries))]
                encoded = []
                def encode(pixels):
                    encoded.append(pixels.shape[-2:])
                    return torch.cat((pixels, pixels[:, :1]), dim=1)
                trainer.sd.encode_images = encode
                trainer.sd.vae = nn.Identity()
                renders = []
                def pipeline(positive, generator, **kwargs):
                    renders.append((positive.value, generator.initial_seed()))
                    return [Image.new('RGB', (128, 128), (90, 140, 180))]
                def get_pipeline():
                    self.assertEqual(mode, 'both', 'provided-only must never create a generation pipeline')
                    return pipeline
                trainer.sd.get_generation_pipeline = get_pipeline
                trainer._build_discovery_bank()
                self.assertEqual(len(encoded), 3 if mode == 'provided' else 4)
                self.assertNotEqual(encoded[0], encoded[1])
                self.assertEqual(renders, [] if mode == 'provided' else [(5, trainer.sliderspace.seed)])
                for index, entry in enumerate(trainer.discovery_entries):
                    latent = load_file(str(trainer._bank_paths(index)[1]))['latent']
                    self.assertEqual(latent.shape[-2:], tuple(reversed(entry.get('size', [128, 128]))))
                # Reuse avoids both generation and encoding, even with mixed sizes.
                trainer._build_discovery_bank()
                self.assertEqual(len(encoded), len(trainer.discovery_entries))
                generator = torch.Generator().manual_seed(trainer.sliderspace.seed + trainer.step_num * 104729)
                selected = torch.randint(len(trainer.discovery_entries), (1,), generator=generator).item()
                trainer.train_single_accumulation(None)
                self.assertEqual(calls[0][2], selected + 2)
                self.assertEqual(calls[2][2], selected + 2)


if __name__ == '__main__':
    unittest.main()
