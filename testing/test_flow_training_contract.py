import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.flow_training import (FlowTrainingProfile, decode_training_latents,
    flow_clean_prediction, guided_flow_prediction, image_only_prompt_embeds,
    noised_flow_state, render_flow_bank_image, training_decode_context)
from toolkit.training_capabilities import (FLOW_TRAINING_MODELS, SPECIALIZED_MODES,
    canonical_training_mode, cfg_reference_mode, validate_edit_references,
    validate_specialized_model)


class CapabilityTests(unittest.TestCase):
    def config(self, arch, mode='flow_dpo', **kwargs):
        return {'type': mode, 'model': {'arch': arch, 'model_kwargs': kwargs},
                'network': {'type': 'lora'}}

    def test_all_architectures_modes_and_legacy_aliases(self):
        for arch in FLOW_TRAINING_MODELS:
            for mode in SPECIALIZED_MODES:
                config = self.config(arch, mode)
                self.assertIs(validate_specialized_model(config), FLOW_TRAINING_MODELS[arch])
                config['network']['type'] = 'dora'
                if mode.startswith('fizgig_'):
                    validate_specialized_model(config)
                else:
                    with self.assertRaises(ValueError):
                        validate_specialized_model(config)
        self.assertEqual(canonical_training_mode('qwen_flow_dpo'), 'flow_dpo')
        validate_specialized_model(self.config('qwen_image_2', 'qwen_guidance_distillation'))

    def test_frozen_training_helpers_supported_in_all_qwen_modes_only(self):
        for arch in FLOW_TRAINING_MODELS:
            for mode in SPECIALIZED_MODES:
                config = self.config(arch, mode)
                config['model']['assistant_lora_path'] = '/helper.safetensors'
                with self.subTest(arch=arch, mode=mode):
                    if arch == 'qwen_image_2':
                        validate_specialized_model(config)
                    else:
                        with self.assertRaisesRegex(ValueError, 'auxiliary LoRA'):
                            validate_specialized_model(config)

    def test_forbidden_model_options_fail_early(self):
        for arch, kwargs in [('anima', {'train_text_conditioner': True}),
                             ('krea2', {'edit': True}), ('krea2', {'kv_cache': True}),
                             ('ideogram4', {'ideogram_cfg_reference': 'blank'})]:
            with self.assertRaises(ValueError):
                validate_specialized_model(self.config(arch, 'sliderspace', **kwargs))
        for arch in ('krea2', 'anima', 'ideogram4'):
            config = self.config(arch)
            config['model']['unconditional_lora_path'] = '/teacher.safetensors'
            with self.assertRaises(ValueError):
                validate_specialized_model(config)
            del config['model']['unconditional_lora_path']
            config['model']['name_or_path'] = f'/models/{arch}_turbo.safetensors'
            with self.assertRaises(ValueError):
                validate_specialized_model(config)
        with self.assertRaises(ValueError):
            validate_specialized_model(self.config('unknown'))

    def test_preview_lora_is_not_blocked_by_specialized_mode_validation(self):
        for arch in FLOW_TRAINING_MODELS:
            for mode in SPECIALIZED_MODES:
                config = self.config(arch, mode)
                config['model']['inference_lora_path'] = '/models/preview.safetensors'
                validate_specialized_model(config)
        for mode in SPECIALIZED_MODES:
            config = self.config('krea2', mode)
            config['model']['name_or_path'] = '/models/raw.safetensors'
            config['model']['inference_lora_path'] = '/models/raw_to_turbo.safetensors'
            validate_specialized_model(config)
            for key in ('assistant_lora_path', 'unconditional_lora_path'):
                config['model'][key] = '/training_adapter.safetensors'
                with self.subTest(mode=mode, key=key), self.assertRaisesRegex(ValueError, f'model.{key}'):
                    validate_specialized_model(config)
                del config['model'][key]

    def test_edit_controls_are_distinct_from_pair_targets(self):
        datasets = [{'control_path_1': '/rejected'}]
        for arch in FLOW_TRAINING_MODELS:
            self.assertFalse(validate_edit_references(self.config(arch), datasets, paired=True))
        datasets[0]['control_path_2'] = '/source'
        for arch in ('anima', 'ideogram4', 'krea2'):
            with self.assertRaises(ValueError):
                validate_edit_references(self.config(arch), datasets, paired=True)
        self.assertTrue(validate_edit_references(self.config('krea2', edit=True), datasets, paired=True))
        self.assertTrue(validate_edit_references(self.config('qwen_image_2'), datasets, paired=True))


class FlowProfileTests(unittest.TestCase):
    def test_qwen_schedule_rng_and_reconstruction_are_legacy_exact(self):
        profile = FlowTrainingProfile.from_model_config({'arch': 'qwen_image_2'})
        clean, noise = torch.randn(2, 4, 8, 12), torch.randn(2, 4, 8, 12)
        generator = torch.Generator().manual_seed(17)
        time = profile.sample_time(clean, generator=generator, minimum=20, maximum=900)
        slope = (.9 - .5) / (8192 - 256)
        mu = 8 * 12 * slope + .5 - slope * 256
        original = torch.sigmoid(torch.randn(2, generator=torch.Generator().manual_seed(17)) + mu)
        original = .02 + (.9 - .02) * original
        self.assertTrue(torch.equal(time, original))
        noisy = noised_flow_state(clean, time, noise)
        torch.testing.assert_close(flow_clean_prediction(noisy, noise - clean, time * 1000), clean)

    def test_model_shifts_and_krea_patch_token_counts(self):
        krea = FlowTrainingProfile.from_model_config({'arch': 'krea2'})
        self.assertAlmostEqual(krea.shift(32, 32), .5)  # 256px / factor8 => latent32; 256 tokens.
        self.assertAlmostEqual(krea.shift(160, 160), 1.15)
        self.assertEqual(krea.shift(32, 64), krea.shift(64, 32))
        pinned = FlowTrainingProfile.from_model_config({'arch': 'krea2', 'model_kwargs': {'schedule_mu': 0}})
        self.assertEqual(pinned.shift(64, 128), 0)
        self.assertEqual(FlowTrainingProfile.from_model_config({'arch': 'anima'}).shift(16, 32), math.log(3))
        self.assertEqual(FlowTrainingProfile.from_model_config({'arch': 'ideogram4'}).shift(16, 32), 0)
        with self.assertRaises(ValueError):
            krea.shift(33, 64)
        for kwargs in ({'schedule_mu': True}, {'schedule_mu': float('inf')},
                       {'schedule_min_res': 1280}, {'schedule_max_res': 1290}):
            with self.assertRaises(ValueError):
                FlowTrainingProfile.from_model_config({'arch': 'krea2', 'model_kwargs': kwargs})

    def test_all_profiles_time_bounds_and_finite_identity(self):
        clean = torch.zeros(20, 4, 8, 12)
        for arch in FLOW_TRAINING_MODELS:
            profile = FlowTrainingProfile.from_model_config({'arch': arch})
            time = profile.sample_time(clean, minimum=100, maximum=800)
            self.assertTrue(((time > .1) & (time < .8)).all())
            self.assertEqual(profile.identity()['arch'], arch)
            with self.assertRaises(ValueError):
                profile.sample_time(clean, minimum=800, maximum=100)


class GuidanceBoundaryTests(unittest.TestCase):
    def embeds(self, value, length=2):
        return AdvancedPromptEmbeds(text_embeds=[torch.full((length, 4), float(value))])

    def test_true_image_only_cache_concat_is_not_encoded_blank(self):
        model = SimpleNamespace(arch='ideogram4')
        positive = self.embeds(2)
        reference = image_only_prompt_embeds(model, positive)
        self.assertEqual(reference.text_embeds[0].shape, (0, 4))
        self.assertEqual(positive.text_embeds[0].shape, (2, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'no_text.safetensors')
            reference.save(path)
            loaded = PromptEmbeds.load(path)
            self.assertIsInstance(loaded, AdvancedPromptEmbeds)
            self.assertEqual(loaded.text_embeds[0].shape, (0, 4))
            self.assertEqual(len(concat_prompt_embeds([loaded, reference]).text_embeds), 2)

    def test_sequential_cfg_modes_and_cfg_one_skips_reference(self):
        calls = []
        def predict(conditional_embeddings, latents, **kwargs):
            features = conditional_embeddings.text_embeds[0]
            value = features.mean() if features.numel() else torch.tensor(-3.)
            calls.append((features.shape[0], kwargs['guidance_scale']))
            return torch.ones_like(latents) * value
        model = SimpleNamespace(arch='ideogram4', model_config=SimpleNamespace(model_kwargs={}), predict_noise=predict)
        noisy, time = torch.zeros(1, 4, 2, 3), torch.tensor([500.])
        positive, negative = self.embeds(2), self.embeds(-1)
        result = guided_flow_prediction(model, noisy, time, positive, negative, 4)
        torch.testing.assert_close(result, torch.full_like(noisy, 17))
        self.assertEqual(calls, [(2, 1.0), (0, 1.0)])
        calls.clear()
        result = guided_flow_prediction(model, noisy, time, positive, negative, 1)
        self.assertEqual(calls, [(2, 1.0)])
        torch.testing.assert_close(result, torch.full_like(noisy, 2))
        model.model_config.model_kwargs['ideogram_cfg_reference'] = 'negative_prompt'
        result = guided_flow_prediction(model, noisy, time, positive, negative, 4)
        torch.testing.assert_close(result, torch.full_like(noisy, 11))
        self.assertEqual(cfg_reference_mode({'arch': 'ideogram4'}), 'image_only')

    def test_render_boundary_uses_model_wrapper_not_raw_krea_cfg(self):
        for arch in FLOW_TRAINING_MODELS:
            calls = []
            def pipeline(*args, **kwargs):
                calls.append((args, kwargs))
                return [Image.new('RGB', (128, 128))]
            def generate(pipe, config, positive, negative, generator, extra):
                self.assertIs(pipe, pipeline)
                self.assertEqual(config.guidance_scale, 4)
                self.assertEqual(generator.initial_seed(), 7)
                if arch == 'anima':
                    self.assertIsNotNone(negative)
                calls.append('wrapper')
                return Image.new('RGB', (128, 128))
            model = SimpleNamespace(arch=arch, generate_single_image=generate)
            image = render_flow_bank_image(model, pipeline, self.embeds(2), None,
                width=128, height=128, steps=2, cfg=4, seed=7)
            self.assertEqual(image.size, (128, 128))
            self.assertEqual(len(calls), 1)
            if arch != 'qwen_image_2':
                self.assertEqual(calls, ['wrapper'])

    def test_decode_scope_restores_depth_placement_and_caches_on_exception(self):
        class VAE:
            device = 'cpu'
            clears = 0
            def to(self, device):
                self.device = device
            def clear_cache(self):
                self.clears += 1
        model = SimpleNamespace(vae=VAE(), decode_latents=lambda latent: latent.sin())
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with training_decode_context(model):
                model.vae.device = 'cuda'
                with training_decode_context(model):
                    self.assertEqual(model._training_decode_depth, 2)
                    latent = torch.randn(1, 3, 4, 4, requires_grad=True)
                    decode_training_latents(model, latent).sum().backward()
                    self.assertTrue(torch.isfinite(latent.grad).all())
                raise RuntimeError('failed')
        self.assertEqual(model._training_decode_depth, 0)
        self.assertEqual(model.vae.device, 'cpu')
        self.assertGreaterEqual(model.vae.clears, 3)


if __name__ == '__main__':
    unittest.main()
