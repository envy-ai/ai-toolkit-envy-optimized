"""Integration fixtures for the shared trainer/model boundary (CPU, no downloads)."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import save_file

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.FizgigSliderTrainer import FizgigSliderTrainer
from extensions_built_in.sd_trainer.QwenFlowDPOTrainer import QwenFlowDPOTrainer
from extensions_built_in.sd_trainer.QwenGuidanceDistillationTrainer import QwenGuidanceDistillationTrainer
from extensions_built_in.sd_trainer.SliderSpaceTrainer import SliderSpaceTrainer
from extensions_built_in.sd_trainer.DiffusionKTOTrainer import DiffusionKTOTrainer
from extensions_built_in.sd_trainer import AI_TOOLKIT_EXTENSIONS
from extensions_built_in.diffusion_models.anima.anima import AnimaModel, AnimaPromptEmbeds
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from extensions_built_in.diffusion_models.ideogram4.src import pipeline as ideogram_pipeline
from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.flow_training import FlowTrainingProfile, render_flow_bank_image, training_decode_context
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.sliderspace import SliderSpaceConfig, discovery_signature, scan_discovery_images

ARCHES = ('qwen_image_2', 'krea2', 'anima', 'ideogram4')


class CrossModelTrainerTests(unittest.TestCase):
    def test_constructors_accept_models_preserve_network_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            positive, negative = Path(directory) / 'positive', Path(directory) / 'negative'
            for folder in (positive, negative):
                folder.mkdir()
                Image.new('RGB', (128, 128)).save(folder / 'a.png')
            for arch, helper in [(arch, None) for arch in ARCHES] + [('qwen_image_2', '/helper.safetensors')]:
                for mode, trainer_type in [('fizgig_image_slider', FizgigSliderTrainer),
                    ('fizgig_prompt_slider', FizgigSliderTrainer), ('flow_dpo', QwenFlowDPOTrainer),
                    ('guidance_distillation', QwenGuidanceDistillationTrainer), ('sliderspace', SliderSpaceTrainer),
                    ('diffusion_kto', DiffusionKTOTrainer)]:
                    with self.subTest(arch=arch, mode=mode, helper=helper):
                        config = {'type': mode, 'model': {'arch': arch}, 'network': {'type': 'lora'},
                            'train': {'noise_scheduler': 'flowmatch', 'cache_text_embeddings': True},
                            'datasets': [{'folder_path': str(positive), 'cache_latents_to_disk': True}],
                            'fizgig_slider': {'positive_prefix': 'large', 'negative_prefix': 'small',
                                'prompt_entries': [{'kind': 'simple', 'prompt': 'cat'}]},
                            'sliderspace': {'concept_prompts': ['cat'], 'num_directions': 2}}
                        if helper:
                            config['model']['assistant_lora_path'] = helper
                        if mode == 'diffusion_kto':
                            config['datasets'][0]['kto_label'] = 'liked'
                        if mode in ('fizgig_image_slider', 'flow_dpo'):
                            config['datasets'][0]['control_path_1'] = str(negative)
                        if mode == 'sliderspace':
                            config['datasets'] = []
                        if mode.startswith('fizgig_'):
                            config['network']['type'] = 'dora'
                        with patch.object(DiffusionTrainer, '__init__',
                            lambda obj, *args, **kwargs: setattr(obj, 'accelerator', SimpleNamespace(num_processes=1))), \
                            patch.object(DiffusionTrainer, 'print'):
                            trainer = trainer_type(0, None, copy.deepcopy(config))
                        self.assertEqual(trainer.flow_profile.arch, arch)
                        if mode == 'flow_dpo':
                            self.assertFalse(trainer.needs_vae_at_train_time)
            extensions = {entry.uid: entry for entry in AI_TOOLKIT_EXTENSIONS}
            self.assertIs(extensions['flow_dpo'].get_process(), extensions['qwen_flow_dpo'].get_process())
            self.assertIs(extensions['guidance_distillation'].get_process(),
                          extensions['qwen_guidance_distillation'].get_process())

    def test_actual_krea_generation_wrapper_converts_standard_cfg_once(self):
        calls = []
        def pipeline(**kwargs):
            calls.append(kwargs)
            return [Image.new('RGB', (128, 128))]
        model = SimpleNamespace(arch='krea2', model=SimpleNamespace(device=torch.device('cpu'), to=lambda *args: None),
            device_torch=torch.device('cpu'), is_edit=False, get_bucket_divisibility=lambda: 16)
        model.generate_single_image = lambda *args: Krea2Model.generate_single_image(model, *args)
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.zeros(2, 4)])
        for cfg in (1, 4):
            render_flow_bank_image(model, pipeline, embeds, embeds, width=128, height=128, steps=2, cfg=cfg, seed=7)
        self.assertEqual([call['guidance_scale'] for call in calls], [0, 3])

    def test_anima_prediction_keeps_velocity_time_and_temporal_singleton_contract(self):
        seen = []
        class Transformer:
            device = torch.device('cpu')
            def __call__(self, **kwargs):
                seen.append(kwargs)
                return (torch.full_like(kwargs['hidden_states'], 3),)
        transformer = Transformer()
        embeds = AnimaPromptEmbeds(torch.ones(1, 2, 4), torch.ones(1, 3, dtype=torch.long),
                                   torch.ones(1, 2, dtype=torch.long), torch.ones(1, 3, dtype=torch.long))
        model = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
            trainable_model=SimpleNamespace(transformer=transformer),
            noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
            _condition_prompt_embeds=lambda value, **kwargs: value.text_embeds)
        output = AnimaModel.get_noise_prediction(model, torch.zeros(1, 16, 4, 6), torch.tensor([250]), embeds)
        torch.testing.assert_close(seen[0]['timestep'], torch.tensor([.25]))
        self.assertEqual(seen[0]['hidden_states'].shape, (1, 16, 1, 4, 6))
        torch.testing.assert_close(output, torch.full((1, 16, 4, 6), 3.))

    def test_model_decoder_checkpoint_context_restores_after_backward_and_errors(self):
        from torch.utils.checkpoint import checkpoint
        class VAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.7), requires_grad=False)
                self.device = torch.device('cpu')
                self.config = SimpleNamespace(z_dim=16, latents_mean=[0] * 16, latents_std=[1] * 16)
                self.moves, self.tiling, self.decodes = [], [], []
            def to(self, device, *args, **kwargs):
                self.moves.append(str(device))
                self.device = torch.device(device)
                return super().to(device, *args, **kwargs)
            def enable_tiling(self): self.tiling.append(True)
            def disable_tiling(self): self.tiling.append(False)
            def clear_cache(self): pass
            def decoder(self, tensor):
                self.decodes.append(torch.is_grad_enabled())
                return torch.sin(tensor[:, :3]) * self.weight
            def decode(self, tensor): return SimpleNamespace(sample=self.decoder(tensor))
        for arch, wrapper, channels in [('krea2', Krea2Model, 16), ('ideogram4', Ideogram4Model, 128)]:
            with self.subTest(arch=arch):
                vae = VAE()
                model = SimpleNamespace(arch=arch, vae=vae, vae_device_torch=torch.device('cpu'), vae_torch_dtype=torch.float32,
                    model_config=SimpleNamespace(low_vram=True), patch_size=2,
                    _latent_shift=torch.zeros(1, 128, 1, 1), _latent_scale=torch.ones(1, 128, 1, 1))
                model.decode_latents = lambda tensor: wrapper.decode_latents(model, tensor)
                latent = torch.randn(1, channels, 4, 6, requires_grad=True)
                with training_decode_context(model):
                    pixels = checkpoint(model.decode_latents, latent, use_reentrant=False)
                    before_backward = len(vae.moves)
                    pixels.square().mean().backward()
                    # CPU decode calls to(cpu) are counted, but inference's final
                    # offload must NOT run in the checkpointed training decode.
                    self.assertEqual(len(vae.moves), before_backward + 1)
                    self.assertEqual(vae.tiling, [])
                self.assertEqual(len(vae.decodes), 2)
                self.assertEqual(model._training_decode_depth, 0)
                self.assertGreater(latent.grad.abs().sum().item(), 0)
                self.assertIsNone(vae.weight.grad)
                if arch == 'krea2':
                    model.decode_latents(latent.detach())
                    self.assertEqual(vae.tiling, [True, False])
                with self.assertRaisesRegex(RuntimeError, 'decode failure'):
                    with training_decode_context(model):
                        raise RuntimeError('decode failure')
                self.assertEqual(model._training_decode_depth, 0)

    def test_guidance_objectives_across_profiles_and_cfg_one_branch_skip(self):
        from testing.test_qwen_guidance_distillation import GuidanceDistillationTests
        fixture = GuidanceDistillationTests()
        for arch in ARCHES:
            for objective, correction in [('full_guidance', 6.), ('negative_only', 3.)]:
                with self.subTest(arch=arch, objective=objective):
                    trainer, parameter, batch, seen = fixture.make_trainer(objective)
                    trainer.sd.arch = arch
                    trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': arch})
                    loss = trainer.train_single_accumulation(batch, accum_scale=.5)
                    error = .25 - correction
                    torch.testing.assert_close(loss, torch.tensor(2 * error ** 2))
                    torch.testing.assert_close(parameter.grad, torch.tensor(2 * error))
                    self.assertTrue(all(call[3]['batch'] is batch for call in seen))
                    self.assertFalse(trainer.network.is_active)
                    trainer.teacher_cfg_scale = 1
                    trainer._teacher_embeds = lambda _: self.fail('CFG 1 must not load reference embeddings')
                    seen.clear()
                    trainer.train_single_accumulation(batch)
                    self.assertEqual(len(seen), 2) # one teacher conditional and one student

    def test_ideogram_distillation_real_no_text_and_text_negative_cache_branches(self):
        trainer = object.__new__(QwenGuidanceDistillationTrainer)
        trainer.sd = SimpleNamespace(arch='ideogram4')
        trainer.cfg_reference = 'image_only'
        positive = AdvancedPromptEmbeds(text_embeds=[torch.ones(2, 4)])
        batch = SimpleNamespace(prompt_embeds=positive)
        image_only, blank = trainer._teacher_embeds(batch)
        self.assertEqual(image_only.text_embeds[0].shape, (0, 4))
        self.assertIsNone(blank)
        with tempfile.TemporaryDirectory() as directory:
            negative = AdvancedPromptEmbeds(text_embeds=[torch.ones(3, 4)])
            filename = str(Path(directory) / 'neg.safetensors')
            negative.save(filename)
            trainer.cfg_reference = 'negative_prompt'
            trainer.distillation_objective = 'negative_only'
            trainer.teacher_prompt_paths = {'positive': (filename,)}
            trainer.train_config = SimpleNamespace(dtype='fp32')
            trainer.device_torch = torch.device('cpu')
            batch.file_items = [SimpleNamespace(get_text_embedding_path=lambda: 'positive')]
            negative, blank = trainer._teacher_embeds(batch)
            self.assertEqual(negative.text_embeds[0].shape, (3, 4))
            self.assertEqual(blank.text_embeds[0].shape, (0, 4))

    def test_anima_prompt_cache_integer_fields_and_unequal_lengths(self):
        def embeds(length):
            return AnimaPromptEmbeds(torch.randn(1, length, 4), torch.ones(1, length + 1, dtype=torch.long),
                                     torch.ones(1, length, dtype=torch.long), torch.ones(1, length + 1, dtype=torch.long))
        first, second = embeds(2), embeds(3)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'prompt.safetensors')
            first.save(path)
            loaded = PromptEmbeds.load(path).detach().to('cpu', dtype=torch.bfloat16)
        self.assertIsInstance(loaded, AnimaPromptEmbeds)
        self.assertEqual(loaded.t5_input_ids.dtype, torch.long)
        self.assertEqual(loaded.attention_mask.dtype, torch.long)
        combined = concat_prompt_embeds([loaded, second.to(dtype=torch.bfloat16)])
        self.assertEqual(combined.text_embeds.shape, (2, 3, 4))
        self.assertEqual(combined.t5_input_ids.shape, (2, 4))
        self.assertEqual(combined.t5_input_ids.dtype, torch.long)

    def test_ideogram_wrapper_reverses_time_and_velocity_exactly_once(self):
        seen = []
        def transformer(**kwargs):
            seen.append(kwargs['t'])
            return torch.full_like(kwargs['x'], -4)
        output = ideogram_pipeline.predict_velocity(transformer, torch.zeros(1, 4, 2, 3),
            torch.tensor([.25]), torch.zeros(1, 2, 8), torch.ones(1, 2, dtype=torch.long))
        torch.testing.assert_close(seen[0], torch.tensor([.75]))
        torch.testing.assert_close(output, torch.full((1, 4, 2, 3), 4.))

    def test_ideogram_native_reference_policy_and_cfg_one(self):
        positive = AdvancedPromptEmbeds(text_embeds=[torch.ones(2, 4)])
        negative = AdvancedPromptEmbeds(text_embeds=[torch.ones(3, 4)])
        seen = []
        def velocity(transformer, latent, time, features, mask):
            seen.append(features.shape[1])
            return torch.zeros_like(latent)
        model = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
            transformer=SimpleNamespace(config=SimpleNamespace(in_channels=4)), patch_size=2, vae_scale_factor=8,
            model_config=SimpleNamespace(model_kwargs={}),
            decode_latents=lambda latent, **kwargs: torch.zeros(1, 3, 32, 32))
        pipeline = ideogram_pipeline.Ideogram4Pipeline(model)
        for policy, cfg, expected in [('image_only', 4, [2, 0]), ('negative_prompt', 4, [2, 3]),
                                      ('negative_prompt', 1, [2])]:
            seen.clear()
            model.model_config.model_kwargs['ideogram_cfg_reference'] = policy
            with patch.object(ideogram_pipeline, 'predict_velocity', velocity):
                pipeline(positive, negative, width=32, height=32, num_inference_steps=1, guidance_scale=cfg,
                         generator=torch.Generator().manual_seed(3))
            self.assertEqual(seen, expected)

    def test_native_bucket_alignment_and_discovery_cache_contracts(self):
        config = SliderSpaceConfig.parse({'discovery_mode': 'provided', 'num_directions': 1,
            'concept_prompts': [], 'discovery_datasets': [{'folder_path': 'placeholder'}],
            'discovery_buckets': True, 'resolution': 128})
        with tempfile.TemporaryDirectory() as directory:
            config.discovery_datasets[0]['folder_path'] = directory
            Image.new('RGB', (80, 48)).save(Path(directory) / 'a.png')
            krea = scan_discovery_images(config, 16)[0]['size']
            anima = scan_discovery_images(config, 32)[0]['size']
            self.assertNotEqual(krea, anima)
            self.assertTrue(all(value % 16 == 0 for value in krea))
            signatures = [discovery_signature(config, {'arch': arch}) for arch in ARCHES]
            self.assertEqual(len(set(signatures)), len(ARCHES))
            first = discovery_signature(config, {'arch': 'ideogram4'})
            self.assertNotEqual(first, discovery_signature(config, {'arch': 'ideogram4',
                'model_kwargs': {'ideogram_cfg_reference': 'negative_prompt'}}))

    def test_sliderspace_all_models_train_checkpointed_student_with_frozen_base(self):
        # Reuse the established tiny adapter-bank fixture; keep model-specific
        # profile/reference semantics real while mocking expensive diffusion weights.
        from testing.test_sliderspace import SliderSpaceTrainerTests
        fixture = SliderSpaceTrainerTests()
        for arch in ARCHES:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                trainer, calls = fixture.make_trainer(directory)
                trainer.sd.arch = arch
                trainer.sd.model_config = SimpleNamespace(model_kwargs={})
                trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': arch})
                trainer._discovery_root().mkdir(parents=True)
                for i in range(len(trainer.discovery_entries)):
                    save_file({'latent': torch.randn(1, 4, 8, 12)}, str(trainer._bank_paths(i)[1]))
                if arch == 'ideogram4':
                    # This fixture's opaque Embed lacks text features; text-negative
                    # mode is appropriate. Image-only branches are tested above.
                    trainer.sd.model_config.model_kwargs['ideogram_cfg_reference'] = 'negative_prompt'
                loss = trainer.train_single_accumulation(None)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(calls[0][:2], (False, False))
                self.assertEqual(calls[-1][:2], (True, True))
                self.assertGreater(trainer.network.directions[1].get_all_modules()[0].lora_up.weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(parameter.grad is None for parameter in trainer.sd.unet.parameters()))


if __name__ == '__main__':
    unittest.main()
