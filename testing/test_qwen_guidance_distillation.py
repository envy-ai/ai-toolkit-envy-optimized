from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from extensions_built_in.sd_trainer import AI_TOOLKIT_EXTENSIONS
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.QwenGuidanceDistillationTrainer import (
    QwenGuidanceDistillationTrainer, guidance_distillation_target,
)
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model
from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import run_transformer
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel


MODULE = 'extensions_built_in.sd_trainer.QwenGuidanceDistillationTrainer'


def embeds(value, length=2):
    result = AdvancedPromptEmbeds(
        text_embeds=[torch.full((length, 1), float(value))],
        attention_mask=[torch.ones(length, dtype=torch.int64)],
        image_slot_mask=[torch.zeros(length, dtype=torch.bool)],
    )
    result.frozen_dtype_keys = ['attention_mask', 'image_slot_mask']
    return result


class CacheItem(TextEmbeddingFileItemDTOMixin):
    def __init__(self, root, name, caption, processed=False):
        self.path = str(root / name)
        Path(self.path).write_bytes(b'image')
        self.caption = caption
        self.text_embedding_space_version = 'qwen21-test-encoder'
        self.text_embedding_version = 1
        self._text_embedding_path = None
        self.encode_control_in_text_embeddings = True
        self.control_path = str(root / 'reference.png') if processed else None
        if processed and not Path(self.control_path).exists():
            Path(self.control_path).write_bytes(b'reference')
        self.cache_processed_control_text_embeddings = processed
        self.text_embedding_uses_target_size = True
        self.scale_to_width = self.scale_to_height = 32
        self.crop_x = self.crop_y = 0
        self.crop_width = self.crop_height = 32
        self.flip_x = self.flip_y = False
        self.control_tensor_list = self.control_tensor = None
        self.load_count = self.cleanup_count = 0

    def load_control_image(self):
        self.load_count += 1
        self.control_tensor = torch.ones(3, 32, 32)

    def cleanup_control(self):
        self.cleanup_count += 1
        self.control_tensor = None


class GuidanceDistillationTests(unittest.TestCase):
    def test_targets_and_cancellation(self):
        pos, neg, blank = torch.tensor([2., 3.]), torch.tensor([-1., 1.]), torch.tensor([0., 2.])
        torch.testing.assert_close(guidance_distillation_target(pos, neg, 4), torch.tensor([11., 9.]))
        torch.testing.assert_close(guidance_distillation_target(pos, neg, 4, blank=blank), torch.tensor([5., 6.]))
        for objective in ({}, {'blank': blank}):
            torch.testing.assert_close(guidance_distillation_target(pos, neg, 1, **objective), pos)
        torch.testing.assert_close(guidance_distillation_target(pos, blank, 4, blank=blank), pos)

    def config(self, directory):
        return {
            'model': {'arch': 'qwen_image_2'}, 'network': {'type': 'lora'},
            'train': {'noise_scheduler': 'flowmatch', 'cache_text_embeddings': True},
            'datasets': [{'folder_path': directory, 'cache_latents_to_disk': True}],
        }

    def test_defaults_reference_vae_and_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with patch.object(DiffusionTrainer, '__init__', return_value=None):
                trainer = QwenGuidanceDistillationTrainer(0, None, config)
                self.assertFalse(trainer.needs_vae_at_train_time)
                config['datasets'][0]['control_path_1'] = directory
                edit = QwenGuidanceDistillationTrainer(0, None, config)
            self.assertTrue(edit.needs_vae_at_train_time)
            self.assertEqual(trainer.teacher_cfg_scale, 4)
            self.assertEqual(trainer.distillation_objective, 'full_guidance')
            self.assertEqual(config['datasets'][0]['caption_dropout_rate'], 0)
        extension = next(ext for ext in AI_TOOLKIT_EXTENSIONS if ext.uid == 'qwen_guidance_distillation')
        self.assertIs(extension.get_process(), QwenGuidanceDistillationTrainer)

    def test_invalid_settings_rejected_before_loading_weights(self):
        cases = [
            ('model', 'arch', 'qwen_image', 'Qwen Image 2.1'),
            ('model', 'unconditional_lora_path', '/frozen.safetensors', 'base teacher'),
            ('network', 'type', 'dora', 'LoRA only'),
            ('train', 'train_text_encoder', True, 'transformer LoRA'),
            ('train', 'cache_text_embeddings', False, 'Cache Text Embeddings'),
            ('train', 'unload_text_encoder', True, 'Unload TE'),
            ('train', 'timestep_type', 'linear', 'shifted'),
            ('train', 'loss_type', 'wavelet', 'mean squared'),
            ('train', 'frequency_loss_type', 'lowpass', 'other training objectives'),
            ('train', 'learnable_snr_gos', True, 'other training objectives'),
            ('train', 'ema_config', {'use_ema': True}, 'EMA'),
            ('train', 'max_denoising_steps', 0, 'min timestep'),
            ('guidance_distillation', 'teacher_cfg_scale', float('inf'), 'Teacher CFG'),
            ('guidance_distillation', 'teacher_cfg_scale', .5, 'Teacher CFG'),
            ('guidance_distillation', 'negative_prompt', [], 'must be text'),
            ('guidance_distillation', 'objective', 'wrong', 'objective'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for section, key, value, error in cases:
                with self.subTest(section=section, key=key, value=value):
                    config = self.config(directory)
                    config.setdefault(section, {})[key] = value
                    with self.assertRaisesRegex(ValueError, error):
                        QwenGuidanceDistillationTrainer(0, None, config)
            for key, value, error in [
                ('cache_latents_to_disk', False, 'Cache Latents'),
                ('mask_path', 'mask', 'masked'), ('is_reg', True, 'ordinary'),
                ('network_weight', .5, 'Weight = 1'), ('shuffle_tokens', True, 'captions'),
                ('random_crop', True, 'presentation'), ('num_frames', 2, 'still images'),
                ('control_from_same_folder', True, 'fixed edit references'),
            ]:
                with self.subTest(key=key):
                    config = self.config(directory)
                    config['datasets'][0][key] = value
                    with self.assertRaisesRegex(ValueError, error):
                        QwenGuidanceDistillationTrainer(0, None, config)

    def make_trainer(self, objective='full_guidance', fail=False):
        trainer = object.__new__(QwenGuidanceDistillationTrainer)
        parameter = torch.nn.Parameter(torch.tensor(.25))
        trainer.network = SimpleNamespace(is_active=False, multiplier=.7)
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=0, max_denoising_steps=999)
        trainer.device_torch = torch.device('cpu')
        trainer.teacher_cfg_scale = 4
        trainer.distillation_objective = objective
        trainer.additional_logs = {}
        trainer._teacher_embeds = lambda batch: (embeds(0), embeds(1) if objective == 'negative_only' else None)
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        unet = torch.nn.Linear(1, 1)
        unet.requires_grad_(False)
        unet.eval()
        seen = []

        def predict(**kwargs):
            seen.append((trainer.network.is_active, torch.is_grad_enabled(), unet.training, kwargs))
            if fail and trainer.network.is_active:
                raise RuntimeError('student failed')
            conditioning = kwargs['conditional_embeddings'].text_embeds[0][0, 0]
            return kwargs['latents'].float() * .1 + conditioning + parameter * float(trainer.network.is_active)

        trainer.sd = SimpleNamespace(unet=unet, predict_noise=predict)
        batch = SimpleNamespace(latents=torch.zeros(1, 1, 2, 2), prompt_embeds=embeds(2), loss_multiplier_list=[2])
        return trainer, parameter, batch, seen

    def test_teacher_is_frozen_student_cfg_one_shared_inputs_and_accumulation(self):
        for objective, correction in [('full_guidance', 6.), ('negative_only', 3.)]:
            with self.subTest(objective=objective):
                trainer, parameter, batch, seen = self.make_trainer(objective)
                loss = trainer.train_single_accumulation(batch, accum_scale=.5)
                expected_error = .25 - correction
                torch.testing.assert_close(loss, torch.tensor(2 * expected_error ** 2))
                torch.testing.assert_close(parameter.grad, torch.tensor(2 * expected_error))
                teacher_count = 2 if objective == 'full_guidance' else 3
                self.assertEqual([s[:3] for s in seen], [(False, False, False)] * teacher_count + [(True, True, True)])
                for _, _, _, call in seen:
                    self.assertIs(call['latents'], seen[0][3]['latents'])
                    self.assertIs(call['timestep'], seen[0][3]['timestep'])
                    self.assertIs(call['batch'], batch)
                    self.assertIsNone(call['unconditional_embeddings'])
                    self.assertEqual(call['guidance_scale'], 1)
                self.assertFalse(trainer.network.is_active)
                self.assertEqual(trainer.network.multiplier, .7)
                self.assertFalse(trainer.sd.unet.training)
                self.assertTrue(all(p.grad is None for p in trainer.sd.unet.parameters()))
                self.assertAlmostEqual(trainer.additional_logs['distill/teacher_correction_rms'], correction, places=5)

    def test_helper_remains_frozen_and_active_for_teacher_and_student(self):
        trainer, parameter, batch, seen = self.make_trainer()
        helper = SimpleNamespace(is_active=True, weight=torch.nn.Parameter(torch.tensor(.75), requires_grad=False))
        trainer.sd.assistant_lora = helper
        original = trainer.sd.predict_noise
        def predict(**kwargs):
            self.assertTrue(helper.is_active)
            return original(**kwargs) + helper.weight
        trainer.sd.predict_noise = predict
        loss = trainer.train_single_accumulation(batch)
        self.assertEqual(len(seen), 3)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(parameter.grad)
        self.assertIsNone(helper.weight.grad)
        self.assertTrue(helper.is_active)
        self.assertEqual(helper.weight.item(), .75)

    def test_state_restored_on_student_failure(self):
        trainer, parameter, batch, _ = self.make_trainer(fail=True)
        trainer.network.is_active = True
        trainer.sd.unet.train()
        with self.assertRaisesRegex(RuntimeError, 'student failed'):
            trainer.train_single_accumulation(batch)
        self.assertTrue(trainer.network.is_active)
        self.assertEqual(trainer.network.multiplier, .7)
        self.assertTrue(trainer.sd.unet.training)
        self.assertIsNone(parameter.grad)

    def test_real_qwen_transformer_checkpointed_lora_backward(self):
        # Exercise the actual Qwen joint-sequence transformer and toolkit LoRA
        # hooks, with tiny randomly initialized weights instead of downloads.
        transformer = QwenImage21Transformer2DModel(
            in_channels=2, out_channels=2, num_layers=1, attention_head_dim=32,
            num_attention_heads=1, context_in_dim=4, axes_dims_rope=(8, 12, 12),
        )
        transformer.requires_grad_(False)
        transformer.enable_gradient_checkpointing()
        base_weights = list(transformer.parameters())
        model = QwenImage2Model.__new__(QwenImage2Model)
        model.use_old_lokr_format = False
        model.model = transformer
        network = LoRASpecialNetwork(
            text_encoder=[], unet=transformer, lora_dim=2, alpha=2,
            train_text_encoder=False, train_unet=True,
            target_lin_modules=['QwenImage21TransformerBlock'],
            network_config=NetworkConfig(linear=2, linear_alpha=2),
            is_transformer=True, base_model=model,
        )
        network.apply_to(None, None, False, True)
        network._update_torch_multiplier()
        trainer, _, batch, _ = self.make_trainer()
        trainer.network = network

        def prompt(seed):
            generator = torch.Generator().manual_seed(seed)
            result = embeds(0)
            result.text_embeds = [torch.randn(2, 4, generator=generator)]
            return result

        positive, negative = prompt(42), prompt(43)
        batch.latents = torch.zeros(1, 2, 2, 2)
        batch.prompt_embeds = positive
        trainer._teacher_embeds = lambda batch: (negative, None)

        def predict(**kwargs):
            pe = kwargs['conditional_embeddings']
            return run_transformer(
                transformer, kwargs['latents'], kwargs['timestep'] / 1000,
                pe.text_embeds[0].unsqueeze(0), pe.attention_mask[0].unsqueeze(0).bool(),
                pe.image_slot_mask[0].unsqueeze(0),
            )

        trainer.sd = SimpleNamespace(unet=transformer, predict_noise=predict)
        loss = trainer.train_single_accumulation(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in base_weights))
        gradients = [p.grad for p in network.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
        self.assertGreater(trainer.additional_logs['distill/teacher_correction_rms'], 0)

    def cache_trainer(self, items):
        trainer = object.__new__(QwenGuidanceDistillationTrainer)
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='bf16')
        trainer.teacher_negative_prompt = 'noise'
        trainer.teacher_cfg_scale = 4
        trainer.distillation_objective = 'negative_only'
        trainer.teacher_prompt_paths = {}
        trainer.data_loader = object()
        events = []

        def encode(caption, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            events.append(('encode', caption, kwargs))
            result = embeds(1 if caption else 0, length=3 if kwargs['control_images'] is not None else 2)
            if kwargs['control_images'] is not None:
                result.image_slot_mask[0][0] = True
            return result

        trainer.sd = SimpleNamespace(
            unet=SimpleNamespace(to=lambda device: events.append(('transformer', str(device)))),
            text_encoder_to=lambda device: events.append(('encoder', str(device))), encode_prompt=encode,
        )
        return trainer, events

    def test_cache_dedup_reuse_and_encoder_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            items = [CacheItem(Path(directory), name, caption) for name, caption in [('a.png', 'cat'), ('b.png', 'dog')]]
            trainer, events = self.cache_trainer(items)
            with patch(MODULE + '.get_dataloader_datasets', return_value=[SimpleNamespace(file_list=items)]), patch(MODULE + '.flush'):
                trainer._cache_teacher_prompts()
                self.assertEqual(len([e for e in events if e[0] == 'encode']), 2)
                self.assertEqual(trainer.teacher_prompt_paths[items[0].get_text_embedding_path()], trainer.teacher_prompt_paths[items[1].get_text_embedding_path()])
                events.clear()
                trainer.teacher_cfg_scale = 7  # CFG changes do not invalidate text caches.
                trainer._cache_teacher_prompts()
                self.assertFalse(any(e[0] == 'encode' for e in events))
                for item in items:
                    item.text_embedding_space_version = 'changed-encoder'
                    item._text_embedding_path = None
                trainer._cache_teacher_prompts()
                self.assertEqual(len([e for e in events if e[0] == 'encode']), 2)
                events.clear()
                trainer.teacher_negative_prompt = 'artifacts'
                trainer._cache_teacher_prompts()
                self.assertEqual([e[1] for e in events if e[0] == 'encode'], ['artifacts'])
            self.assertEqual(events[-1], ('encoder', 'cpu'))

    def test_setup_reuses_optimizer_offload_and_parent_unloading_lifecycle(self):
        trainer = object.__new__(QwenGuidanceDistillationTrainer)
        events = []
        trainer._cache_teacher_prompts = lambda: events.append('teacher cache')

        def offload(callback):
            events.append('optimizer to cpu')
            callback()
            events.append('optimizer restored')

        trainer._run_with_optimizer_state_offload = offload
        with patch.object(DiffusionTrainer, 'hook_before_train_loop', side_effect=lambda: events.append('parent setup')):
            trainer.hook_before_train_loop()
        self.assertEqual(events, ['optimizer to cpu', 'teacher cache', 'optimizer restored', 'parent setup'])

    def test_edit_cache_matches_presentation_and_batched_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            items = [CacheItem(Path(directory), 'a.png', 'cat', processed=True), CacheItem(Path(directory), 'b.png', 'dog')]
            trainer, events = self.cache_trainer(items)
            with patch(MODULE + '.get_dataloader_datasets', return_value=[SimpleNamespace(file_list=items)]), patch(MODULE + '.flush'):
                trainer._cache_teacher_prompts()
            calls = [e for e in events if e[0] == 'encode']
            self.assertEqual(items[0].load_count, 1)
            self.assertEqual(items[0].cleanup_count, 1)
            self.assertIs(calls[0][2]['control_images'], calls[1][2]['control_images'])
            self.assertEqual(calls[0][2]['target_size'], (32, 32))
            negative, blank = trainer._teacher_embeds(SimpleNamespace(file_items=items))
            self.assertEqual([len(t) for t in negative.text_embeds], [3, 2])
            self.assertEqual(negative.text_embeds[0].dtype, torch.bfloat16)
            self.assertEqual(negative.attention_mask[0].dtype, torch.int64)
            self.assertEqual(negative.image_slot_mask[0].dtype, torch.bool)
            self.assertEqual(blank.text_embeds[0][0, 0], 0)
            torch.testing.assert_close(negative.image_slot_mask[0], torch.tensor([True, False, False]))
            torch.testing.assert_close(blank.image_slot_mask[0], negative.image_slot_mask[0])
            original = items[0]._build_text_embedding_path(caption_override='noise')
            items[0].flip_x = True
            self.assertNotEqual(original, items[0]._build_text_embedding_path(caption_override='noise'))
            items[0].flip_x = False
            reference = Path(items[0].control_path)
            reference.write_bytes(b'changed reference')
            self.assertNotEqual(original, items[0]._build_text_embedding_path(caption_override='noise'))

    def test_cache_failure_cleans_controls_and_offloads_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            item = CacheItem(Path(directory), 'a.png', 'cat', processed=True)
            trainer, events = self.cache_trainer([item])

            def fail(*args, **kwargs):
                raise RuntimeError('encoding failed')

            trainer.sd.encode_prompt = fail
            with patch(MODULE + '.get_dataloader_datasets', return_value=[SimpleNamespace(file_list=[item])]), patch(MODULE + '.flush'):
                with self.assertRaisesRegex(RuntimeError, 'encoding failed'):
                    trainer._cache_teacher_prompts()
            self.assertEqual(item.cleanup_count, 1)
            self.assertIsNone(item.control_tensor)
            self.assertEqual(events[0], ('transformer', 'cpu'))
            self.assertEqual(events[-1], ('encoder', 'cpu'))


if __name__ == '__main__':
    unittest.main()
