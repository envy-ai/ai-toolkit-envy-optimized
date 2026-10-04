"""Actual Krea prediction boundary with stochastic tiny-VAE edit references."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from extensions_built_in.diffusion_models.krea2 import krea2 as wrapper
from extensions_built_in.sd_trainer.QwenFlowDPOTrainer import QwenFlowDPOTrainer, flow_dpo_terms
from extensions_built_in.sd_trainer.QwenGuidanceDistillationTrainer import QwenGuidanceDistillationTrainer
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.flow_training import FlowTrainingProfile
from toolkit.flow_cache_identity import prepare_specialized_conditioning
from toolkit.dataloader_mixins import (TextEmbeddingFileItemDTOMixin, ControlFileItemDTOMixin,
                                      TextEmbeddingCachingMixin)


class ReferenceItem(TextEmbeddingFileItemDTOMixin, ControlFileItemDTOMixin):
    """Real cache identity/control loader, with unused DTO plumbing omitted."""
    def __init__(self, root, model):
        self.path, self.control_path = str(root / 'target.png'), str(root / 'source.png')
        self.caption = self.caption_dop = 'edit this'
        self.text_embedding_space_version = 'krea2-test'
        self.text_embedding_version = 1
        self._text_embedding_path = None
        self.encode_control_in_text_embeddings = True
        self.cache_processed_control_text_embeddings = model.cache_processed_control_text_embeddings
        self.text_embedding_control_presentation = model._specialized_control_presentation
        self.caption_dropout_keeps_control_images = False
        self.text_embedding_uses_target_size = False
        self.dataset_config = SimpleNamespace(control_from_same_folder=False, caption_dropout_rate=0,
            diff_output_preservation=False, do_i2v=False, buckets=True, control_transparent_color=[19, 71, 139])
        self.scale_to_width = self.crop_width = 16
        self.scale_to_height = self.crop_height = 32
        self.crop_x = self.crop_y = 0
        self.flip_x = self.flip_y = self.load_rgba = False
        self.use_raw_control_images = self.full_size_control_images = True
        self.aug_replay_spatial_transforms = []
        self.control_tensor = self.control_tensor_list = None
        self.control_video_paths = None
        self.is_video = self.dopsd_self_ref = False


def embeds(value):
    return AdvancedPromptEmbeds(text_embeds=[torch.full((2, 4), float(value))])


class KreaEditObjectiveTests(unittest.TestCase):
    def fixture(self, trainer_type):
        trainer = object.__new__(trainer_type)
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=0, max_denoising_steps=999)
        trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': 'krea2'})
        trainer.network = SimpleNamespace(is_active=False, multiplier=.7)
        trainer.additional_logs = {}
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        parameter = torch.nn.Parameter(torch.tensor(.25))
        model = object.__new__(wrapper.Krea2Model)
        model.is_edit, model.kv_cache = True, True
        model.device_torch, model.torch_dtype = torch.device('cpu'), torch.float32
        model.vae_scale_factor = 8
        model.patch_size = 2
        model.model_config = SimpleNamespace(model_kwargs={}, low_vram=False)
        model.model = torch.nn.Linear(1, 1)
        model.model.requires_grad_(False)
        # Real Krea get_noise_prediction uses .device, as supplied by its DiT.
        model.model.device = torch.device('cpu')
        encodes, seen = [], []
        def encode(images, **kwargs):
            encodes.append(images.detach().clone())
            return torch.rand(1, 1, 2, 2)  # posterior sample changes on every encode
        model.encode_images = encode
        def velocity(transformer, noisy, time, context, mask, **kwargs):
            refs = kwargs['ref_latents']
            seen.append((trainer.network.is_active, torch.is_grad_enabled(), refs, kwargs))
            reference = refs[0][0].mean()
            # Make both the value AND parameter derivative depend on the refs.
            return noisy * .1 + context.mean() + reference + (
                parameter * (1 + reference) if trainer.network.is_active else 0)
        def predict(**kwargs):
            return model.get_noise_prediction(kwargs['latents'], kwargs['timestep'],
                kwargs['conditional_embeddings'], batch=kwargs['batch'])
        model.predict_noise = predict
        trainer.sd = model
        item = SimpleNamespace(path='preferred/a.png', unconditional_path='rejected/a.png',
            scale_to_width=16, scale_to_height=16, crop_x=0, crop_y=0,
            crop_width=16, crop_height=16, flip_x=False, flip_y=False)
        control = torch.full((3, 16, 16), .8)
        batch = SimpleNamespace(latents=torch.zeros(1, 1, 2, 2), prompt_embeds=embeds(2),
            control_tensor_list=[[control]], control_tensor=None, file_items=[item], loss_multiplier_list=[2])
        return trainer, parameter, batch, encodes, seen, velocity

    def test_dpo_exact_replay_gradient_matches_full_graph_with_same_refs(self):
        trainer, parameter, batch, encodes, seen, velocity = self.fixture(QwenFlowDPOTrainer)
        trainer.dpo_beta, trainer.dpo_sft_weight = 1.3, .1
        trainer.rejected_latents = {trainer._pair_key(batch.file_items[0]): torch.ones(1, 2, 2)}
        torch.manual_seed(42)
        with patch.object(wrapper, 'predict_velocity', velocity):
            actual = trainer.train_single_accumulation(batch, accum_scale=.5)
        self.assertEqual(len(encodes), 1)
        torch.testing.assert_close(encodes[0], torch.full((1, 3, 16, 16), .6))
        self.assertEqual([call[0] for call in seen], [False, False, True, True, True, True])
        self.assertTrue(all(call[2] is seen[0][2] for call in seen))
        self.assertTrue(all(call[3]['isolate_refs'] for call in seen))
        self.assertTrue(all('ref_kv_cache' not in call[3] for call in seen))
        self.assertFalse(seen[0][2][0][0].requires_grad)
        self.assertFalse(hasattr(trainer.sd, '_training_reference_scope'))
        self.assertFalse(trainer.network.is_active)
        self.assertTrue(all(p.grad is None for p in trainer.sd.model.parameters()))
        actual_grad = parameter.grad.clone()
        reference = seen[0][2][0][0].mean()
        # Match the trainer's RNG before the one reference VAE sample.
        torch.manual_seed(42)
        time = torch.sigmoid(torch.randn(1) + trainer.flow_profile.shift(2, 2)) * .999
        noise = torch.randn(1, 1, 2, 2)
        oracle_parameter = parameter.detach().clone().requires_grad_()
        win, lose = batch.latents, torch.ones_like(batch.latents)
        def error(clean, active):
            noisy = (1 - time.view(1, 1, 1, 1)) * clean + time.view(1, 1, 1, 1) * noise
            pred = noisy * .1 + 2 + reference + oracle_parameter * (1 + reference) * active
            return (pred - (noise - clean)).square().flatten(1).mean(1)
        expected, _, _ = flow_dpo_terms(error(win, 1), error(lose, 1),
            error(win, 0).detach(), error(lose, 0).detach(), 1.3, .1)
        weight = torch.tensor(batch.loss_multiplier_list)
        ((expected * weight).mean() * .5).backward()
        torch.testing.assert_close(actual, (expected * weight).mean().detach())
        torch.testing.assert_close(actual_grad, oracle_parameter.grad)
        with patch.object(wrapper, 'predict_velocity', velocity):
            trainer.train_single_accumulation(batch)
        self.assertEqual(len(encodes), 2)  # scoped cache must not survive optimizer steps

    def test_distillation_teacher_student_share_refs_and_restore_on_failure(self):
        for objective in ('full_guidance', 'negative_only'):
            with self.subTest(objective=objective):
                trainer, parameter, batch, encodes, seen, velocity = self.fixture(QwenGuidanceDistillationTrainer)
                trainer.teacher_cfg_scale = 4
                trainer.distillation_objective = objective
                trainer._teacher_embeds = lambda batch: (embeds(0), embeds(1) if objective == 'negative_only' else None)
                with patch.object(wrapper, 'predict_velocity', velocity):
                    loss = trainer.train_single_accumulation(batch, accum_scale=.5)
                reference = seen[0][2][0][0].mean()
                correction = 6 if objective == 'full_guidance' else 3
                error = parameter.detach() * (1 + reference) - correction
                torch.testing.assert_close(loss, 2 * error.square())
                torch.testing.assert_close(parameter.grad, 2 * error * (1 + reference))
                self.assertEqual(len(encodes), 1)
                self.assertTrue(all(call[2] is seen[0][2] for call in seen))
                self.assertFalse(hasattr(trainer.sd, '_training_reference_scope'))
                with patch.object(wrapper, 'predict_velocity', side_effect=RuntimeError('failed prediction')):
                    with self.assertRaisesRegex(RuntimeError, 'failed prediction'):
                        trainer.train_single_accumulation(batch)
                self.assertFalse(hasattr(trainer.sd, '_training_reference_scope'))
                self.assertFalse(trainer.network.is_active)
                self.assertEqual(trainer.network.multiplier, .7)

    def test_reference_scope_nested_and_rejects_different_batch(self):
        trainer, _, batch, encodes, _, _ = self.fixture(QwenFlowDPOTrainer)
        model = trainer.sd
        with model.training_reference_context(batch):
            first = model._batch_ref_latents_from_batch(batch, 1, 256)
            with model.training_reference_context(batch):
                self.assertIs(first, model._batch_ref_latents_from_batch(batch, 1, 256))
            with self.assertRaisesRegex(ValueError, 'different batches'):
                with model.training_reference_context(SimpleNamespace()):
                    pass
            with self.assertRaisesRegex(ValueError, 'original training batch'):
                model._batch_ref_latents_from_batch(SimpleNamespace(), 1, 256)
            self.assertIs(first, model._batch_ref_latents_from_batch(batch, 1, 256))
        self.assertEqual(len(encodes), 1)
        self.assertFalse(hasattr(model, '_training_reference_scope'))

    def test_positive_negative_blank_cache_use_same_composited_vlm_pixels(self):
        trainer, _, _, _, _, _ = self.fixture(QwenGuidanceDistillationTrainer)
        model = trainer.sd
        model.torch_dtype = torch.bfloat16
        prepare_specialized_conditioning(model)
        model.max_text_length = 256
        model.tokenizer = model.processor = model.vl_processor = None
        model.text_encoder = SimpleNamespace(device=torch.device('cpu'), to=lambda device: None)
        model.has_multiple_control_images = True
        model.device = 'cpu'
        model.set_device_state_preset = lambda name: None
        model.text_encoder_to = lambda device: None
        model.encode_prompt = lambda caption, control_images=None, **kwargs: model.get_prompt_embeds(caption, control_images)
        trainer.teacher_cfg_scale = 4
        trainer.teacher_negative_prompt = 'noise'
        trainer.distillation_objective = 'negative_only'
        trainer.teacher_prompt_paths = {}
        calls = []
        def vlm(encoder, tokenizer, processor, caption, **kwargs):
            calls.append((caption, kwargs['images'][0].detach().clone()))
            return torch.ones(2, 1, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (16, 32)).save(root / 'target.png')
            Image.new('RGBA', (16, 32), (250, 3, 7, 0)).save(root / 'source.png')
            item = ReferenceItem(root, model)
            dataset = TextEmbeddingCachingMixin.__new__(TextEmbeddingCachingMixin)
            dataset.dataset_path, dataset.sd = directory, model
            dataset.dataset_config = item.dataset_config
            dataset.transform, dataset.file_list = None, [item]
            trainer.data_loader = object()
            with patch.object(wrapper, 'encode_krea_prompt', vlm), patch(
                'extensions_built_in.sd_trainer.QwenGuidanceDistillationTrainer.get_dataloader_datasets',
                return_value=[dataset]):
                dataset.cache_text_embeddings()
                trainer._cache_teacher_prompts()
            self.assertEqual([caption for caption, _ in calls], ['edit this', 'noise', ''])
            item.load_control_image()
            # True latent refs and VLM start from the same composited raw image.
            live = item.control_tensor.unsqueeze(0).to(dtype=torch.bfloat16)
            expected = model._prep_vlm_images([live])[0]
            for _, pixels in calls:
                torch.testing.assert_close(pixels, expected)
            torch.testing.assert_close(item.control_tensor[:, 0, 0], torch.tensor([19, 71, 139]) / 255)
            identity = item.get_text_embedding_info_dict()
            item.dataset_config.control_transparent_color = [0, 0, 0]
            self.assertNotEqual(item.get_text_embedding_info_dict(), identity)
            del item.text_embedding_control_presentation
            self.assertNotIn('specialized_control_presentation', item.get_text_embedding_info_dict())

    def test_conditioning_opt_in_leaves_other_models_and_nonedit_krea_unchanged(self):
        for arch, edit in [('qwen_image_2', True), ('anima', False), ('ideogram4', False), ('krea2', False)]:
            model = SimpleNamespace(arch=arch, is_edit=edit, cache_processed_control_text_embeddings=False)
            prepare_specialized_conditioning(model)
            self.assertFalse(model.cache_processed_control_text_embeddings)
            self.assertFalse(hasattr(model, '_specialized_control_presentation'))


if __name__ == '__main__':
    unittest.main()
