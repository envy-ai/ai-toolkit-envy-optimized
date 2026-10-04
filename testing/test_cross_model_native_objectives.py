"""Specialized objectives through native tiny models and checkpointed adapters.

No downloads/GPU: these exercise the real packing, conditioning and time/velocity
boundaries, including Anima's learned frozen conditioner and Ideogram no-text CFG.
Full-graph oracles disable checkpointing so mutable signed adapter strengths are
not re-read during recomputation; production paths retain sequential checkpointing.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from testing import test_diffusion_kto_native as native_fixtures
from extensions_built_in.sd_trainer.FizgigSliderTrainer import FizgigSliderTrainer, pair_difference_weights
from extensions_built_in.sd_trainer.QwenFlowDPOTrainer import QwenFlowDPOTrainer, flow_dpo_terms
from extensions_built_in.sd_trainer.QwenGuidanceDistillationTrainer import QwenGuidanceDistillationTrainer
from extensions_built_in.diffusion_models.anima.anima import AnimaPromptEmbeds
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.flow_training import FlowTrainingProfile
from toolkit.prompt_utils import concat_prompt_embeds

ARCHES = ('krea2', 'anima', 'ideogram4')


class NativeObjectiveTests(unittest.TestCase):
    def setup_native(self, trainer_type, arch, network_type='lora', cfg_reference='negative_prompt'):
        torch.set_num_threads(2)
        torch.manual_seed(193)
        model, network, prompt_dim = native_fixtures.NativeDiffusionKTOTests().native_model(arch, network_type)
        model.model_config = SimpleNamespace(model_kwargs={'ideogram_cfg_reference': cfg_reference})
        # Nonzero learned adapters catch errors that no-op initialization hides.
        with torch.no_grad():
            for module in network.get_all_modules():
                module.lora_up.weight.normal_(0, .025)
        network.is_active, network.multiplier = False, .7
        trainer = object.__new__(trainer_type)
        trainer.sd, trainer.network = model, network
        trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': arch})
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=0, max_denoising_steps=999)
        trainer.additional_logs = {}
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        params = [parameter for parameter in network.parameters() if parameter.requires_grad]

        def prompt(length, offset=0):
            features = torch.randn(1, length, prompt_dim) + offset
            if arch == 'anima':
                return AnimaPromptEmbeds(features, torch.arange(3).unsqueeze(0),
                    torch.ones(1, length, dtype=torch.long), torch.ones(1, 3, dtype=torch.long))
            return AdvancedPromptEmbeds(text_embeds=list(features))

        return trainer, params, prompt

    def compare_gradients(self, trainer, params, expected, atol=2e-5):
        actual = [torch.zeros_like(param) if param.grad is None else param.grad.detach().clone() for param in params]
        for param in params:
            param.grad = None
        transformer = trainer.sd.model.transformer if trainer.sd.arch == 'anima' else trainer.sd.model
        transformer.disable_gradient_checkpointing()
        oracle = expected()
        oracle.backward()
        for param, value in zip(params, actual):
            grad = torch.zeros_like(param) if param.grad is None else param.grad
            torch.testing.assert_close(grad, value, atol=atol, rtol=3e-4)
        self.assertGreater(sum(value.abs().sum().item() for value in actual), 0)
        self.assertTrue(all(param.grad is None for param in trainer.sd.model.parameters()))
        return oracle.detach()

    def test_dpo_weighted_exact_sequential_recomputation_native_all_three(self):
        for arch in ARCHES:
            with self.subTest(arch=arch):
                trainer, params, prompt = self.setup_native(QwenFlowDPOTrainer, arch)
                trainer.network.multiplier = 1.
                item = SimpleNamespace(path='win/a.png', unconditional_path='lose/a.png',
                    scale_to_width=64, scale_to_height=64, crop_x=0, crop_y=0,
                    crop_width=64, crop_height=64, flip_x=False, flip_y=False)
                second = SimpleNamespace(**vars(item))
                second.path, second.unconditional_path = 'win/b.png', 'lose/b.png'
                preferred, rejected = torch.randn(2, 4, 4, 6), torch.randn(2, 4, 4, 6)
                trainer.rejected_latents = {trainer._pair_key(value): tensor for value, tensor in zip((item, second), rejected)}
                trainer.dpo_beta, trainer.dpo_sft_weight = 1.4, .2
                batch = SimpleNamespace(latents=preferred, prompt_embeds=concat_prompt_embeds([prompt(2), prompt(4)]),
                    file_items=[item, second], loss_multiplier_list=[.25, 2.5])
                records = []
                original = trainer._predict
                def predict(noisy, time, embeds, batch):
                    output = original(noisy, time, embeds, batch)
                    records.append((noisy.detach(), time.detach(), output.detach()))
                    return output
                trainer._predict = predict
                loss = trainer.train_single_accumulation(batch, accum_scale=.3)
                self.assertEqual(len(records), 6)
                self.assertFalse(trainer.network.is_active)
                self.assertTrue(torch.equal(records[0][0], records[2][0]))
                self.assertTrue(torch.equal(records[2][0], records[4][0]))
                self.assertTrue(torch.equal(records[1][0], records[3][0]))
                torch.testing.assert_close(records[2][2], records[4][2])
                torch.testing.assert_close(records[3][2], records[5][2])
                t = records[0][1].view(-1, 1, 1, 1) / 1000
                noise = (records[0][0] - (1 - t) * preferred) / t
                targets = [noise - preferred, noise - rejected]
                refs = [(record[2] - target).square().flatten(1).mean(1)
                        for record, target in zip(records[:2], targets)]
                def oracle():
                    trainer.network.is_active = True
                    errors = [(original(record[0], record[1], batch.prompt_embeds, batch) - target)
                              .square().flatten(1).mean(1) for record, target in zip(records[:2], targets)]
                    terms, _, _ = flow_dpo_terms(*errors, *refs, trainer.dpo_beta, trainer.dpo_sft_weight)
                    return (terms * torch.tensor(batch.loss_multiplier_list)).mean() * .3
                expected = self.compare_gradients(trainer, params, oracle)
                torch.testing.assert_close(loss * .3, expected, atol=2e-5, rtol=2e-5)

    def test_prompt_multipoint_and_anchors_native_lora_dora_all_three(self):
        for arch in ARCHES:
            for adapter in ('lora', 'dora'):
                with self.subTest(arch=arch, adapter=adapter):
                    trainer, params, prompt = self.setup_native(FizgigSliderTrainer, arch, adapter,
                        cfg_reference='image_only' if arch == 'ideogram4' else 'negative_prompt')
                    trainer.slider_cfg = 2.5
                    trainer.slider_mode, trainer.slider_multipoint = 'prompt_pairs', True
                    trainer.slider_points = [{'id': name, 'strength': strength} for name, strength in
                        [('small', -1.), ('middle', .5), ('large', 2.), ('neutral', 0.)]]
                    trainer.slider_bank = [(torch.randn(1, 4, 4, 6), 0)]
                    neutral, neg = prompt(3), prompt(2, -1)
                    trainer.slider_embeds, trainer.slider_cfg_negative_embeds = [(neutral,)], [(neg,)]
                    trainer.slider_target_embeds = [{name: (prompt(2 + index, index), prompt(3 + index, -index))
                        for index, name in enumerate(('small', 'middle', 'large'))}]
                    trainer.slider_preservation_weight = .8
                    trainer.slider_anchor_bank = [(torch.randn(1, 4, 4, 6), prompt(4), prompt(2))]
                    states, teacher_targets, strengths = [], [], []
                    noising, predict = trainer._noised_state, trainer._predict
                    def noised(clean):
                        result = noising(clean)
                        states.append(result)
                        return result
                    def prediction(*args):
                        result = predict(*args)
                        if not trainer.network.is_active:
                            teacher_targets.append(result.detach())
                        return result
                    def backward(loss):
                        strengths.append(trainer.network.multiplier)
                        loss.backward()
                    trainer._noised_state, trainer._predict = noised, prediction
                    trainer.accelerator.backward = backward
                    with patch('extensions_built_in.sd_trainer.FizgigSliderTrainer.random.uniform', return_value=.25):
                        loss = trainer.train_single_accumulation(None, accum_scale=.4)
                    self.assertEqual(strengths, [-1., .5, 2., -1., .5, 2., .25])
                    self.assertEqual(len(states), 2)
                    self.assertEqual(len(teacher_targets), 4)
                    self.assertFalse(trainer.network.is_active)
                    self.assertEqual(trainer.network.multiplier, .7)
                    def oracle():
                        trainer.network.is_active = True
                        total = 0.
                        for strength, target in zip((-1., .5, 2.), teacher_targets[:3]):
                            trainer.network.multiplier = strength
                            total = total + (predict(*states[0][:2], neutral, neg) - target).square().mean() / 3
                        _, positive, negative = trainer.slider_anchor_bank[0]
                        for strength in (-1., .5, 2., .25):
                            trainer.network.multiplier = strength
                            total = total + (predict(*states[1][:2], positive, negative) - teacher_targets[3]).square().mean() * .8 / 4
                        return total * .4
                    expected = self.compare_gradients(trainer, params, oracle)
                    torch.testing.assert_close(loss * .4, expected, atol=2e-5, rtol=2e-5)

    def test_image_multipoint_difference_weights_neutral_and_envelope_native(self):
        for arch in ARCHES:
            for adapter in ('lora', 'dora'):
                for explicit_neutral in (False, True):
                    with self.subTest(arch=arch, adapter=adapter, neutral=explicit_neutral):
                        trainer, params, prompt = self.setup_native(FizgigSliderTrainer, arch, adapter)
                        trainer.slider_cfg = 1.
                        trainer.slider_mode, trainer.slider_multipoint = 'image_pairs', True
                        points = [('small', -1.), ('middle', .5), ('large', 2.)]
                        trainer.slider_points = [{'id': name, 'strength': strength} for name, strength in points]
                        if explicit_neutral:
                            trainer.slider_points.append({'id': 'neutral', 'strength': 0.})
                        trainer.slider_diff_weight = 1.2
                        trainer.slider_anchor_bank = []
                        item = SimpleNamespace(path='cat.png', scale_to_width=64, scale_to_height=64,
                            crop_x=0, crop_y=0, crop_width=64, crop_height=64, flip_x=False, flip_y=False)
                        key = trainer._multipoint_key(item)
                        clean = {name: torch.randn(1, 4, 4, 6) for name in ('small', 'middle', 'large', 'neutral')}
                        trainer.slider_multipoint_latents = {name: {key: tensor[0]} for name, tensor in clean.items()}
                        batch = SimpleNamespace(latents=clean['neutral'], prompt_embeds=prompt(3), file_items=[item])
                        states = []
                        noising, predict = trainer._noised_state, trainer._predict
                        def noised(tensor):
                            result = noising(tensor)
                            states.append(result)
                            return result
                        trainer._noised_state = noised
                        loss = trainer.train_single_accumulation(batch, accum_scale=.4)
                        self.assertEqual(len(states), 3)
                        self.assertFalse(trainer.network.is_active)
                        self.assertEqual(trainer.network.multiplier, .7)
                        stack = torch.stack([clean[name] for name, _ in points])
                        envelope = pair_difference_weights(stack.max(0).values, stack.min(0).values, 1.2)
                        def oracle():
                            trainer.network.is_active = True
                            total = 0.
                            for (name, strength), (noisy, time, target) in zip(points, states):
                                trainer.network.multiplier = strength
                                weights = (pair_difference_weights(clean[name], clean['neutral'], 1.2)
                                           if explicit_neutral else envelope)
                                errors = (predict(noisy, time, batch.prompt_embeds) - target).square().mean(1).flatten(1)
                                total = total + (errors * weights).mean() / 3
                            return total * .4
                        expected = self.compare_gradients(trainer, params, oracle)
                        torch.testing.assert_close(loss * .4, expected, atol=2e-5, rtol=2e-5)

    def test_distillation_actual_subclass_caches_and_ideogram_both_reference_policies(self):
        for arch in ARCHES:
            policies = ('image_only', 'negative_prompt') if arch == 'ideogram4' else ('negative_prompt',)
            for policy in policies:
                objectives = ('full_guidance',) if policy == 'image_only' else ('full_guidance', 'negative_only')
                for objective in objectives:
                    with self.subTest(arch=arch, policy=policy, objective=objective), tempfile.TemporaryDirectory() as directory:
                        trainer, params, prompt = self.setup_native(QwenGuidanceDistillationTrainer, arch, cfg_reference=policy)
                        trainer.cfg_reference, trainer.distillation_objective = policy, objective
                        trainer.teacher_cfg_scale = 3.
                        positive, negative, blank = prompt(3), prompt(2, -1), prompt(4, .1)
                        neg_path, blank_path = [str(Path(directory) / name) for name in ('negative.safetensors', 'blank.safetensors')]
                        negative.save(neg_path)
                        blank.save(blank_path)
                        trainer.teacher_prompt_paths = {'positive': (neg_path, blank_path)}
                        item = SimpleNamespace(get_text_embedding_path=lambda: 'positive')
                        batch = SimpleNamespace(latents=torch.randn(1, 4, 4, 6), prompt_embeds=positive,
                            file_items=[item], loss_multiplier_list=[1.7])
                        records = []
                        predict = trainer._predict
                        def prediction(noisy, time, embeds, batch):
                            result = predict(noisy, time, embeds, batch)
                            records.append((noisy.detach(), time.detach(), embeds, result.detach()))
                            return result
                        trainer._predict = prediction
                        loss = trainer.train_single_accumulation(batch, accum_scale=.2)
                        self.assertEqual(len(records), 4 if objective == 'negative_only' else 3)
                        if arch == 'anima':
                            self.assertIsInstance(records[1][2], AnimaPromptEmbeds)
                            self.assertEqual(records[1][2].t5_input_ids.dtype, torch.long)
                        if arch == 'ideogram4':
                            zero = records[1][2] if policy == 'image_only' else records[2][2] if objective == 'negative_only' else None
                            if zero is not None:
                                self.assertEqual(zero.text_embeds[0].shape[0], 0)
                        target = (records[0][3] + 2 * (records[2][3] - records[1][3]) if objective == 'negative_only'
                                  else records[1][3] + 3 * (records[0][3] - records[1][3]))
                        self.assertFalse(trainer.network.is_active)
                        self.assertEqual(trainer.network.multiplier, .7)
                        def oracle():
                            trainer.network.is_active, trainer.network.multiplier = True, 1.
                            return (predict(records[0][0], records[0][1], positive, batch) - target).square().mean() * 1.7 * .2
                        expected = self.compare_gradients(trainer, params, oracle)
                        torch.testing.assert_close(loss * .2, expected, atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
