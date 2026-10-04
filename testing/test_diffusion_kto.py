"""CPU KTO runtime/config/replay fixtures; no diffusion weights downloaded."""
from collections import deque
from contextlib import nullcontext
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from extensions_built_in.sd_trainer import AI_TOOLKIT_EXTENSIONS
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions_built_in.sd_trainer.DiffusionKTOTrainer import DiffusionKTOTrainer
from extensions_built_in.diffusion_models.anima.anima import AnimaPromptEmbeds
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.flow_kto import FlowKTOSettings, validate_kto_config, flow_kto_terms, kto_reference_point
from toolkit.flow_training import FlowTrainingProfile

ARCHES = ('qwen_image_2', 'krea2', 'anima', 'ideogram4')


class DiffusionKTOTests(unittest.TestCase):
    def config(self, directory, arch='qwen_image_2'):
        return {'type': 'diffusion_kto', 'model': {'arch': arch}, 'network': {'type': 'lora'},
            'train': {'noise_scheduler': 'flowmatch', 'cache_text_embeddings': True},
            'datasets': [{'folder_path': directory, 'kto_label': 'liked', 'cache_latents_to_disk': True}]}

    def test_registration_and_all_model_constructors_with_unpaired_data(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new('RGB', (48, 32)).save(Path(directory) / 'a.png')
            for arch in ARCHES:
                with self.subTest(arch=arch), patch.object(DiffusionTrainer, '__init__', return_value=None), patch.object(DiffusionKTOTrainer, 'print'):
                    config = self.config(directory, arch)
                    trainer = DiffusionKTOTrainer(0, None, config)
                    self.assertEqual(trainer.kto_source_counts, {'liked': 1, 'disliked': 0})
                    self.assertEqual(trainer.flow_profile.arch, arch)
                    self.assertFalse(trainer.needs_vae_at_train_time)
                    self.assertEqual(config['datasets'][0]['caption_dropout_rate'], 0)
            extension = next(entry for entry in AI_TOOLKIT_EXTENSIONS if entry.uid == 'diffusion_kto')
            self.assertIs(extension.get_process(), DiffusionKTOTrainer)
        self.assertEqual(DatasetConfig(kto_label='disliked').kto_label, 'disliked')
        with self.assertRaises(ValueError):
            DatasetConfig(kto_label='negative')

    def test_counts_recursive_disabled_files_and_validation_before_base_init(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'nested').mkdir()
            (root / '.hidden').mkdir()
            Image.new('RGB', (16, 32)).save(root / 'nested' / 'small.png')
            Image.new('RGB', (64, 16)).save(root / 'large.jpg')
            Image.new('RGB', (16, 16)).save(root / '.hidden' / 'ignored.png')
            Image.new('RGB', (16, 16)).save(root / 'off.png.disabled', format='PNG')
            (root / 'broken.png').write_bytes(b'not an image')
            config = self.config(directory)
            config['datasets'][0]['num_repeats'] = 3
            _, counts = validate_kto_config(config)
            self.assertEqual(counts, {'liked': 2, 'disliked': 0})
            config['datasets'][0]['kto_label'] = 'bad'
            with patch.object(DiffusionTrainer, '__init__') as initialize:
                with self.assertRaisesRegex(ValueError, 'liked or disliked'):
                    DiffusionKTOTrainer(0, None, config)
                initialize.assert_not_called()

    def test_invalid_reference_replay_and_objective_settings(self):
        cases = [('network', 'type', 'dora'), ('network', 'dropout', .1),
            ('network', 'pretrained_lora_path', 'old.safetensors'),
            ('model', 'inference_lora_path', 'teacher.safetensors'),
            ('train', 'train_text_encoder', True), ('train', 'cache_text_embeddings', False),
            ('train', 'frequency_loss_type', 'lowpass'), ('train', 'min_snr_gamma', 5),
            ('train', 'gradient_accumulation_steps', 2), ('train', 'max_denoising_steps', 0)]
        cases.extend([('network', 'network_kwargs', {'rank_dropout': .1}),
            ('network', 'network_kwargs', {'module_dropout': .1}),
            ('network', 'network_kwargs', {'full_train_in_out': True}),
            ('network', 'all_layers', True), ('model', 'name_or_path', '/models/qwen-turbo')])
        with tempfile.TemporaryDirectory() as directory:
            Image.new('RGB', (16, 16)).save(Path(directory) / 'a.png')
            for section, key, value in cases:
                config = self.config(directory)
                config[section][key] = value
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    validate_kto_config(config)
            for key in ('control_path_1', 'unconditional_path', 'mask_path', 'random_crop', 'is_reg', 'shuffle_tokens', 'num_repeats'):
                config = self.config(directory)
                config['datasets'][0][key] = True
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_kto_config(config)
            config = self.config(directory)
            config['diffusion_kto'] = {'reference_estimator': 'score_window', 'score_window_size': 2}
            with self.assertRaisesRegex(ValueError, 'gradient_accumulation >= 2'):
                validate_kto_config(config)
            config['train']['gradient_accumulation'] = 2
            validate_kto_config(config)

    def test_real_recursive_dataset_buckets_preserve_labels_and_match_source_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subfolder in ('nested', '.hidden', '_controls/nested'):
                (root / subfolder).mkdir(parents=True)
            for path, size in [('wide.png', (160, 64)), ('nested/tall.png', (64, 160)),
                               ('.hidden/off.png', (64, 64)), ('_controls/nested/off.png', (64, 64))]:
                Image.new('RGB', size).save(root / path)
            Image.new('RGB', (64, 64)).save(root / 'disabled.png.disabled', format='PNG')
            config = self.config(directory)
            config['datasets'][0].update(kto_label='disliked', num_repeats=2, resolution=128, loss_multiplier=.5)
            _, counts = validate_kto_config(config)
            self.assertEqual(counts, {'liked': 0, 'disliked': 2})
            sd = SimpleNamespace(arch='qwen_image_2', vae=None, unet=None, load_rgba=False,
                use_raw_control_images=False, adapter=None,
                get_bucket_divisibility=lambda: 32, get_latent_space_version=lambda: 'test-latents',
                get_text_embedding_space_version=lambda: 'test-encoder', encode_control_in_text_embeddings=False,
                te_padding_side='right')
            with patch.object(AiToolkitDataset, 'cache_latents_all_latents'):
                dataset = AiToolkitDataset(DatasetConfig(**config['datasets'][0]), batch_size=1, sd=sd)
            self.assertEqual(len(dataset.file_list), 4)  # repeats, not invented pairs
            self.assertEqual({Path(item.path).name for item in dataset.file_list}, {'wide.png', 'tall.png'})
            self.assertGreater(len(dataset.buckets), 1)
            for item in dataset.file_list:
                self.assertEqual(item.dataset_config.kto_label, 'disliked')
                self.assertEqual(item.dataset_config.loss_multiplier, .5)
                self.assertEqual(item.crop_width % 32, 0)
                self.assertEqual(item.crop_height % 32, 0)

    def fixture(self, arch, estimator='score_window', window=2):
        trainer = DiffusionKTOTrainer.__new__(DiffusionKTOTrainer)
        trainer.kto_settings = FlowKTOSettings.parse({'beta': 1.3, 'liked_weight': 1.2,
            'disliked_weight': .7, 'reference_estimator': estimator, 'score_window_size': window})
        trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': arch})
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=0, max_denoising_steps=999,
            gradient_accumulation_steps=1, optimizer='adamw', max_grad_norm=1000)
        trainer.device_torch = torch.device('cpu')
        trainer.network = SimpleNamespace(is_active=False, multiplier=.6)
        trainer._kto_replay_queue = deque()
        trainer._kto_preparing = False
        trainer.additional_logs = {}
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward(), clip_grad_norm_=lambda *args: None)
        parameter = torch.nn.Parameter(torch.tensor(.25))
        frozen = torch.nn.Linear(1, 1).requires_grad_(False)
        frozen.train()  # do not turn checkpointing off while scoring/replaying
        calls = []
        def predict(**kwargs):
            prompt = kwargs['conditional_embeddings']
            if arch == 'anima':
                self.assertIsInstance(prompt, AnimaPromptEmbeds)
                self.assertEqual(prompt.t5_input_ids.dtype, torch.long)
            random = torch.rand_like(kwargs['latents']) * .2
            prediction = kwargs['latents'] * .1 + random + parameter * float(trainer.network.is_active)
            calls.append((trainer.network.is_active, torch.is_grad_enabled(), kwargs['latents'].detach().clone(),
                kwargs['timestep'].clone(), random.detach().clone(), kwargs['batch'], frozen.training))
            return prediction
        trainer.sd = SimpleNamespace(arch=arch, unet=frozen, predict_noise=predict, is_multistage=False)
        trainer.optimizer = torch.optim.SGD([parameter], lr=.05)
        trainer.params = [parameter]
        trainer.timer = lambda name: nullcontext()
        trainer.lr_scheduler = SimpleNamespace(step=lambda: None)
        trainer.is_grad_accumulation_step = False
        trainer.adapter = trainer.ema = trainer.embedding = None
        trainer.model_config = SimpleNamespace(low_vram=False)
        trainer.steps_this_boundary = 0
        trainer.end_of_training_loop = lambda: None
        return trainer, parameter, calls

    def batches(self, arch, counts=(1, 3)):
        result = []
        for index, count in enumerate(counts):
            if arch == 'anima':
                prompt = AnimaPromptEmbeds(torch.ones(count, 2 + index, 4),
                    torch.ones(count, 3, dtype=torch.long), torch.ones(count, 2 + index, dtype=torch.long),
                    torch.ones(count, 3, dtype=torch.long))
            else:
                prompt = AdvancedPromptEmbeds(text_embeds=[torch.ones(2 + index, 4) for _ in range(count)])
            result.append(SimpleNamespace(latents=torch.full((count, 1, 2, 2 + 2 * index), .1 * index),
                prompt_embeds=prompt, file_items=[SimpleNamespace(dataset_config=SimpleNamespace(
                    kto_label='liked' if (index + i) % 2 == 0 else 'disliked')) for i in range(count)],
                loss_multiplier_list=[1 + i * .2 for i in range(count)]))
        return result

    def test_helper_is_shared_by_reference_scoring_policy_and_replay(self):
        for estimator in ('batch_mean', 'score_window'):
            with self.subTest(estimator=estimator):
                trainer, parameter, calls = self.fixture('qwen_image_2', estimator=estimator)
                helper = SimpleNamespace(is_active=True,
                    weight=torch.nn.Parameter(torch.tensor(.75), requires_grad=False))
                trainer.sd.assistant_lora = helper
                original = trainer.sd.predict_noise
                def predict(**kwargs):
                    self.assertTrue(helper.is_active)
                    return original(**kwargs) + helper.weight
                trainer.sd.predict_noise = predict
                before = parameter.detach().clone()
                trainer.hook_train_loop(self.batches('qwen_image_2'))
                self.assertTrue(any(not active for active, *_ in calls))
                self.assertTrue(any(active for active, *_ in calls))
                self.assertFalse(torch.equal(before, parameter))
                self.assertIsNone(helper.weight.grad)
                self.assertTrue(helper.is_active)
                self.assertEqual(helper.weight.item(), .75)

    def test_real_optimizer_hook_matches_full_graph_oracle_all_models(self):
        for arch in ARCHES:
            with self.subTest(arch=arch):
                trainer, parameter, seen = self.fixture(arch)
                batches = self.batches(arch)
                captured = []
                original = trainer._prepare_replay
                def prepare(values):
                    loss = original(values)
                    captured.extend(list(trainer._kto_replay_queue))
                    return loss
                trainer._prepare_replay = prepare
                torch.manual_seed(42)
                result = trainer.hook_train_loop(batches)  # REAL SDTrainer clipping/optimizer/scheduler hook
                oracle = torch.tensor(.25, requires_grad=True)
                policy, reference, labels, weights = [], [], [], []
                for record, score_call in zip(captured, (seen[1], seen[3])):
                    base = record['noisy'] * .1 + score_call[4]
                    policy.append((base + oracle - record['target']).square().flatten(1).mean(1))
                    reference.append((base - record['target']).square().flatten(1).mean(1))
                    labels.append(record['liked'])
                    weights.append(record['weights'])
                policy, reference = torch.cat(policy), torch.cat(reference)
                loss, _, _ = flow_kto_terms(policy, reference, torch.cat(labels), trainer.kto_settings)
                expected = (loss * torch.cat(weights)).mean()
                expected.backward()
                torch.testing.assert_close(parameter, .25 - .05 * oracle.grad)
                self.assertAlmostEqual(result['loss'], expected.item(), places=6)
                self.assertEqual(len(seen), 6)
                for reference_i, policy_i, replay_i in ((0, 1, 4), (2, 3, 5)):
                    for j in (2, 3, 4):
                        torch.testing.assert_close(seen[reference_i][j], seen[policy_i][j])
                        torch.testing.assert_close(seen[policy_i][j], seen[replay_i][j])
                self.assertTrue(all(call[6] for call in seen))
                self.assertFalse(trainer.network.is_active)
                self.assertEqual(trainer.network.multiplier, .6)
                self.assertFalse(trainer._kto_replay_queue)
                self.assertTrue(all(not r['noisy'].requires_grad and r['noisy'].device.type == 'cpu' for r in captured))
                self.assertTrue(all(p.grad is None for p in trainer.sd.unet.parameters()))
                self.assertEqual(trainer.additional_logs['kto/liked_count'], 2)
                self.assertEqual(trainer.additional_logs['kto/disliked_count'], 2)

    def test_batch_mean_standalone_short_tail_and_failure_cleanup(self):
        trainer, parameter, _ = self.fixture('qwen_image_2', 'batch_mean')
        batch = self.batches('qwen_image_2', (1,))[0]
        loss = trainer.train_single_accumulation(batch, .5)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(parameter.grad)
        self.assertFalse(trainer._kto_replay_queue)
        trainer, parameter, _ = self.fixture('ideogram4')
        batches = self.batches('ideogram4', (1, 1, 1))
        trainer.hook_train_loop(batches)
        self.assertEqual(trainer.additional_logs['kto/window_count'], 2)
        self.assertEqual(trainer.additional_logs['kto/short_tail_batches'], 1)
        before = parameter.detach().clone()
        with patch.object(trainer.accelerator, 'backward', side_effect=RuntimeError('failed backward')):
            with self.assertRaisesRegex(RuntimeError, 'failed backward'):
                trainer.hook_train_loop(batches)
        torch.testing.assert_close(parameter, before)
        self.assertIsNone(parameter.grad)
        self.assertFalse(trainer._kto_replay_queue)
        self.assertFalse(trainer._kto_preparing)
        self.assertFalse(trainer.network.is_active)
        self.assertEqual(trainer.network.multiplier, .6)
        self.assertTrue(trainer.sd.unet.training)

    def test_save_only_at_completed_window_and_resume_has_no_estimator_state(self):
        trainer, parameter, _ = self.fixture('anima')
        batches = self.batches('anima')
        trainer.network.multiplier = 1
        trainer._prepare_replay(batches)
        with patch.object(DiffusionTrainer, 'save') as save:
            with self.assertRaisesRegex(RuntimeError, 'completed score/replay'):
                trainer.save(step=1)
            save.assert_not_called()
            trainer._kto_replay_queue.clear()
            trainer.save(step=0)
            save.assert_called_once_with(step=0)
        # A completed-boundary restart needs model/optimizer and normal RNG only.
        trainer, parameter, _ = self.fixture('anima')
        torch.manual_seed(17)
        trainer.hook_train_loop(batches)
        state, rng = copy.deepcopy(trainer.optimizer.state_dict()), torch.get_rng_state()
        saved = parameter.detach().clone()
        trainer.hook_train_loop(batches)
        expected = parameter.detach().clone()
        restored, restored_parameter, _ = self.fixture('anima')
        with torch.no_grad():
            restored_parameter.copy_(saved)
        restored.optimizer.load_state_dict(state)
        torch.set_rng_state(rng)
        restored.hook_train_loop(batches)
        torch.testing.assert_close(restored_parameter, expected)


if __name__ == '__main__':
    unittest.main()
