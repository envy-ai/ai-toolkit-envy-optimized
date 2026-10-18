"""Focused filesystem tests for copying a job from its latest complete state."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from toolkit import job_fork

REPO_TMP = Path(__file__).resolve().parents[1] / 'tmp'


class JobForkTests(unittest.TestCase):
    def temp_dir(self):
        REPO_TMP.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=REPO_TMP)

    def ordinary_config(self, name='source'):
        return {
            'config': {
                'name': name,
                'process': [{
                    'type': 'sd_trainer',
                    'model': {'arch': 'qwen_image_2'},
                    'network': {'type': 'lora'},
                }],
            },
        }

    def sliderspace_config(self, name='source'):
        config = self.ordinary_config(name)
        config['config']['process'][0].update(
            type='sliderspace', sliderspace={'num_directions': 2},
        )
        return config

    def write_checkpoint(self, folder, name, step, *, mtime=None):
        path = Path(folder) / name
        save_file(
            {'weight': torch.tensor([float(step)])},
            str(path),
            metadata={'training_info': json.dumps({'step': step})},
        )
        if mtime is not None:
            os.utime(path, ns=(mtime, mtime))
        return path

    def write_optimizer(self, folder, *, mtime=None, value=1):
        path = Path(folder) / 'optimizer.pt'
        torch.save({'state': {}, 'param_groups': [{'params': [], 'lr': 0.001, 'marker': value}]}, path)
        if mtime is not None:
            os.utime(path, ns=(mtime, mtime))
        return path

    @staticmethod
    def stamp(path):
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns]

    def write_sliderspace_generation(self, folder, step, *, signature='sig'):
        folder = Path(folder)
        state = folder / 'sliderspace_state'
        state.mkdir(parents=True, exist_ok=True)
        bank = state / f'bank_{step:09d}.safetensors'
        save_file(
            {'direction_1.weight': torch.tensor([float(step)])},
            str(bank),
            metadata={
                'training_info': json.dumps({'step': step}),
                'ss_sliderspace_signature': signature,
                'ss_sliderspace_directions': '2',
            },
        )
        optimizer = bank.with_suffix('.optimizer.pt')
        torch.save({'state': {}, 'param_groups': [{'params': [], 'lr': 0.001}]}, optimizer)
        exports = []
        for direction in range(1, 3):
            export = folder / f'source_direction_{direction:02d}_{step:09d}.safetensors'
            save_file(
                {f'direction_{direction}.weight': torch.tensor([float(step)])},
                str(export),
                metadata={'ss_sliderspace_direction': str(direction)},
            )
            exports.append(export.name)
        record_files = [bank, optimizer, *(folder / name for name in exports)]
        record = {
            'step': step,
            'exports': exports,
            'signature': signature,
            'files': {path.name: self.stamp(path) for path in record_files},
        }
        (state / f'checkpoint_{step:09d}.json').write_text(json.dumps(record))
        (state / 'checkpoint_final.json').write_text(json.dumps(record))
        discovery = folder / 'sliderspace_discovery' / signature
        discovery.mkdir(parents=True, exist_ok=True)
        feature_file = discovery / 'features.safetensors'
        save_file({'features': torch.tensor([float(step)])}, str(feature_file))
        cache_manifest = {
            'signature': signature,
            'entries': [],
            'files': {},
            'feature_file': self.stamp(feature_file),
        }
        (discovery / 'manifest.json').write_text(json.dumps(cache_manifest))
        return record

    def test_ordinary_selects_newest_ctime_checkpoint_and_reads_its_metadata_step(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            # The final export is newest by ctime but its step is lower than an
            # older numbered save; metadata supplies its actual resume step.
            self.write_checkpoint(source, 'source_000000002.safetensors', 2)
            self.write_checkpoint(source, 'source_000000031.safetensors', 31)
            self.write_checkpoint(source, 'source.safetensors', 12)
            self.write_optimizer(source)
            (source / 'unrelated.txt').write_text('history should not be copied')
            source_config = self.ordinary_config()
            target_config = self.ordinary_config('fork')

            selected = job_fork.inspect_job(str(source), 'source', source_config)
            self.assertEqual(selected['step'], 12)
            self.assertEqual(selected['checkpoint'], 'source.safetensors')

            summary = job_fork.fork_job(
                str(source), str(target), 'source', 'fork', source_config, target_config,
            )
            checkpoint = target / 'fork.safetensors'
            self.assertEqual(summary['step'], 12)
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((target / 'optimizer.pt').is_file())
            self.assertEqual({path.name for path in target.iterdir()}, {
                'fork.safetensors', 'optimizer.pt', '.fork_resume.json',
            })
            self.assertNotEqual(source.joinpath('source.safetensors').stat().st_ino,
                                checkpoint.stat().st_ino)
            self.assertEqual(
                source.joinpath('source.safetensors').read_bytes(),
                checkpoint.read_bytes(),
            )
            self.assertNotEqual((source / 'optimizer.pt').stat().st_ino,
                                (target / 'optimizer.pt').stat().st_ino)
            self.assertEqual(torch.load(target / 'optimizer.pt', weights_only=True)['param_groups'][0]['marker'], 1)
            self.assertTrue((source / 'unrelated.txt').is_file())
            self.assertTrue((source / 'source.safetensors').is_file())

    def test_final_basename_is_valid_and_optimizer_may_have_a_later_internal_step(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            checkpoint = self.write_checkpoint(source, 'source.safetensors', 47)
            optimizer = source / 'optimizer.pt'
            # Adam can have fewer per-parameter updates than global steps
            # after resume or conditional parameter updates.
            torch.save({
                'state': {0: {
                    'step': torch.tensor(3.),
                    'exp_avg': torch.ones(2),
                    'exp_avg_sq': torch.ones(2),
                }},
                'param_groups': [{'params': [0], 'lr': 0.001}],
            }, optimizer)
            older_mtime = checkpoint.stat().st_mtime_ns - 1_000_000
            os.utime(optimizer, ns=(older_mtime, older_mtime))
            result = job_fork.inspect_job(str(source), 'source', self.ordinary_config())
            self.assertEqual(result['step'], 47)
            self.assertEqual(result['checkpoint'], 'source.safetensors')
            loaded = torch.load(optimizer, weights_only=True)
            self.assertEqual(loaded['state'][0]['step'].item(), 3)
            self.assertTrue(any('optimizer file predates' in warning for warning in result['warnings']))

    def test_copy_failure_removes_partial_target_and_preserves_source(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            checkpoint = self.write_checkpoint(source, 'source_000000005.safetensors', 5)
            self.write_optimizer(source)
            original = job_fork.shutil.copy2
            calls = 0

            def fail_during_copy(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError('injected copy failure')
                return original(*args, **kwargs)

            with patch.object(job_fork.shutil, 'copy2', side_effect=fail_during_copy):
                with self.assertRaisesRegex(OSError, 'injected copy failure'):
                    job_fork.fork_job(
                        str(source), str(target), 'source', 'fork',
                        self.ordinary_config(), self.ordinary_config('fork'),
                    )
            self.assertFalse(target.exists())
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((source / 'optimizer.pt').is_file())

    def test_source_checkpoint_change_during_copy_rolls_back_target(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            checkpoint = self.write_checkpoint(source, 'source_000000005.safetensors', 5)
            optimizer = self.write_optimizer(source)
            original = job_fork.shutil.copy2
            changed = False

            def mutate_after_checkpoint_copy(source_path, destination_path, *args, **kwargs):
                nonlocal changed
                result = original(source_path, destination_path, *args, **kwargs)
                if Path(source_path) == checkpoint and not changed:
                    changed = True
                    updated = max(checkpoint.stat().st_mtime_ns, optimizer.stat().st_mtime_ns) + 1_000_000
                    os.utime(checkpoint, ns=(updated, updated))
                    os.utime(optimizer, ns=(updated + 1, updated + 1))
                return result

            with patch.object(job_fork.shutil, 'copy2', side_effect=mutate_after_checkpoint_copy):
                with self.assertRaisesRegex(ValueError, 'changed while forking'):
                    job_fork.fork_job(
                        str(source), str(target), 'source', 'fork',
                        self.ordinary_config(), self.ordinary_config('fork'),
                    )
            self.assertTrue(changed)
            self.assertFalse(target.exists())
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(optimizer.is_file())

    def test_diffusers_full_model_checkpoint_is_copied_with_metadata_and_nested_files(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source, target = root / 'source', root / 'fork'
            checkpoint = source / 'source_000000009'
            weights = checkpoint / 'unet' / 'diffusion_pytorch_model.bin'
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b'tiny model payload')
            (checkpoint / 'aitk_meta.yaml').write_text('training_info:\n  step: 9\n')
            (checkpoint / 'unet' / 'config.json').write_text('{"layers": 2}')
            source_config = {
                'config': {'name': 'source', 'process': [{
                    'type': 'sd_trainer', 'model': {'arch': 'sdxl'},
                    'network': None, 'train': {'train_unet': True},
                }]},
            }
            target_config = json.loads(json.dumps(source_config))
            target_config['config']['name'] = 'fork'
            self.write_optimizer(source)

            summary = job_fork.fork_job(
                str(source), str(target), 'source', 'fork', source_config, target_config,
            )
            copied = target / 'fork_000000009'
            self.assertEqual(summary, {
                'step': 9, 'checkpoint': 'fork_000000009', 'layout': 'diffusers', 'warnings': [],
            })
            self.assertEqual((copied / 'aitk_meta.yaml').read_text(), 'training_info:\n  step: 9\n')
            self.assertEqual((copied / 'unet' / 'config.json').read_text(), '{"layers": 2}')
            copied_weights = copied / 'unet' / 'diffusion_pytorch_model.bin'
            self.assertEqual(copied_weights.read_bytes(), b'tiny model payload')
            self.assertNotEqual(weights.stat().st_ino, copied_weights.stat().st_ino)
            self.assertTrue((target / 'optimizer.pt').is_file())

    def test_fork_allows_guidance_loss_performance_and_memory_edits_but_rejects_rank_or_model(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            self.write_checkpoint(source, 'source_000000005.safetensors', 5)
            self.write_optimizer(source)
            source_config = self.ordinary_config()
            source_process = source_config['config']['process'][0]
            source_process['network']['layer_offloading'] = False
            source_process['train'] = {'optimizer': 'adamw', 'learning_rate': 0.001}
            allowed_config = json.loads(json.dumps(source_config))
            allowed_config['config']['name'] = 'fork_allowed'
            allowed_process = allowed_config['config']['process'][0]
            allowed_process['network']['layer_offloading'] = True
            allowed_process['train'].update({
                'learning_rate': 0.0005, 'do_guidance_loss': True, 'loss_weight': 0.25,
            })
            allowed_process['performance'] = {'low_vram': True, 'gradient_checkpointing': True}
            summary = job_fork.fork_job(
                str(source), str(root / 'fork_allowed'), 'source', 'fork_allowed',
                source_config, allowed_config,
            )
            self.assertEqual(summary['step'], 5)

            for mutate in (
                lambda process: process['network'].update(linear=32),
                lambda process: process['model'].update(arch='sdxl'),
            ):
                config = json.loads(json.dumps(source_config))
                config['config']['name'] = 'fork_rejected'
                mutate(config['config']['process'][0])
                with self.assertRaisesRegex(ValueError, 'preserve the source model, network'):
                    job_fork.fork_job(
                        str(source), str(root / 'fork_rejected'), 'source', 'fork_rejected',
                        source_config, config,
                    )
                self.assertFalse((root / 'fork_rejected').exists())

    def test_sliderspace_forks_the_manifest_published_generation_and_renames_exports(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source, target = root / 'source', root / 'fork'
            source.mkdir()
            record = self.write_sliderspace_generation(source, 24)
            # An older generation exists but the final marker selects step 24.
            (source / 'sliderspace_state' / 'checkpoint_000000004.json').write_text(
                json.dumps({'step': 4, 'exports': [], 'signature': 'sig'})
            )
            summary = job_fork.fork_job(
                str(source), str(target), 'source', 'fork',
                self.sliderspace_config(), self.sliderspace_config('fork'),
            )
            self.assertEqual(summary['step'], 24)
            copied_state = target / 'sliderspace_state'
            self.assertTrue((copied_state / 'bank_000000024.safetensors').is_file())
            self.assertTrue((copied_state / 'bank_000000024.optimizer.pt').is_file())
            self.assertTrue((copied_state / 'checkpoint_final.json').is_file())
            copied_marker = json.loads((copied_state / 'checkpoint_final.json').read_text())
            self.assertEqual(copied_marker['step'], 24)
            self.assertEqual(copied_marker['signature'], record['signature'])
            self.assertEqual(len(copied_marker['exports']), 2)
            self.assertEqual(set(copied_marker['exports']), {
                'fork_direction_01_000000024.safetensors',
                'fork_direction_02_000000024.safetensors',
            })
            for export in copied_marker['exports']:
                self.assertTrue((target / export).is_file())
            self.assertFalse((copied_state / 'checkpoint_000000004.json').exists())
            copied_discovery = target / 'sliderspace_discovery' / 'sig'
            self.assertTrue((copied_discovery / 'features.safetensors').is_file())
            self.assertTrue((copied_discovery / 'manifest.json').is_file())

    def test_rejects_missing_optimizer_invalid_optimizer_and_corrupt_checkpoint(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            self.write_checkpoint(source, 'source_000000005.safetensors', 5)
            with self.assertRaises((ValueError, FileNotFoundError)):
                job_fork.inspect_job(str(source), 'source', self.ordinary_config())

            torch.save({'state': []}, source / 'optimizer.pt')
            with self.assertRaises((ValueError, KeyError, TypeError)):
                job_fork.inspect_job(str(source), 'source', self.ordinary_config())

            (source / 'optimizer.pt').unlink()
            corrupt = source / 'source_000000006.safetensors'
            corrupt.write_bytes(b'not a safetensors file')
            with self.assertRaises(Exception):
                job_fork.inspect_job(str(source), 'source', self.ordinary_config())

    def test_corrupt_older_history_does_not_override_latest_valid_checkpoint(self):
        with self.temp_dir() as temporary:
            source = Path(temporary) / 'source'
            source.mkdir()
            (source / 'source_000000001.safetensors').write_bytes(b'corrupt retained history')
            latest = self.write_checkpoint(source, 'source_000000002.safetensors', 2)
            self.write_optimizer(source)
            result = job_fork.inspect_job(str(source), 'source', self.ordinary_config())
            self.assertEqual(result['step'], 2)
            self.assertEqual(result['checkpoint'], latest.name)

    def test_missing_checkpoint_never_falls_back_to_an_empty_resume(self):
        with self.temp_dir() as temporary:
            source = Path(temporary) / 'source'
            source.mkdir()
            with self.assertRaisesRegex(ValueError, 'No supported resumable checkpoint'):
                job_fork.inspect_job(str(source), 'source', self.ordinary_config())

    def test_rejects_unsupported_training_layout_and_changed_sliderspace_manifest(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            self.write_checkpoint(source, 'source_000000005.safetensors', 5)
            self.write_optimizer(source)
            unsupported = self.ordinary_config()
            unsupported['config']['process'][0]['adapter'] = {'type': 'controlnet'}
            with self.assertRaises((ValueError, NotImplementedError)):
                job_fork.inspect_job(str(source), 'source', unsupported)

            sliderspace = root / 'slider'
            sliderspace.mkdir()
            self.write_sliderspace_generation(sliderspace, 24)
            marker = sliderspace / 'sliderspace_state' / 'checkpoint_final.json'
            marker_data = json.loads(marker.read_text())
            marker_data['files']['bank_000000024.safetensors'][0] += 1
            marker.write_text(json.dumps(marker_data))
            with self.assertRaisesRegex(ValueError, 'manifest|files|changed|incomplete'):
                job_fork.inspect_job(str(sliderspace), 'source', self.sliderspace_config())

    def test_fork_resume_accepts_new_complete_generations_and_allowed_training_edits(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source, target = root / 'source', root / 'fork'
            source.mkdir()
            self.write_checkpoint(source, 'source_000000005.safetensors', 5)
            self.write_optimizer(source)
            config = self.ordinary_config()
            config['config']['process'][0]['train'] = {'optimizer': 'adamw'}
            fork_config = self.ordinary_config('fork')
            fork_config['config']['process'][0]['train'] = {
                'optimizer': 'adamw', 'learning_rate': 0.001,
            }
            job_fork.fork_job(
                str(source), str(target), 'source', 'fork', config, fork_config,
            )
            self.assertEqual(job_fork.validate_fork_resume(str(target), fork_config)['step'], 5)

            next_checkpoint = self.write_checkpoint(target, 'fork_000000006.safetensors', 6)
            optimizer = target / 'optimizer.pt'
            latest_time = max(next_checkpoint.stat().st_mtime_ns, optimizer.stat().st_mtime_ns) + 1_000_000
            os.utime(optimizer, ns=(latest_time, latest_time))
            fork_config['config']['process'][0]['train']['learning_rate'] = 0.0005
            self.assertEqual(job_fork.validate_fork_resume(str(target), fork_config)['step'], 6)

            changed_model = json.loads(json.dumps(fork_config))
            changed_model['config']['process'][0]['model']['arch'] = 'sdxl'
            with self.assertRaisesRegex(ValueError, 'model/network/optimizer layout changed'):
                job_fork.validate_fork_resume(str(target), changed_model)

            start_step = json.loads(json.dumps(fork_config))
            start_step['config']['process'][0]['train']['start_step'] = 7
            with self.assertRaisesRegex(ValueError, 'without train.start_step'):
                job_fork.validate_fork_resume(str(target), start_step)

    def test_pretrained_lora_setting_is_preserved_while_forking_and_resuming(self):
        with self.temp_dir() as temporary:
            root = Path(temporary)
            source, target = root / 'source', root / 'fork'
            source.mkdir()
            self.write_checkpoint(source, 'source_000000005.safetensors', 5)
            self.write_optimizer(source)
            source_config = self.ordinary_config()
            source_config['config']['process'][0]['network']['pretrained_lora_path'] = '/missing/retained.safetensors'
            target_config = json.loads(json.dumps(source_config))
            target_config['config']['name'] = 'fork'
            result = job_fork.fork_job(
                str(source), str(target), 'source', 'fork', source_config, target_config,
            )
            self.assertEqual(result['step'], 5)
            self.assertEqual(job_fork.validate_fork_resume(str(target), target_config)['step'], 5)


if __name__ == '__main__':
    unittest.main()
