import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from toolkit.config_modules import SampleConfig, SaveConfig
from toolkit.lowest_loss import (
    checkpoint_step, is_record_low_save_due, load_record_low_checkpoints,
    record_low_retention, write_record_low_checkpoints,
)


class RecordLowCheckpointTests(unittest.TestCase):
    def test_sampling_toggle_and_scheduled_overlap(self):
        from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

        cases = [
            # step, record low, sample record lows, disabled, interval, start, reason
            (51, True, False, False, 400, 0, None),
            (51, True, True, False, 400, 0, 'record low'),
            (51, False, True, False, 400, 0, None),
            (400, True, False, False, 400, 0, 'scheduled'),
            (400, True, True, False, 400, 0, 'scheduled and record low'),
            (400, False, False, False, 400, 0, 'scheduled'),
            (400, True, True, True, 400, 0, None),
            (400, True, False, False, 400, 500, None),
            (400, True, True, False, 400, 500, 'record low'),
            (400, True, False, False, 0, 0, None),
            (400, True, True, False, 0, 0, 'record low'),
        ]
        for step, record_low, enabled, disabled, interval, start, reason in cases:
            with self.subTest(step=step, enabled=enabled, disabled=disabled, interval=interval, start=start):
                process = object.__new__(BaseSDTrainProcess)
                process.step_num = step
                process.train_config = SimpleNamespace(disable_sampling=disabled)
                process.sample_config = SampleConfig(sample_every=interval, sample_start_step=start)
                process.save_config = SaveConfig(sample_on_record_low=enabled)
                self.assertEqual(process._get_step_sample_reason(record_low), reason)

    def test_live_config_reloads_record_low_sampling_toggle(self):
        from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.job_config.json'
            config = {
                'job': 'extension',
                'config': {'name': 'model', 'process': [{
                    'sample': {'sample_every': 400},
                    'save': {'sample_on_record_low': False, 'save_every': 999},
                }]},
            }
            path.write_text(json.dumps(config))
            process = object.__new__(BaseSDTrainProcess)
            process.job = SimpleNamespace(config_path=str(path))
            process.process_id = 0
            process.step_num = 125
            process._live_sample_config_mtime = None
            process.train_config = SimpleNamespace(disable_sampling=False)
            process.save_config = SaveConfig(sample_on_record_low=True, save_every=100)

            process._refresh_live_sample_config()

            self.assertFalse(process.save_config.sample_on_record_low)
            self.assertIsNone(process._get_step_sample_reason(True))
            self.assertEqual(process.save_config.save_every, 100)

            config['config']['process'][0]['save']['sample_on_record_low'] = True
            previous_mtime = path.stat().st_mtime
            path.write_text(json.dumps(config))
            os.utime(path, (previous_mtime + 1, previous_mtime + 1))
            process._refresh_live_sample_config()

            self.assertTrue(process.save_config.sample_on_record_low)
            self.assertEqual(process._get_step_sample_reason(True), 'record low')
            self.assertEqual(process.save_config.save_every, 100)

    def test_comparison_uses_only_existing_saves_in_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / '.record_low_checkpoints.json'
            old = root / 'model_000000050.safetensors'
            recent = root / 'model_000003100.safetensors'
            old.touch()
            recent.touch()
            write_record_low_checkpoints(str(index), [
                {'step': 50, 'loss': 0.01, 'checkpoint': old.name, 'scheduled': False},
                {'step': 3100, 'loss': 0.3, 'checkpoint': recent.name, 'scheduled': False},
            ])
            records = load_record_low_checkpoints(str(index), str(root))
            self.assertFalse(is_record_low_save_due(records, 49, 0.001, 3000, 50))
            self.assertTrue(is_record_low_save_due(records, 3101, 0.2, 3000, 50))
            self.assertFalse(is_record_low_save_due(records, 3101, 0.3, 3000, 50))
            recent.unlink()
            records = load_record_low_checkpoints(str(index), str(root))
            self.assertEqual([record['step'] for record in records], [50])
            self.assertTrue(is_record_low_save_due(records, 3101, 0.8, 3000, 50))

    def test_retention_keeps_lowest_records_across_run_and_scheduled_overlap(self):
        records = [
            {'step': 50, 'loss': 0.01, 'scheduled': False},
            {'step': 3100, 'loss': 0.4, 'scheduled': False},
            {'step': 3200, 'loss': 0.3, 'scheduled': True},
            {'step': 3300, 'loss': 0.35, 'scheduled': False},
        ]
        best, record_only = record_low_retention(records, 2)
        self.assertEqual(best, {50, 3200})
        self.assertEqual(record_only, {50, 3100, 3300})

    def test_invalid_losses_and_checkpoint_names(self):
        self.assertFalse(is_record_low_save_due([], 50, float('nan'), 3000, 50))
        self.assertEqual(checkpoint_step('model_000003100.safetensors'), 3100)
        self.assertEqual(checkpoint_step('model_000003100'), 3100)
        self.assertIsNone(checkpoint_step('model-lora.safetensors'))

    def test_config_defaults_and_invalid_limits(self):
        config = SaveConfig()
        self.assertEqual(config.record_low_window_size, 3000)
        self.assertEqual(config.record_low_start_step, 50)
        self.assertTrue(config.sample_on_record_low)
        self.assertEqual(config.max_record_low_saves_to_keep, 5)
        with self.assertRaises(ValueError):
            SaveConfig(record_low_window_size=0)
        with self.assertRaises(ValueError):
            SaveConfig(max_record_low_saves_to_keep=0)

    def test_cleanup_retains_best_record_separately_from_latest_scheduled(self):
        from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for step in (100, 200, 250, 300):
                paths[step] = root / f'model_{step:09d}.safetensors'
                paths[step].touch()
            index = root / '.record_low_checkpoints.json'
            write_record_low_checkpoints(str(index), [
                {'step': 200, 'loss': 0.2, 'checkpoint': paths[200].name, 'scheduled': False},
                {'step': 250, 'loss': 0.3, 'checkpoint': paths[250].name, 'scheduled': False},
            ])
            process = object.__new__(BaseSDTrainProcess)
            process.accelerator = SimpleNamespace(is_main_process=True)
            process.save_root = str(root)
            process.job = SimpleNamespace(name='model')
            process.embed_config = None
            process.sd = SimpleNamespace()
            process.step_num = 300
            process.save_config = SaveConfig(max_step_saves_to_keep=1, max_record_low_saves_to_keep=1)
            process._record_low_index_path = str(index)
            process._record_low_pending_step = None
            process._comfy_background_threads = []

            process.clean_up_saves()

            self.assertFalse(paths[100].exists())
            self.assertTrue(paths[200].exists())
            self.assertFalse(paths[250].exists())
            self.assertTrue(paths[300].exists())
            self.assertEqual(
                [record['step'] for record in load_record_low_checkpoints(str(index), str(root))],
                [200],
            )

    def test_ui_save_overrides_forward_record_low_arguments(self):
        from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
        from extensions_built_in.sd_trainer.UITrainer import UITrainer

        for trainer_class in (DiffusionTrainer, UITrainer):
            with self.subTest(trainer=trainer_class.__name__):
                trainer = object.__new__(trainer_class)
                trainer.maybe_stop = Mock()
                trainer.update_status = Mock()
                with patch.object(SDTrainer, 'save') as base_save:
                    trainer.save(50, record_low_loss=0.2, scheduled_save=True)
                base_save.assert_called_once_with(
                    50, record_low_loss=0.2, scheduled_save=True,
                )


if __name__ == '__main__':
    unittest.main()
