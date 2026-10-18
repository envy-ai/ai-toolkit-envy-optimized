import sqlite3
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from safetensors.torch import save_file

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.UITrainer import UITrainer
from toolkit.ui_checkpoint import UICheckpointControls
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


class PauseCheckpointTests(unittest.TestCase):
    def trainer(self, root, cls, stop=True, step=123, db_step=None):
        db = Path(root) / 'job.db'
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE Job (id TEXT, stop INTEGER, save_now INTEGER, step INTEGER)')
            conn.execute('INSERT INTO Job VALUES (?, ?, 1, ?)',
                         ('test', int(stop), step if db_step is None else db_step))
        trainer = object.__new__(cls)
        trainer.is_ui_trainer = True
        trainer.job_id = 'test'
        trainer.sqlite_db_path = str(db)
        trainer.is_grad_accumulation_step = False
        trainer.step_num = step
        trainer.progress_bar = None
        trainer.optimizer = Mock()
        trainer.ensure_params_requires_grad = Mock()
        trainer.save = Mock()

        def update(key, value):
            with sqlite3.connect(db) as conn:
                conn.execute(f'UPDATE Job SET {key} = ? WHERE id = ?', (value, 'test'))
        trainer.update_db_key = update
        return trainer

    @patch('toolkit.ui_checkpoint.flush')
    def test_pause_defers_interrupt_until_completed_save_in_both_trainers(self, flush):
        for cls in (DiffusionTrainer, UITrainer):
            with self.subTest(trainer=cls.__name__), tempfile.TemporaryDirectory() as root:
                trainer = self.trainer(root, cls)
                self.assertFalse(trainer.should_stop())
                trainer.is_grad_accumulation_step = True
                trainer.maybe_save()
                trainer.save.assert_not_called()
                self.assertFalse(trainer.should_stop())

                def save(step):
                    self.assertEqual(step, 123)
                    self.assertTrue(trainer.should_save())
                    self.assertFalse(trainer.should_stop())
                    self.assertTrue(trainer._pause_save_in_progress)
                trainer.save.side_effect = save
                trainer.is_grad_accumulation_step = False
                trainer.maybe_save()
                trainer.save.assert_called_once_with(123)
                self.assertTrue(trainer.should_stop())
                self.assertFalse(trainer.should_save())
                self.assertFalse(trainer._pause_save_in_progress)

    @patch('toolkit.ui_checkpoint.flush')
    def test_save_now_without_pause_continues(self, flush):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root, DiffusionTrainer, stop=False)
            trainer.maybe_save()
            trainer.save.assert_called_once_with(123)
            self.assertFalse(trainer.should_stop())
            self.assertFalse(trainer.should_save())

    def test_zero_step_pause_stops_without_saving_in_both_trainers(self):
        for cls in (DiffusionTrainer, UITrainer):
            with self.subTest(trainer=cls.__name__), tempfile.TemporaryDirectory() as root:
                trainer = self.trainer(root, cls, step=0)
                self.assertTrue(trainer.should_stop())
                self.assertFalse(trainer.should_save())
                trainer.maybe_save()
                trainer.save.assert_not_called()
                trainer.optimizer.zero_grad.assert_not_called()

    def test_resumed_pause_waits_for_save_before_checkpoint_step_is_restored(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root, DiffusionTrainer, step=0, db_step=123)
            self.assertFalse(trainer.should_stop())
            self.assertTrue(trainer.should_save())

    @patch('toolkit.ui_checkpoint.flush')
    def test_save_now_at_zero_without_pause_still_saves(self, flush):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root, DiffusionTrainer, stop=False, step=0)
            trainer.maybe_save()
            trainer.save.assert_called_once_with(0)
            self.assertFalse(trainer.should_stop())

    def test_failed_save_does_not_acknowledge_pause(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root, DiffusionTrainer)
            trainer.save.side_effect = OSError('disk full')
            with self.assertRaisesRegex(OSError, 'disk full'):
                trainer.maybe_save()
            self.assertTrue(trainer.should_save())
            self.assertFalse(trainer.should_stop())
            self.assertFalse(trainer._pause_save_in_progress)

    def test_end_step_saves_then_stops_without_sampling(self):
        trainer = object.__new__(DiffusionTrainer)
        trainer.is_ui_trainer = True
        events = []
        trainer.update_step = lambda: events.append('update')
        trainer.maybe_save = lambda: events.append('save')
        def stop():
            events.append('stop')
            raise RuntimeError('paused')
        trainer.maybe_stop = stop
        trainer.maybe_sample = Mock()
        with self.assertRaisesRegex(RuntimeError, 'paused'):
            trainer.end_step_hook()
        self.assertEqual(events, ['update', 'save', 'stop'])
        trainer.maybe_sample.assert_not_called()

    def test_non_ui_training_ignores_controls(self):
        trainer = SimpleNamespace(is_ui_trainer=False)
        self.assertEqual(UICheckpointControls._checkpoint_flags(trainer), (False, False))

    def test_resume_discovers_pause_checkpoint_and_restores_completed_step(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'experiment_000000123.safetensors')
            save_file({'weight': torch.ones(1)}, path,
                      metadata={'training_info': json.dumps({'step': 123, 'epoch': 7})})
            trainer = object.__new__(BaseSDTrainProcess)
            trainer.job = SimpleNamespace(name='experiment')
            trainer.save_root = root
            trainer.network_config = SimpleNamespace(pretrained_lora_path=None)
            trainer.train_config = SimpleNamespace(start_step=None)
            trainer.accelerator = SimpleNamespace(is_main_process=True)
            latest = trainer.get_latest_save_path()
            self.assertEqual(latest, path)
            trainer.load_training_state_from_metadata(latest)
            self.assertEqual((trainer.step_num, trainer.start_step, trainer.epoch_num), (123, 123, 7))
