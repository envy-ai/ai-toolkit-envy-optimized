import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from toolkit.config_modules import LoggingConfig
from toolkit.logging_aitk import UILogger, MultiLogger, WandbLogger, create_logger
from toolkit.training_examples import capture_training_rng
from toolkit.training_examples import (source_item, record_step_inputs, record_batch_inputs,
    log_image_losses, save_practice_source, remember_bank_source)
from types import SimpleNamespace


def example(loss=1., weight=1., caption='training caption', path='/datasets/a.jpg'):
    return {'metadata': {'path': path, 'dataset_path': '/datasets', 'caption': caption},
            'presentation': {'crop_width': 64, 'crop_height': 64, 'flip_x': False},
            'loss': loss, 'weighted_loss': loss * weight, 'loss_weight': weight,
            'timestep': 450., 'teacher_correction_rms': 2., 'noise_mean': .01, 'noise_std': 1.}


class TrainingExampleTests(unittest.TestCase):
    def trainer(self, enabled=True):
        return SimpleNamespace(logger=self.logger, logging_config=LoggingConfig(record_training_examples=enabled),
            accelerator=SimpleNamespace(is_main_process=True), sd=SimpleNamespace(arch='sdxl'),
            train_config=SimpleNamespace(dtype='fp32', loss_type='mse'), save_root=self.temp.name)

    def test_shared_steps_link_all_inputs_without_fabricating_individual_losses(self):
        trainer = self.trainer()
        item = source_item('/datasets/preferred/a.png', 'caption', unconditional_path='/datasets/rejected/a.png')
        batch = SimpleNamespace(file_items=[item], latents=None)
        record_step_inputs(trainer, [batch], {'loss': 9.})
        self.logger.commit(20)
        self.assertEqual(self.rows('SELECT loss,weighted_loss,loss_kind FROM training_examples'), [(None, None, 'unattributed')] * 2)
        self.assertEqual(self.rows("SELECT value_real FROM metrics WHERE key='loss/loss'"), [(9.,)])
        self.assertEqual({row[0] for row in self.rows('SELECT path FROM training_example_metadata')}, {'/datasets/preferred/a.png', '/datasets/rejected/a.png'})

    def test_disabling_reporting_skips_records_rng_and_practice_image_writes(self):
        from PIL import Image
        trainer = self.trainer(False)
        record_step_inputs(trainer, [None], {'loss': 9.})
        save_practice_source(trainer, torch.ones(1), Image.new('RGB', (8, 8)), 'test', 'prompt')
        self.assertEqual(self.logger.pending_training_examples(), 0)
        self.assertFalse(Path(self.filename).exists())
        self.assertFalse((Path(self.temp.name) / 'loss_report_practice').exists())

    def test_practice_bank_logs_selected_image_without_resampling_or_rng_changes(self):
        from PIL import Image
        trainer = self.trainer()
        latent = torch.ones(1, 1, 2, 2)
        rng = torch.get_rng_state().clone()
        save_practice_source(trainer, latent, Image.new('RGB', (8, 8)), 'selected', 'actual prompt')
        remember_bank_source(trainer, latent)
        record_step_inputs(trainer, [None], {'loss': 2.})
        self.logger.commit(0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        path, caption = self.rows('SELECT path,caption FROM training_example_metadata')[0]
        self.assertTrue(Path(path).exists()); self.assertEqual(caption, 'actual prompt')

    def test_pair_objective_records_both_images_with_shared_attribution(self):
        trainer = self.trainer()
        trainer._loss_report_step_rng = {'cpu': 'observed-step-state'}
        batch = SimpleNamespace(file_items=[source_item('/datasets/win.png', 'caption', unconditional_path='/datasets/lose.png')],
            latents=torch.ones(1, 1, 2, 2), loss_multiplier_list=[2.])
        log_image_losses(trainer, batch, torch.tensor([3.]), torch.tensor([6.]), scope='pair_objective')
        self.logger.commit(0)
        rows = self.rows('SELECT loss,weighted_loss,presentation_json FROM training_examples')
        self.assertEqual(len(rows), 2)
        for loss, weighted, presentation in rows:
            self.assertEqual((loss, weighted), (3., 6.)); self.assertEqual(json.loads(presentation)['loss_attribution'], 'pair_objective')
        self.assertEqual(self.rows("SELECT value_real FROM metrics WHERE key='loss/loss'"), [(6.,)])
        self.assertEqual(json.loads(self.rows('SELECT rng_state_json FROM training_batches')[0][0]), trainer._loss_report_step_rng)

    def test_standard_diffusion_reduction_gradients_and_raw_weighted_vectors_unchanged(self):
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
        from toolkit.config_modules import TrainConfig
        results = []
        for enabled in (False, True):
            trainer = SDTrainer.__new__(SDTrainer)
            trainer.logger = self.logger; trainer.logging_config = LoggingConfig(record_training_examples=enabled)
            trainer.accelerator = SimpleNamespace(is_main_process=True)
            trainer.train_config = TrainConfig(dtype='fp32', loss_type='mse')
            trainer.device_torch = torch.device('cpu')
            trainer.sd = SimpleNamespace(prediction_type='epsilon', is_flow_matching=False, scale_loss=lambda x: x)
            trainer.adapter = trainer.dfe = None; trainer.additional_logs = {}
            parameter = torch.tensor([1., 3.], requires_grad=True)
            batch = SimpleNamespace(mask_tensor=None, loss_multiplier_list=[1., 2.], get_is_reg_list=lambda: [False, False])
            loss = trainer.calculate_loss(parameter[:, None, None, None], torch.zeros(2,1,1,1),
                torch.zeros(2,1,1,1), torch.tensor([100., 200.]), batch)
            loss.backward(); results.append((loss.detach(), parameter.grad))
            if enabled:
                raw, weighted, times = trainer._loss_report_components
                torch.testing.assert_close(raw, torch.tensor([1., 9.]))
                torch.testing.assert_close(weighted, torch.tensor([1., 18.]))
        torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
        torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)

    def test_kto_reporting_preserves_rng_and_optimizer_update_and_uses_actual_window_loss(self):
        from testing.test_diffusion_kto import DiffusionKTOTests
        outputs = []
        fixture = DiffusionKTOTests()
        for enabled in (False, True):
            torch.manual_seed(94)
            trainer, parameter, _ = fixture.fixture('qwen_image_2')
            trainer.logging_config = LoggingConfig(record_training_examples=enabled); trainer.logger = self.logger
            batches = fixture.batches('qwen_image_2')
            for bi, batch in enumerate(batches):
                for ii, item in enumerate(batch.file_items):
                    item.path = f'/datasets/{bi}-{ii}.png'; item.caption = f'caption {bi}-{ii}'
                    item.dataset_config.folder_path = '/datasets'
            result = trainer.hook_train_loop(batches)
            record_step_inputs(trainer, batches, result)
            outputs.append((result['loss'], parameter.detach().clone(), torch.get_rng_state().clone()))
        self.assertEqual(outputs[0][0], outputs[1][0])
        torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
        self.assertTrue(torch.equal(outputs[0][2], outputs[1][2]))
        self.logger.commit(0)
        self.assertEqual(self.rows('SELECT COUNT(*) FROM training_examples'), [(4,)])
        self.assertEqual(self.rows("SELECT value_real FROM metrics WHERE key='loss/loss'"), [(outputs[1][0],)])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.filename = str(Path(self.temp.name) / 'loss_log.db')
        self.logger = UILogger(self.filename, flush_every_n=10000, flush_every_secs=10000)

    def tearDown(self):
        self.logger.finish()
        self.temp.cleanup()

    def rows(self, sql):
        self.logger._flush()
        with sqlite3.connect(self.filename) as con:
            return con.execute(sql).fetchall()

    def test_weighted_microbatches_match_graph_and_metadata_deduplicates(self):
        self.logger.log_training_examples([example(1, 2), example(3, 2)], {'cpu': 'state'})
        self.logger.log_training_examples([example(10, 1)])
        self.logger.commit(20)
        # mean([mean([2, 6]), mean([10])]) = 7, NOT mean([2, 6, 10]).
        self.assertEqual(self.rows("SELECT value_real FROM metrics WHERE key='loss/loss'"), [(7.,)])
        self.assertEqual(self.rows('SELECT microbatch,item_index,loss,weighted_loss FROM training_examples ORDER BY microbatch,item_index'),
                         [(0, 0, 1., 2.), (0, 1, 3., 6.), (1, 0, 10., 10.)])
        self.assertEqual(self.rows('SELECT COUNT(*) FROM training_example_metadata'), [(1,)])
        self.assertEqual(json.loads(self.rows('SELECT rng_state_json FROM training_batches WHERE microbatch=0')[0][0]), {'cpu': 'state'})

    def test_buffered_writes_and_scalar_same_step_preserve_images(self):
        self.logger.log_training_examples([example()]); self.logger.commit(5)
        with sqlite3.connect(self.filename) as con:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM training_examples').fetchone()[0], 0)
        self.logger.log({'learning_rate': .001}); self.logger.commit(5)
        self.assertEqual(self.rows('SELECT COUNT(*) FROM training_examples'), [(1,)])
        self.logger.log_training_examples([example(4, caption='new caption')]); self.logger.commit(5)
        self.assertEqual(self.rows('SELECT loss FROM training_examples'), [(4.,)])
        self.assertEqual(self.rows('SELECT m.caption FROM training_examples e JOIN training_example_metadata m ON m.id=e.metadata_id'), [('new caption',)])

    def test_failed_partial_batch_discarded(self):
        self.logger.log_training_examples([example(99)])
        self.logger.discard_training_examples()
        self.logger.log_training_examples([example(2)]); self.logger.commit(0)
        self.assertEqual(self.rows('SELECT loss FROM training_examples'), [(2.,)])

    def test_nonfinite_values_are_explicit_and_json_is_valid(self):
        self.logger.log_training_examples([example(float('inf')), example(float('nan')), example(-float('inf'))]); self.logger.commit(0)
        self.assertEqual(self.rows('SELECT loss,weighted_loss,loss_kind FROM training_examples ORDER BY item_index'),
                         [(None, None, '+inf'), (None, None, 'nan'), (None, None, '-inf')])

    def test_resume_prunes_examples_including_resumed_step_even_when_disabled(self):
        for step in range(4):
            self.logger.log_training_examples([example(caption=f'step {step}')]); self.logger.commit(step)
        self.logger.finish()
        self.logger = UILogger(self.filename)
        self.logger.log({'loss/loss': 8}); self.logger.commit(2)
        self.assertEqual(self.rows('SELECT step FROM training_examples ORDER BY step'), [(0,), (1,)])
        self.assertEqual(self.rows('SELECT COUNT(*) FROM training_example_metadata'), [(2,)])
        self.assertEqual(self.rows('SELECT step FROM training_batches ORDER BY step'), [(0,), (1,)])
        self.assertEqual(self.rows('SELECT step FROM steps ORDER BY step'), [(0,), (1,), (2,)])

    def test_existing_scalar_database_migrates(self):
        with sqlite3.connect(self.filename) as con:
            con.executescript('CREATE TABLE steps(step INTEGER PRIMARY KEY,wall_time REAL NOT NULL);'
                              'INSERT INTO steps VALUES(0,123);')
        self.logger.log_training_examples([example()]); self.logger.commit(1)
        self.assertEqual(self.rows('SELECT step FROM steps ORDER BY step'), [(0,), (1,)])

    def test_rng_snapshot_replays_draws_without_mutating_state(self):
        import base64
        state = torch.get_rng_state().clone()
        snapshot = capture_training_rng('cpu')
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        first = torch.randn(12)
        torch.set_rng_state(torch.frombuffer(bytearray(base64.b64decode(snapshot['cpu'])), dtype=torch.uint8).clone())
        torch.testing.assert_close(torch.randn(12), first)
        torch.set_rng_state(state)

    def test_logging_flags_and_wandb_composition(self):
        for config in ({'record_training_examples': 'true'}, {'record_training_rng': True, 'record_training_examples': False}):
            with self.assertRaises(ValueError): LoggingConfig(**config)
        config = LoggingConfig(use_wandb=True, record_training_examples=True)
        self.assertTrue(LoggingConfig().record_training_examples)
        self.assertFalse(LoggingConfig().record_training_rng)
        logger = create_logger(config, {}, self.temp.name)
        self.assertIsInstance(logger, MultiLogger)
        self.assertIsInstance(logger.loggers[0], WandbLogger)
        self.assertIsInstance(logger.loggers[1], UILogger)
        with patch.object(logger.loggers[0], 'log_training_examples') as wandb, patch.object(logger.loggers[1], 'log_training_examples') as local:
            logger.log_training_examples([example()]); wandb.assert_called_once(); local.assert_called_once()


if __name__ == '__main__':
    unittest.main()
