"""Create real Python-logger databases for the TypeScript report contract tests."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolkit.logging_aitk import UILogger
from testing.test_training_examples import example
from toolkit.training_examples import source_item, record_step_inputs
from toolkit.config_modules import LoggingConfig
from types import SimpleNamespace


def create(root):
    root = Path(root)
    for name in ('spikes', 'zero', 'nonfinite'):
        logger = UILogger(str(root / name / 'loss_log.db'), flush_every_n=10000)
        for step in range(24):
            loss = 1.
            if name == 'zero': loss = 0. if step < 20 else (2. if step == 20 else 0.)
            if name == 'nonfinite' and step == 3: loss = float('inf')
            if name == 'nonfinite' and step == 21: loss = float('nan')
            losses = [loss]
            if name == 'spikes':
                losses = {5: [100.], 20: [2., 18.], 21: [1.5], 22: [4.]}.get(step, losses)
            records = [example(value, caption='stored training caption', path=str(root / 'datasets' / f'image-{i}.png')) for i, value in enumerate(losses)]
            logger.log_training_examples(records, {'cpu': 'example-state'} if step == 20 else None)
            logger.commit(step)
        logger.finish()
    logger = UILogger(str(root / 'shared' / 'loss_log.db'), flush_every_n=10000)
    trainer = SimpleNamespace(logger=logger, logging_config=LoggingConfig(), sd=SimpleNamespace(arch='qwen_image_2'),
        train_config=SimpleNamespace(dtype='fp32', loss_type='mse'))
    batch = SimpleNamespace(file_items=[source_item(root / 'datasets' / 'shared.png', 'shared caption'),
        source_item('prompt://slider-target', 'prompt-only input')])
    for step in range(24):
        record_step_inputs(trainer, [batch], {'loss': 10. if step == 20 else 1.})
        logger.commit(step)
    logger.finish()
    (root / 'legacy').mkdir(parents=True)
    with sqlite3.connect(root / 'legacy' / 'loss_log.db') as db:
        db.executescript('CREATE TABLE steps(step INTEGER PRIMARY KEY,wall_time REAL);'
                        'CREATE TABLE metrics(step INTEGER,key TEXT,value_real REAL,value_text TEXT);')


if __name__ == '__main__':
    create(sys.argv[1])
