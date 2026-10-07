from typing import OrderedDict, Optional
from PIL import Image

from toolkit.config_modules import LoggingConfig
import os
import sqlite3
import time
import json
import hashlib
import math
from typing import Any, Dict, Tuple, List


# Base logger class
# This class does nothing, it's just a placeholder
class EmptyLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass

    # start logging the training
    def start(self):
        pass

    # collect the log to send
    def log(self, *args, **kwargs):
        pass

    # send the log
    def commit(self, step: Optional[int] = None):
        pass

    # log image
    def log_image(self, *args, **kwargs):
        pass

    # finish logging
    def finish(self):
        pass

    def log_training_examples(self, records, rng_state=None):
        pass

    def discard_training_examples(self):
        pass

    def pending_training_examples(self):
        return 0


class MultiLogger(EmptyLogger):
    """Keep local per-image diagnostics when scalar metrics also go to WandB."""
    def __init__(self, loggers):
        self.loggers = loggers

    def start(self):
        for logger in self.loggers:
            logger.start()

    def log(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log(*args, **kwargs)

    def commit(self, *args, **kwargs):
        for logger in self.loggers:
            logger.commit(*args, **kwargs)

    def log_image(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_image(*args, **kwargs)

    def log_training_examples(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_training_examples(*args, **kwargs)

    def discard_training_examples(self):
        for logger in self.loggers:
            logger.discard_training_examples()

    def finish(self):
        for logger in self.loggers:
            logger.finish()

    def pending_training_examples(self):
        return max((logger.pending_training_examples() for logger in self.loggers), default=0)


# Wandb logger class
# This class logs the data to wandb
class WandbLogger(EmptyLogger):
    def __init__(self, project: str, run_name: str | None, config: OrderedDict) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config

    def start(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "Failed to import wandb. Please install wandb by running `pip install wandb`"
            )

        # send the whole config to wandb
        run = wandb.init(project=self.project, name=self.run_name, config=self.config)
        self.run = run
        self._log = wandb.log  # log function
        self._image = wandb.Image  # image object

    def log(self, *args, **kwargs):
        # when commit is False, wandb increments the step,
        # but we don't want that to happen, so we set commit=False
        self._log(*args, **kwargs, commit=False)

    def commit(self, step: Optional[int] = None):
        # after overall one step is done, we commit the log
        # by log empty object with commit=True
        self._log({}, step=step, commit=True)

    def log_image(
        self,
        image: Image,
        id,  # sample index
        caption: str | None = None,  # positive prompt
        *args,
        **kwargs,
    ):
        # create a wandb image object and log it
        image = self._image(image, caption=caption, *args, **kwargs)
        self._log({f"sample_{id}": image}, commit=False)

    def finish(self):
        self.run.finish()


class UILogger:
    def __init__(
        self,
        log_file: str,
        flush_every_n: int = 256,
        flush_every_secs: float = 0.25,
    ) -> None:
        self.log_file = log_file
        self._log_to_commit: Dict[str, Any] = {}

        self._con: Optional[sqlite3.Connection] = None
        self._started = False

        self._step_counter = 0

        # buffered writes
        self._pending_steps: List[Tuple[int, float]] = []
        self._pending_metrics: List[
            Tuple[int, str, Optional[float], Optional[str]]
        ] = []
        self._pending_key_minmax: Dict[str, Tuple[int, int]] = {}

        self._flush_every_n = int(flush_every_n)
        self._flush_every_secs = float(flush_every_secs)
        self._last_flush = time.time()

        self._first_commit_done = False
        self._examples_to_commit = []
        self._pending_example_steps = set()
        self._pending_example_metadata = {}
        self._pending_examples = []
        self._pending_batches = []

    # start logging the training
    def start(self):
        if self._started:
            return

        parent = os.path.dirname(os.path.abspath(self.log_file))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        self._con = sqlite3.connect(self.log_file, timeout=30.0, isolation_level=None)
        self._con.execute("PRAGMA journal_mode=WAL;")
        self._con.execute("PRAGMA synchronous=NORMAL;")
        self._con.execute("PRAGMA temp_store=MEMORY;")
        self._con.execute("PRAGMA foreign_keys=ON;")
        self._con.execute("PRAGMA busy_timeout=30000;")

        self._init_schema(self._con)

        self._started = True
        self._last_flush = time.time()

    # collect the log to send
    def log(self, log_dict):
        # log_dict is like {'learning_rate': learning_rate}
        if not isinstance(log_dict, dict):
            raise TypeError("log_dict must be a dict")
        self._log_to_commit.update(log_dict)

    def log_training_examples(self, records, rng_state=None):
        # Records are detached CPU scalars/text, never GPU tensors or images.
        if records:
            self._examples_to_commit.append((records, rng_state))

    def discard_training_examples(self):
        self._examples_to_commit.clear()

    def pending_training_examples(self):
        return len(self._examples_to_commit)

    # send the log
    def commit(self, step: Optional[int] = None):
        if not self._started:
            self.start()

        if not self._log_to_commit and not self._examples_to_commit:
            return

        if step is None:
            step = self._step_counter
            self._step_counter += 1
        else:
            step = int(step)
            if step >= self._step_counter:
                self._step_counter = step + 1

        # On the first commit of this run, prune any rows from a prior run
        # whose step is greater than where we are resuming from.
        if not self._first_commit_done:
            self._prune_future_steps(step)
            self._first_commit_done = True

        wall_time = time.time()

        if self._examples_to_commit:
            self._pending_example_steps.add(step)
            self._pending_examples = [row for row in self._pending_examples if row[0] != step]
            self._pending_batches = [row for row in self._pending_batches if row[0] != step]
            batch_losses = []
            for microbatch, (records, rng_state) in enumerate(self._examples_to_commit):
                values = [record['weighted_loss'] for record in records]
                if all(value is not None for value in values):
                    batch_losses.append(sum(values) / len(values))
                self._pending_batches.append((step, microbatch, json.dumps(rng_state) if rng_state else None))
                for item_index, record in enumerate(records):
                    metadata_json = json.dumps(record['metadata'], sort_keys=True, ensure_ascii=False, allow_nan=False)
                    metadata_id = hashlib.sha256(metadata_json.encode()).hexdigest()
                    metadata = record['metadata']
                    self._pending_example_metadata[metadata_id] = (
                        metadata_id, metadata['path'], metadata['dataset_path'], metadata['caption'], metadata_json)
                    loss = record['loss']
                    weighted = record['weighted_loss']
                    kind = 'unattributed' if weighted is None else ('finite' if math.isfinite(weighted) else ('nan' if math.isnan(weighted) else ('+inf' if weighted > 0 else '-inf')))
                    finite = lambda value: value if value is None or math.isfinite(value) else None
                    self._pending_examples.append((
                        step, microbatch, item_index, metadata_id, finite(loss), finite(weighted), kind,
                        finite(record['loss_weight']), finite(record['timestep']), finite(record['teacher_correction_rms']),
                        finite(record['noise_mean']), finite(record['noise_std']),
                        json.dumps(record['presentation'], sort_keys=True, allow_nan=False)))
            # Match the graph: average the microbatch means, including uneven batches.
            if len(batch_losses) == len(self._examples_to_commit):
                self._log_to_commit.setdefault('loss/loss', sum(batch_losses) / len(batch_losses))
            self._examples_to_commit.clear()

        # buffer step row (upsert later)
        self._pending_steps.append((step, wall_time))

        # buffer metrics rows + key min/max updates
        for k, v in self._log_to_commit.items():
            k = k if isinstance(k, str) else str(k)
            vr, vt = self._coerce_value(v)

            self._pending_metrics.append((step, k, vr, vt))

            if k in self._pending_key_minmax:
                lo, hi = self._pending_key_minmax[k]
                if step < lo:
                    lo = step
                if step > hi:
                    hi = step
                self._pending_key_minmax[k] = (lo, hi)
            else:
                self._pending_key_minmax[k] = (step, step)

        self._log_to_commit = {}

        # flush conditions
        now = time.time()
        if (
            len(self._pending_metrics) + len(self._pending_examples) >= self._flush_every_n
            or (now - self._last_flush) >= self._flush_every_secs
        ):
            self._flush()

    # log image
    def log_image(self, *args, **kwargs):
        # this doesnt log images for now
        pass

    # finish logging
    def finish(self):
        if not self._started:
            return

        self._flush()

        assert self._con is not None
        self._con.close()
        self._con = None
        self._started = False

    # -------------------------
    # internal
    # -------------------------

    def _init_schema(self, con: sqlite3.Connection) -> None:
        con.execute("BEGIN;")

        con.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step      INTEGER PRIMARY KEY,
                wall_time REAL NOT NULL
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS metric_keys (
                key             TEXT PRIMARY KEY,
                first_seen_step INTEGER,
                last_seen_step  INTEGER
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                step       INTEGER NOT NULL,
                key        TEXT NOT NULL,
                value_real REAL,
                value_text TEXT,
                PRIMARY KEY (step, key),
                FOREIGN KEY (step) REFERENCES steps(step) ON DELETE CASCADE
            );
        """)

        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_key_step ON metrics (key, step);"
        )

        con.execute("""
            CREATE TABLE IF NOT EXISTS training_example_metadata (
                id TEXT PRIMARY KEY, path TEXT NOT NULL, dataset_path TEXT NOT NULL,
                caption TEXT NOT NULL, metadata_json TEXT NOT NULL
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS training_batches (
                step INTEGER NOT NULL, microbatch INTEGER NOT NULL, rng_state_json TEXT,
                PRIMARY KEY(step, microbatch),
                FOREIGN KEY(step) REFERENCES steps(step) ON DELETE CASCADE
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS training_examples (
                step INTEGER NOT NULL, microbatch INTEGER NOT NULL, item_index INTEGER NOT NULL,
                metadata_id TEXT NOT NULL, loss REAL, weighted_loss REAL, loss_kind TEXT NOT NULL,
                loss_weight REAL, timestep REAL, teacher_correction_rms REAL,
                noise_mean REAL, noise_std REAL, presentation_json TEXT NOT NULL,
                PRIMARY KEY(step, microbatch, item_index),
                FOREIGN KEY(step) REFERENCES steps(step) ON DELETE CASCADE,
                FOREIGN KEY(metadata_id) REFERENCES training_example_metadata(id)
            );
        """)
        con.execute('CREATE INDEX IF NOT EXISTS idx_training_examples_metadata ON training_examples(metadata_id, step);')

        con.execute("COMMIT;")

    def _coerce_value(self, v: Any) -> Tuple[Optional[float], Optional[str]]:
        if v is None:
            return None, None
        if isinstance(v, bool):
            return float(int(v)), None
        if isinstance(v, (int, float)):
            return float(v), None
        try:
            return float(v), None  # type: ignore[arg-type]
        except Exception:
            return None, str(v)

    def _prune_future_steps(self, current_step: int) -> None:
        assert self._con is not None
        con = self._con

        con.execute("BEGIN;")
        # metrics rows cascade via FK ON DELETE CASCADE
        con.execute("DELETE FROM steps WHERE step > ?;", (current_step,))
        # The resumed step may have a different batch or logging may be disabled.
        con.execute('DELETE FROM training_examples WHERE step >= ?;', (current_step,))
        con.execute('DELETE FROM training_batches WHERE step >= ?;', (current_step,))
        con.execute('DELETE FROM training_example_metadata WHERE NOT EXISTS '
                    '(SELECT 1 FROM training_examples WHERE metadata_id = training_example_metadata.id);')
        # drop any keys that no longer have any metrics, and clamp last_seen_step
        con.execute(
            "DELETE FROM metric_keys "
            "WHERE NOT EXISTS (SELECT 1 FROM metrics WHERE metrics.key = metric_keys.key);"
        )
        con.execute(
            "UPDATE metric_keys "
            "SET last_seen_step = (SELECT MAX(step) FROM metrics WHERE metrics.key = metric_keys.key) "
            "WHERE last_seen_step > ?;",
            (current_step,),
        )
        con.execute("COMMIT;")

    def _flush(self) -> None:
        if not self._pending_steps and not self._pending_metrics:
            return

        assert self._con is not None
        con = self._con

        con.execute("BEGIN;")

        # steps upsert
        if self._pending_steps:
            con.executemany(
                "INSERT INTO steps(step, wall_time) VALUES(?, ?) "
                "ON CONFLICT(step) DO UPDATE SET wall_time=excluded.wall_time;",
                self._pending_steps,
            )

        # keys table upsert (maintains list of keys + seen range)
        if self._pending_key_minmax:
            con.executemany(
                "INSERT INTO metric_keys(key, first_seen_step, last_seen_step) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "first_seen_step=MIN(metric_keys.first_seen_step, excluded.first_seen_step), "
                "last_seen_step=MAX(metric_keys.last_seen_step, excluded.last_seen_step);",
                [(k, lo, hi) for k, (lo, hi) in self._pending_key_minmax.items()],
            )

        # metrics upsert
        if self._pending_metrics:
            con.executemany(
                "INSERT INTO metrics(step, key, value_real, value_text) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(step, key) DO UPDATE SET "
                "value_real=excluded.value_real, value_text=excluded.value_text;",
                self._pending_metrics,
            )

        if self._pending_example_steps:
            for step in self._pending_example_steps:
                con.execute('DELETE FROM training_examples WHERE step = ?;', (step,))
                con.execute('DELETE FROM training_batches WHERE step = ?;', (step,))
            con.executemany('INSERT OR IGNORE INTO training_example_metadata VALUES (?, ?, ?, ?, ?);',
                            self._pending_example_metadata.values())
            con.executemany('INSERT INTO training_batches VALUES (?, ?, ?);', self._pending_batches)
            con.executemany('INSERT INTO training_examples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);',
                            self._pending_examples)

        con.execute("COMMIT;")

        self._pending_steps.clear()
        self._pending_metrics.clear()
        self._pending_key_minmax.clear()
        self._pending_example_steps.clear()
        self._pending_example_metadata.clear()
        self._pending_examples.clear()
        self._pending_batches.clear()
        self._last_flush = time.time()


# create logger based on the logging config
def create_logger(
    logging_config: LoggingConfig,
    all_config: OrderedDict,
    save_root: Optional[str] = None,
):
    loggers = []
    if logging_config.use_wandb:
        project_name = logging_config.project_name
        run_name = logging_config.run_name
        loggers.append(WandbLogger(project=project_name, run_name=run_name, config=all_config))
    if (logging_config.use_ui_logger and not logging_config.use_wandb) or getattr(logging_config, 'record_training_examples', False):
        if save_root is None:
            raise ValueError("save_root must be provided when using UILogger")
        log_file = os.path.join(save_root, "loss_log.db")
        loggers.append(UILogger(log_file=log_file))
    return MultiLogger(loggers) if len(loggers) > 1 else (loggers[0] if loggers else EmptyLogger())
