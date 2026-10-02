"""Checkpoint-backed rolling loss records for training."""

import json
import math
import os
import re
import tempfile
from typing import Optional


def checkpoint_step(path: str) -> Optional[int]:
    """Extract the training step from a checkpoint file or directory name."""
    match = re.search(r"_(\d{9})(?:\.(?:safetensors|pt))?$", os.path.basename(path))
    return int(match.group(1)) if match else None


def load_record_low_checkpoints(index_path: str, save_root: str):
    """Only return indexed checkpoints that still exist on disk."""
    try:
        with open(index_path, 'r') as stream:
            records = json.load(stream)
    except (OSError, ValueError):
        return []
    if not isinstance(records, list):
        return []
    valid = []
    for record in records:
        if not isinstance(record, dict):
            continue
        name = record.get('checkpoint')
        loss = record.get('loss')
        step = record.get('step')
        if (
            not isinstance(name, str) or os.path.basename(name) != name
            or not isinstance(step, int) or not isinstance(loss, (int, float))
            or not math.isfinite(loss) or not os.path.exists(os.path.join(save_root, name))
        ):
            continue
        valid.append({
            'step': step,
            'loss': float(loss),
            'scheduled': bool(record.get('scheduled', False)),
            'checkpoint': name,
        })
    return valid


def write_record_low_checkpoints(index_path: str, records):
    """Atomically persist the index after a successful checkpoint save."""
    parent = os.path.dirname(index_path)
    os.makedirs(parent, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=parent, prefix='.record_low_', delete=False) as stream:
            temporary_path = stream.name
            json.dump(records, stream, indent=2)
        os.replace(temporary_path, index_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def is_record_low_save_due(records, step: int, loss: Optional[float], window_size: int, start_step: int):
    """Compare against existing record-low checkpoints in the preceding window."""
    if step < start_step or loss is None or not math.isfinite(float(loss)):
        return False
    comparison_losses = [
        record['loss'] for record in records
        if step - window_size <= record['step'] < step
    ]
    return not comparison_losses or float(loss) < min(comparison_losses)


def record_low_retention(records, max_record_lows: int):
    """Keep the lowest existing record-low checkpoints across the run."""
    best = sorted(records, key=lambda record: (record['loss'], record['step']))[:max_record_lows]
    best_steps = {record['step'] for record in best}
    record_only_steps = {
        record['step'] for record in records if not record['scheduled']
    }
    return best_steps, record_only_steps
