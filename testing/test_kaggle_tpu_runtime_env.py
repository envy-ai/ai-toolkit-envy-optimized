"""Import preflight for the Kaggle Torch/XLA worker environment."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_training_import_preflight_skips_flax_jax_and_tensorflow():
    script = r'''
import builtins
import json
import os
import sys

os.environ.update(USE_TORCH="0", USE_FLAX="1", USE_TF="1")
attempted = []
blocked = {"flax", "jax", "tensorflow"}

real_import = builtins.__import__
def tracked_import(name, *args, **kwargs):
    if name.partition(".")[0] in blocked:
        attempted.append(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = tracked_import
from workers.kaggle_tpu.runtime_env import check_training_imports
report = check_training_imports()
assert os.environ["USE_TORCH"] == "1"
assert os.environ["USE_FLAX"] == "0"
assert os.environ["USE_TF"] == "0"
assert not attempted, attempted
assert not (blocked & set(sys.modules))
print(json.dumps(report))
'''
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = str(ROOT) + os.pathsep + worker_env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=worker_env,
        text=True, capture_output=True, check=True,
    )
    reports = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert reports == [{"status": "pass", "optional_backends": "torch_only"}]
