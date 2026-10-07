"""CPU-only regression runner, safe while the GPU is occupied by training.

The unrelated eager OmniGen registry imports construct Triton autotuners using
CUDA properties at import time. Stub ONLY that import-time hardware discovery;
tests themselves run with real torch operations and CUDA hidden.
"""
import contextlib
import importlib
import io
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton

DEFAULT_TESTS = [
    'testing.test_flow_training_contract', 'testing.test_fizgig_slider',
    'testing.test_fizgig_multipoint', 'testing.test_fizgig_anchors',
    'testing.test_qwen_flow_dpo', 'testing.test_qwen_guidance_distillation',
    'testing.test_sliderspace', 'testing.test_cross_model_flow_trainers',
    'testing.test_krea_edit_objectives', 'testing.test_flow_cache_identity',
    'testing.test_krea_edit_previews',
    'testing.test_krea_preview_v3',
    'testing.test_qwen_image_2_processor',
    'testing.test_gpu_smoke_safety',
    'testing.test_cross_model_loha',
    'testing.test_flow_kto',
    'testing.test_diffusion_kto', 'testing.test_diffusion_kto_native', 'testing.test_cross_model_examples',
    'testing.test_sliderspace_components',
    'testing.test_cross_model_native_objectives',
    'testing.test_adapter_offload_restore',
    'testing.test_training_examples',
]


def main():
    if '--isolate' in sys.argv:
        # Some legacy tests install persistent sys.modules stubs. Run modules in
        # separate CPU-only processes instead of letting those mocks alter later tests.
        names = [name for name in sys.argv[1:] if name != '--isolate'] or DEFAULT_TESTS
        totals = [0, 0, 0, 0]
        successful = True
        for name in names:
            child = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True)
            summary = re.search(r'CPU regressions: (\d+) tests, (\d+) failures, (\d+) errors, (\d+) skipped', child.stdout)
            if summary:
                totals = [total + int(value) for total, value in zip(totals, summary.groups())]
            else:
                totals[2] += 1
            print(f'{name}: {child.stdout.strip()}', flush=True)
            if child.returncode:
                successful = False
                if child.stderr:
                    print(child.stderr[:2000], flush=True)
        print(f'Isolated CPU regressions: {totals[0]} tests, {totals[1]} failures, {totals[2]} errors, {totals[3]} skipped')
        return 0 if successful else 1
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        with patch('torch.cuda.current_device', return_value=0), patch(
            'torch.cuda.get_device_properties', return_value=SimpleNamespace(warp_size=32, major=8)
        ), patch('triton.autotune', side_effect=lambda **kwargs: lambda function: function):
            # Some tests import a native wrapper inside the test method, rather
            # than at module load. Resolve the eager registry here too, so its
            # unrelated Triton hardware discovery remains import-only: none of
            # the patches above are active during numerical test execution.
            importlib.import_module('extensions_built_in.diffusion_models')
            suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:] or DEFAULT_TESTS)
        result = unittest.TextTestRunner(stream=output).run(suite)
    print(f'CPU regressions: {result.testsRun} tests, {len(result.failures)} failures, '
          f'{len(result.errors)} errors, {len(result.skipped)} skipped')
    for test, detail in result.errors + result.failures:
        print(str(test))
        # Keep the exception/cause when native wrapper traces are deep.
        print(detail[-4000:])
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
