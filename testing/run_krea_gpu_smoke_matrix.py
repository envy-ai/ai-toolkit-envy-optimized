"""Explicit sequential smoke matrix; run with conda run -n ai-toolkit python.

No autonomous scheduling, parallel GPU work, UI writes or services. Each child
uses the real training startup path and an existing local checkpoint; hard
timeout is 15 minutes per child. Stops at the first failure for diagnosis.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--quant-cache', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', help='Run only named cases, preserving earlier evidence')
    args = parser.parse_args()
    root = args.root.resolve()
    if not str(root).startswith('/tmp/'):
        parser.error('Use an isolated matrix root under /tmp')
    if root.exists():
        parser.error('Use a fresh matrix root; do not overwrite earlier evidence')
    root.mkdir(parents=True)
    script = Path(__file__).with_name('smoke_flow_training_gpu.py')
    cases = [('flow_dpo', 'lora', None), ('fizgig_image_slider', 'dora', None),
             ('fizgig_prompt_slider', 'dora', None), ('loha', 'lora', None), ('doha', 'lora', None),
             ('sliderspace', 'lora', 'provided'), ('sliderspace', 'lora', 'generated'),
             ('sliderspace', 'lora', 'both')]
    if args.cases:
        available = {mode + ('_' + discovery if discovery else '') for mode, _, discovery in cases}
        unknown = set(args.cases) - available
        if unknown:
            parser.error('Unknown cases: ' + ', '.join(sorted(unknown)))
        cases = [(mode, adapter, discovery) for mode, adapter, discovery in cases
                 if mode + ('_' + discovery if discovery else '') in args.cases]
    results = []
    for mode, adapter, discovery in cases:
        name = mode + ('_' + discovery if discovery else '')
        command = [sys.executable, '-u', str(script), '--arch', 'krea2', '--mode', mode,
                   '--root', str(root / name), '--quant-cache', str(args.quant_cache.resolve()),
                   '--adapter-type', adapter, '--steps', '2']
        if discovery:
            command += ['--discovery-mode', discovery]
        started = time.perf_counter()
        print('GPU_SMOKE_MATRIX_START ' + name, flush=True)
        log = root / f'{name}.log'
        with log.open('w') as stream:
            try:
                child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
                returncode = child.returncode
            except subprocess.TimeoutExpired:
                returncode = 'timeout'
        result = {'case': name, 'returncode': returncode, 'seconds': time.perf_counter() - started,
                  'log': str(log)}
        results.append(result)
        (root / 'matrix.json').write_text(json.dumps(results, indent=2))
        print('GPU_SMOKE_MATRIX_RESULT ' + json.dumps(result), flush=True)
        if returncode:
            print(log.read_text()[-5000:], flush=True)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
