#!/usr/bin/env python3
"""Preview adaptive ordinary-LoRA rank reduction; write only a new output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.lora_trimming import format_trimming_text, trim_lora


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', help='Standalone .safetensors LoRA')
    parser.add_argument('--input', dest='input_path', help='Input path (alternative to positional INPUT)')
    threshold = parser.add_mutually_exclusive_group()
    threshold.add_argument('--retained-energy', type=float, help='Retained squared singular-value energy (default .9999)')
    threshold.add_argument('--relative-error', type=float, help='Maximum relative Frobenius error, including storage rounding')
    parser.add_argument('--min-rank', type=int, default=1)
    parser.add_argument('--max-rank', type=int)
    parser.add_argument('--output', help='Write a new .safetensors file; existing paths are never overwritten')
    parser.add_argument('--expected-preview', help='Require source/settings to match this preview token')
    parser.add_argument('--json', action='store_true', help='Print a compact JSON report on the final stdout line')
    return parser


def main(argv=None, *, require_preview=False, json_output=False):
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.path) == bool(args.input_path):
        parser.error('Specify exactly one positional INPUT or --input')
    if require_preview and args.output and not args.expected_preview:
        parser.error('Saving from the UI requires --expected-preview; preview the source/settings first')
    report = trim_lora(args.path or args.input_path, retained_energy=args.retained_energy,
        relative_error=args.relative_error, min_rank=args.min_rank, max_rank=args.max_rank,
        output=args.output, expected_preview=args.expected_preview,
        progress=lambda message: print(message, file=sys.stderr, flush=True))
    if args.json or json_output:
        print(json.dumps(report, ensure_ascii=False, separators=(',', ':'), allow_nan=False), flush=True)
    else:
        print(format_trimming_text(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
