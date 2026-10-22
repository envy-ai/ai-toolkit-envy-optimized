#!/usr/bin/env python3
"""Preview CPU/CUDA adapter rank reduction; write only a new output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.lora_trimming import BASE_DEQUANT_DTYPES, RANK_POLICIES, format_trimming_text, trim_lora


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', help='Standalone .safetensors LoRA')
    parser.add_argument('--input', dest='input_path', help='Input path (alternative to positional INPUT)')
    threshold = parser.add_mutually_exclusive_group()
    threshold.add_argument('--retained-energy', type=float, help='Retained squared singular-value energy (default .9999)')
    threshold.add_argument('--relative-error', type=float, help='Maximum relative Frobenius error, including storage rounding')
    parser.add_argument('--rank-policy', choices=RANK_POLICIES, help='Explicit bounded or kohya-style spectrum rank policy')
    parser.add_argument('--rank-value', type=float, help='Value for --rank-policy (fixed rank, ratio, or retention/error fraction)')
    parser.add_argument('--convert-to-lora', action='store_true', help='Explicitly convert supported 2D LoHa/LoKr to ordinary LoRA')
    parser.add_argument('--max-working-mb', type=float, default=512, help='Conversion working-memory admission budget in MiB (default 512)')
    parser.add_argument('--device', default='auto', help='Linear-algebra device: auto (CUDA when available, otherwise CPU), cpu, cuda, or cuda:<index>; CUDA errors do not retry on CPU')
    parser.add_argument('--bake-dora-to-lora', action='store_true', help='Bake full-strength DoRA with matching base weights before reducing to ordinary LoRA')
    parser.add_argument('--base-model-path', help='Single local floating or supported Comfy-quantized safetensors base checkpoint, required for DoRA baking')
    parser.add_argument('--dora-convention', choices=('comfy', 'toolkit'), help='Explicit usual DoRA consumer: ComfyUI or AI Toolkit/native (required for baking)')
    parser.add_argument('--base-dequant-dtype', choices=BASE_DEQUANT_DTYPES, default='float32',
                        help='Decoded quantized-base weight view: float32 (default), bfloat16 or float16; floating base storage is unchanged')
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
        rank_policy=args.rank_policy, rank_value=args.rank_value,
        convert_to_lora=args.convert_to_lora, max_working_mb=args.max_working_mb,
        bake_dora_to_lora=args.bake_dora_to_lora, base_model_path=args.base_model_path,
        dora_convention=args.dora_convention,
        base_dequant_dtype=args.base_dequant_dtype,
        device=args.device,
        output=args.output, expected_preview=args.expected_preview,
        progress=lambda message: print(message, file=sys.stderr, flush=True))
    if args.json or json_output:
        print(json.dumps(report, ensure_ascii=False, separators=(',', ':'), allow_nan=False), flush=True)
    else:
        print(format_trimming_text(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
