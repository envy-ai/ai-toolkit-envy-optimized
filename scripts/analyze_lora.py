#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.lora_analysis import analyze_lora, format_analysis_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank the effective changes in an AI Toolkit LoRA, separated by "
            "layer role. Local base weights are used when they can be resolved."
        )
    )
    parser.add_argument("lora", help="AI Toolkit .safetensors LoRA to analyze")
    parser.add_argument(
        "--base",
        help=(
            "Base .safetensors file, model directory, or locally cached Hugging "
            "Face repository ID. Overrides metadata/config discovery."
        ),
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Skip local base-model discovery and report LoRA-only update metrics",
    )
    parser.add_argument(
        "--base-search-root",
        action="append",
        default=[],
        help="Additional directory containing base-model .safetensors files; repeatable",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum layers printed per type (default: 20; JSON always contains all)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        metavar="PATH",
        help="Write the complete analysis as JSON; omit PATH to print only JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top < 1:
        raise SystemExit("--top must be at least 1")

    try:
        analysis = analyze_lora(
            args.lora,
            base=args.base,
            use_base=not args.no_base,
            search_roots=args.base_search_root,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    json_text = json.dumps(analysis, indent=2, sort_keys=True)
    if args.json == "-":
        print(json_text)
    else:
        print(format_analysis_text(analysis, top=args.top))
        if args.json:
            output_path = Path(args.json).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_text + "\n")
            print(f"\nJSON written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
