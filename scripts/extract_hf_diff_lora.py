#!/usr/bin/env python3
"""Extract a LoRA by subtracting one Hugging Face model from another.

This script is aimed at transformer submodules (for example ERNIE-Image) and
emits PEFT-style LoRA weights that can be loaded as an inference LoRA.

Default use case:
- base:  baidu/ERNIE-Image
- tuned: baidu/ERNIE-Image-Turbo
- rank:  128
- alpha: 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, Tuple, Type

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from toolkit.lorm import extract_conv, extract_linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a low-rank diff LoRA from two HF models")
    parser.add_argument(
        "--base",
        type=str,
        default="baidu/ERNIE-Image",
        help="Base model path or HF repo id",
    )
    parser.add_argument(
        "--tuned",
        type=str,
        default="baidu/ERNIE-Image-Turbo",
        help="Tuned model path or HF repo id",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="transformer",
        help="Subfolder containing the transformer weights/config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/ernie_image_turbo_diff_lora.safetensors",
        help="Output safetensors path",
    )
    parser.add_argument("--dim", type=int, default=128, help="Target LoRA rank")
    parser.add_argument(
        "--alpha",
        type=float,
        default=64.0,
        help="LoRA alpha. In PEFT output this is baked into lora_B scaling (alpha/rank)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device used for SVD extraction (cpu or cuda)",
    )
    parser.add_argument(
        "--load-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model load dtype",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="fp16",
        choices=["bf16", "fp16", "fp32"],
        help="LoRA tensor dtype in the output file",
    )
    parser.add_argument(
        "--module-name-filter",
        type=str,
        default="",
        help="Only include modules whose full name contains this substring",
    )
    parser.add_argument(
        "--extract-conv",
        action="store_true",
        help="Also extract Conv2d weights (linear-only by default)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Skip modules where max absolute diff <= eps",
    )
    return parser.parse_args()


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def _load_json_config(model_id_or_path: str, subfolder: str) -> dict:
    if os.path.isdir(model_id_or_path):
        cfg_path = os.path.join(model_id_or_path, subfolder, "config.json")
    else:
        cfg_path = hf_hub_download(repo_id=model_id_or_path, filename="config.json", subfolder=subfolder)

    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_transformer_class(class_name: str) -> Type[torch.nn.Module]:
    import diffusers

    if hasattr(diffusers, class_name):
        return getattr(diffusers, class_name)

    # Repo-local fallback for custom ERNIE transformer integration.
    if class_name == "ErnieImageTransformer2DModel":
        from extensions_built_in.diffusion_models.ernie_image.transformer import ErnieImageTransformer2DModel

        return ErnieImageTransformer2DModel

    raise ValueError(f"Could not resolve transformer class '{class_name}' from diffusers or local extensions")


def _get_transformer_class(base: str, tuned: str, subfolder: str) -> Type[torch.nn.Module]:
    base_cfg = _load_json_config(base, subfolder)
    tuned_cfg = _load_json_config(tuned, subfolder)

    base_class = base_cfg.get("_class_name")
    tuned_class = tuned_cfg.get("_class_name")

    if not base_class or not tuned_class:
        raise ValueError("Missing _class_name in model config.json")
    if base_class != tuned_class:
        raise ValueError(f"Model class mismatch: base={base_class}, tuned={tuned_class}")

    return _resolve_transformer_class(base_class)


def _module_target_type(module: torch.nn.Module, extract_conv: bool) -> str | None:
    name = module.__class__.__name__
    if isinstance(module, torch.nn.Linear) or name == "LoRACompatibleLinear":
        return "linear"
    if extract_conv and (isinstance(module, torch.nn.Conv2d) or name == "LoRACompatibleConv"):
        return "conv"
    return None


@torch.no_grad()
def extract_lora_diff(
    base_model: torch.nn.Module,
    tuned_model: torch.nn.Module,
    dim: int,
    alpha: float,
    device: str,
    save_dtype: torch.dtype,
    module_name_filter: str,
    extract_conv_modules: bool,
    eps: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    base_modules = dict(base_model.named_modules())
    lora_sd: Dict[str, torch.Tensor] = {}

    stats = {
        "seen": 0,
        "matched": 0,
        "extracted": 0,
        "skipped_identical": 0,
        "skipped_shape": 0,
        "skipped_filter": 0,
    }

    iterator = list(tuned_model.named_modules())
    for name, tuned_module in tqdm(iterator, desc="Extracting"):
        target_type = _module_target_type(tuned_module, extract_conv_modules)
        if target_type is None:
            continue

        stats["seen"] += 1

        if module_name_filter and module_name_filter not in name:
            stats["skipped_filter"] += 1
            continue

        base_module = base_modules.get(name)
        if base_module is None or not hasattr(base_module, "weight"):
            continue

        tuned_weight = tuned_module.weight
        base_weight = base_module.weight

        if tuned_weight.shape != base_weight.shape:
            stats["skipped_shape"] += 1
            continue

        stats["matched"] += 1

        diff = tuned_weight.detach().to(device=device, dtype=torch.float32) - base_weight.detach().to(
            device=device, dtype=torch.float32
        )

        if eps > 0.0 and float(diff.abs().max().item()) <= eps:
            stats["skipped_identical"] += 1
            continue

        if target_type == "linear":
            down, up, rank, _ = extract_linear(diff, mode="fixed", mode_param=dim, device=device)
        else:
            down, up, rank, _ = extract_conv(diff, mode="fixed", mode_param=dim, device=device)

        # For PEFT-style output we bake alpha/rank into lora_B so inference scale at 1.0 works as expected.
        scale = float(alpha) / float(rank)
        up = up * scale

        key_prefix = f"transformer.{name}"
        lora_sd[f"{key_prefix}.lora_A.weight"] = down.detach().to("cpu", dtype=save_dtype).contiguous()
        lora_sd[f"{key_prefix}.lora_B.weight"] = up.detach().to("cpu", dtype=save_dtype).contiguous()

        stats["extracted"] += 1

        del diff, down, up

    return lora_sd, stats


def main() -> None:
    args = parse_args()
    load_dtype = _torch_dtype(args.load_dtype)
    save_dtype = _torch_dtype(args.save_dtype)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    transformer_cls = _get_transformer_class(args.base, args.tuned, args.subfolder)
    print(f"Using transformer class: {transformer_cls.__name__}")

    print(f"Loading base model: {args.base}")
    base_model = transformer_cls.from_pretrained(args.base, subfolder=args.subfolder, torch_dtype=load_dtype)

    print(f"Loading tuned model: {args.tuned}")
    tuned_model = transformer_cls.from_pretrained(args.tuned, subfolder=args.subfolder, torch_dtype=load_dtype)

    base_model.to("cpu").eval()
    tuned_model.to("cpu").eval()

    lora_sd, stats = extract_lora_diff(
        base_model=base_model,
        tuned_model=tuned_model,
        dim=args.dim,
        alpha=args.alpha,
        device=args.device,
        save_dtype=save_dtype,
        module_name_filter=args.module_name_filter,
        extract_conv_modules=args.extract_conv,
        eps=args.eps,
    )

    metadata = OrderedDict(
        {
            "format": "pt",
            "lora_format": "peft",
            "base": args.base,
            "tuned": args.tuned,
            "subfolder": args.subfolder,
            "dim": str(args.dim),
            "alpha": str(args.alpha),
            "alpha_baked_into_lora_B": "true",
        }
    )

    save_file(lora_sd, args.output, metadata=metadata)

    print("Extraction complete")
    print(f"Saved LoRA diff: {args.output}")
    print(
        "Stats: "
        f"seen={stats['seen']} matched={stats['matched']} extracted={stats['extracted']} "
        f"skipped_identical={stats['skipped_identical']} skipped_shape={stats['skipped_shape']} "
        f"skipped_filter={stats['skipped_filter']}"
    )


if __name__ == "__main__":
    main()
