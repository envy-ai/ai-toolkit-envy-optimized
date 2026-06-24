#!/usr/bin/env python3
"""Extract a Krea 2 Raw -> Turbo difference LoRA.

The Krea 2 checkpoints are plain safetensors files (raw.safetensors and
turbo.safetensors), not a Diffusers transformer subfolder. This script streams
those files one tensor at a time and emits PEFT-style transformer LoRA weights
that ai-toolkit can load as an inference LoRA.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Mapping, MutableMapping, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a rank-limited LoRA diff between Krea 2 checkpoints"
    )
    parser.add_argument(
        "--base",
        type=str,
        default="krea/Krea-2-Raw",
        help="Base model path, directory, safetensors file, or HF repo id",
    )
    parser.add_argument(
        "--tuned",
        type=str,
        default="krea/Krea-2-Turbo",
        help="Tuned model path, directory, safetensors file, or HF repo id",
    )
    parser.add_argument(
        "--base-filename",
        type=str,
        default=None,
        help="Checkpoint filename for --base when it is a directory or HF repo",
    )
    parser.add_argument(
        "--tuned-filename",
        type=str,
        default=None,
        help="Checkpoint filename for --tuned when it is a directory or HF repo",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/krea2_raw_to_turbo_r256.safetensors",
        help="Output PEFT LoRA safetensors path",
    )
    parser.add_argument("--dim", type=int, default=256, help="Target LoRA rank")
    parser.add_argument(
        "--alpha",
        type=float,
        default=256.0,
        help="LoRA alpha. This is baked into lora_B as alpha/rank.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for SVD extraction. Use cpu for lowest VRAM usage.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="LoRA tensor dtype in the output file",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="tmp/krea2_diff_lora/hf",
        help="Hugging Face cache dir for downloaded checkpoints",
    )
    parser.add_argument(
        "--module-name-filter",
        type=str,
        default="",
        help="Only include modules whose full name contains this substring",
    )
    parser.add_argument(
        "--all-linear",
        action="store_true",
        help="Extract all linear weights instead of only transformer blocks",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Skip modules where max absolute diff <= eps",
    )
    parser.add_argument(
        "--svd-lowrank-niter",
        type=int,
        default=2,
        help=(
            "Power iterations for torch.svd_lowrank on large matrices. "
            "Set 0 for exact torch.linalg.svd."
        ),
    )
    return parser.parse_args()


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def default_checkpoint_filename(model_id_or_path: str) -> str:
    name = model_id_or_path.rstrip("/").split("/")[-1]
    return name.split("-")[-1].lower() + ".safetensors"


def resolve_checkpoint_path(
    model_id_or_path: str,
    filename: str | None = None,
    cache_dir: str = "tmp/krea2_diff_lora/hf",
) -> str:
    if model_id_or_path.endswith(".safetensors") and os.path.isfile(model_id_or_path):
        return os.path.abspath(model_id_or_path)

    if os.path.isdir(model_id_or_path):
        if filename is not None:
            path = os.path.join(model_id_or_path, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            return os.path.abspath(path)
        candidates = [
            os.path.join(model_id_or_path, f)
            for f in os.listdir(model_id_or_path)
            if f.endswith(".safetensors")
        ]
        if len(candidates) == 1:
            return os.path.abspath(candidates[0])
        default_name = default_checkpoint_filename(model_id_or_path)
        default_path = os.path.join(model_id_or_path, default_name)
        if os.path.isfile(default_path):
            return os.path.abspath(default_path)
        raise FileNotFoundError(
            f"Could not pick a checkpoint in {model_id_or_path}: found {candidates}. "
            "Set the matching --base-filename or --tuned-filename."
        )

    os.makedirs(cache_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=model_id_or_path,
        filename=filename or default_checkpoint_filename(model_id_or_path),
        cache_dir=cache_dir,
        token=os.getenv("HF_TOKEN", None),
    )


def _should_extract_module(
    module_name: str,
    module_name_filter: str,
    blocks_only: bool,
) -> bool:
    if blocks_only and not module_name.startswith("blocks."):
        return False
    if module_name_filter and module_name_filter not in module_name:
        return False
    return True


def _is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _factorize_linear_diff(
    diff: torch.Tensor,
    dim: int,
    device: str,
    svd_lowrank_niter: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    matrix = diff.to(device=device, dtype=torch.float32)
    out_ch, in_ch = matrix.shape
    rank = max(1, min(int(dim), out_ch, in_ch))
    use_lowrank = svd_lowrank_niter > 0 and rank < min(out_ch, in_ch)

    if use_lowrank:
        q = min(max(rank + 16, rank * 2), out_ch, in_ch)
        u, s, v = torch.svd_lowrank(matrix, q=q, niter=svd_lowrank_niter)
        vh = v.transpose(0, 1)
    else:
        u, s, vh = torch.linalg.svd(matrix)

    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    up = u @ torch.diag(s)
    down = vh
    return down.detach(), up.detach(), rank


def _factorize_with_cuda_fallback(
    diff: torch.Tensor,
    dim: int,
    device: str,
    svd_lowrank_niter: int,
    stats: MutableMapping[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    try:
        return _factorize_linear_diff(diff, dim, device, svd_lowrank_niter)
    except RuntimeError as exc:
        if not device.startswith("cuda") or not _is_cuda_oom(exc):
            raise
        stats["cuda_oom_fallbacks"] += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _factorize_linear_diff(diff, dim, "cpu", svd_lowrank_niter)


@torch.no_grad()
def extract_krea2_lora_diff_from_state_dicts(
    base_state_dict: Mapping[str, torch.Tensor],
    tuned_state_dict: Mapping[str, torch.Tensor],
    dim: int,
    alpha: float,
    device: str,
    save_dtype: torch.dtype,
    module_name_filter: str = "",
    blocks_only: bool = True,
    eps: float = 0.0,
    svd_lowrank_niter: int = 0,
    show_progress: bool = False,
) -> Tuple[OrderedDict[str, torch.Tensor], dict[str, int]]:
    lora_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    stats = {
        "seen": 0,
        "matched": 0,
        "extracted": 0,
        "skipped_filter": 0,
        "skipped_missing": 0,
        "skipped_shape": 0,
        "skipped_identical": 0,
        "skipped_non_linear": 0,
        "cuda_oom_fallbacks": 0,
    }

    iterator = tqdm(
        list(tuned_state_dict.keys()),
        desc="Extracting",
        disable=not show_progress,
    )
    for key in iterator:
        tuned_weight = tuned_state_dict[key]
        if not key.endswith(".weight") or tuned_weight.ndim != 2:
            stats["skipped_non_linear"] += 1
            continue

        stats["seen"] += 1
        module_name = key[: -len(".weight")]
        if not _should_extract_module(module_name, module_name_filter, blocks_only):
            stats["skipped_filter"] += 1
            continue

        base_weight = base_state_dict.get(key)
        if base_weight is None:
            stats["skipped_missing"] += 1
            continue
        if tuned_weight.shape != base_weight.shape:
            stats["skipped_shape"] += 1
            continue

        stats["matched"] += 1
        diff = tuned_weight.detach().to(dtype=torch.float32) - base_weight.detach().to(
            dtype=torch.float32
        )
        if eps > 0.0 and float(diff.abs().max().item()) <= eps:
            stats["skipped_identical"] += 1
            continue

        down, up, rank = _factorize_with_cuda_fallback(
            diff,
            dim,
            device,
            svd_lowrank_niter,
            stats,
        )
        up = up * (float(alpha) / float(rank))

        key_prefix = f"transformer.{module_name}"
        lora_state_dict[f"{key_prefix}.lora_A.weight"] = (
            down.detach().to("cpu", dtype=save_dtype).contiguous()
        )
        lora_state_dict[f"{key_prefix}.lora_B.weight"] = (
            up.detach().to("cpu", dtype=save_dtype).contiguous()
        )
        stats["extracted"] += 1

        del diff, down, up
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return lora_state_dict, stats


@torch.no_grad()
def extract_krea2_lora_diff_from_safetensors(
    base_path: str,
    tuned_path: str,
    dim: int,
    alpha: float,
    device: str,
    save_dtype: torch.dtype,
    module_name_filter: str = "",
    blocks_only: bool = True,
    eps: float = 0.0,
    svd_lowrank_niter: int = 2,
) -> Tuple[OrderedDict[str, torch.Tensor], dict[str, int]]:
    lora_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    stats = {
        "seen": 0,
        "matched": 0,
        "extracted": 0,
        "skipped_filter": 0,
        "skipped_missing": 0,
        "skipped_shape": 0,
        "skipped_identical": 0,
        "skipped_non_linear": 0,
        "cuda_oom_fallbacks": 0,
    }

    with safe_open(base_path, framework="pt", device="cpu") as base_file:
        base_keys = set(base_file.keys())
        with safe_open(tuned_path, framework="pt", device="cpu") as tuned_file:
            for key in tqdm(list(tuned_file.keys()), desc="Extracting"):
                tuned_weight = tuned_file.get_tensor(key)
                if not key.endswith(".weight") or tuned_weight.ndim != 2:
                    stats["skipped_non_linear"] += 1
                    continue

                stats["seen"] += 1
                module_name = key[: -len(".weight")]
                if not _should_extract_module(
                    module_name, module_name_filter, blocks_only
                ):
                    stats["skipped_filter"] += 1
                    continue

                if key not in base_keys:
                    stats["skipped_missing"] += 1
                    continue
                base_weight = base_file.get_tensor(key)
                if tuned_weight.shape != base_weight.shape:
                    stats["skipped_shape"] += 1
                    continue

                stats["matched"] += 1
                diff = tuned_weight.to(dtype=torch.float32) - base_weight.to(
                    dtype=torch.float32
                )
                if eps > 0.0 and float(diff.abs().max().item()) <= eps:
                    stats["skipped_identical"] += 1
                    continue

                down, up, rank = _factorize_with_cuda_fallback(
                    diff,
                    dim,
                    device,
                    svd_lowrank_niter,
                    stats,
                )
                up = up * (float(alpha) / float(rank))

                key_prefix = f"transformer.{module_name}"
                lora_state_dict[f"{key_prefix}.lora_A.weight"] = (
                    down.detach().to("cpu", dtype=save_dtype).contiguous()
                )
                lora_state_dict[f"{key_prefix}.lora_B.weight"] = (
                    up.detach().to("cpu", dtype=save_dtype).contiguous()
                )
                stats["extracted"] += 1

                del tuned_weight, base_weight, diff, down, up
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return lora_state_dict, stats


def main() -> None:
    args = parse_args()
    save_dtype = _torch_dtype(args.save_dtype)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Resolving base checkpoint: {args.base}")
    base_path = resolve_checkpoint_path(
        args.base,
        filename=args.base_filename,
        cache_dir=args.cache_dir,
    )
    print(f"Base checkpoint: {base_path}")

    print(f"Resolving tuned checkpoint: {args.tuned}")
    tuned_path = resolve_checkpoint_path(
        args.tuned,
        filename=args.tuned_filename,
        cache_dir=args.cache_dir,
    )
    print(f"Tuned checkpoint: {tuned_path}")

    lora_state_dict, stats = extract_krea2_lora_diff_from_safetensors(
        base_path=base_path,
        tuned_path=tuned_path,
        dim=args.dim,
        alpha=args.alpha,
        device=args.device,
        save_dtype=save_dtype,
        module_name_filter=args.module_name_filter,
        blocks_only=not args.all_linear,
        eps=args.eps,
        svd_lowrank_niter=args.svd_lowrank_niter,
    )
    if not lora_state_dict:
        raise RuntimeError(f"No LoRA tensors were extracted. Stats: {stats}")

    metadata = OrderedDict(
        [
            ("format", "pt"),
            ("lora_format", "peft"),
            ("architecture", "krea2"),
            ("base", args.base),
            ("tuned", args.tuned),
            ("base_checkpoint", base_path),
            ("tuned_checkpoint", tuned_path),
            ("dim", str(args.dim)),
            ("alpha", str(args.alpha)),
            ("alpha_baked_into_lora_B", "true"),
            ("blocks_only", str(not args.all_linear).lower()),
            ("svd_lowrank_niter", str(args.svd_lowrank_niter)),
        ]
    )

    print(f"Saving {len(lora_state_dict)} LoRA tensors to {args.output}")
    save_file(lora_state_dict, args.output, metadata=metadata)
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
