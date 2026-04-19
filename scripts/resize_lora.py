#!/usr/bin/env python3
"""Resize ai-toolkit LoRA/DoRA checkpoints using SVD.

This follows the algorithm described in tmp/resize_lora_algorithm_spec.md:
- detect down/up pairs from supported key conventions
- reconstruct unscaled dense update
- SVD re-factorize at a new rank (fixed or dynamic)
- set alpha to preserve runtime scale alpha/rank

Supported key pairs:
- lora_down / lora_up
- lora_A / lora_B
- down / up

DoRA checkpoints are supported because they still store LoRA A/B tensors;
DoRA-specific tensors (for example magnitude vectors) are preserved untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from toolkit.metadata import add_model_hash_to_meta  # noqa: E402


SUPPORTED_PAIRS = [
    ("lora_down", "lora_up"),
    ("lora_A", "lora_B"),
    ("down", "up"),
]
MIN_SV = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize LoRA/DoRA checkpoint rank")
    parser.add_argument("--input", required=True, help="Input checkpoint (.safetensors/.pt/.pth/.ckpt)")
    parser.add_argument("--save_to", required=True, help="Output checkpoint path")
    parser.add_argument("--new_rank", required=True, type=int, help="Target rank cap for non-conv layers")
    parser.add_argument("--new_conv_rank", type=int, default=None, help="Target rank cap for conv layers")
    parser.add_argument(
        "--dynamic_method",
        choices=["sv_ratio", "sv_cumulative", "sv_fro"],
        default=None,
        help="Optional dynamic rank policy",
    )
    parser.add_argument("--dynamic_param", type=float, default=None, help="Threshold for dynamic method")
    parser.add_argument(
        "--save_precision",
        choices=["float", "fp16", "bf16"],
        default="float",
        help="Output tensor precision",
    )
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--svd_lowrank_niter",
        type=int,
        default=2,
        help="Power iterations for torch.svd_lowrank. Set 0 to disable.",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "float":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def load_state_dict_with_metadata(path: str, dtype: torch.dtype) -> Tuple[OrderedDict, OrderedDict]:
    if path.endswith(".safetensors"):
        sd = load_file(path)
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        state_dict = OrderedDict((k, v.to(dtype=dtype)) for k, v in sd.items())
        metadata = OrderedDict(meta)
        return state_dict, metadata

    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("Expected a dict-like checkpoint")
    state_dict = OrderedDict()
    for k, v in raw.items():
        if torch.is_tensor(v):
            state_dict[k] = v.to(dtype=dtype)
        else:
            state_dict[k] = v
    return state_dict, OrderedDict()


def detect_down_key(key: str) -> Optional[Tuple[str, str, str, str]]:
    for down_name, up_name in SUPPORTED_PAIRS:
        suffix_weight = f".{down_name}.weight"
        suffix_plain = f".{down_name}"

        if key.endswith(suffix_weight):
            block = key[: -len(suffix_weight)]
            return block, down_name, up_name, ".weight"

        if key.endswith(suffix_plain):
            block = key[: -len(suffix_plain)]
            return block, down_name, up_name, ""

    return None


def merge_linear(lora_down: torch.Tensor, lora_up: torch.Tensor, device: str) -> torch.Tensor:
    rank_down = lora_down.shape[0]
    rank_up = lora_up.shape[1]
    if rank_down != rank_up:
        raise ValueError(f"Rank mismatch: down={rank_down}, up={rank_up}")
    return lora_up.to(device=device, dtype=torch.float32) @ lora_down.to(device=device, dtype=torch.float32)


def merge_conv(lora_down: torch.Tensor, lora_up: torch.Tensor, device: str) -> torch.Tensor:
    rank_down, in_ch, kh, kw = lora_down.shape
    out_ch, rank_up, uh, uw = lora_up.shape
    if rank_down != rank_up:
        raise ValueError(f"Rank mismatch: down={rank_down}, up={rank_up}")
    if kh != kw:
        raise ValueError(f"Conv kernel must be square, got ({kh}, {kw})")
    if uh != 1 or uw != 1:
        raise ValueError(f"lora_up conv must be 1x1, got ({uh}, {uw})")

    up_flat = lora_up.to(device=device, dtype=torch.float32).reshape(out_ch, rank_down)
    down_flat = lora_down.to(device=device, dtype=torch.float32).reshape(rank_down, in_ch * kh * kw)
    merged = up_flat @ down_flat
    return merged.reshape(out_ch, in_ch, kh, kw)


def index_sv_cumulative(S: torch.Tensor, target: float) -> int:
    total = torch.sum(S)
    if total <= 0:
        return 0
    cumulative = torch.cumsum(S, dim=0) / total
    idx = int(torch.searchsorted(cumulative, torch.tensor(target, device=S.device), right=False).item())
    return max(0, min(idx, S.shape[0] - 1))


def index_sv_fro(S: torch.Tensor, target: float) -> int:
    sq = S * S
    total = torch.sum(sq)
    if total <= 0:
        return 0
    cumulative = torch.cumsum(sq, dim=0) / total
    threshold = target * target
    idx = int(torch.searchsorted(cumulative, torch.tensor(threshold, device=S.device), right=False).item())
    return max(0, min(idx, S.shape[0] - 1))


def index_sv_ratio(S: torch.Tensor, target: float) -> int:
    max_sv = float(S[0].item())
    if max_sv <= 0:
        return 0
    min_sv = max_sv / target
    count = int(torch.sum(S > min_sv).item())
    idx = count - 1
    return max(0, min(idx, S.shape[0] - 1))


def rank_resize(
    S: torch.Tensor,
    rank_cap: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    scale: float = 1.0,
) -> Dict[str, float]:
    if dynamic_method is None:
        new_rank = rank_cap
    elif dynamic_method == "sv_ratio":
        new_rank = index_sv_ratio(S, float(dynamic_param)) + 1
    elif dynamic_method == "sv_cumulative":
        new_rank = index_sv_cumulative(S, float(dynamic_param)) + 1
    elif dynamic_method == "sv_fro":
        new_rank = index_sv_fro(S, float(dynamic_param)) + 1
    else:
        raise ValueError(f"Unknown dynamic method: {dynamic_method}")

    if float(S[0].item()) <= MIN_SV:
        new_rank = 1

    new_rank = min(max(1, int(new_rank)), int(rank_cap))
    new_alpha = float(scale * new_rank)

    sum_total = torch.sum(S)
    sum_used = torch.sum(S[:new_rank])
    sum_retained = float((sum_used / sum_total).item()) if sum_total > 0 else float("nan")

    fro_total = torch.sqrt(torch.sum(S * S))
    fro_used = torch.sqrt(torch.sum(S[:new_rank] * S[:new_rank]))
    fro_retained = float((fro_used / fro_total).item()) if fro_total > 0 else float("nan")

    denom = float(S[new_rank - 1].item())
    max_ratio = float(S[0].item() / denom) if denom != 0.0 else float("nan")

    return {
        "new_rank": float(new_rank),
        "new_alpha": new_alpha,
        "sum_retained": sum_retained,
        "fro_retained": fro_retained,
        "max_ratio": max_ratio,
    }


def _svd_2d(weight_2d: torch.Tensor, rank_cap: int, svd_lowrank_niter: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = weight_2d.shape
    if svd_lowrank_niter > 0 and m > 2048 and n > 2048:
        q = min(2 * rank_cap, m, n)
        U, S, V = torch.svd_lowrank(weight_2d, q=q, niter=svd_lowrank_niter)
        Vh = V.transpose(0, 1)
        return U, S, Vh
    U, S, Vh = torch.linalg.svd(weight_2d)
    return U, S, Vh


def extract_linear(
    weight: torch.Tensor,
    rank_cap: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    device: str,
    scale: float = 1.0,
    svd_lowrank_niter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, float]]:
    matrix = weight.to(device=device, dtype=torch.float32)
    U, S, Vh = _svd_2d(matrix, rank_cap, svd_lowrank_niter)

    params = rank_resize(S, rank_cap, dynamic_method, dynamic_param, scale=scale)
    r = int(params["new_rank"])

    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]

    up = U @ torch.diag(S)
    down = Vh
    return down, up, params["new_alpha"], params


def extract_conv(
    weight: torch.Tensor,
    rank_cap: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    device: str,
    scale: float = 1.0,
    svd_lowrank_niter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, float]]:
    out_ch, in_ch, kh, kw = weight.shape
    matrix = weight.to(device=device, dtype=torch.float32).reshape(out_ch, in_ch * kh * kw)
    U, S, Vh = _svd_2d(matrix, rank_cap, svd_lowrank_niter)

    params = rank_resize(S, rank_cap, dynamic_method, dynamic_param, scale=scale)
    r = int(params["new_rank"])

    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]

    up = (U @ torch.diag(S)).reshape(out_ch, r, 1, 1)
    down = Vh.reshape(r, in_ch, kh, kw)
    return down, up, params["new_alpha"], params


def resize_lora_model(
    lora_sd: OrderedDict,
    new_rank: int,
    new_conv_rank: int,
    save_dtype: torch.dtype,
    device: str,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    verbose: bool,
    svd_lowrank_niter: int,
) -> Tuple[OrderedDict, float]:
    out_sd = OrderedDict(lora_sd)
    processed_blocks = set()
    fro_list = []
    last_alpha = float(new_rank)

    for key, value in lora_sd.items():
        if not torch.is_tensor(value):
            continue

        parsed = detect_down_key(key)
        if parsed is None:
            continue

        block_prefix, down_name, up_name, weight_suffix = parsed
        block_id = f"{block_prefix}|{down_name}|{weight_suffix}"
        if block_id in processed_blocks:
            continue
        processed_blocks.add(block_id)

        down_key = f"{block_prefix}.{down_name}{weight_suffix}"
        up_key = f"{block_prefix}.{up_name}{weight_suffix}"
        alpha_key = f"{block_prefix}.alpha"

        if down_key not in lora_sd or up_key not in lora_sd:
            continue

        down = lora_sd[down_key]
        up = lora_sd[up_key]
        if not (torch.is_tensor(down) and torch.is_tensor(up)):
            continue

        old_rank = int(down.shape[0])
        alpha = lora_sd.get(alpha_key, None)
        if torch.is_tensor(alpha):
            scale = float(alpha.detach().float().item()) / float(old_rank)
        else:
            scale = 1.0

        if down.ndim == 4:
            merged = merge_conv(down, up, device=device)
            new_down, new_up, new_alpha, info = extract_conv(
                merged,
                rank_cap=new_conv_rank,
                dynamic_method=dynamic_method,
                dynamic_param=dynamic_param,
                device=device,
                scale=scale,
                svd_lowrank_niter=svd_lowrank_niter,
            )
        else:
            merged = merge_linear(down, up, device=device)
            new_down, new_up, new_alpha, info = extract_linear(
                merged,
                rank_cap=new_rank,
                dynamic_method=dynamic_method,
                dynamic_param=dynamic_param,
                device=device,
                scale=scale,
                svd_lowrank_niter=svd_lowrank_niter,
            )

        out_sd[down_key] = new_down.detach().to("cpu", dtype=save_dtype).contiguous()
        out_sd[up_key] = new_up.detach().to("cpu", dtype=save_dtype).contiguous()
        out_sd[alpha_key] = torch.tensor(new_alpha, dtype=save_dtype)
        last_alpha = float(new_alpha)

        fro_ret = float(info["fro_retained"])
        if not (fro_ret != fro_ret):  # NaN check without numpy
            fro_list.append(fro_ret)

        if verbose:
            print(
                f"{block_prefix}: old_rank={old_rank} new_rank={int(info['new_rank'])} "
                f"alpha={new_alpha:.6g} sum_retained={info['sum_retained']:.6f} "
                f"fro_retained={info['fro_retained']:.6f} max_ratio={info['max_ratio']:.6f}"
            )

    if verbose and len(fro_list) > 0:
        fro_t = torch.tensor(fro_list, dtype=torch.float32)
        print(f"Fro retained mean={float(fro_t.mean().item()):.6f} std={float(fro_t.std(unbiased=False).item()):.6f}")

    return out_sd, last_alpha


def update_metadata(
    metadata: OrderedDict,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    new_rank: int,
    new_alpha: float,
) -> OrderedDict:
    out = OrderedDict(metadata)
    comment = out.get("ss_training_comment", "")
    if dynamic_method is None:
        suffix = f"resized_lora(new_rank={new_rank})"
        out["ss_network_dim"] = str(new_rank)
        out["ss_network_alpha"] = str(new_alpha)
    else:
        suffix = f"resized_lora(dynamic_method={dynamic_method},dynamic_param={dynamic_param})"
        out["ss_network_dim"] = "Dynamic"
        out["ss_network_alpha"] = "Dynamic"

    out["ss_training_comment"] = f"{comment}; {suffix}" if comment else suffix
    return out


def save_checkpoint(path: str, state_dict: OrderedDict, metadata: OrderedDict):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        save_file(state_dict, path, metadata={k: str(v) for k, v in metadata.items()})
        return
    torch.save(state_dict, path)


def main() -> None:
    args = parse_args()

    ext = os.path.splitext(args.save_to)[1].lower()
    if ext not in {".safetensors", ".ckpt", ".pt", ".pth"}:
        raise ValueError("--save_to must end with .ckpt/.pt/.pth/.safetensors")

    if args.dynamic_method is not None and args.dynamic_param is None:
        raise ValueError("--dynamic_param is required when --dynamic_method is set")

    compute_dtype = torch.float32
    save_dtype = get_dtype(args.save_precision)
    new_conv_rank = args.new_conv_rank if args.new_conv_rank is not None else args.new_rank

    print(f"Loading: {args.input}")
    lora_sd, metadata = load_state_dict_with_metadata(args.input, dtype=compute_dtype)

    resized_sd, new_alpha = resize_lora_model(
        lora_sd=lora_sd,
        new_rank=args.new_rank,
        new_conv_rank=new_conv_rank,
        save_dtype=save_dtype,
        device=args.device,
        dynamic_method=args.dynamic_method,
        dynamic_param=args.dynamic_param,
        verbose=args.verbose,
        svd_lowrank_niter=args.svd_lowrank_niter,
    )

    metadata = update_metadata(
        metadata=metadata,
        dynamic_method=args.dynamic_method,
        dynamic_param=args.dynamic_param,
        new_rank=args.new_rank,
        new_alpha=new_alpha,
    )

    if ext == ".safetensors":
        hash_sd = OrderedDict()
        for k, v in resized_sd.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                hash_sd[k] = v.to(dtype=save_dtype)
            else:
                hash_sd[k] = v
        metadata = add_model_hash_to_meta(hash_sd, metadata)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_to)), exist_ok=True)
    print(f"Saving: {args.save_to}")
    save_checkpoint(args.save_to, resized_sd, metadata)
    print("Done")


if __name__ == "__main__":
    main()
