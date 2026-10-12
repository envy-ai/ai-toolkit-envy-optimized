"""Kaggle TPU ConvRot feasibility probe. Run as a subprocess, never in a notebook kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def preflight():
    """Inspect installed wheels without importing Torch or initializing PJRT."""
    versions = {name: _package_version(name) for name in ("torch", "torch-xla", "libtpu")}
    torch_version, xla_version = versions["torch"], versions["torch-xla"]
    if torch_version is None or xla_version is None:
        raise RuntimeError("Kaggle runtime needs installed torch and torch-xla; do not guess wheel pins")
    if torch_version.split("+", 1)[0].split(".")[:2] != xla_version.split("+", 1)[0].split(".")[:2]:
        raise RuntimeError(f"Torch/XLA major.minor mismatch: {torch_version} vs {xla_version}")
    for name in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16"):
        if os.environ.get(name, "").lower() not in ("", "0", "false", "no"):
            raise RuntimeError(f"{name} must be unset to preserve FP32 ConvRot scale math")
    if os.environ.get("PJRT_DEVICE") != "TPU":
        raise RuntimeError("Set PJRT_DEVICE=TPU before starting the probe")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "pjrt_device": "TPU",
        "host_ram_bytes": _host_ram(),
        "disk_free_bytes": os.statvfs("/kaggle/working").f_bavail * os.statvfs("/kaggle/working").f_frsize,
        "native_int8_throughput_proven": False,
    }


def _host_ram():
    with open("/proc/meminfo", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemTotal")


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _rotate(x, rot_size):
    if rot_size == 1:
        return x
    import torch

    r4 = torch.tensor(
        [[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    h = r4
    while h.shape[0] < rot_size:
        h = torch.kron(h, r4)
    if h.shape[0] != rot_size:
        raise ValueError("rotation size must be a power of four")
    h = (h / rot_size**0.5).to(device=x.device, dtype=x.dtype)
    return (x.reshape(-1, x.shape[-1] // rot_size, rot_size) @ h).reshape(x.shape)


def _quantize_rows(x, qmax):
    import torch

    xf = x.float()
    scales = xf.abs().amax(dim=1) / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(xf / scales.unsqueeze(1)).clamp_(-qmax, qmax).to(torch.int8)
    return codes, scales


def _rank(index, report_dir):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    from convrot_xla import _int8_dot, convrot8_linear_xla, prepare_convrot8_scales_for_xla_

    rank = xr.global_ordinal()
    result = {"rank": rank, "launcher_index": index, "status": "error"}
    try:
        device = xm.xla_device()
        result.update(
            device=str(device),
            world_size=xr.world_size(),
            addressable_device_count=xr.addressable_device_count(),
            global_logical_device_count=xr.global_device_count(),
            local_ordinal=xr.local_ordinal(),
        )
        a = torch.tensor([[1, -2, 3, -4]], dtype=torch.int8, device=device)
        b = torch.tensor([[3, -2, 1, -4]], dtype=torch.int8, device=device)
        dot = _int8_dot(a, b)
        result["int8_dot_dtype"] = str(dot.dtype)
        result["int8_dot_value"] = int(dot.cpu()[0, 0])
        if result["int8_dot_value"] != 26:
            raise AssertionError("INT8 dot returned an incorrect value")

        rank_sum = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(rank + 1, dtype=torch.float32, device=device))
        result["collective_sum"] = float(rank_sum.cpu())
        expected_sum = result["world_size"] * (result["world_size"] + 1) / 2
        if result["collective_sum"] != expected_sum:
            raise AssertionError("all-reduce did not include every replica")

        cpu_gen = torch.Generator(device="cpu").manual_seed(20261012)
        layer = torch.nn.Module()
        layer.in_features = 256
        layer.out_features = 512
        layer.register_buffer("cr8_qdata", torch.randint(-127, 128, (512, 256), generator=cpu_gen, dtype=torch.int8))
        layer.register_buffer("cr8_scales", torch.full((512,), 1 / 127, dtype=torch.float32).view(torch.uint8))
        layer.register_parameter("bias", None)
        prepare_convrot8_scales_for_xla_(layer)
        layer.to(device)
        x_host = torch.randn((2, 256), generator=cpu_gen, dtype=torch.float32).to(torch.bfloat16)
        a_host = (torch.randn((256, 4), generator=cpu_gen) * 0.01).to(torch.bfloat16)
        b_host = (torch.randn((4, 512), generator=cpu_gen) * 0.01).to(torch.bfloat16)
        x = x_host.to(device).requires_grad_()
        lora_a = torch.nn.Parameter(a_host.to(device))
        lora_b = torch.nn.Parameter(b_host.to(device))
        saved = []

        def pack(tensor):
            saved.append((tuple(tensor.shape), str(tensor.dtype)))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            y = convrot8_linear_xla(layer, x, 256, 127, _rotate, _quantize_rows)
            loss = (y.float() + ((x @ lora_a) @ lora_b).float()).square().mean()
            loss.backward()
        xm.mark_step()
        result["projection"] = {
            "forward_finite": bool(torch.isfinite(y).all().cpu()),
            "input_grad_finite": bool(torch.isfinite(x.grad).all().cpu()),
            "lora_a_grad_finite": bool(torch.isfinite(lora_a.grad).all().cpu()),
            "lora_b_grad_finite": bool(torch.isfinite(lora_b.grad).all().cpu()),
            "saved_int8_codes": ((512, 256), "torch.int8") in saved,
            "saved_fp32_scales": ((512,), "torch.float32") in saved,
            "saved_full_float_weight": any(shape == (512, 256) and dtype != "torch.int8" for shape, dtype in saved),
        }
        # Compare to the same ConvRot arithmetic on CPU. The base gradient is
        # the existing straight-through dequantized-weight estimate.
        x_ref = x_host.detach().clone().requires_grad_()
        a_ref = a_host.detach().clone().requires_grad_()
        b_ref = b_host.detach().clone().requires_grad_()
        x_rot_ref = _rotate(x_ref, 256)
        aq_ref, as_ref = _quantize_rows(x_rot_ref.detach(), 127)
        codes = layer.cr8_qdata.cpu()
        scales = layer.cr8_xla_scales.cpu()
        i32_ref = aq_ref.to(torch.int32) @ codes.to(torch.int32).T
        y_ref = ((i32_ref.float() * scales) * as_ref[:, None]).to(torch.bfloat16)
        w_ref = codes.to(torch.bfloat16) * scales.to(torch.bfloat16)[:, None]
        base_ref = x_rot_ref @ w_ref.T
        base_ref = base_ref + (y_ref - base_ref).detach()
        ref_loss = (base_ref.float() + ((x_ref @ a_ref) @ b_ref).float()).square().mean()
        ref_loss.backward()
        def comparison(actual, expected):
            actual = actual.detach().cpu().float().reshape(-1)
            expected = expected.detach().cpu().float().reshape(-1)
            delta = actual - expected
            cosine = float(torch.nn.functional.cosine_similarity(actual, expected, dim=0))
            return {"max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt()), "cosine": cosine}
        result["cpu_comparison"] = {
            "forward": comparison(y, y_ref),
            "input_gradient": comparison(x.grad, x_ref.grad),
            "lora_a_gradient": comparison(lora_a.grad, a_ref.grad),
            "lora_b_gradient": comparison(lora_b.grad, b_ref.grad),
        }
        result["comparison_tolerances"] = {"forward_max_abs": 0.5, "gradient_cosine_min": 0.99}
        if result["cpu_comparison"]["forward"]["max_abs"] > 0.5 or any(
            result["cpu_comparison"][name]["cosine"] < 0.99
            for name in ("input_gradient", "lora_a_gradient", "lora_b_gradient")
        ):
            raise AssertionError("ConvRot XLA forward or gradient differs from the CPU reference beyond declared tolerance")
        if not all(result["projection"][name] for name in (
            "forward_finite", "input_grad_finite", "lora_a_grad_finite", "lora_b_grad_finite",
            "saved_int8_codes", "saved_fp32_scales",
        )) or result["projection"]["saved_full_float_weight"]:
            raise AssertionError("ConvRot forward, gradients, or compact saved tensor contract failed")
        try:
            result["memory_info"] = xm.get_memory_info(device)
        except (AttributeError, RuntimeError) as exc:
            result["memory_info_error"] = str(exc)
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        _atomic_json(Path(report_dir) / f"rank-{rank}.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": 1, "status": "error", "started_unix": time.time(), "ranks": []}
    source_manifest = Path(__file__).with_name("source_manifest.json")
    if source_manifest.exists():
        report["source_manifest"] = json.loads(source_manifest.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Probe output directory must be new and empty; stale rank reports are unsafe")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report["environment"] = preflight()
        import torch_xla
        # Do not query a device in the launcher parent: v2/v3 may use threaded
        # replicas and the runtime must first initialize inside torch_xla.launch.
        torch_xla.launch(_rank, args=(str(args.output),))
        report["ranks"] = [json.loads(p.read_text()) for p in sorted(args.output.glob("rank-*.json"))]
        if not report["ranks"]:
            raise RuntimeError("PJRT launcher returned without writing any rank report")
        expected = report["ranks"][0]["world_size"]
        report["expected_logical_devices"] = expected
        if any(item["global_logical_device_count"] != expected or item["world_size"] != expected for item in report["ranks"]):
            raise RuntimeError("Launcher world size differs from discovered logical TPU count")
        if len(report["ranks"]) != expected or any(item["status"] != "pass" for item in report["ranks"]):
            raise RuntimeError(f"Only {len(report['ranks'])}/{expected} logical devices completed the probe")
        ordinals = {item["rank"] for item in report["ranks"]}
        if ordinals != set(range(expected)):
            raise RuntimeError("Rank ordinals do not cover every logical TPU device")
        report["status"] = "pass"
    except Exception:
        report["error"] = traceback.format_exc()
        report["ranks"] = [json.loads(p.read_text()) for p in sorted(args.output.glob("rank-*.json"))]
    finally:
        report["completed_unix"] = time.time()
        report["completed_logical_devices"] = len([r for r in report["ranks"] if r["status"] == "pass"])
        report["full_qwen_training_gate"] = "pending: requires model, cache, attention, and one-device sustained run"
        _atomic_json(args.output / "diagnostic.json", report)
    if report["status"] != "pass":
        raise RuntimeError(report.get("error", "TPU probe failed"))


if __name__ == "__main__":
    main()
