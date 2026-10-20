"""One-process/one-device exact Qwen 2.1 signed-DoRA training feasibility gate.

Run from the repository root in a TPU subprocess. There is deliberately no
data-parallel launcher, scheduler, sample generation, or remote publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from workers.kaggle_tpu.runtime_env import configure_torch_only

configure_torch_only()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_file(root, record):
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Cache/asset path must name a file inside its bundle")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Artifact SHA-256 mismatch: {path.name}")
    return path


def load_fixture(cache, pair_id=None):
    from safetensors.torch import load_file
    from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds

    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("kind") != "qwen21_fizgig_paired_cache":
        raise ValueError("Expected version 1 Qwen 2.1 Fizgig paired-cache manifest")
    pairs = manifest["pairs"]
    pair = pairs[0] if pair_id is None else next(p for p in pairs if p["id"] == pair_id)
    if pair.get("loss_multiplier", 1) != 1:
        raise ValueError("Smoke supports only the original unit loss multiplier")
    latents = load_file(str(verified_file(cache, pair["latents"])))
    if set(latents) != {"positive", "negative"}:
        raise ValueError("Pair fixture must contain positive and negative latents only")
    positive, negative = latents["positive"], latents["negative"]
    if positive.shape != negative.shape or positive.ndim != 3 or positive.shape[0] != 64:
        raise ValueError("Expected identical CHW Qwen 2.1 pair latents with 64 channels")
    if not positive.isfinite().all() or not negative.isfinite().all():
        raise ValueError("Cached latents must be finite")
    embeds = AdvancedPromptEmbeds.load(str(verified_file(cache, pair["conditioning"])))
    if set(embeds.keys()) != {"text_embeds", "attention_mask", "image_slot_mask"}:
        raise ValueError("Smoke requires native Qwen prompt, validity and image-slot tensors")
    prompt, mask, slots = [embeds[key][0] for key in ("text_embeds", "attention_mask", "image_slot_mask")]
    if prompt.ndim != 2 or prompt.shape[1] != 4096 or mask.shape != prompt.shape[:1] or slots.shape != mask.shape:
        raise ValueError("Cached Qwen prompt layout is invalid")
    if bool(slots.any()):
        raise ValueError("Image-slider negative targets must not become edit references")
    if not bool(mask.any()) or not prompt.isfinite().all():
        raise ValueError("Cached prompt must be finite and contain valid tokens")
    return manifest, pair, positive.unsqueeze(0), negative.unsqueeze(0), prompt.unsqueeze(0), mask.unsqueeze(0)


def validate_profile(profile):
    """Admit only the selected cached image-slider objective and explicit policies."""
    from toolkit.config_modules import TrainConfig, NetworkConfig

    model, train, network, slider = [profile[key] for key in ("model", "train", "network", "fizgig_slider")]
    if profile["type"] != "fizgig_image_slider" or model.get("arch") != "qwen_image_2" or model.get("qtype") != "convrot8":
        raise ValueError("Smoke requires Qwen 2.1 ConvRot8 Fizgig image slider")
    if network.get("type") != "dora" or slider.get("multipoint") or slider.get("anchor_prompts"):
        raise ValueError("Smoke supports signed DoRA paired images without multipoint or anchors")
    if model.get("layer_offloading") or model.get("layer_offloading_transformer_percent", 0):
        raise ValueError("Layer streaming/offloading is unsupported on this one-rank gate")
    if model.get("model_kwargs"):
        raise ValueError("Non-default Qwen architecture/model kwargs require a separate TPU qualification")
    tc, nc = TrainConfig(**train), NetworkConfig(**network)
    if not tc.train_unet or tc.train_text_encoder or tc.dtype != "bf16" or tc.batch_size != 1:
        raise ValueError("Smoke requires frozen encoders, BF16 transformer activations and batch one")
    for key in ("do_cfg", "do_random_cfg", "do_guidance_loss", "train_refiner", "diff_output_preservation", "blank_prompt_preservation"):
        if train.get(key):
            raise ValueError(f"Unsupported smoke training option: {key}")
    if train.get("loss_type", "mse") != "mse" or train.get("frequency_loss_type", "none") != "none" or train.get("ema_config", {}).get("use_ema"):
        raise ValueError("Smoke only supports MSE Fizgig objective without EMA or frequency losses")
    if not tc.gradient_checkpointing or tc.gradient_accumulation != 1 or tc.gradient_accumulation_steps != 1:
        raise ValueError("Smoke requires block checkpointing and one accumulation step")
    if tc.optimizer not in ("adamw", "adamw8bit"):
        raise ValueError("Only the user-approved AdamW replacement is supported")
    if tc.unet_lr != tc.lr or tc.lr_scheduler != "constant" or tc.lr_scheduler_params:
        raise ValueError("Smoke requires the original single constant learning rate")
    params = train.get("optimizer_params") or {}
    if set(params) - {"weight_decay"}:
        raise ValueError("Smoke requires default AdamW betas/epsilon with explicit original weight decay")
    if nc.dropout or network.get("rank_dropout") or network.get("module_dropout"):
        raise ValueError("Adapter dropout is unqualified in this smoke profile")
    if not 0 <= float(slider.get("diff_weight", 1)) <= 1 or float(slider.get("cfg_scale", 1)) != 1:
        raise ValueError("Smoke requires CFG 1 and a valid original pair-difference weight")
    return tc, nc


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def initialize_training(base, helper, nc, mask, latent_shape, device, seed):
    """Shared exact model/adapter construction; all-device callers serialize host loading."""
    import torch
    from toolkit.util.convrot_xla import prepare_convrot8_scales_for_xla_
    from toolkit.config_modules import ModelConfig
    from toolkit.lora_special import LoRASpecialNetwork
    from toolkit.assistant_lora import load_assistant_lora_from_path
    from workers.kaggle_tpu.export_paired_cache import qwen_module
    QwenImage2Model = qwen_module("qwen_image_2").QwenImage2Model
    QwenImage21Transformer2DModel = qwen_module("src.transformer").QwenImage21Transformer2DModel
    from workers.kaggle_tpu.static_qwen import StaticQwen

    torch.manual_seed(seed)
    architecture = {"patch_size": 1, "in_channels": 64, "out_channels": 64, "num_layers": 32,
                    "attention_head_dim": 128, "num_attention_heads": 32, "context_in_dim": 4096,
                    "mlp_ratio": 3, "axes_dims_rope": [16, 56, 56], "eps": 1e-6, "causal_condition": True}
    model = QwenImage21Transformer2DModel.load(str(base), qtype="convrot8", dtype=torch.bfloat16,
                                             device=torch.device("cpu"), config=architecture)
    model.to(dtype=torch.bfloat16).requires_grad_(False)
    quantized = [m for m in model.modules() if getattr(m, "is_ostris_quantized", False)]
    if not quantized or any(not hasattr(m, "cr8_qdata") for m in quantized):
        raise RuntimeError("Base did not load as the original frozen ConvRot8 checkpoint")
    holder = QwenImage2Model("cpu", ModelConfig(arch="qwen_image_2", name_or_path=str(base), qtype="convrot8"), dtype="bf16")
    holder.model, holder.text_encoder = model, []
    holder.pipeline = SimpleNamespace(transformer=model, text_encoder=None)
    assistant = load_assistant_lora_from_path(str(helper), holder, strict=True)
    network = LoRASpecialNetwork(text_encoder=[], unet=model, lora_dim=nc.linear, alpha=nc.linear_alpha,
                                conv_lora_dim=nc.conv, conv_alpha=nc.conv_alpha, network_type="dora",
                                network_config=nc, train_unet=True, train_text_encoder=False,
                                is_transformer=True, base_model=holder, transformer_only=nc.transformer_only,
                                target_lin_modules=holder.target_lora_modules, **nc.network_kwargs)
    network.force_to("cpu", dtype=torch.float32)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.signed_dora_slider, network.is_active, network.can_merge_in = True, True, False
    static = StaticQwen(model, mask, *latent_shape[-2:], checkpoint_blocks=True)
    count = prepare_convrot8_scales_for_xla_(model)
    static.to(device)
    assistant.force_to(device, dtype=torch.bfloat16)
    assistant._update_torch_multiplier()
    network.force_to(device, dtype=torch.float32)
    network._update_torch_multiplier()
    for owner in (static, assistant, network):
        if any(t.device != device for t in list(owner.parameters()) + list(owner.buffers())):
            raise RuntimeError("Training model contains tensors outside its owning XLA device")
    return SimpleNamespace(holder=holder, model=model, assistant=assistant, network=network, static=static, convrot_layers=count)


def run(args, report, update):
    # Validate installed TPU wheels before loading the model/trainer imports.
    from workers.kaggle_tpu.probe import preflight
    report["runtime"] = preflight()
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.debug.metrics as metrics
    from toolkit.fizgig_image_objective import _noised_state, _image_loss

    device = xm.xla_device()
    if xr.world_size() != 1 or device.type != "xla":
        raise RuntimeError("This smoke requires exactly one XLA process/rank; do not launch all-device training")
    report["logical_device"] = str(device)
    report["world_size"] = xr.world_size()
    manifest, pair, positive, negative, prompt, mask = load_fixture(args.cache, args.pair_id)
    tc, nc = validate_profile(manifest["profile"])
    assets = json.loads(args.asset_manifest.read_text())
    base = verified_file(args.asset_manifest.parent, {"path": args.base_name, **assets["files"][args.base_name]})
    helper = verified_file(args.asset_manifest.parent, {"path": args.helper_name, **assets["files"][args.helper_name]})
    for field, asset_name in (("name_or_path", args.base_name), ("assistant_lora_path", args.helper_name)):
        if manifest["components"][field]["sha256"] != assets["files"][asset_name]["sha256"]:
            raise ValueError(f"Cached profile component identity differs from supplied {field}")
    identity = {"base_sha256": assets["files"][args.base_name]["sha256"],
                "helper_sha256": assets["files"][args.helper_name]["sha256"],
                "cache_manifest_sha256": sha256(args.cache / "manifest.json"), "pair_id": pair["id"],
                "world_size": 1, "per_device_batch": 1, "optimizer": "AdamW", "optimizer_epsilon": 1e-6, "adapter_dtype": "float32",
                "quantization_contract": "convrot8_w8a8_xla_v1"}
    report["identity"] = identity
    report["source_files_sha256"] = {name: sha256(ROOT / name) for name in (
        "workers/kaggle_tpu/full_model_smoke.py", "workers/kaggle_tpu/static_qwen.py",
        "workers/kaggle_tpu/runtime_env.py",
        "toolkit/lora_special.py",
        "toolkit/util/convrot_quant.py", "toolkit/util/convrot_xla.py",
        "toolkit/fizgig_image_objective.py",
        "extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py")}
    report["policies"] = {"low_vram": "CPU checkpoint staging, then every training layer placed on XLA",
                           "cuda_compile": "replaced by XLA lazy compilation; no torch.compile",
                           "attention": "dense segmented SDPA correctness gate; fit unproven until execution",
                           "optimizer": "user-approved AdamW; original LR/weight decay, FP32 moments and parameters",
                           "scope": "one optimizer update, local checkpoint roundtrip; no bulk training or remote publication"}
    report["latent_shape"] = list(positive.shape)
    report["prompt_shape"] = list(prompt.shape)
    update("load_frozen_transformer")
    started = time.monotonic()
    components = initialize_training(base, helper, nc, mask, positive.shape, device, args.seed)
    model, assistant, network, static = components.model, components.assistant, components.network, components.static
    report["convrot_layers"] = components.convrot_layers
    positive, negative, prompt = [t.to(device, dtype=torch.bfloat16) for t in (positive, negative, prompt)]
    # Reuse the trainer's exact signed image loss and noise sampler. The shim
    # replaces prediction/accelerator/UI plumbing only; it never copies losses.
    trainer = SimpleNamespace(train_config=tc, device_torch=device, slider_diff_weight=float(manifest["profile"]["fizgig_slider"].get("diff_weight", 1)))
    trainer._noised_state = MethodType(_noised_state, trainer)
    trainer._predict = lambda noisy, timestep, embeds: static(noisy, timestep.to(torch.bfloat16) / 1000, embeds).float()
    optimizer = torch.optim.AdamW(network.parameters(), lr=tc.lr,
                                 weight_decay=tc.optimizer_params.get("weight_decay", 0.01), eps=1e-6, foreach=False, fused=False)
    xm.mark_step()
    xm.wait_device_ops()
    report["load_seconds"] = time.monotonic() - started
    report["trainable_parameters"] = sum(p.numel() for p in network.parameters() if p.requires_grad)
    metrics.clear_all()
    xm.set_rng_state(args.seed, device)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.monotonic()
    for strength, clean, other in ((1.0, positive, negative), (-1.0, negative, positive)):
        update(f"compile_forward_backward_strength_{strength:+g}")
        network.multiplier = strength
        loss = _image_loss(trainer, clean, other, prompt)
        (loss * 0.5).backward()
        # Complete recomputation before changing the mutable adapter strength.
        xm.mark_step()
        xm.wait_device_ops()
        losses.append(float(loss.detach().cpu()))
    report["forward_backward_seconds"] = time.monotonic() - started
    parameters = [p for p in network.parameters() if p.requires_grad]
    if any(p.grad is None for p in parameters):
        raise RuntimeError("An adapter parameter was conditionally unused")
    finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
    if not bool(finite.cpu()) or not all(__import__("math").isfinite(value) for value in losses):
        raise FloatingPointError("Nonfinite signed image loss or DoRA gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, tc.max_grad_norm, foreach=False)
    # A small representative layer supplies direct update evidence without a
    # complete second adapter snapshot on the device.
    update_probe = network.get_all_modules()[0].lora_up.weight
    probe_before = update_probe.detach().cpu().clone()
    update("optimizer_step")
    optimizer.step()
    xm.mark_step()
    xm.wait_device_ops()
    report["losses"] = losses
    report["gradient_norm"] = float(norm.cpu())
    report["optimizer_changed_probe"] = not torch.equal(probe_before, update_probe.detach().cpu())
    if not report["optimizer_changed_probe"]:
        raise RuntimeError("AdamW did not change the representative adapter projection")
    if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all().cpu()):
        raise FloatingPointError("AdamW produced nonfinite adapter parameters")
    report["xla_metrics"] = metrics.metrics_report()
    fallback = {name: metrics.counter_value(name) for name in metrics.counter_names() if name.startswith("aten::")}
    report["aten_fallback_counters"] = fallback
    if any(value for value in fallback.values()):
        raise RuntimeError(f"Unqualified XLA CPU fallbacks occurred: {fallback}")
    report["memory_info"] = xm.get_memory_info(device)
    report["peak_hbm_measured"] = False
    update("checkpoint_save_reload")
    checkpoint = {"schema": 1, "identity": identity, "completed_optimizer_updates": 1,
                  "adapter": _cpu_tree(network.state_dict()), "optimizer": _cpu_tree(optimizer.state_dict()),
                  "cpu_rng": torch.get_rng_state(), "xla_rng": xm.get_rng_state(device)}
    checkpoint_path = args.output / "checkpoint.pt"
    pending = checkpoint_path.with_suffix(".partial")
    torch.save(checkpoint, pending)
    pending.replace(checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if loaded["identity"] != identity or loaded["completed_optimizer_updates"] != 1:
        raise RuntimeError("Checkpoint identity/update count did not roundtrip")
    # Prove loading restores adapter values, rather than merely reading bytes.
    with torch.no_grad():
        parameters[0].add_(1)
    network.load_state_dict(loaded["adapter"], strict=True)
    optimizer.load_state_dict(loaded["optimizer"])
    torch.set_rng_state(loaded["cpu_rng"])
    xm.set_rng_state(loaded["xla_rng"], device)
    xm.mark_step()
    xm.wait_device_ops()
    for key, value in network.state_dict().items():
        if not torch.equal(value.cpu(), loaded["adapter"][key]):
            raise RuntimeError(f"Reloaded adapter differs at {key}")
    report["checkpoint"] = {"path": checkpoint_path.name, "sha256": sha256(checkpoint_path),
                             "adapter_restored": True, "optimizer_loaded": True, "rng_loaded": True,
                             "fresh_process_resume_tested": False}
    report["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["status"] = "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--base-name", default="Qwenvy2.1-v1.0.safetensors")
    parser.add_argument("--helper-name", default="fizgig_qwen_image_2.1_training_adapter.safetensors")
    parser.add_argument("--pair-id")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Smoke output must be new and empty")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "status": "error", "started_unix": time.time()}

    def update(phase):
        report["phase"] = phase
        report["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        pending = args.output / "diagnostic.json.partial"
        pending.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        pending.replace(args.output / "diagnostic.json")
        print(f"TPU smoke: {phase}", flush=True)

    try:
        update("preflight")
        run(args, report, update)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        update("complete" if report["status"] == "pass" else report.get("phase", "preflight"))


if __name__ == "__main__":
    main()
