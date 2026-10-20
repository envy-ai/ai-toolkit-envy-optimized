"""Eight-replica Qwen/Fizgig 40,000-update runner with durable checkpoint relay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from workers.kaggle_tpu.runtime_env import configure_torch_only

configure_torch_only()

from workers.kaggle_tpu.full_model_smoke import sha256, verified_file


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".partial")
    pending.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def process_group_rss(group):
    """Observed aggregate RSS; shared pages may be counted in multiple processes."""
    resident, count = 0, 0
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != group:
                continue
            for line in (process / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    resident += int(line.split()[1]) * 1024
                    count += 1
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {"rss_bytes": resident, "process_count": count}


def session_deadline(config, now_unix):
    """Preserve allocation time: Python installation/loading never resets the budget."""
    if config.get("session_deadline_unix") is not None:
        deadline = float(config["session_deadline_unix"])
    elif config.get("allocation_started_unix") is not None:
        deadline = float(config["allocation_started_unix"]) + config["session_seconds"]
    else:
        raise ValueError("Pass actual --session-deadline-unix or --allocation-started-unix; runner startup is not allocation start")
    if not math.isfinite(deadline) or deadline - now_unix <= config["final_save_reserve_seconds"]:
        raise ValueError("Remaining allocation time is no longer greater than the final-save reserve")
    return deadline


def full_resume_boundary(completed, *, stop=False):
    """Regular 1,000-update state or a terminal session boundary."""
    return completed % 1000 == 0 or stop or completed == 40000


def terminal_resume(config):
    """Accept a verified completed input without launching or publishing again."""
    if not config.get("resume"):
        return None
    import torch
    from workers.kaggle_tpu.checkpoint_relay import verify_checkpoint
    directory = Path(config["resume"])
    manifest = verify_checkpoint(directory)
    if manifest["job_id"] != config["job_id"]:
        raise ValueError("Resume checkpoint belongs to another job")
    if manifest["step"] != 40000:
        return None
    state = torch.load(directory / "resume/training.pt", map_location="cpu", weights_only=True)
    required = {"identity", "adapter", "optimizer", "scheduler", "rank_states", "epoch", "cursor"}
    if state.get("schema") != 1 or not required.issubset(state) or state.get("completed_updates") != 40000 or state.get("samples") != 320000:
        raise ValueError("Terminal checkpoint payload is incomplete or has inconsistent update/sample counts")
    identity = state["identity"]
    assets = json.loads(Path(config["asset_manifest"]).read_text())
    if (identity != manifest["metadata"]["identity"] or identity["total_updates"] != 40000
            or identity["cache_manifest_sha256"] != config["manifest_sha256"]
            or identity["schedule"] != config["schedule_contract"]
            or identity["schedule_fingerprint"] != config["schedule_fingerprint"]
            or identity["source_files_sha256"] != config["smoke_source_files_sha256"]
            or identity["base_sha256"] != assets["files"][config["base_name"]]["sha256"]
            or identity["helper_sha256"] != assets["files"][config["helper_name"]]["sha256"]
            or sorted(item["rank"] for item in state["rank_states"]) != list(range(8))):
        raise ValueError("Terminal checkpoint identity or per-rank state differs from the selected profile")
    return {"schema": 1, "status": "completed", "completed_updates": 40000, "samples": 320000,
            "verified_terminal_checkpoint": str(directory.resolve()), "checkpoint_manifest_sha256": sha256(directory / "manifest.json"),
            "launched_replicas": False, "new_publication": False}


def epoch_schedule(exposures, layouts, seed, epoch, world_size=8):
    """Shuffle canonical exposures, retaining repeats and dropping bucket remainders."""
    generator = random.Random(seed + epoch)
    buckets = defaultdict(list)
    for exposure in exposures:
        buckets[tuple(layouts[exposure])].append(exposure)
    batches, dropped = [], 0
    for key in sorted(buckets):
        members = buckets[key]
        generator.shuffle(members)
        while len(set(members)) >= world_size:
            batch, selected, remainder = [], set(), []
            for member in members:
                if len(batch) < world_size and member not in selected:
                    batch.append(member)
                    selected.add(member)
                else:
                    remainder.append(member)
            batches.append(batch)
            members = remainder
        dropped += len(members)
    generator.shuffle(batches)
    if not batches:
        raise ValueError("No complete global batch remains after static-layout bucketing")
    return batches, dropped


def gate_smoke(path, cache, asset_manifest):
    smoke = json.loads(Path(path).read_text())
    if smoke.get("status") != "pass" or smoke.get("world_size") != 1 or not smoke.get("checkpoint", {}).get("adapter_restored"):
        raise ValueError("All-device training requires a real passing one-rank full-model smoke and checkpoint roundtrip")
    if smoke["runtime"]["pjrt_device"] != "TPU" or not smoke.get("optimizer_changed_probe"):
        raise ValueError("Smoke did not prove the TPU optimizer update")
    identity = smoke["identity"]
    if identity["cache_manifest_sha256"] != sha256(Path(cache) / "manifest.json"):
        raise ValueError("Smoke used a different paired cache")
    assets = json.loads(Path(asset_manifest).read_text())
    if identity["base_sha256"] not in {value["sha256"] for value in assets["files"].values()} or identity["helper_sha256"] not in {value["sha256"] for value in assets["files"].values()}:
        raise ValueError("Smoke used different model/helper assets")
    for name, digest in smoke["source_files_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Training source changed after smoke: {name}")
    return smoke


def prepare(config):
    """CPU-only bundle validation; no PJRT initialization before launch."""
    from safetensors import safe_open
    from workers.kaggle_tpu.full_model_smoke import validate_profile
    cache = Path(config["cache"])
    manifest = json.loads((cache / "manifest.json").read_text())
    validate_profile(manifest["profile"])
    if manifest.get("initialization") != "fresh_adapter":
        raise ValueError("Expected an exported fresh-adapter profile")
    if int(manifest["profile"]["train"].get("steps", 40000)) != 40000:
        raise ValueError("The authorized profile requires 40,000 optimizer updates")
    layouts = {}
    for pair in manifest["pairs"]:
        if pair.get("loss_multiplier", 1) != 1:
            raise ValueError("Only the exact unit-loss image-slider profile is supported")
        latent = verified_file(cache, pair["latents"])
        conditioning = verified_file(cache, pair["conditioning"])
        with safe_open(latent, framework="pt") as tensors:
            shape = tensors.get_slice("positive").get_shape()
            if shape != tensors.get_slice("negative").get_shape() or len(shape) != 3 or shape[0] != 64:
                raise ValueError("Incompatible paired Qwen latents")
        with safe_open(conditioning, framework="pt") as tensors:
            length, dim = tensors.get_slice("text_embeds").get_shape()
            if dim != 4096 or bool(tensors.get_tensor("image_slot_mask").any()):
                raise ValueError("Only no-reference Qwen image-slider conditioning is qualified")
        layouts[pair["id"]] = [shape[-2], shape[-1], length]
    if len(set(map(tuple, layouts.values()))) != 1:
        raise ValueError("Initial replicated runner requires one exact image/text layout; mixed layouts need qualification")
    if any(pair not in layouts for pair in manifest["exposures"]):
        raise ValueError("Exposure schedule refers to an unknown paired presentation")
    schedule, dropped = epoch_schedule(manifest["exposures"], layouts, config["seed"], 0)
    config["layouts"] = layouts
    config["manifest_sha256"] = sha256(cache / "manifest.json")
    config["schedule_contract"] = {"version": 1, "seed": config["seed"], "world_size": 8, "per_device_batch": 1,
                                   "accumulation": 1, "global_batch": 8, "remainder": "drop_per_layout_per_epoch",
                                   "exposure_count": len(manifest["exposures"]), "unique_presentations": len(layouts),
                                   "batches_per_epoch": len(schedule), "dropped_exposures_per_epoch": dropped}
    config["schedule_fingerprint"] = hashlib.sha256(json.dumps({"exposures": manifest["exposures"], "layouts": layouts,
                                                             "contract": config["schedule_contract"]}, sort_keys=True).encode()).hexdigest()
    return manifest


def _rank(index, config_path):
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.debug.metrics as metrics
    from workers.kaggle_tpu.full_model_smoke import load_fixture, validate_profile, initialize_training, _cpu_tree
    from workers.kaggle_tpu.checkpoint_relay import build_checkpoint, publish_checkpoint, KaggleDataset, verify_checkpoint
    from toolkit.fizgig_image_objective import _noised_state, _image_loss

    config = json.loads(Path(config_path).read_text())
    output, cache = Path(config["output"]), Path(config["cache"])
    rank, world = xr.global_ordinal(), xr.world_size()
    device = xm.xla_device()
    status = {"rank": rank, "pid": os.getpid(), "status": "running", "completed_updates": 0}

    def heartbeat(phase):
        status.update(phase=phase, updated_unix=time.time())
        write_json(output / f"rank-{rank}.json", status)

    def exchange(tag, value):
        return [json.loads(item.decode()) for item in xm.rendezvous(tag, json.dumps(value, sort_keys=True).encode())]

    try:
        if world != 8 or xr.global_device_count() != 8:
            raise RuntimeError(f"Expected all eight discovered logical TPU devices; launched {world}")
        heartbeat("validate")
        manifest, first, positive, negative, prompt, mask = load_fixture(cache)
        tc, nc = validate_profile(manifest["profile"])
        assets_path = Path(config["asset_manifest"])
        assets = json.loads(assets_path.read_text())
        base_name, helper_name = config["base_name"], config["helper_name"]
        identity = {"schema": 1, "base_sha256": assets["files"][base_name]["sha256"],
                    "helper_sha256": assets["files"][helper_name]["sha256"],
                    "cache_manifest_sha256": config["manifest_sha256"], "schedule": config["schedule_contract"],
                    "schedule_fingerprint": config["schedule_fingerprint"], "total_updates": 40000,
                    "optimizer": "AdamW", "adapter_dtype": "float32", "lr": tc.lr,
                    "weight_decay": tc.optimizer_params.get("weight_decay", 0.01), "optimizer_epsilon": 1e-6,
                    "quantization_contract": "convrot8_w8a8_xla_v1", "runtime": {"versions": config["runtime"]["versions"], "device_count": 8},
                    "source_files_sha256": config["smoke_source_files_sha256"], "replicated_runner_sha256": sha256(__file__)}
        resume = None
        if config.get("resume"):
            checkpoint_dir = Path(config["resume"])
            checkpoint_manifest = verify_checkpoint(checkpoint_dir)
            if checkpoint_manifest["job_id"] != config["job_id"]:
                raise ValueError("Resume checkpoint belongs to another job")
            resume = checkpoint_dir / "resume/training.pt"
        components = None
        # PJRT may use multiple Python threads. Serialize CPU RNG initialization
        # and host staging, then discard each checkpoint staging copy promptly.
        for owner in range(world):
            heartbeat(f"load_replica_{owner}")
            if owner == rank:
                base = verified_file(assets_path.parent, {"path": base_name, **assets["files"][base_name]})
                helper = verified_file(assets_path.parent, {"path": helper_name, **assets["files"][helper_name]})
                for field, name in (("name_or_path", base_name), ("assistant_lora_path", helper_name)):
                    if manifest["components"][field]["sha256"] != assets["files"][name]["sha256"]:
                        raise ValueError("Model component identity differs from exported profile")
                components = initialize_training(base, helper, nc, mask, positive.shape, device, config["seed"])
                xm.mark_step()
                xm.wait_device_ops()
            xm.rendezvous(f"replica-loaded-{owner}")
        network, static = components.network, components.static
        parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=tc.lr, weight_decay=tc.optimizer_params.get("weight_decay", 0.01), eps=1e-6, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        completed, epoch, cursor = 0, 0, 0
        xm.set_rng_state(config["seed"] + 100003 * (rank + 1), device)
        if resume:
            # One CPU resume staging copy at a time as well.
            for owner in range(world):
                if owner == rank:
                    state = torch.load(resume, map_location="cpu", weights_only=True)
                    if state.get("schema") != 1 or state["identity"] != identity:
                        raise ValueError("Resume identity/topology/batch/runtime differs from current run")
                    if state["completed_updates"] != checkpoint_manifest["step"]:
                        raise ValueError("Resume payload update differs from relay checkpoint manifest")
                    network.load_state_dict(state["adapter"], strict=True)
                    optimizer.load_state_dict(state["optimizer"])
                    scheduler.load_state_dict(state["scheduler"])
                    completed, epoch, cursor = state["completed_updates"], state["epoch"], state["cursor"]
                    if state["samples"] != completed * 8 or not 0 <= completed <= 40000:
                        raise ValueError("Resume sample/update accounting is inconsistent")
                    rank_state = state["rank_states"][rank]
                    if rank_state["rank"] != rank:
                        raise ValueError("Resume per-rank RNG order is inconsistent")
                    xm.set_rng_state(rank_state["xla_rng"], device)
                    del state
                    xm.mark_step()
                    xm.wait_device_ops()
                xm.rendezvous(f"resume-loaded-{owner}")
        # Compare every adapter tensor, not only a sample/probe.
        digest = hashlib.sha256()
        for name, value in network.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        checks = exchange("initial-adapter-and-schedule", {"rank": rank, "adapter": digest.hexdigest(),
                           "schedule": config["schedule_fingerprint"], "completed": completed})
        if len({item["adapter"] for item in checks}) != 1 or len({item["schedule"] for item in checks}) != 1 or len({item["completed"] for item in checks}) != 1:
            raise RuntimeError("Replica adapter, schedule or update initialization differs")
        trainer = SimpleNamespace(train_config=tc, device_torch=device,
                                  slider_diff_weight=float(manifest["profile"]["fizgig_slider"].get("diff_weight", 1)))
        trainer._noised_state = MethodType(_noised_state, trainer)
        trainer._predict = lambda noisy, timestep, embeds: static(noisy, timestep.to(torch.bfloat16) / 1000, embeds).float()
        metrics.clear_all()
        last_saved = completed if resume else -1

        def save_boundary(*, full_resume=False):
            nonlocal last_saved
            heartbeat("checkpoint_collect")
            rank_states = (sorted(exchange(f"checkpoint-rng-{completed}", {"rank": rank, "xla_rng": xm.get_rng_state(device)}),
                                  key=lambda value: value["rank"]) if full_resume else None)
            failure = None
            receipt = None
            if rank == 0:
                try:
                    staging = output / f"save-{completed:09d}"
                    staging.mkdir(exist_ok=False)
                    if full_resume:
                        state = {"schema": 1, "identity": identity, "completed_updates": completed, "samples": completed * 8,
                                 "epoch": epoch, "cursor": cursor, "rank_states": rank_states,
                                 "adapter": _cpu_tree(network.state_dict()), "optimizer": _cpu_tree(optimizer.state_dict()),
                                 "scheduler": scheduler.state_dict(), "data_rng": "epoch_schedule(seed+epoch); cursor indexes completed global batches",
                                 "cpu_rng_used_during_steps": False}
                        torch.save(state, staging / "training.pt.partial")
                        (staging / "training.pt.partial").replace(staging / "training.pt")
                        del state
                    # Existing toolkit serializer preserves adapter naming,
                    # alpha, signed DoRA magnitude and model metadata.
                    network.save_weights(str(staging / "adapter.safetensors"), dtype=torch.bfloat16,
                                         metadata={"ss_slider": "image_pairs", "ss_slider_diff_weight": str(trainer.slider_diff_weight),
                                                   "ss_network_module": "networks.dora", "ss_training_updates": str(completed)})
                    files = {"export/adapter.safetensors": staging / "adapter.safetensors"}
                    if full_resume:
                        files["resume/training.pt"] = staging / "training.pt"
                    checkpoint = build_checkpoint(output / "checkpoints", job_id=config["job_id"], attempt_id=config["attempt_id"],
                                                  step=completed, files=files,
                                                  metadata={"identity": identity, "completed_samples": completed * 8,
                                                            "fresh_process_resume_tested": bool(resume) if full_resume else False})
                    shutil.rmtree(staging)
                    heartbeat("checkpoint_publish")
                    try:
                        dataset = KaggleDataset(config["relay_dataset"])
                        receipt = publish_checkpoint(dataset, checkpoint, create=bool(config["relay_create"] and last_saved < 0), timeout=config["publish_timeout"])
                    except Exception as error:
                        # Kaggle HTTP exceptions may contain signed URLs.
                        # Retain type and local recovery path, never response
                        # bodies or credential-bearing traceback chains.
                        raise RuntimeError(f"Private checkpoint publication failed ({type(error).__name__}); local checkpoint preserved at {checkpoint}") from None
                    write_json(output / "durable-latest.json", receipt)
                    # Remote versions are immutable and retained. Preserve two
                    # full local resume states even after export-only saves.
                    local = sorted((output / "checkpoints").iterdir(), key=lambda path: path.name)
                    resume_local = [path for path in local if (path / "resume/training.pt").is_file()]
                    keep = {checkpoint, *resume_local[-2:]}
                    for previous in local:
                        if previous.is_dir() and previous not in keep:
                            shutil.rmtree(previous)
                except Exception:
                    failure = traceback.format_exc()
            outcomes = exchange(f"checkpoint-outcome-{completed}", {"rank": rank, "error": failure, "receipt": receipt})
            leader = next(value for value in outcomes if value["rank"] == 0)
            if leader["error"]:
                raise RuntimeError("Checkpoint publication failed; local completed checkpoint preserved: " + leader["error"])
            last_saved = completed

        while completed < 40000:
            batches, dropped = epoch_schedule(manifest["exposures"], config["layouts"], config["seed"], epoch)
            if not 0 <= cursor <= len(batches):
                raise ValueError("Resume global bucket cursor is out of range")
            if cursor == len(batches):
                epoch, cursor = epoch + 1, 0
                continue
            batch = batches[cursor]
            if len(set(batch)) != 8:
                # Configured repeats are preserved, but the exact same cached
                # presentation must not be silently used on multiple ranks.
                raise ValueError("A global batch contains duplicate canonical presentations; revise the deterministic schedule")
            pair_id = batch[rank]
            _, _, positive, negative, prompt, pair_mask = load_fixture(cache, pair_id)
            if not torch.equal(pair_mask, mask):
                raise ValueError("Prompt validity changed inside the qualified static layout")
            positive, negative, prompt = [tensor.to(device, dtype=torch.bfloat16) for tensor in (positive, negative, prompt)]
            optimizer.zero_grad(set_to_none=True)
            heartbeat("forward_backward")
            started = time.monotonic()
            loss_total = torch.zeros((), device=device, dtype=torch.float32)
            for strength, clean, other in ((1.0, positive, negative), (-1.0, negative, positive)):
                network.multiplier = strength
                loss = _image_loss(trainer, clean, other, prompt)
                (0.5 * loss).backward()
                loss_total = loss_total + loss.detach() * 0.5
                xm.mark_step()
                xm.wait_device_ops()
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError("Conditionally unused adapter parameter would break collective ordering")
            finite = (torch.stack([torch.isfinite(parameter.grad).all() for parameter in parameters]).all()
                      & torch.isfinite(loss_total)).to(torch.int32)
            finite = xm.all_reduce(xm.REDUCE_SUM, finite)
            if int(finite.cpu()) != world:
                raise FloatingPointError("At least one replica has nonfinite adapter gradients")
            # The sole reduction owner. No DDP, xm.optimizer_step or second
            # scale/reduction is applied. Each local loss is already endpoint-averaged.
            gradients = [parameter.grad for parameter in parameters]
            xm.all_reduce(xm.REDUCE_SUM, gradients, scale=1 / world)
            if not bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().cpu()):
                raise FloatingPointError("Gradient reduction produced nonfinite values")
            norm = torch.nn.utils.clip_grad_norm_(parameters, tc.max_grad_norm, foreach=False)
            optimizer.step()
            scheduler.step()
            mean_loss = xm.all_reduce(xm.REDUCE_SUM, loss_total, scale=1 / world)
            xm.mark_step()
            xm.wait_device_ops()
            if not bool(torch.stack([torch.isfinite(parameter).all() for parameter in parameters]).all().cpu()):
                raise FloatingPointError("Optimizer produced nonfinite adapter parameters")
            completed, cursor = completed + 1, cursor + 1
            elapsed = time.monotonic() - started
            status.update(completed_updates=completed, samples=completed * 8, epoch=epoch, cursor=cursor)
            heartbeat("update_complete")
            fallback = {name: metrics.counter_value(name) for name in metrics.counter_names() if name.startswith("aten::")}
            if any(fallback.values()):
                raise RuntimeError(f"XLA CPU fallback counters are nonzero: {fallback}")
            if rank == 0:
                print(json.dumps({"update": completed, "samples": completed * 8, "loss": float(mean_loss.cpu()),
                                  "gradient_norm": float(norm.cpu()), "update_seconds": elapsed,
                                  "global_samples_per_second": 8 / elapsed, "epoch_dropped_exposures": dropped}), flush=True)
            control = {"rank": rank, "stop": rank == 0 and (output / "stop.json").exists()}
            stop = any(item["stop"] for item in exchange(f"control-{completed}", control))
            if completed % config["save_every"] == 0 or stop or completed == 40000:
                save_boundary(full_resume=full_resume_boundary(completed, stop=stop))
            if stop:
                break
        if completed != last_saved:
            save_boundary(full_resume=True)
        status.update(status="completed" if completed == 40000 else "session_checkpointed", memory_info=xm.get_memory_info(device))
        heartbeat("finished")
    except BaseException:
        status.update(status="error", error=traceback.format_exc())
        heartbeat("error")
        raise


def launch(config):
    import torch_xla
    torch_xla.launch(_rank, args=(str(config),))


def supervise(args):
    from workers.kaggle_tpu.probe import preflight
    from workers.kaggle_tpu.smoke_gate import SOURCE_FILES, VERSIONS
    smoke = gate_smoke(args.smoke_report, args.cache, args.asset_manifest) if args.smoke_report else None
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    # Retain this configuration key so existing resume identity validation
    # uses the same source contract even when no smoke report was requested.
    config["smoke_source_files_sha256"] = (smoke["source_files_sha256"] if smoke else
                                            {name: sha256(ROOT / name) for name in SOURCE_FILES})
    prepare(config)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Each attempt requires a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    terminal = terminal_resume(config)
    if terminal:
        write_json(output / "result.json", terminal)
        return
    runtime = preflight()
    if runtime.get("pjrt_device") != "TPU" or any(
        str(runtime["versions"].get(name, "")).split("+", 1)[0] != version
        for name, version in VERSIONS.items()
    ):
        raise ValueError("Training requires the qualified Kaggle Torch/XLA 2.8.0 + libtpu 0.0.17 TPU runtime")
    if smoke and runtime["versions"] != smoke["runtime"]["versions"]:
        raise ValueError("All-device runtime differs from the passing one-rank gate")
    config["runtime"] = runtime
    absolute_deadline = session_deadline(config, time.time())
    config["effective_session_deadline_unix"] = absolute_deadline
    write_json(output / "run-config.json", config)
    env = os.environ.copy()
    for name in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
        env.pop(name, None)
    env["PJRT_DEVICE"] = "TPU"
    deadline = time.monotonic() + absolute_deadline - time.time()
    result = None
    started = time.monotonic()
    peak_observed_group_rss = 0
    with (output / "training.log").open("w") as log:
        process = subprocess.Popen([sys.executable, __file__, "--launch-config", str(output / "run-config.json")],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while result is None:
                now = time.monotonic()
                observed = process_group_rss(process.pid)
                peak_observed_group_rss = max(peak_observed_group_rss, observed["rss_bytes"])
                write_json(output / "host-memory.json", {"sampled_unix": time.time(), **observed,
                           "peak_observed_group_rss_bytes": peak_observed_group_rss, "sample_interval_seconds": 5,
                           "shared_pages_may_be_counted_multiple_times": True})
                if now >= deadline - args.final_save_reserve_seconds and not (output / "stop.json").exists():
                    write_json(output / "stop.json", {"reason": "session_deadline", "requested_unix": time.time()})
                ranks = [json.loads(path.read_text()) for path in output.glob("rank-*.json")]
                if len(ranks) < 8 and now - started > args.rank_startup_timeout_seconds:
                    raise TimeoutError("Not every logical TPU rank reported startup before the startup deadline")
                if any(rank["status"] == "error" for rank in ranks):
                    raise RuntimeError("A rank failed; terminating the complete training process group")
                if any(rank["status"] == "running" and not Path(f"/proc/{rank['pid']}").exists() for rank in ranks):
                    raise RuntimeError("A rank process exited before completion; terminating remaining replicas")
                if any(time.time() - rank["updated_unix"] > args.rank_watchdog_seconds for rank in ranks):
                    raise TimeoutError("A rank stopped reporting within the compilation/step watchdog")
                if now >= deadline:
                    raise TimeoutError("Session deadline expired before a complete acknowledged final checkpoint")
                try:
                    result = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
            if result:
                raise RuntimeError(f"Replicated launcher exited with code {result}")
            ranks = [json.loads(path.read_text()) for path in output.glob("rank-*.json")]
            if len(ranks) != 8 or any(rank["status"] not in ("completed", "session_checkpointed") for rank in ranks):
                raise RuntimeError("Missing or incomplete rank completion reports")
            if not (output / "durable-latest.json").is_file():
                raise RuntimeError("Training ended without an acknowledged durable checkpoint")
            write_json(output / "result.json", {"schema": 1, "status": "completed" if all(rank["status"] == "completed" for rank in ranks) else "session_checkpointed", "ranks": ranks,
                                               "durable": json.loads((output / "durable-latest.json").read_text())})
        except BaseException:
            write_json(output / "result.json", {"schema": 1, "status": "error", "error": traceback.format_exc()})
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--launch-config":
        launch(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, help="Optional passing one-rank smoke report; direct training is the default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--relay-dataset", required=True)
    parser.add_argument("--relay-create", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--session-seconds", type=int, default=9 * 3600)
    allocation = parser.add_mutually_exclusive_group()
    allocation.add_argument("--session-deadline-unix", type=float)
    allocation.add_argument("--allocation-started-unix", type=float)
    parser.add_argument("--final-save-reserve-seconds", type=int, default=1800)
    parser.add_argument("--publish-timeout", type=int, default=900)
    parser.add_argument("--rank-watchdog-seconds", type=int, default=7200)
    parser.add_argument("--rank-startup-timeout-seconds", type=int, default=600)
    parser.add_argument("--base-name", default="Qwenvy2.1-v1.0.safetensors")
    parser.add_argument("--helper-name", default="fizgig_qwen_image_2.1_training_adapter.safetensors")
    args = parser.parse_args()
    if args.save_every < 1 or not 0 < args.publish_timeout < args.final_save_reserve_seconds < args.session_seconds or min(args.rank_watchdog_seconds, args.rank_startup_timeout_seconds) < 1:
        parser.error("Require positive checkpoint/watchdog limits and publish timeout < final-save reserve < session duration")
    if args.resume and args.relay_create:
        parser.error("A resumed session must append versions to the existing private relay")
    supervise(args)


if __name__ == "__main__":
    main()
