"""CPU diagnostics for canonical exposure partitioning and exact resume cursor."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workers.kaggle_tpu.train_replicated import epoch_schedule, prepare, gate_smoke, session_deadline, terminal_resume, full_resume_boundary
from workers.kaggle_tpu.cache_bundle import extract_cache_bundle


def run(cache=None):
    # Configured repeats remain explicit; a batch never assigns the same
    # canonical cached presentation to two ranks.
    exposures = [f"pair-{index}" for index in range(25)] + [f"pair-{index}" for index in range(10)]
    layouts = {pair: [64, 64, 9] for pair in exposures}
    batches, dropped = epoch_schedule(exposures, layouts, 17, 0)
    assert len(batches) == 4 and dropped == 3
    assert all(len(batch) == len(set(batch)) == 8 for batch in batches)
    consumed = Counter(pair for batch in batches for pair in batch)
    assert not (consumed - Counter(exposures))
    assert batches == epoch_schedule(exposures, layouts, 17, 0)[0]
    assert batches != epoch_schedule(exposures, layouts, 17, 1)[0]
    # Saving after optimizer update cursor=2 resumes at batch2, without
    # replaying the last completed batch or skipping the next exposure.
    resumed = epoch_schedule(exposures, layouts, 17, 0)[0][2:]
    assert batches[:2] + resumed == batches
    assert not full_resume_boundary(100)
    assert full_resume_boundary(1000)
    assert full_resume_boundary(155, stop=True)
    assert full_resume_boundary(40000)
    try:
        epoch_schedule(["one"] * 16, {"one": [64, 64, 9]}, 17, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("A bucket without eight distinct presentations must be rejected")
    if cache:
        config = {"cache": str(cache), "seed": 20261016}
        manifest = prepare(config)
        for epoch in range(10):
            batches, dropped = epoch_schedule(manifest["exposures"], config["layouts"], config["seed"], epoch)
            assert len(batches) == 44 and dropped == 3
            assert all(len(batch) == len(set(batch)) == 8 for batch in batches)
            assert sum(map(len, batches)) + dropped == len(manifest["exposures"]) == 355
        assert config["schedule_contract"]["global_batch"] == 8
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        deadline_config = {"allocation_started_unix": 1000, "session_seconds": 32400, "final_save_reserve_seconds": 1800}
        assert session_deadline(deadline_config, 1900) == 33400
        assert session_deadline({**deadline_config, "session_deadline_unix": 33000}, 1900) == 33000
        for invalid in ({"session_seconds": 32400, "final_save_reserve_seconds": 1800},
                        {**deadline_config, "session_deadline_unix": 2000}):
            try:
                session_deadline(invalid, 1900)
            except ValueError:
                pass
            else:
                raise AssertionError("Absent or insufficient allocation budget must be rejected")
        import torch
        from workers.kaggle_tpu.checkpoint_relay import build_checkpoint
        assets = {"files": {"base": {"sha256": "base-hash"}, "helper": {"sha256": "helper-hash"}}}
        (root / "assets.json").write_text(json.dumps(assets))
        identity = {"total_updates": 40000, "cache_manifest_sha256": "cache-hash", "schedule": {"global_batch": 8},
                    "schedule_fingerprint": "schedule-hash", "source_files_sha256": {"module.py": "source-hash"},
                    "base_sha256": "base-hash", "helper_sha256": "helper-hash"}
        state = {"schema": 1, "identity": identity, "completed_updates": 40000, "samples": 320000,
                 "adapter": {"test": torch.ones(1)}, "optimizer": {}, "scheduler": {}, "epoch": 909, "cursor": 4,
                 "rank_states": [{"rank": rank, "xla_rng": rank} for rank in range(8)]}
        torch.save(state, root / "state.pt")
        (root / "export.safetensors").write_bytes(b"opaque export bytes for relay contract test")
        checkpoint = build_checkpoint(root / "checkpoints", job_id="unit-test", attempt_id="terminal", step=40000,
                                      files={"resume/training.pt": root / "state.pt", "export/adapter.safetensors": root / "export.safetensors"},
                                      metadata={"identity": identity})
        terminal_config = {"resume": str(checkpoint), "job_id": "unit-test", "asset_manifest": str(root / "assets.json"),
                           "base_name": "base", "helper_name": "helper", "manifest_sha256": "cache-hash",
                           "schedule_contract": identity["schedule"], "schedule_fingerprint": "schedule-hash",
                           "smoke_source_files_sha256": identity["source_files_sha256"]}
        result = terminal_resume(terminal_config)
        assert result["status"] == "completed" and result["completed_updates"] == 40000
        assert result["launched_replicas"] is False and result["new_publication"] is False
        try:
            terminal_resume({**terminal_config, "manifest_sha256": "changed-cache"})
        except ValueError:
            pass
        else:
            raise AssertionError("Terminal checkpoint profile drift must be rejected")
        (root / "false-smoke.json").write_text(json.dumps({"status": "error"}))
        try:
            gate_smoke(root / "false-smoke.json", root, root)
        except ValueError:
            pass
        else:
            raise AssertionError("Training must reject an unsuccessful hardware gate")
        unsafe = root / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("../manifest.json", "{}")
        try:
            extract_cache_bundle(unsafe, root / "extracted")
        except ValueError:
            pass
        else:
            raise AssertionError("Cache ZIP traversal must be rejected")
    print("PASS: repeat-aware schedule/resume cursor, mixed save cadence and stop boundary, real allocation deadline, terminal checkpoint acceptance/profile validation, failed hardware gate and ZIP traversal rejection")


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
