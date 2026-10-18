"""Adopt one already-submitted private Kaggle smoke, then relay its proof privately."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from workers.kaggle_tpu import checkpoint_relay as relay

SCHEMA = 1
SOURCE_FILES = (
    "workers/kaggle_tpu/full_model_smoke.py",
    "workers/kaggle_tpu/static_qwen.py",
    "toolkit/util/convrot_quant.py",
    "toolkit/util/convrot_xla.py",
    "toolkit/fizgig_image_objective.py",
    "extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py",
)
ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {"torch": "2.8.0", "torch-xla": "2.8.0", "libtpu": "0.0.17"}


def _notebook_code_digest(value):
    notebook = json.loads(value) if isinstance(value, str) else value
    sources = ["".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
               for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return hashlib.sha256(json.dumps(sources, separators=(",", ":")).encode()).hexdigest()


def verify_smoke_output(directory, *, cache_manifest, asset_manifest):
    """Verify bytes and one-rank training claims without unpickling output."""
    directory = Path(directory)
    reports = list(directory.rglob("diagnostic.json"))
    if len(reports) != 1:
        raise ValueError("Smoke output must contain exactly one diagnostic.json")
    path = reports[0]
    if path.is_symlink() or not path.is_file():
        raise ValueError("Smoke diagnostic must be a regular file")
    report = json.loads(path.read_text())
    if report.get("schema") != 1 or report.get("status") != "pass" or report.get("phase") != "complete":
        raise ValueError("One-rank full-model smoke did not pass")
    if report.get("world_size") != 1 or not str(report.get("logical_device", "")).startswith("xla"):
        raise ValueError("Smoke did not execute one XLA rank")
    runtime = report.get("runtime", {})
    if runtime.get("pjrt_device") != "TPU" or any(
        str(runtime.get("versions", {}).get(name, "")).split("+", 1)[0] != version
        for name, version in VERSIONS.items()
    ):
        raise ValueError("Smoke did not use the qualified Kaggle TPU runtime")
    if (not report.get("optimizer_changed_probe") or
            not isinstance(report.get("losses"), list) or len(report["losses"]) != 2 or
            not all(type(loss) in (int, float) and math.isfinite(loss) for loss in report["losses"])):
        raise ValueError("Smoke did not prove a finite signed optimizer update")
    if any(report.get("aten_fallback_counters", {}).values()):
        raise ValueError("Smoke reported unqualified XLA CPU fallbacks")
    checkpoint = report.get("checkpoint", {})
    if not all(checkpoint.get(key) is True for key in ("adapter_restored", "optimizer_loaded", "rng_loaded")):
        raise ValueError("Smoke did not verify its checkpoint roundtrip")
    if checkpoint.get("path") != "checkpoint.pt":
        raise ValueError("Smoke checkpoint path is unexpected")
    checkpoint_path = path.parent / "checkpoint.pt"
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink() or relay.sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError("Downloaded smoke checkpoint does not match diagnostic hash")
    cache_manifest = Path(cache_manifest)
    asset_manifest = Path(asset_manifest)
    cache = json.loads(cache_manifest.read_text())
    assets = json.loads(asset_manifest.read_text())
    identity = report.get("identity", {})
    if (identity.get("cache_manifest_sha256") != relay.sha256(cache_manifest) or
            identity.get("base_sha256") != assets["files"]["Qwenvy2.1-v1.0.safetensors"]["sha256"] or
            identity.get("helper_sha256") != assets["files"]["fizgig_qwen_image_2.1_training_adapter.safetensors"]["sha256"] or
            identity.get("world_size") != 1 or identity.get("per_device_batch") != 1 or
            identity.get("optimizer") != "AdamW" or identity.get("optimizer_epsilon") != 1e-6 or
            identity.get("adapter_dtype") != "float32" or
            identity.get("quantization_contract") != "convrot8_w8a8_xla_v1" or
            identity.get("pair_id") not in {pair["id"] for pair in cache["pairs"]}):
        raise ValueError("Smoke base/helper/cache or training identity differs from local inputs")
    sources = report.get("source_files_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES) or any(
        sources[name] != relay.sha256(ROOT / name) for name in SOURCE_FILES
    ):
        raise ValueError("Smoke source hashes differ from this training worktree")
    return path, relay.sha256(path)


class SmokeReportDataset:
    """Private one-version dataset containing only the verified diagnostic."""

    def __init__(self, reference, dataset=None):
        self.delegate = dataset if dataset is not None else relay.KaggleDataset(reference)
        self.reference = reference

    def info(self, *, allow_missing=False):
        return self.delegate.info(allow_missing=allow_missing)

    def ready(self):
        return self.delegate.ready()

    def upload(self, directory, *, create):
        if not create:
            raise ValueError("Smoke report dataset is immutable after its first version")
        return self.delegate.upload(directory, create=True)

    def download(self, version, destination):
        if self.info() != version:
            raise ValueError("Smoke report dataset has an unexpected version")
        destination = Path(destination)
        self.delegate._call(self.delegate.api.dataset_download_files,
                            f"{self.reference}/{version}", path=str(destination),
                            force=True, quiet=True, unzip=False)
        archive = destination / (self.reference.split("/")[1] + ".zip")
        relay._extract(archive, destination, {"diagnostic.json"})
        archive.unlink()


class SmokeGate:
    """Crash-safe state machine; no smoke submission and no queue deadline."""

    def __init__(self, *, destination, smoke_kernel_ref, report_dataset_ref,
                 cache_manifest, asset_manifest, kernel, dataset=None, now=time.time,
                 publication_wait_seconds=900):
        if publication_wait_seconds <= 0:
            raise ValueError("Smoke report publication wait must be positive")
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=True)
        self.smoke_kernel_ref, self.report_dataset_ref = smoke_kernel_ref, report_dataset_ref
        self.cache_manifest, self.asset_manifest = Path(cache_manifest), Path(asset_manifest)
        self.kernel = kernel
        self.dataset = dataset if dataset is not None else SmokeReportDataset(report_dataset_ref)
        self.now, self.publication_wait_seconds = now, publication_wait_seconds
        self.path = self.destination / "smoke-gate-state.json"
        owner = {"schema": SCHEMA, "smoke_kernel": smoke_kernel_ref, "report_dataset": report_dataset_ref,
                 "cache_manifest_sha256": relay.sha256(self.cache_manifest),
                 "asset_manifest_sha256": relay.sha256(self.asset_manifest)}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if {key: self.state.get(key) for key in owner} != owner:
                raise ValueError("Smoke gate inputs or ownership changed")
        else:
            self.state = {**owner, "phase": "observe", "version": None}
            self._save()

    def _save(self):
        relay._json(self.path, self.state)

    def _fence(self):
        snapshot = self.kernel.snapshot()
        if (snapshot["version"] != self.state["version"] or
                _notebook_code_digest(snapshot["source"]) != self.state["source_sha256"]):
            raise RuntimeError("Smoke notebook version changed outside this controller")

    def advance(self):
        phase = self.state["phase"]
        if phase == "ready":
            if self.dataset.info() != 1:
                raise ValueError("Pinned private smoke report dataset changed")
            diagnostic = Path(self.state["verified_path"]) / "diagnostic.json"
            if relay.sha256(diagnostic) != self.state["diagnostic_sha256"]:
                raise ValueError("Local verified smoke diagnostic changed")
            sources = json.loads(diagnostic.read_text())["source_files_sha256"]
            if any(sources[name] != relay.sha256(ROOT / name) for name in SOURCE_FILES):
                raise ValueError("Training source changed after the verified smoke")
            return "ready"
        if phase == "observe":
            snapshot = self.kernel.snapshot()
            if not snapshot or type(snapshot["version"]) is not int or snapshot["version"] < 1:
                raise ValueError("Submit a private one-rank smoke notebook before starting the controller")
            source_digest = _notebook_code_digest(snapshot["source"])
            template_digest = _notebook_code_digest(json.loads((Path(__file__).with_name("full_model_smoke.ipynb")).read_text()))
            if source_digest != template_digest:
                raise ValueError("Submitted smoke notebook code differs from the qualified local template")
            if self.dataset.info(allow_missing=True):
                raise ValueError("Smoke report dataset must be new and dedicated to this gate")
            self.state.update(phase="waiting", version=snapshot["version"], source_sha256=source_digest)
            self._save()
            return "smoke_observed"
        if phase == "waiting":
            self._fence()
            status = self.kernel.status()
            if status in {"QUEUED", "RUNNING", "NEW_SCRIPT", "CANCEL_REQUESTED"}:
                return "smoke_" + status.lower()
            if status != "COMPLETE":
                raise RuntimeError(f"Smoke notebook finished with {status}; training is blocked")
            with tempfile.TemporaryDirectory(prefix=".smoke-download-", dir=self.destination) as temporary:
                self.kernel.download_output(Path(temporary))
                self._fence()  # Kaggle output API selects latest; reject a version race.
                diagnostic, digest = verify_smoke_output(temporary, cache_manifest=self.cache_manifest,
                                                         asset_manifest=self.asset_manifest)
                checkpoint = diagnostic.parent / "checkpoint.pt"
                verified = self.destination / "smoke" / f"version-{self.state['version']}"
                verified.parent.mkdir(exist_ok=True)
                if verified.exists():
                    if (relay.sha256(verified / "diagnostic.json") != digest or
                            relay.sha256(verified / "checkpoint.pt") != relay.sha256(checkpoint)):
                        raise ValueError("Existing smoke proof folder differs from this download")
                else:
                    with tempfile.TemporaryDirectory(prefix=".smoke-proof-", dir=verified.parent) as staging:
                        proof = Path(staging) / "proof"
                        proof.mkdir()
                        shutil.copyfile(diagnostic, proof / "diagnostic.json")
                        shutil.copyfile(checkpoint, proof / "checkpoint.pt")
                        os.rename(proof, verified)
            self.state.update(phase="verified", diagnostic_sha256=digest,
                              verified_path=str(verified.resolve()))
            self._save()
            return "smoke_verified"
        if phase == "verified":
            if self.dataset.info(allow_missing=True):
                raise ValueError("Smoke report dataset exists before this gate published it")
            stage = self.destination / "smoke" / "report-dataset-stage"
            stage.mkdir(exist_ok=True)
            if {path.name for path in stage.iterdir()} - {"diagnostic.json", "dataset-metadata.json"}:
                raise ValueError("Unexpected file in smoke report dataset stage")
            diagnostic = Path(self.state["verified_path"]) / "diagnostic.json"
            if relay.sha256(diagnostic) != self.state["diagnostic_sha256"]:
                raise ValueError("Verified smoke diagnostic changed before publication")
            checkpoint = Path(self.state["verified_path"]) / "checkpoint.pt"
            if relay.sha256(checkpoint) != json.loads(diagnostic.read_text())["checkpoint"]["sha256"]:
                raise ValueError("Verified smoke checkpoint changed before publication")
            shutil.copyfile(diagnostic, stage / "diagnostic.json")
            relay._json(stage / "dataset-metadata.json", {"id": self.report_dataset_ref,
                        "title": "AI Toolkit private TPU smoke proof", "licenses": [{"name": "other"}]})
            self.state.update(phase="publishing", publication_started_unix=self.now())
            self._save()  # Write intent before the ambiguous remote mutation.
            self.dataset.upload(stage, create=True)
            return "smoke_publishing"
        if phase == "publishing":
            version = self.dataset.info(allow_missing=True)
            if version == 1 and self.dataset.ready():
                with tempfile.TemporaryDirectory(prefix=".smoke-ack-", dir=self.destination) as temporary:
                    self.dataset.download(1, Path(temporary))
                    if relay.sha256(Path(temporary) / "diagnostic.json") != self.state["diagnostic_sha256"]:
                        raise ValueError("Private smoke report dataset differs from verified diagnostic")
                self.state.update(phase="ready", report_dataset_version=1)
                self._save()
                return "ready"
            if version not in (0, 1):
                raise RuntimeError("Concurrent publisher changed the dedicated smoke report dataset")
            if self.now() - self.state["publication_started_unix"] >= self.publication_wait_seconds:
                raise RuntimeError("Smoke report publication unacknowledged; inspect private dataset before recovery")
            return "smoke_publishing"
        raise ValueError(f"Unsupported smoke gate phase: {phase}")
