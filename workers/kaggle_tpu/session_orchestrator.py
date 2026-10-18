"""Local, restartable Kaggle TPU session controller for one private 40k job.

The queue and overall job have no deadline. One ambiguous notebook push is
never retried automatically because the remote mutation may have committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from workers.kaggle_tpu import checkpoint_relay as relay
from workers.kaggle_tpu.smoke_gate import SmokeGate, _notebook_code_digest

SCHEMA = 1
GOAL = 40000
TERMINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
PENDING = {"QUEUED", "RUNNING", "NEW_SCRIPT", "CANCEL_REQUESTED"}
REF = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+\Z")
TPU_IMAGE = re.compile(r"gcr\.io/kaggle-private-byod/python-tpuvm@sha256:[0-9a-f]{64}\Z")


def _safe_api(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except Exception as error:
        raise RuntimeError(f"Kaggle {operation.__name__} failed ({type(error).__name__}); credential-bearing details suppressed") from None


class KaggleKernel:
    """Small official-API adapter; methods return only non-secret fields."""

    def __init__(self, reference, api=None):
        if not REF.fullmatch(reference):
            raise ValueError("Kernel reference must be owner/slug")
        self.reference = reference
        if api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            _safe_api(api.authenticate)
        self.api = api

    def snapshot(self, *, allow_missing=False, allow_cpu_draft=False):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        owner, slug = self.reference.split("/")
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        try:
            with self.api.build_kaggle_client() as client:
                response = client.kernels.kernels_api_client.get_kernel(request)
        except Exception as error:
            if allow_missing and getattr(getattr(error, "response", None), "status_code", None) == 404:
                return None
            raise RuntimeError(f"Kaggle notebook lookup failed ({type(error).__name__}); credential-bearing details suppressed") from None
        metadata = response.metadata
        if not metadata.is_private or (not metadata.enable_tpu and not allow_cpu_draft):
            raise ValueError("Notebook must remain private with TPU enabled")
        if metadata.ref and metadata.ref.lower() != self.reference.lower():
            raise ValueError("Notebook reference differs from configured kernel")
        return {"version": metadata.current_version_number, "source": response.blob.source,
                "is_private": metadata.is_private, "enable_tpu": metadata.enable_tpu,
                "enable_gpu": metadata.enable_gpu, "title": metadata.title,
                "kernel_type": metadata.kernel_type, "machine_shape": metadata.machine_shape,
                "docker_image": metadata.docker_image}

    def status(self):
        response = _safe_api(self.api.kernels_status, self.reference)
        status = getattr(response.status, "name", None)
        if status not in TERMINAL | PENDING:
            raise ValueError(f"Unknown Kaggle notebook status: {status!r}")
        return status

    def push(self, folder):
        # Passing timeout=None leaves Kaggle's session limit unchanged. It is
        # unrelated to this controller's indefinitely long queue wait. The
        # explicit accelerator is required: TPU metadata alone previously
        # produced a CPU image on Kaggle CLI pushes.
        response = _safe_api(self.api.kernels_push, str(folder), timeout=None, acc="TpuV5E8")
        if response is None or response.error:
            raise RuntimeError("Kaggle rejected private notebook push; inspect remote notebook status")
        return response.version_number

    def download_output(self, destination):
        # The installed API resolves the latest version even when a version is
        # written in the reference. SmokeGate fences versions around this call.
        _safe_api(self.api.kernels_output, self.reference, str(destination),
                  file_pattern=r"(^|/)(diagnostic\.json|checkpoint\.pt)$",
                  force=True, quiet=True)


def _canonical_notebook(value):
    value = json.loads(value) if isinstance(value, str) else value
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    return value


def _source_digest(source):
    return hashlib.sha256(json.dumps(_canonical_notebook(source), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stage_submission(template, metadata_template, destination, *, kernel, session):
    """Build one immutable, credential-free notebook with a pinned resume."""
    notebook = json.loads(Path(template).read_text())
    if "dockerImageVersionId" in notebook.get("metadata", {}).get("kaggle", {}):
        raise ValueError("Training notebook metadata must not pin a Docker image")
    cells = [cell for cell in notebook["cells"] if "aitk-session-config" in cell.get("metadata", {}).get("tags", [])]
    if len(cells) != 1:
        raise ValueError("Training notebook needs exactly one aitk-session-config cell")
    config = {key: session[key] for key in ("session_id", "job_id", "attempt_id", "relay_dataset",
              "resume_version", "resume_step", "resume_manifest_sha256")}
    if session.get("smoke_report_sha256"):
        config["smoke_report_sha256"] = session["smoke_report_sha256"]
    cells[0]["source"] = "AITK_SESSION_CONFIG = " + repr(config) + "\nimport time\nallocation_started_unix = time.time()\n"
    cells[0]["outputs"] = []
    cells[0]["execution_count"] = None
    metadata = json.loads(Path(metadata_template).read_text())
    if metadata.get("is_private") is not True or metadata.get("enable_tpu") is not True:
        raise ValueError("Kernel metadata must describe a private TPU notebook")
    if metadata.get("code_file") != "train_session.ipynb" or metadata.get("kernel_type") != "notebook":
        raise ValueError("Kernel metadata must reference train_session.ipynb")
    if metadata.get("machine_shape") != "TpuV5E8" or metadata.get("enable_internet") is not True:
        raise ValueError("Kernel metadata must request TPU v5e-8 and Internet")
    if metadata.get("docker_image") or metadata.get("docker_image_pinning_type"):
        raise ValueError("Pinned Docker image can silently select Kaggle's CPU runtime")
    # The same slug starts as a CPU draft. An explicit empty value tells the
    # push API to clear its inherited CPU image pin for the TPU notebook.
    metadata["docker_image"] = ""
    metadata["id"] = kernel
    metadata["kernel_sources"] = []
    metadata["dataset_sources"] = list(dict.fromkeys([
        *metadata.get("dataset_sources", []),
        *([session["smoke_report_dataset"]] if session.get("smoke_report_dataset") else [])]))
    folder = Path(destination) / "submissions" / session["attempt_id"]
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "train_session.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder, _source_digest(notebook)


def verified_latest(destination, *, job_id, dataset_ref):
    pointer = Path(destination) / "latest.json"
    if not pointer.exists():
        return None
    record = json.loads(pointer.read_text())
    if record["job_id"] != job_id or record["dataset"] != dataset_ref:
        raise ValueError("Resume pointer belongs to another job or dataset")
    checkpoint = Path(record["checkpoint_path"])
    if checkpoint.resolve().parent != (Path(destination) / "checkpoints").resolve():
        raise ValueError("Resume checkpoint is outside the relay folder")
    manifest = relay.verify_checkpoint(checkpoint)
    if any(manifest[key] != record[key] for key in ("job_id", "attempt_id", "step")):
        raise ValueError("Resume pointer and verified checkpoint differ")
    if relay.sha256(checkpoint / "manifest.json") != record["manifest_sha256"]:
        raise ValueError("Resume pointer manifest hash differs")
    if record["step"] > GOAL:
        raise ValueError("Checkpoint exceeds authorized 40,000-update goal")
    return record


class Orchestrator:
    def __init__(self, *, destination, job_id, dataset_ref, kernel_ref, current_job,
                 notebook, metadata, dataset=None, kernel=None, poll_seconds=30,
                 checkpoint_wait_seconds=1800, sleep=time.sleep, now=time.time,
                 secret_ready_file, smoke_gate=None):
        if not REF.fullmatch(dataset_ref) or not REF.fullmatch(kernel_ref):
            raise ValueError("Dataset and kernel must be owner/slug")
        if not relay.IDENTIFIER.fullmatch(job_id):
            raise ValueError("Job ID must be a safe identifier")
        if poll_seconds <= 0 or checkpoint_wait_seconds <= 0:
            raise ValueError("Poll and checkpoint wait intervals must be positive")
        self.destination = relay.initialize_destination(destination, job_id=job_id,
                             dataset_ref=dataset_ref, current_job=current_job)
        self.secret_ready_file = Path(secret_ready_file).absolute()
        self.job_id, self.dataset_ref, self.kernel_ref = job_id, dataset_ref, kernel_ref
        self.notebook, self.metadata = Path(notebook), Path(metadata)
        self.dataset = dataset if dataset is not None else relay.KaggleDataset(dataset_ref)
        self.kernel = kernel if kernel is not None else KaggleKernel(kernel_ref)
        self.poll_seconds, self.checkpoint_wait_seconds = poll_seconds, checkpoint_wait_seconds
        self.sleep, self.now = sleep, now
        self.smoke_gate = smoke_gate
        self.state_path = self.destination / "orchestrator-state.json"
        owner = {"schema": SCHEMA, "job_id": job_id, "dataset": dataset_ref, "kernel": kernel_ref,
                 "secret_ready_file": str(self.secret_ready_file)}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if {key: self.state.get(key) for key in owner} != owner:
                raise ValueError("Orchestrator state belongs to another job, dataset or kernel")
        else:
            self.state = {**owner, "phase": "ready", "attempt": None,
                          "last_terminal_version": None, "secret_ready_confirmed": False}
            self._save()

    def _save(self):
        relay._json(self.state_path, self.state)

    def _snapshot(self, *, allow_missing=False, allow_cpu_draft=False):
        return self.kernel.snapshot(allow_missing=allow_missing, allow_cpu_draft=allow_cpu_draft)

    def _draft_status(self, snapshot):
        if snapshot is None:
            return "waiting_for_draft"
        if (not isinstance(snapshot["version"], int) or snapshot["version"] < 1 or
                snapshot.get("is_private") is not True or
                snapshot.get("enable_tpu") is not False or snapshot.get("enable_gpu") is not False or
                snapshot.get("title") != "Qwen21 Fizgig Train" or
                snapshot.get("kernel_type") != "notebook"):
            raise ValueError("Initial training kernel must be the private CPU-only draft")
        template = json.loads((Path(__file__).with_name("train_draft") / "train_draft.ipynb").read_text())
        if _notebook_code_digest(snapshot["source"]) != _notebook_code_digest(template):
            raise ValueError("Initial training kernel code differs from the private CPU draft")
        status = self.kernel.status()
        if status in PENDING:
            return "waiting_for_draft"
        if status != "COMPLETE":
            raise RuntimeError(f"Private CPU draft finished with {status}; training is blocked")
        return "ready"

    def _match_submission(self, snapshot, attempt):
        if snapshot is None or snapshot["version"] != attempt["version"]:
            raise RuntimeError("Notebook version changed outside this controller; refusing duplicate launch")
        if _source_digest(snapshot["source"]) != attempt["source_sha256"]:
            raise RuntimeError("Remote notebook source differs from the prepared session")
        image = snapshot.get("docker_image")
        # Kaggle resolves the empty submitted image to its pinned TPU image.
        # Accept that known TPU registry/repository while rejecting CPU pins.
        if (snapshot.get("machine_shape") != "TpuV5E8" or
                (image and not TPU_IMAGE.fullmatch(image))):
            raise RuntimeError("Remote notebook retained a CPU image or lacks TPU v5e-8 shape; refusing session adoption")

    def _secret_ready(self):
        if self.state["secret_ready_confirmed"]:
            return True
        if not self.secret_ready_file.exists():
            return False
        if self.secret_ready_file.is_symlink() or not self.secret_ready_file.is_file():
            raise ValueError("Secret readiness marker must be a regular JSON file")
        marker = json.loads(self.secret_ready_file.read_text())
        expected = {"schema": SCHEMA, "job_id": self.job_id, "kernel": self.kernel_ref,
                    "secret_name": "KAGGLE_API_TOKEN", "confirmed": True}
        if marker != expected:
            raise ValueError("Secret readiness marker must confirm this job, kernel and Kaggle Secret name")
        self.state["secret_ready_confirmed"] = True
        self._save()
        return True

    def advance(self):
        """Perform at most one durable state transition; caller owns watcher lock."""
        if self.smoke_gate is not None:
            smoke_outcome = self.smoke_gate.advance()
            if smoke_outcome != "ready":
                return smoke_outcome
        relay.poll_once(self.dataset, self.destination, job_id=self.job_id)
        latest = verified_latest(self.destination, job_id=self.job_id, dataset_ref=self.dataset_ref)
        phase, attempt = self.state["phase"], self.state["attempt"]
        if phase == "completed":
            return "completed"
        if phase == "ready":
            if latest and latest["step"] >= GOAL:
                self.state.update(phase="completed", completed_checkpoint=latest["checkpoint_path"])
                self._save()
                return "completed"
            snapshot = self._snapshot(allow_missing=True, allow_cpu_draft=True)
            if self.state["last_terminal_version"] is None:
                draft_status = self._draft_status(snapshot)
                if draft_status != "ready":
                    return draft_status
            if not self._secret_ready():
                return "waiting_for_secret"
            baseline = snapshot["version"] if snapshot else 0
            if self.state["last_terminal_version"] is not None and baseline != self.state["last_terminal_version"]:
                raise RuntimeError("Notebook changed outside this controller")
            if self.state["last_terminal_version"] is not None and not snapshot.get("enable_tpu"):
                raise RuntimeError("Training notebook no longer has TPU enabled")
            if snapshot and self.kernel.status() not in TERMINAL:
                raise RuntimeError("Existing Kaggle notebook session is not terminal; refusing another launch")
            if latest is None and self.dataset.info(allow_missing=True):
                raise ValueError("Relay dataset already exists without a verified local checkpoint")
            session_id = uuid.uuid4().hex
            attempt = {"session_id": session_id, "attempt_id": "session-" + session_id,
                       "job_id": self.job_id, "relay_dataset": self.dataset_ref,
                       "resume_version": latest["dataset_version"] if latest else None,
                       "resume_step": latest["step"] if latest else 0,
                       "resume_manifest_sha256": latest["manifest_sha256"] if latest else None,
                       "baseline_version": baseline, "version": baseline + 1}
            if self.smoke_gate is not None:
                attempt.update(smoke_report_dataset=self.smoke_gate.report_dataset_ref,
                               smoke_report_sha256=self.smoke_gate.state["diagnostic_sha256"])
            folder, digest = stage_submission(self.notebook, self.metadata, self.destination,
                                              kernel=self.kernel_ref, session=attempt)
            attempt.update(folder=str(folder), source_sha256=digest)
            self.state.update(phase="pushing", attempt=attempt)
            self._save()  # Write intent before a remote mutation.
            version = self.kernel.push(folder)
            if version != attempt["version"]:
                raise RuntimeError("Kaggle returned an unexpected notebook version; reconcile before retrying")
            self._match_submission(self._snapshot(), attempt)
            self.state["phase"] = "submitted"
            self._save()
            return "submitted"
        if phase == "pushing":
            snapshot = self._snapshot(allow_missing=True)
            if snapshot is None or snapshot["version"] == attempt["baseline_version"]:
                raise RuntimeError("Notebook push outcome is ambiguous; never retry automatically. Inspect Kaggle and the staged submission")
            self._match_submission(snapshot, attempt)
            self.state["phase"] = "submitted"
            self._save()
            return "submitted"
        if phase == "submitted":
            snapshot = self._snapshot()
            self._match_submission(snapshot, attempt)
            status = self.kernel.status()
            if status in PENDING:
                return status.lower()
            self.state.update(phase="terminal", terminal_status=status, terminal_unix=self.now())
            self._save()
            return "terminal"
        if phase == "terminal":
            snapshot = self._snapshot()
            self._match_submission(snapshot, attempt)
            if self.kernel.status() not in TERMINAL:
                raise RuntimeError("Kaggle notebook ceased to be terminal; refusing relaunch")
            if latest and latest["step"] > attempt["resume_step"]:
                if latest["step"] == GOAL:
                    self.state.update(phase="completed", completed_checkpoint=latest["checkpoint_path"])
                    self._save()
                    return "completed"
                self.state.update(phase="ready", attempt=None, last_terminal_version=attempt["version"])
                self._save()
                return "checkpointed"
            if self.now() - self.state["terminal_unix"] >= self.checkpoint_wait_seconds:
                raise RuntimeError("Terminal notebook produced no newer verified checkpoint; refusing replay from the same step")
            return "waiting_for_checkpoint"
        raise ValueError(f"Unsupported orchestrator phase: {phase}")

    def run(self):
        with relay.watcher_lock(self.destination):
            while True:
                outcome = self.advance()
                print(json.dumps({"outcome": outcome, "phase": self.state["phase"],
                                  "step": (verified_latest(self.destination, job_id=self.job_id,
                                           dataset_ref=self.dataset_ref) or {}).get("step", 0)}), flush=True)
                if outcome == "completed":
                    return
                self.sleep(self.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--current-job", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=Path(__file__).with_name("train_session.ipynb"))
    parser.add_argument("--metadata", type=Path, default=Path(__file__).with_name("train_session.kernel-metadata.json"))
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--checkpoint-wait-seconds", type=float, default=1800)
    parser.add_argument("--smoke-kernel", help="Optional already-submitted private smoke notebook owner/slug")
    parser.add_argument("--smoke-report-dataset", help="Optional dedicated private smoke diagnostic dataset")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--secret-ready-file", required=True, type=Path,
                        help="JSON marker created only after the private notebook KAGGLE_API_TOKEN Secret is enabled")
    args = parser.parse_args()
    destination = relay.initialize_destination(args.destination, job_id=args.job_id,
                                               dataset_ref=args.dataset, current_job=args.current_job)
    smoke_gate = None
    if args.smoke_kernel:
        if not all((args.smoke_report_dataset, args.cache_manifest, args.asset_manifest)):
            parser.error("Optional smoke gate requires report dataset, cache manifest and asset manifest")
        smoke_gate = SmokeGate(destination=destination, smoke_kernel_ref=args.smoke_kernel,
                               report_dataset_ref=args.smoke_report_dataset,
                               cache_manifest=args.cache_manifest, asset_manifest=args.asset_manifest,
                               kernel=KaggleKernel(args.smoke_kernel))
    Orchestrator(destination=destination, job_id=args.job_id, dataset_ref=args.dataset,
                 kernel_ref=args.kernel, current_job=args.current_job, notebook=args.notebook,
                 metadata=args.metadata, poll_seconds=args.poll_seconds,
                 checkpoint_wait_seconds=args.checkpoint_wait_seconds,
                 smoke_gate=smoke_gate, secret_ready_file=args.secret_ready_file).run()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Kaggle session orchestrator stopped: {type(error).__name__}: {error}") from None
