"""Offline state-transition tests for the local Kaggle session controller."""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from workers.kaggle_tpu import checkpoint_relay as relay
from workers.kaggle_tpu import session_orchestrator as orchestrator_module
from workers.kaggle_tpu import smoke_gate as smoke_gate_module


class FakeDataset:
    reference = "owner/relay"

    def info(self, *, allow_missing=False):
        return 0

    def ready(self):
        return True

    def download(self, _version, _destination):  # pragma: no cover - version stays at zero
        raise AssertionError("offline fake dataset has no remote versions")


class FakeKernel:
    def __init__(self, *, status="COMPLETE", push_then_raise=False,
                 push_without_commit=False, draft_status="COMPLETE",
                 snapshot_missing=False, draft_metadata=None, draft_source=None,
                 draft_version=1):
        draft_path = (Path(orchestrator_module.__file__).with_name("train_draft") /
                      "train_draft.ipynb")
        self.version = draft_version
        self.source = json.loads(draft_path.read_text()) if draft_source is None else draft_source
        self.is_private = True
        self.enable_tpu = False
        self.enable_gpu = False
        self.machine_shape = ""
        self.docker_image = ""
        self.title = "Qwen21 Fizgig Train"
        self.kernel_type = "notebook"
        if draft_metadata:
            for name, value in draft_metadata.items():
                setattr(self, name, value)
        self.current_status = draft_status
        self.post_push_status = status
        self.push_calls = 0
        self.push_then_raise = push_then_raise
        self.push_without_commit = push_without_commit
        self.snapshot_missing = snapshot_missing

    def snapshot(self, *, allow_missing=False, allow_cpu_draft=False):
        if self.snapshot_missing and allow_missing:
            return None
        return {"version": self.version, "source": self.source,
                "is_private": self.is_private, "enable_tpu": self.enable_tpu,
                "enable_gpu": self.enable_gpu, "title": self.title,
                "kernel_type": self.kernel_type, "machine_shape": self.machine_shape,
                "docker_image": self.docker_image}

    def status(self):
        return self.current_status

    def push(self, folder):
        self.push_calls += 1
        if self.push_without_commit:
            raise RuntimeError("simulated lost response before remote commit")
        self.version += 1
        self.source = json.loads((Path(folder) / "train_session.ipynb").read_text())
        self.enable_tpu = True
        self.machine_shape = "TpuV5E8"
        self.docker_image = ""
        self.current_status = self.post_push_status
        if self.push_then_raise:
            raise RuntimeError("simulated lost response after remote commit")
        return self.version


def test_kaggle_kernel_push_pins_tpu_accelerator_and_preserves_timeout(tmp_path):
    class FakeApi:
        def __init__(self):
            self.calls = []

        def kernels_push(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return SimpleNamespace(error=False, version_number=17)

    api = FakeApi()
    kernel = orchestrator_module.KaggleKernel("owner/training-kernel", api=api)
    folder = tmp_path / "submission"

    assert kernel.push(folder) == 17
    assert api.calls == [((str(folder),), {"timeout": None, "acc": "TpuV5E8"})]


def test_kaggle_kernel_snapshot_exposes_runtime_metadata(monkeypatch):
    class Request:
        pass

    response = SimpleNamespace(
        metadata=SimpleNamespace(
            is_private=True,
            enable_tpu=True,
            enable_gpu=False,
            ref="owner/training-kernel",
            current_version_number=2,
            title="Qwen21 Fizgig Train",
            kernel_type="notebook",
            machine_shape="TpuV5E8",
            docker_image="",
        ),
        blob=SimpleNamespace(source={"cells": []}),
    )

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        kernels = SimpleNamespace(kernels_api_client=SimpleNamespace(
            get_kernel=lambda _request: response))

    class Api:
        def build_kaggle_client(self):
            return Client()

    root = ModuleType("kagglesdk")
    kernels = ModuleType("kagglesdk.kernels")
    type_package = ModuleType("kagglesdk.kernels.types")
    service = ModuleType("kagglesdk.kernels.types.kernels_api_service")
    service.ApiGetKernelRequest = Request
    for name, module in (("kagglesdk", root), ("kagglesdk.kernels", kernels),
                         ("kagglesdk.kernels.types", type_package),
                         ("kagglesdk.kernels.types.kernels_api_service", service)):
        monkeypatch.setitem(sys.modules, name, module)

    snapshot = orchestrator_module.KaggleKernel(
        "owner/training-kernel", api=Api()).snapshot()
    assert snapshot["machine_shape"] == "TpuV5E8"
    assert snapshot["docker_image"] == ""


class FakeClock:
    def __init__(self, value=100):
        self.value = value

    def now(self):
        return self.value


class FakeReadyGate:
    report_dataset_ref = "owner/smoke-report"
    state = {"diagnostic_sha256": "a" * 64}

    def advance(self):
        return "ready"


def write_templates(root):
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["train\n"]},
            {"cell_type": "code", "metadata": {"tags": ["aitk-session-config"]},
             "source": ["old config\n"], "outputs": [{"text": "stale"}], "execution_count": 4},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": "owner/training-kernel",
        "is_private": True,
        "enable_tpu": True,
        "enable_internet": True,
        "machine_shape": "TpuV5E8",
        "code_file": "train_session.ipynb",
        "kernel_type": "notebook",
        "docker_image": "",
    }
    notebook_path = root / "template.ipynb"
    metadata_path = root / "kernel-metadata.json"
    notebook_path.write_text(json.dumps(notebook))
    metadata_path.write_text(json.dumps(metadata))
    return notebook_path, metadata_path


def make_controller(root, *, kernel=None, clock=None, checkpoint_wait_seconds=30,
                    smoke_gate=None, secret_ready=True, secret_ready_file=None):
    root.mkdir(parents=True, exist_ok=True)
    notebook, metadata = write_templates(root)
    current_job = root / "current-job"
    current_job.mkdir(exist_ok=True)
    secret_ready_file = secret_ready_file or root / "secret-ready.json"
    if secret_ready and not secret_ready_file.exists():
        secret_ready_file.write_text(json.dumps({
            "schema": 1,
            "job_id": "job-a",
            "kernel": "owner/training-kernel",
            "secret_name": "KAGGLE_API_TOKEN",
            "confirmed": True,
        }))
    clock = clock or FakeClock()
    controller = orchestrator_module.Orchestrator(
        destination=root / "relay-local",
        job_id="job-a",
        dataset_ref="owner/relay",
        kernel_ref="owner/training-kernel",
        current_job=current_job,
        notebook=notebook,
        metadata=metadata,
        dataset=FakeDataset(),
        kernel=kernel or FakeKernel(),
        poll_seconds=1,
        checkpoint_wait_seconds=checkpoint_wait_seconds,
        sleep=lambda _seconds: None,
        now=clock.now,
        smoke_gate=smoke_gate,
        secret_ready_file=secret_ready_file,
    )
    return controller


def install_local_checkpoint(controller, *, step, attempt="attempt-a"):
    root = controller.destination
    inputs = root / "checkpoint-inputs"
    inputs.mkdir(exist_ok=True)
    export = inputs / "export.bin"
    resume = inputs / "resume.bin"
    export.write_bytes(f"export-{step}".encode())
    resume.write_bytes(f"resume-{step}".encode())
    checkpoint = relay.build_checkpoint(
        root / "checkpoints",
        job_id="job-a",
        attempt_id=attempt,
        step=step,
        files={"export/model.bin": export, "resume/state.bin": resume},
        metadata={"test": True},
    )
    manifest_sha256 = relay.sha256(checkpoint / "manifest.json")
    relay._json(root / "latest.json", {
        "job_id": "job-a",
        "dataset": "owner/relay",
        "attempt_id": attempt,
        "step": step,
        "dataset_version": 1,
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": str(checkpoint.resolve()),
    })
    return checkpoint


def test_ambiguous_push_is_reconciled_after_restart_without_duplicate_push(tmp_path):
    kernel = FakeKernel(push_then_raise=True)
    controller = make_controller(tmp_path / "run", kernel=kernel)

    with pytest.raises(RuntimeError, match="simulated lost response"):
        controller.advance()
    assert controller.state["phase"] == "pushing"
    assert kernel.push_calls == 1

    # A new process loads the durable intent, observes the committed version,
    # and adopts it instead of issuing another remote mutation.
    restarted = make_controller(tmp_path / "run", kernel=kernel)
    assert restarted.state["phase"] == "pushing"
    assert restarted.advance() == "submitted"
    assert restarted.state["phase"] == "submitted"
    assert kernel.push_calls == 1


def test_ambiguous_push_without_remote_change_never_retries(tmp_path):
    kernel = FakeKernel(push_without_commit=True)
    controller = make_controller(tmp_path / "run", kernel=kernel)
    with pytest.raises(RuntimeError, match="before remote commit"):
        controller.advance()
    assert controller.state["phase"] == "pushing"
    assert kernel.push_calls == 1

    with pytest.raises(RuntimeError, match="push outcome is ambiguous"):
        controller.advance()
    assert kernel.push_calls == 1


def test_exact_private_cpu_draft_is_accepted_before_version_two_push(tmp_path):
    kernel = FakeKernel()
    controller = make_controller(tmp_path / "run", kernel=kernel)

    assert kernel.version == 1
    assert kernel.is_private is True
    assert kernel.enable_tpu is False
    assert kernel.enable_gpu is False
    assert kernel.title == "Qwen21 Fizgig Train"
    assert controller.advance() == "submitted"
    assert kernel.version == 2
    assert kernel.push_calls == 1


def test_cpu_draft_accepts_any_positive_baseline_version(tmp_path):
    kernel = FakeKernel(draft_version=7)
    controller = make_controller(tmp_path / "run", kernel=kernel)

    assert controller.advance() == "submitted"
    assert kernel.version == 8
    assert kernel.push_calls == 1


@pytest.mark.parametrize("metadata", [
    {"is_private": False},
    {"enable_tpu": True},
    {"enable_gpu": True},
])
def test_public_or_accelerated_draft_is_rejected_before_push(tmp_path, metadata):
    kernel = FakeKernel(draft_metadata=metadata)
    controller = make_controller(tmp_path / "run", kernel=kernel)

    with pytest.raises(ValueError, match="private CPU-only draft"):
        controller.advance()
    assert kernel.version == 1
    assert kernel.push_calls == 0


def test_changed_draft_code_is_rejected_before_push(tmp_path):
    draft_path = (Path(orchestrator_module.__file__).with_name("train_draft") /
                  "train_draft.ipynb")
    changed_source = json.loads(draft_path.read_text())
    code = next(cell for cell in changed_source["cells"] if cell["cell_type"] == "code")
    code["source"] = "print('modified draft')\n"
    kernel = FakeKernel(draft_source=changed_source)
    controller = make_controller(tmp_path / "run", kernel=kernel)

    with pytest.raises(ValueError, match="code differs from the private CPU draft"):
        controller.advance()
    assert kernel.version == 1
    assert kernel.push_calls == 0


def test_training_push_waits_for_draft_terminal_status_and_secret_marker(tmp_path):
    kernel = FakeKernel(draft_status="QUEUED")
    controller = make_controller(tmp_path / "run", kernel=kernel, secret_ready=False)

    assert controller.advance() == "waiting_for_draft"
    assert kernel.push_calls == 0
    kernel.current_status = "COMPLETE"
    assert controller.advance() == "waiting_for_secret"
    assert kernel.push_calls == 0

    controller.secret_ready_file.write_text(json.dumps({
        "schema": 1,
        "job_id": "job-a",
        "kernel": "owner/training-kernel",
        "secret_name": "KAGGLE_API_TOKEN",
        "confirmed": True,
    }))
    assert controller.advance() == "submitted"
    assert kernel.version == 2
    assert kernel.push_calls == 1


def test_missing_draft_waits_without_push(tmp_path):
    kernel = FakeKernel(snapshot_missing=True)
    controller = make_controller(tmp_path / "run", kernel=kernel)
    assert controller.advance() == "waiting_for_draft"
    assert kernel.push_calls == 0


def test_failed_draft_status_blocks_training_push(tmp_path):
    kernel = FakeKernel(draft_status="ERROR")
    controller = make_controller(tmp_path / "run", kernel=kernel)

    with pytest.raises(RuntimeError, match="draft finished with ERROR"):
        controller.advance()
    assert kernel.version == 1
    assert kernel.push_calls == 0


@pytest.mark.parametrize("metadata_change", [
    {"machine_shape": ""},
    {"docker_image": "cpu-runtime-image"},
])
def test_post_push_submission_rejects_runtime_metadata_drift(tmp_path, metadata_change):
    kernel = FakeKernel()
    controller = make_controller(tmp_path / "run", kernel=kernel)
    assert controller.advance() == "submitted"
    assert kernel.machine_shape == "TpuV5E8"
    assert kernel.docker_image == ""
    for key, value in metadata_change.items():
        setattr(kernel, key, value)

    with pytest.raises(RuntimeError, match="runtime|machine|Docker|docker|TPU"):
        controller.advance()
    assert kernel.push_calls == 1


def test_first_push_with_inherited_cpu_image_is_not_adopted(tmp_path):
    kernel = FakeKernel()
    original_push = kernel.push

    def cpu_pinned_push(folder):
        version = original_push(folder)
        kernel.docker_image = "sha256:cpu-image"
        return version

    kernel.push = cpu_pinned_push
    controller = make_controller(tmp_path / "run", kernel=kernel)
    with pytest.raises(RuntimeError, match="CPU image"):
        controller.advance()
    assert controller.state["phase"] == "pushing"
    assert kernel.push_calls == 1

    with pytest.raises(RuntimeError, match="CPU image"):
        controller.advance()
    assert kernel.push_calls == 1


def test_first_push_adopts_kaggle_tpuvm_image_and_keeps_it_valid(tmp_path):
    kernel = FakeKernel()
    original_push = kernel.push
    tpu_image = "gcr.io/kaggle-private-byod/python-tpuvm@sha256:" + "a" * 64

    def kaggle_tpu_image_push(folder):
        version = original_push(folder)
        kernel.docker_image = tpu_image
        return version

    kernel.push = kaggle_tpu_image_push
    controller = make_controller(tmp_path / "run", kernel=kernel)

    assert controller.advance() == "submitted"
    assert kernel.docker_image == tpu_image
    assert controller.advance() == "terminal"
    assert kernel.push_calls == 1


def test_pending_queue_has_no_controller_timeout(tmp_path):
    clock = FakeClock(value=10_000_000)
    kernel = FakeKernel(status="QUEUED")
    controller = make_controller(tmp_path / "run", kernel=kernel, clock=clock,
                                 checkpoint_wait_seconds=1)
    assert controller.advance() == "submitted"

    for _ in range(3):
        assert controller.advance() == "queued"
    assert controller.state["phase"] == "submitted"
    assert kernel.push_calls == 1


def test_terminal_attempt_waits_for_a_newer_verified_checkpoint(tmp_path):
    clock = FakeClock(value=100)
    kernel = FakeKernel(status="RUNNING")
    controller = make_controller(tmp_path / "run", kernel=kernel,
                                 clock=clock, checkpoint_wait_seconds=30)
    install_local_checkpoint(controller, step=20)
    assert controller.advance() == "submitted"

    kernel.current_status = "COMPLETE"
    assert controller.advance() == "terminal"
    assert controller.advance() == "waiting_for_checkpoint"
    assert controller.state["phase"] == "terminal"

    clock.value += 31
    with pytest.raises(RuntimeError, match="no newer verified checkpoint"):
        controller.advance()
    assert controller.state["phase"] == "terminal"


def test_newer_verified_checkpoint_releases_terminal_attempt(tmp_path):
    kernel = FakeKernel(status="COMPLETE")
    controller = make_controller(tmp_path / "run", kernel=kernel)
    install_local_checkpoint(controller, step=20)
    assert controller.advance() == "submitted"
    assert controller.advance() == "terminal"
    assert controller.advance() == "waiting_for_checkpoint"

    install_local_checkpoint(controller, step=21, attempt="attempt-b")
    assert controller.advance() == "checkpointed"
    assert controller.state["phase"] == "ready"
    assert controller.state["attempt"] is None


def test_goal_checkpoint_completes_only_after_notebook_is_terminal(tmp_path):
    kernel = FakeKernel(status="QUEUED")
    controller = make_controller(tmp_path / "run", kernel=kernel)
    assert controller.advance() == "submitted"
    install_local_checkpoint(controller, step=orchestrator_module.GOAL)

    assert controller.advance() == "queued"
    assert controller.state["phase"] == "submitted"

    kernel.current_status = "COMPLETE"
    assert controller.advance() == "terminal"
    assert controller.advance() == "completed"
    assert controller.state["phase"] == "completed"
    assert controller.advance() == "completed"


def test_stage_submission_pins_resume_config_and_scrubs_outputs(tmp_path):
    notebook, metadata = write_templates(tmp_path)
    session = {
        "session_id": "session-1",
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "relay_dataset": "owner/relay",
        "resume_version": 3,
        "resume_step": 120,
        "resume_manifest_sha256": "a" * 64,
        "smoke_report_dataset": "owner/smoke-report",
        "smoke_report_sha256": "b" * 64,
    }
    folder, digest = orchestrator_module.stage_submission(
        notebook, metadata, tmp_path / "stage", kernel="owner/training-kernel",
        session=session,
    )

    staged = json.loads((folder / "train_session.ipynb").read_text())
    config_cell = staged["cells"][1]
    source_lines = config_cell["source"].splitlines()
    config = ast.literal_eval(source_lines[0].removeprefix("AITK_SESSION_CONFIG = "))
    assert config == {key: value for key, value in session.items()
                      if key not in {"smoke_report_dataset"}}
    assert source_lines[1:] == ["import time", "allocation_started_unix = time.time()"]
    assert config_cell["outputs"] == []
    assert config_cell["execution_count"] is None
    assert digest == orchestrator_module._source_digest(staged)
    staged_metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert staged_metadata["enable_tpu"] is True
    assert staged_metadata["docker_image"] == ""
    assert "owner/smoke-report" in staged_metadata["dataset_sources"]


def test_direct_40k_launch_then_resume_uses_verified_checkpoint_without_smoke(tmp_path):
    root = tmp_path / "run"
    kernel = FakeKernel()
    controller = make_controller(root, kernel=kernel)

    assert controller.smoke_gate is None
    assert controller.advance() == "submitted"
    first = controller.state["attempt"]
    first_notebook = json.loads((Path(first["folder"]) / "train_session.ipynb").read_text())
    first_source = first_notebook["cells"][1]["source"]
    first_lines = first_source.splitlines()
    first_config = ast.literal_eval(first_lines[0].removeprefix("AITK_SESSION_CONFIG = "))
    assert first_config["resume_step"] == 0
    assert "smoke_report_sha256" not in first_config
    assert first_lines[1:] == ["import time", "allocation_started_unix = time.time()"]
    first_metadata = json.loads((Path(first["folder"]) / "kernel-metadata.json").read_text())
    assert not any("smoke" in source for source in first_metadata["dataset_sources"])
    assert not (controller.destination / "smoke-gate-state.json").exists()

    install_local_checkpoint(controller, step=1000, attempt=first["attempt_id"])
    checkpoint_record = orchestrator_module.verified_latest(
        controller.destination, job_id="job-a", dataset_ref="owner/relay")
    assert controller.advance() == "terminal"
    assert controller.advance() == "checkpointed"
    assert controller.advance() == "submitted"
    resumed = controller.state["attempt"]
    resumed_notebook = json.loads((Path(resumed["folder"]) / "train_session.ipynb").read_text())
    resumed_config = ast.literal_eval(
        resumed_notebook["cells"][1]["source"].splitlines()[0].removeprefix("AITK_SESSION_CONFIG = "))
    assert resumed_config["resume_step"] == 1000
    assert resumed_config["resume_version"] == 1
    assert resumed_config["resume_manifest_sha256"] == checkpoint_record["manifest_sha256"]
    assert "smoke_report_sha256" not in resumed_config
    assert kernel.push_calls == 2


def test_repository_training_notebook_stages_in_direct_mode(tmp_path):
    template = Path(orchestrator_module.__file__).with_name("train_session.ipynb")
    metadata = template.with_name("train_session.kernel-metadata.json")
    session = {
        "session_id": "session-direct", "job_id": "job-a", "attempt_id": "attempt-direct",
        "relay_dataset": "owner/relay", "resume_version": None, "resume_step": 0,
        "resume_manifest_sha256": None,
    }
    folder, _ = orchestrator_module.stage_submission(
        template, metadata, tmp_path / "stage", kernel="owner/training-kernel", session=session)
    notebook = json.loads((folder / "train_session.ipynb").read_text())
    config_source = notebook["cells"][1]["source"]
    config_lines = config_source.splitlines()
    staged_config = ast.literal_eval(
        config_lines[0].removeprefix("AITK_SESSION_CONFIG = "))
    assert staged_config == session
    assert config_lines[1:] == ["import time", "allocation_started_unix = time.time()"]
    training_source = "".join(notebook["cells"][4]["source"])
    assert "if config.get('smoke_report_sha256'):" in training_source
    assert "if smoke_reports:" in training_source
    assert "'--smoke-report'" in training_source
    metadata_source = json.loads((folder / "kernel-metadata.json").read_text())
    assert "owner/smoke-report" not in metadata_source["dataset_sources"]


def test_stage_rejects_notebook_docker_image_pin(tmp_path):
    notebook, metadata = write_templates(tmp_path)
    contents = json.loads(notebook.read_text())
    contents["metadata"]["kaggle"] = {"dockerImageVersionId": "cpu-image"}
    notebook.write_text(json.dumps(contents))
    session = {
        "session_id": "session-1", "job_id": "job-a", "attempt_id": "attempt-a",
        "relay_dataset": "owner/relay", "resume_version": None, "resume_step": 0,
        "resume_manifest_sha256": None, "smoke_report_dataset": "owner/smoke-report",
        "smoke_report_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="must not pin a Docker image"):
        orchestrator_module.stage_submission(
            notebook, metadata, tmp_path / "stage", kernel="owner/training-kernel",
            session=session,
        )


def test_missing_secret_marker_waits_indefinitely_after_smoke_ready(tmp_path):
    clock = FakeClock(value=5)
    controller = make_controller(tmp_path / "run", clock=clock, secret_ready=False)

    for _ in range(4):
        clock.value += 1_000_000
        assert controller.advance() == "waiting_for_secret"
    assert controller.state["secret_ready_confirmed"] is False
    assert controller.state["phase"] == "ready"
    assert controller.kernel.push_calls == 0


def test_invalid_secret_marker_fails_before_training_push(tmp_path):
    controller = make_controller(tmp_path / "run", secret_ready=False)
    controller.secret_ready_file.write_text(json.dumps({
        "schema": 1,
        "job_id": "job-a",
        "kernel": "owner/another-kernel",
        "secret_name": "KAGGLE_API_TOKEN",
        "confirmed": True,
    }))

    with pytest.raises(ValueError, match="confirm this job, kernel"):
        controller.advance()
    assert controller.state["secret_ready_confirmed"] is False
    assert controller.kernel.push_calls == 0


def test_secret_confirmation_persists_across_restart_after_push(tmp_path):
    root = tmp_path / "run"
    kernel = FakeKernel()
    controller = make_controller(root, kernel=kernel)
    assert controller.advance() == "submitted"
    assert controller.state["secret_ready_confirmed"] is True
    controller.secret_ready_file.unlink()

    restarted = make_controller(root, kernel=kernel, secret_ready=False)
    assert restarted.state["secret_ready_confirmed"] is True
    assert restarted.advance() == "terminal"
    assert kernel.push_calls == 1


def test_cli_initializes_direct_training_without_smoke_gate(tmp_path, monkeypatch):
    destination = tmp_path / "new-relay-destination"
    current_job = tmp_path / "current-job"
    current_job.mkdir()
    secret_ready_file = tmp_path / "secret-ready.json"
    argv = [
        "session_orchestrator",
        "--dataset", "owner/relay",
        "--kernel", "owner/training-kernel",
        "--job-id", "job-a",
        "--current-job", str(current_job),
        "--destination", str(destination),
        "--secret-ready-file", str(secret_ready_file),
    ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(orchestrator_module, "KaggleKernel",
                        lambda _reference: FakeKernel())
    monkeypatch.setattr(orchestrator_module.relay, "KaggleDataset",
                        lambda _reference: FakeDataset())
    monkeypatch.setattr(orchestrator_module.Orchestrator, "run", lambda _self: None)

    assert not destination.exists()
    orchestrator_module.main()

    assert (destination / "relay-owner.json").is_file()
    assert not (destination / "smoke-gate-state.json").exists()
    assert (destination / "orchestrator-state.json").is_file()


def test_verified_latest_rejects_pointer_to_modified_checkpoint(tmp_path):
    controller = make_controller(tmp_path / "run")
    checkpoint = install_local_checkpoint(controller, step=7)
    (checkpoint / "resume/state.bin").write_bytes(b"modified")

    with pytest.raises(ValueError, match="integrity failure"):
        orchestrator_module.verified_latest(controller.destination, job_id="job-a",
                                            dataset_ref="owner/relay")


class FakeSmokeKernel:
    def __init__(self, *, status="QUEUED", output=None):
        template = Path(smoke_gate_module.__file__).with_name("full_model_smoke.ipynb")
        self.source = json.loads(template.read_text())
        self.version = 4
        self.current_status = status
        self.output = output or {}
        self.download_calls = 0

    def snapshot(self):
        return {"version": self.version, "source": self.source}

    def status(self):
        return self.current_status

    def download_output(self, destination):
        self.download_calls += 1
        destination = Path(destination)
        for name, contents in self.output.items():
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            (destination / name).write_bytes(contents)


class FakeSmokeDataset:
    reference = "owner/smoke-report"

    def __init__(self, *, upload_then_raise=False):
        self.version = 0
        self.upload_calls = 0
        self.upload_then_raise = upload_then_raise
        self.ready_value = True
        self.files = {}
        self.stage_metadata = None

    def info(self, *, allow_missing=False):
        return self.version

    def ready(self):
        return self.ready_value

    def upload(self, directory, *, create):
        assert create is True
        self.upload_calls += 1
        directory = Path(directory)
        self.files = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
        self.stage_metadata = json.loads((directory / "dataset-metadata.json").read_text())
        # The fake models a server commit followed by a lost response.
        self.version = 1
        if self.upload_then_raise:
            raise RuntimeError("simulated lost report-publication response")

    def download(self, version, destination):
        assert version == self.version == 1
        destination = Path(destination)
        (destination / "diagnostic.json").write_bytes(self.files["diagnostic.json"])


def make_smoke_materials(root, *, report_changes=None):
    root.mkdir(parents=True, exist_ok=True)
    cache_manifest = root / "cache-manifest.json"
    asset_manifest = root / "asset-manifest.json"
    cache_manifest.write_text(json.dumps({"pairs": [{"id": "pair-a"}]}))
    base_sha, helper_sha = "b" * 64, "c" * 64
    asset_manifest.write_text(json.dumps({"files": {
        "Qwenvy2.1-v1.0.safetensors": {"sha256": base_sha},
        "fizgig_qwen_image_2.1_training_adapter.safetensors": {"sha256": helper_sha},
    }}))
    checkpoint = b"verified fake checkpoint bytes"
    report = {
        "schema": 1,
        "status": "pass",
        "phase": "complete",
        "world_size": 1,
        "logical_device": "xla:0",
        "runtime": {"pjrt_device": "TPU", "versions": {
            "torch": "2.8.0", "torch-xla": "2.8.0", "libtpu": "0.0.17",
        }},
        "optimizer_changed_probe": True,
        "losses": [1.25, 0.75],
        "aten_fallback_counters": {},
        "checkpoint": {"adapter_restored": True, "optimizer_loaded": True,
                       "rng_loaded": True, "path": "checkpoint.pt",
                       "sha256": hashlib.sha256(checkpoint).hexdigest()},
        "identity": {
            "cache_manifest_sha256": relay.sha256(cache_manifest),
            "base_sha256": base_sha,
            "helper_sha256": helper_sha,
            "world_size": 1,
            "per_device_batch": 1,
            "optimizer": "AdamW",
            "optimizer_epsilon": 1e-6,
            "adapter_dtype": "float32",
            "quantization_contract": "convrot8_w8a8_xla_v1",
            "pair_id": "pair-a",
        },
        "source_files_sha256": {
            name: relay.sha256(smoke_gate_module.ROOT / name)
            for name in smoke_gate_module.SOURCE_FILES
        },
    }
    if report_changes:
        report.update(report_changes)
    output = {"diagnostic.json": json.dumps(report).encode(), "checkpoint.pt": checkpoint}
    return cache_manifest, asset_manifest, output


def make_smoke_gate(root, *, kernel=None, dataset=None, clock=None,
                    report_changes=None, output=None):
    cache_manifest, asset_manifest, generated_output = make_smoke_materials(
        root / "inputs", report_changes=report_changes,
    )
    kernel = kernel or FakeSmokeKernel(output=output or generated_output)
    clock = clock or FakeClock()
    gate = smoke_gate_module.SmokeGate(
        destination=root / "smoke-state",
        smoke_kernel_ref="owner/one-rank-smoke",
        report_dataset_ref="owner/smoke-report",
        cache_manifest=cache_manifest,
        asset_manifest=asset_manifest,
        kernel=kernel,
        dataset=dataset or FakeSmokeDataset(),
        now=clock.now,
        publication_wait_seconds=5,
    )
    return gate, kernel, gate.dataset, clock


def test_smoke_queue_waits_without_timeout(tmp_path):
    clock = FakeClock(value=1)
    gate, kernel, dataset, _ = make_smoke_gate(
        tmp_path / "smoke", kernel=FakeSmokeKernel(status="QUEUED"), clock=clock,
    )
    assert gate.advance() == "smoke_observed"
    for _ in range(4):
        clock.value += 1_000_000
        assert gate.advance() == "smoke_queued"
    kernel.current_status = "RUNNING"
    assert gate.advance() == "smoke_running"
    assert dataset.upload_calls == 0
    assert gate.state["phase"] == "waiting"


def test_cpu_smoke_diagnostic_blocks_training_push(tmp_path):
    gate, smoke_kernel, report_dataset, _ = make_smoke_gate(
        tmp_path / "smoke",
        report_changes={"runtime": {"pjrt_device": "CPU", "versions": {
            "torch": "2.8.0", "torch-xla": "2.8.0", "libtpu": "0.0.17",
        }}},
    )
    training_kernel = FakeKernel()
    controller = make_controller(tmp_path / "controller", kernel=training_kernel,
                                 smoke_gate=gate)
    assert controller.advance() == "smoke_observed"
    smoke_kernel.current_status = "COMPLETE"

    with pytest.raises(ValueError, match="qualified Kaggle TPU runtime"):
        controller.advance()
    assert training_kernel.push_calls == 0
    assert report_dataset.upload_calls == 0


def test_verified_smoke_stages_private_report_and_acknowledges_hash(tmp_path):
    gate, kernel, dataset, _ = make_smoke_gate(tmp_path / "smoke")
    assert gate.advance() == "smoke_observed"
    kernel.current_status = "COMPLETE"
    assert gate.advance() == "smoke_verified"
    assert dataset.upload_calls == 0

    assert gate.advance() == "smoke_publishing"
    assert dataset.upload_calls == 1
    assert set(dataset.files) == {"diagnostic.json", "dataset-metadata.json"}
    assert dataset.stage_metadata["id"] == "owner/smoke-report"
    assert gate.state["phase"] == "publishing"
    assert gate.advance() == "ready"
    assert gate.state["report_dataset_version"] == 1
    assert gate.advance() == "ready"


def test_ambiguous_smoke_dataset_publication_is_reconciled_without_retry(tmp_path):
    dataset = FakeSmokeDataset(upload_then_raise=True)
    gate, kernel, _, _ = make_smoke_gate(tmp_path / "smoke", dataset=dataset)
    assert gate.advance() == "smoke_observed"
    kernel.current_status = "COMPLETE"
    assert gate.advance() == "smoke_verified"

    with pytest.raises(RuntimeError, match="lost report-publication response"):
        gate.advance()
    assert gate.state["phase"] == "publishing"
    assert dataset.upload_calls == 1

    restarted, _, _, _ = make_smoke_gate(tmp_path / "smoke", kernel=kernel,
                                         dataset=dataset)
    assert restarted.state["phase"] == "publishing"
    assert restarted.advance() == "ready"
    assert dataset.upload_calls == 1


def test_training_push_waits_for_smoke_gate_ready(tmp_path):
    gate, smoke_kernel, report_dataset, _ = make_smoke_gate(tmp_path / "smoke")
    training_kernel = FakeKernel()
    controller = make_controller(tmp_path / "controller", kernel=training_kernel,
                                 smoke_gate=gate)

    assert controller.advance() == "smoke_observed"
    smoke_kernel.current_status = "COMPLETE"
    assert controller.advance() == "smoke_verified"
    assert controller.advance() == "smoke_publishing"
    assert training_kernel.push_calls == 0
    assert report_dataset.upload_calls == 1

    assert controller.advance() == "submitted"
    assert training_kernel.push_calls == 1
    attempt = controller.state["attempt"]
    assert attempt["smoke_report_dataset"] == "owner/smoke-report"
    assert attempt["smoke_report_sha256"] == gate.state["diagnostic_sha256"]
    staged = Path(attempt["folder"])
    metadata = json.loads((staged / "kernel-metadata.json").read_text())
    notebook = json.loads((staged / "train_session.ipynb").read_text())
    assert "owner/smoke-report" in metadata["dataset_sources"]
    config = ast.literal_eval(
        notebook["cells"][1]["source"].splitlines()[0].removeprefix("AITK_SESSION_CONFIG = "))
    assert config["smoke_report_sha256"] == gate.state["diagnostic_sha256"]
