import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from workers.kaggle_tpu import checkpoint_relay as relay


def make_checkpoint(tmp_path, *, attempt="attempt-a", step=12, export=b"adapter", state=b"optimizer"):
    export_path = tmp_path / "adapter.bin"
    state_path = tmp_path / "state.bin"
    export_path.write_bytes(export)
    if state is not None:
        state_path.write_bytes(state)
    files = {"export/adapter.bin": export_path}
    if state is not None:
        files["resume/state.bin"] = state_path
    return relay.build_checkpoint(
        tmp_path / "checkpoints",
        job_id="job-a",
        attempt_id=attempt,
        step=step,
        files=files,
        metadata={"base_hash": "abc123"},
    )


def make_download(checkpoint, destination):
    destination.mkdir(parents=True, exist_ok=True)
    record = relay.package_checkpoint(checkpoint, destination)
    return record


class FakeDataset:
    reference = "owner/private-relay"

    def __init__(self, *, ready=True, version_step=1, corrupt_ack=False):
        self.version = 0
        self.ready_value = ready
        self.version_step = version_step
        self.corrupt_ack = corrupt_ack
        self.versions = {}
        self.downloaded_versions = []
        self.upload_calls = []

    def info(self, *, allow_missing=False):
        return self.version

    def ready(self):
        return self.ready_value

    def upload(self, directory, *, create=False):
        self.upload_calls.append(create)
        self.version += self.version_step
        self.versions[self.version] = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}

    def download(self, version, destination):
        self.downloaded_versions.append(version)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in self.versions[version].items():
            (destination / name).write_bytes(content)
        if self.corrupt_ack:
            archive = destination / relay.ARCHIVE_NAME
            archive.write_bytes(archive.read_bytes() + b"changed")


def test_checkpoint_rejects_duplicate_identity_and_verifies_archive_integrity(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    with pytest.raises(FileExistsError, match="immutable"):
        make_checkpoint(tmp_path)

    download = tmp_path / "download"
    make_download(checkpoint, download)
    archive = download / relay.ARCHIVE_NAME
    archive.write_bytes(archive.read_bytes() + b"tamper")
    destination = tmp_path / "watcher"
    destination.mkdir()
    with pytest.raises(ValueError, match="archive integrity"):
        relay.install_download(download, destination, job_id="job-a", dataset_ref="owner/private-relay", version=1)
    assert not (destination / "latest.json").exists()


def test_archive_rejects_traversal_and_duplicate_members(tmp_path):
    for members, expected_error in [
        ([("../escape", b"bad")], "normalized relative POSIX"),
        ([("safe", b"one"), ("safe", b"two")], "duplicate or undeclared"),
    ]:
        archive = tmp_path / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, contents in members:
                bundle.writestr(name, contents)
        destination = tmp_path / "extract"
        destination.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match=expected_error):
            relay._extract(archive, destination)
        assert not (tmp_path / "escape").exists()


def test_replay_is_idempotent_but_identity_cannot_change_content(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    first = tmp_path / "first"
    make_download(checkpoint, first)
    destination = tmp_path / "watcher"
    destination.mkdir()
    installed = relay.install_download(first, destination, job_id="job-a", dataset_ref="owner/private-relay", version=1)

    replay = tmp_path / "replay"
    make_download(checkpoint, replay)
    again = relay.install_download(replay, destination, job_id="job-a", dataset_ref="owner/private-relay", version=1)
    assert again["manifest_sha256"] == installed["manifest_sha256"]
    assert json.loads((destination / "latest.json").read_text())["step"] == 12

    changed = make_checkpoint(tmp_path, attempt="attempt-b", step=12, export=b"different")
    # Reuse the original identity to prove the immutable-name collision fence.
    changed_manifest = json.loads((changed / "manifest.json").read_text())
    changed_manifest["attempt_id"] = "attempt-a"
    relay._json(changed / "manifest.json", changed_manifest)
    changed_dir = tmp_path / "changed"
    make_download(changed, changed_dir)
    with pytest.raises(ValueError, match="identity was reused"):
        relay.install_download(changed_dir, destination, job_id="job-a", dataset_ref="owner/private-relay", version=2)


def test_older_step_is_recorded_without_moving_latest_backwards(tmp_path):
    destination = tmp_path / "watcher"
    destination.mkdir()
    older = tmp_path / "older"
    make_download(make_checkpoint(tmp_path, attempt="old", step=20), older)
    relay.install_download(older, destination, job_id="job-a", dataset_ref="owner/private-relay", version=1)
    newest = tmp_path / "newest"
    make_download(make_checkpoint(tmp_path, attempt="new", step=10), newest)
    relay.install_download(newest, destination, job_id="job-a", dataset_ref="owner/private-relay", version=2)
    assert json.loads((destination / "latest.json").read_text())["step"] == 20


def test_export_only_versions_are_installed_without_advancing_resume(tmp_path):
    destination = tmp_path / "watcher"
    destination.mkdir()
    first = tmp_path / "first"
    make_download(make_checkpoint(tmp_path, attempt="run", step=100, state=None), first)
    export_record = relay.install_download(first, destination, job_id="job-a",
                                            dataset_ref="owner/private-relay", version=1)
    assert export_record["resume_files"] == []
    assert not (destination / "latest.json").exists()
    assert json.loads((destination / "latest-export.json").read_text())["step"] == 100

    full = tmp_path / "full"
    make_download(make_checkpoint(tmp_path, attempt="run", step=1000), full)
    relay.install_download(full, destination, job_id="job-a",
                           dataset_ref="owner/private-relay", version=2)
    later = tmp_path / "later"
    make_download(make_checkpoint(tmp_path, attempt="run", step=1100, state=None), later)
    relay.install_download(later, destination, job_id="job-a",
                           dataset_ref="owner/private-relay", version=3)

    assert json.loads((destination / "latest-export.json").read_text())["step"] == 1100
    resume = json.loads((destination / "latest.json").read_text())
    assert resume["step"] == 1000
    assert len(resume["resume_files"]) == 1
    assert (destination / "checkpoints/run--step-000000100/export/adapter.bin").read_bytes() == b"adapter"
    assert (destination / "checkpoints/run--step-000001100/export/adapter.bin").read_bytes() == b"adapter"


def test_publish_pins_sequential_version_and_waits_for_byte_ack(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    dataset = FakeDataset()
    receipt = relay.publish_checkpoint(dataset, checkpoint, create=True, timeout=1, poll_seconds=0.001)
    assert receipt["dataset_version"] == 1
    assert dataset.downloaded_versions == [1]
    assert dataset.upload_calls == [True]

    second = make_checkpoint(tmp_path, attempt="attempt-b", step=13)
    receipt2 = relay.publish_checkpoint(dataset, second, create=False, timeout=1, poll_seconds=0.001)
    assert receipt2["dataset_version"] == 2
    assert dataset.downloaded_versions == [1, 2]
    assert dataset.upload_calls == [True, False]

    bad_ack = FakeDataset(corrupt_ack=True)
    with pytest.raises(ValueError, match="acknowledgement differs"):
        relay.publish_checkpoint(bad_ack, checkpoint, create=True, timeout=1, poll_seconds=0.001)


def test_publish_rejects_version_gaps_and_private_dataset_check(tmp_path, monkeypatch):
    checkpoint = make_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="Concurrent publisher"):
        relay.publish_checkpoint(FakeDataset(version_step=2), checkpoint, create=True,
                                 timeout=1, poll_seconds=0.001)

    class Request:
        pass

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        class datasets:
            class dataset_api_client:
                @staticmethod
                def get_dataset(_request):
                    class Dataset:
                        is_private = False
                        current_version_number = 1
                    return Dataset()

    class Api:
        mutations = []

        @staticmethod
        def build_kaggle_client():
            return Client()

        def dataset_create_version(self, *_args, **_kwargs):
            self.mutations.append("upload")

        def dataset_download_files(self, *_args, **_kwargs):
            self.mutations.append("download")

    import sys
    import types
    sdk = types.ModuleType("kagglesdk")
    datasets = types.ModuleType("kagglesdk.datasets")
    types_module = types.ModuleType("kagglesdk.datasets.types")
    dataset_api = types.ModuleType("kagglesdk.datasets.types.dataset_api_service")
    dataset_api.ApiGetDatasetRequest = Request
    monkeypatch.setitem(sys.modules, "kagglesdk", sdk)
    monkeypatch.setitem(sys.modules, "kagglesdk.datasets", datasets)
    monkeypatch.setitem(sys.modules, "kagglesdk.datasets.types", types_module)
    monkeypatch.setitem(sys.modules, "kagglesdk.datasets.types.dataset_api_service", dataset_api)
    api = Api()
    dataset = relay.KaggleDataset("owner/slug", api=api)
    with pytest.raises(ValueError, match="public dataset"):
        dataset.upload(tmp_path, create=False)
    with pytest.raises(ValueError, match="public dataset"):
        dataset.download(1, tmp_path)
    assert api.mutations == []


def test_missing_private_slug_403_requires_owned_absent_listing():
    class Forbidden(Exception):
        response = SimpleNamespace(status_code=403)

    class Client:
        def __enter__(self):
            return SimpleNamespace(datasets=SimpleNamespace(dataset_api_client=self))

        def __exit__(self, *_args):
            pass

        def get_dataset(self, _request):
            raise Forbidden()

    class Api:
        def __init__(self, username="owner", present=False):
            self.username, self.present = username, present
            self.pages_seen = []

        def build_kaggle_client(self):
            return Client()

        def get_config_value(self, name):
            assert name == "username"
            return self.username

        def dataset_list_with_response(self, *, mine, page_size, page_token):
            assert mine and page_size == 100
            self.pages_seen.append(page_token)
            if page_token is None:
                return SimpleNamespace(datasets=[SimpleNamespace(ref="owner/other")], next_page_token="second")
            assert page_token == "second"
            datasets = [SimpleNamespace(ref="owner/slug")] if self.present else []
            return SimpleNamespace(datasets=datasets, next_page_token=None)

    absent = Api()
    assert relay.KaggleDataset("owner/slug", api=absent).info(allow_missing=True) == 0
    assert absent.pages_seen == [None, "second"]
    with pytest.raises(RuntimeError, match="lookup failed"):
        relay.KaggleDataset("owner/slug", api=Api(present=True)).info(allow_missing=True)
    wrong_owner = Api(username="someone-else")
    with pytest.raises(RuntimeError, match="lookup failed"):
        relay.KaggleDataset("owner/slug", api=wrong_owner).info(allow_missing=True)
    assert wrong_owner.pages_seen == []
