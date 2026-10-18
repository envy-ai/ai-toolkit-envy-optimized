"""Private Kaggle dataset checkpoint transport; no training or notebook control.

Only explicitly supplied files enter the relay. Credentials are read by Kaggle
in memory, never placed in a checkpoint or dataset metadata. See checkpoint_relay.md.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import zipfile

SCHEMA = 1
KIND = "aitk_private_checkpoint"
ARCHIVE_NAME = "checkpoint.bin"  # ZIP bytes under an opaque extension; Kaggle expands uploaded .zip files.
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    """Commit a JSON record on the same filesystem after writing its contents."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError("Checkpoint names must be normalized relative POSIX paths")
    return path


def _identity(job_id, attempt_id, step):
    if not all(isinstance(item, str) and IDENTIFIER.fullmatch(item) for item in (job_id, attempt_id)):
        raise ValueError("Job and attempt IDs must be safe nonempty identifiers")
    if type(step) is not int or step < 0:
        raise ValueError("Completed optimizer step must be a nonnegative integer")
    return f"{attempt_id}--step-{step:09d}"


def _no_secret_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if re.search(r"(^|_)(token|secret|password|credential|api_key)(_|$)", str(key), re.I):
                raise ValueError("Credentials must not appear in checkpoint metadata")
            _no_secret_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _no_secret_keys(child)


def build_checkpoint(root, *, job_id, attempt_id, step, files, metadata):
    """Atomically stage an immutable export, optionally with resume state.

    When supplied, resume files must contain the full runner-owned resume
    contract (optimizer, scheduler, per-rank RNG, cursor and identities).
    This function hashes opaque bytes; it never unpickles resume state.
    """
    key = _identity(job_id, attempt_id, step)
    _no_secret_keys(metadata)
    names = [_relative(name) for name in files]
    if not names or any(path.parts[0] not in {"export", "resume"} or len(path.parts) < 2 for path in names):
        raise ValueError("Files must be explicitly named below export/ or resume/")
    if "export" not in {path.parts[0] for path in names}:
        raise ValueError("A checkpoint requires an inference export")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    if target.exists():
        raise FileExistsError(f"Checkpoint is immutable: {key}")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=root) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        records = {}
        for name, source in sorted(files.items()):
            source = Path(source)
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Checkpoint input must be a regular file: {name}")
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            records[name] = {"size": destination.stat().st_size, "sha256": sha256(destination)}
        manifest = {"schema": SCHEMA, "kind": KIND, "job_id": job_id, "attempt_id": attempt_id,
                    "step": step, "files": records, "metadata": metadata}
        _json(staging / "manifest.json", manifest)
        verify_checkpoint(staging)
        os.rename(staging, target)
    return target


def verify_checkpoint(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("kind") != KIND:
        raise ValueError("Unsupported checkpoint manifest")
    _identity(manifest["job_id"], manifest["attempt_id"], manifest["step"])
    _no_secret_keys(manifest["metadata"])
    files = manifest["files"]
    roles = set()
    for name, record in files.items():
        relative = _relative(name)
        roles.add(relative.parts[0])
        if relative.parts[0] not in {"export", "resume"} or len(relative.parts) < 2:
            raise ValueError("Invalid checkpoint file role")
        path = root / name
        if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
            raise ValueError(f"Checkpoint file is missing or a symlink: {name}")
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if "export" not in roles or actual != set(files) | {"manifest.json"}:
        raise ValueError("Checkpoint requires exactly its declared export and optional resume files")
    return manifest


def package_checkpoint(checkpoint, destination):
    manifest = verify_checkpoint(checkpoint)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name in ["manifest.json", *sorted(manifest["files"])]:
            bundle.write(Path(checkpoint) / name, name)
    relay = {"schema": SCHEMA, "kind": KIND, "job_id": manifest["job_id"],
             "attempt_id": manifest["attempt_id"], "step": manifest["step"],
             "archive_sha256": sha256(archive), "archive_size": archive.stat().st_size,
             "manifest_sha256": sha256(Path(checkpoint) / "manifest.json")}
    _json(destination / "relay.json", relay)
    return relay


def _extract(archive, destination, expected=None):
    """Extract regular files only, disallowing duplicate and traversal members."""
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or (expected is not None and set(names) != set(expected)):
            raise ValueError("Archive contains duplicate or undeclared members")
        for member in members:
            _relative(member.filename)
            mode = member.external_attr >> 16
            if member.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError("Archive may contain regular files only")
        for member in members:
            target = Path(destination) / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)


class KaggleDataset:
    """Adapter for the installed official Kaggle CLI API and kagglesdk."""

    def __init__(self, reference, api=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", reference):
            raise ValueError("Dataset reference must be owner/slug")
        self.reference = reference
        if api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        self.api = api

    def info(self, *, allow_missing=False):
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = self.reference.split("/")
        try:
            with self.api.build_kaggle_client() as client:
                dataset = client.datasets.dataset_api_client.get_dataset(request)
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if allow_missing:
                if status == 404:
                    return 0
                if status == 403 and self._owned_slug_absent():
                    return 0
            raise RuntimeError(f"Kaggle dataset lookup failed ({type(error).__name__}); credential-bearing details suppressed") from None
        if not dataset.is_private:
            raise ValueError("Checkpoint relay refuses a public dataset")
        return dataset.current_version_number

    def _owned_slug_absent(self):
        """Disambiguate Kaggle's 403 for a missing owned private dataset.

        A 403 on an unowned or listed dataset remains an access error. Exhaust
        the authenticated `mine` listing so pagination cannot imply absence.
        """
        owner = self.reference.split("/", 1)[0]
        username = self._call(self.api.get_config_value, "username")
        if not username or owner.casefold() != username.casefold():
            return False
        token, seen = None, set()
        while True:
            page = self._call(self.api.dataset_list_with_response, mine=True,
                              page_size=100, page_token=token)
            if page is None:
                raise RuntimeError("Kaggle authenticated dataset listing returned no response")
            for dataset in page.datasets or []:
                if dataset is not None and str(dataset.ref).casefold() == self.reference.casefold():
                    return False
            token = page.next_page_token
            if not token:
                return True
            if token in seen:
                raise RuntimeError("Kaggle authenticated dataset listing repeated a page token")
            seen.add(token)

    def ready(self):
        status = self._call(self.api.dataset_status, self.reference)
        if status in {"error", "failed"}:
            raise RuntimeError("Kaggle checkpoint dataset processing failed")
        return status == "ready"

    @staticmethod
    def _call(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            raise RuntimeError(f"Kaggle {operation.__name__} failed ({type(error).__name__}); credential-bearing details suppressed") from None

    def download(self, version, destination):
        self.info()  # Check privacy on every transfer, including old versions.
        self._call(self.api.dataset_download_files, f"{self.reference}/{version}", path=str(destination),
                   force=True, quiet=True, unzip=False)
        archive = Path(destination) / (self.reference.split("/")[1] + ".zip")
        _extract(archive, destination, {ARCHIVE_NAME, "relay.json"})
        archive.unlink()

    def upload(self, directory, *, create=False):
        if not create:
            self.info()
        elif self.info(allow_missing=True):
            raise FileExistsError("Refusing to create over an existing relay dataset")
        if create:
            response = self._call(self.api.dataset_create_new, str(directory), public=False,
                                  quiet=True, convert_to_csv=False)
        else:
            response = self._call(self.api.dataset_create_version, str(directory), version_notes="AI Toolkit checkpoint",
                                  quiet=True, convert_to_csv=False, delete_old_versions=False)
        if response.error:
            raise RuntimeError("Kaggle rejected checkpoint publication; inspect private dataset status")


def publish_checkpoint(dataset, checkpoint, *, create=False, timeout=900, poll_seconds=10):
    """Synchronous save: return only after a pinned version verifies remotely.

    Exactly one publisher must own this dedicated dataset. Ambiguous uploads
    propagate rather than retrying a mutation; the watcher can still recover a
    remotely committed version. Prior versions are never deleted.
    """
    if timeout <= 0 or poll_seconds <= 0:
        raise ValueError("Publication timeout and poll interval must be positive")
    previous = 0 if create else dataset.info()
    with tempfile.TemporaryDirectory(prefix="checkpoint-upload-") as temporary:
        directory = Path(temporary)
        relay = package_checkpoint(checkpoint, directory)
        _json(directory / "dataset-metadata.json", {"id": dataset.reference,
              "title": "AI Toolkit private checkpoint relay", "licenses": [{"name": "other"}]})
        dataset.upload(directory, create=create)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Queue/session limits are separate; this only bounds one save ack.
            # A newly created private dataset may briefly return Kaggle's
            # missing-slug 403 before it appears in the authenticated listing.
            version = dataset.info(allow_missing=True)
            if version > previous and dataset.ready():
                if version != previous + 1:
                    raise RuntimeError("Concurrent publisher detected; dedicated dataset required")
                with tempfile.TemporaryDirectory(prefix="checkpoint-ack-") as downloaded:
                    dataset.download(version, Path(downloaded))
                    received = json.loads((Path(downloaded) / "relay.json").read_text())
                    if received != relay or sha256(Path(downloaded) / ARCHIVE_NAME) != relay["archive_sha256"]:
                        raise ValueError("Published checkpoint acknowledgement differs from local bytes")
                return {**relay, "dataset": dataset.reference, "dataset_version": version}
            time.sleep(poll_seconds)
    raise TimeoutError("Kaggle checkpoint publication was not acknowledged within save timeout")


@contextmanager
def watcher_lock(destination):
    with (Path(destination) / ".watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def initialize_destination(destination, *, job_id, dataset_ref, current_job):
    destination, current_job = Path(destination).resolve(), Path(current_job).resolve()
    if destination == current_job or destination.is_relative_to(current_job) or current_job.is_relative_to(destination):
        raise ValueError("Relay destination must be separate from the current local job folder")
    marker = destination / "relay-owner.json"
    owner = {"schema": SCHEMA, "job_id": job_id, "dataset": dataset_ref}
    if destination.exists() and not marker.exists() and any(destination.iterdir()):
        raise ValueError("First watcher start requires a new or empty local destination")
    destination.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text()) != owner:
            raise ValueError("Local relay folder belongs to another job or dataset")
    else:
        _json(marker, owner)
    return destination


def install_download(download, destination, *, job_id, dataset_ref, version):
    """Verify then atomically install an immutable checkpoint and update pointer."""
    download, destination = Path(download), Path(destination)
    relay = json.loads((download / "relay.json").read_text())
    if relay.get("schema") != SCHEMA or relay.get("kind") != KIND or relay.get("job_id") != job_id:
        raise ValueError("Dataset checkpoint belongs to another job or schema")
    key = _identity(job_id, relay["attempt_id"], relay["step"])
    archive = download / ARCHIVE_NAME
    if archive.stat().st_size != relay["archive_size"] or sha256(archive) != relay["archive_sha256"]:
        raise ValueError("Dataset checkpoint archive integrity failure")
    with tempfile.TemporaryDirectory(prefix=".install-", dir=destination) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        _extract(archive, staging)
        if sha256(staging / "manifest.json") != relay["manifest_sha256"]:
            raise ValueError("Dataset checkpoint manifest integrity failure")
        manifest = verify_checkpoint(staging)
        if any(manifest[key] != relay[key] for key in ("job_id", "attempt_id", "step")):
            raise ValueError("Relay and checkpoint identities differ")
        target = destination / "checkpoints" / key
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            verify_checkpoint(target)
            if sha256(target / "manifest.json") != relay["manifest_sha256"]:
                raise ValueError("Immutable checkpoint identity was reused with different content")
        else:
            os.rename(staging, target)
    record = {**relay, "dataset": dataset_ref, "dataset_version": version,
              "checkpoint_path": str(target.resolve()), "resume_files": [str((target / name).resolve())
                  for name in manifest["files"] if name.startswith("resume/")]}
    # Every save advances the export pointer. Only a complete resume save may
    # advance the orchestrator's durable resume pointer.
    for pointer in ("latest-export.json", "latest.json") if record["resume_files"] else ("latest-export.json",):
        latest_path = destination / pointer
        latest = json.loads(latest_path.read_text()) if latest_path.exists() else None
        if latest is None or record["step"] > latest["step"]:
            _json(latest_path, record)
        elif record["step"] == latest["step"] and record["manifest_sha256"] != latest["manifest_sha256"]:
            raise ValueError("Two different checkpoints claim the same completed step")
    _json(destination / "watcher-state.json", {"schema": SCHEMA, "last_version": version})
    return record


def poll_once(dataset, destination, *, job_id, on_checkpoint=None):
    destination = Path(destination)
    state = destination / "watcher-state.json"
    previous = json.loads(state.read_text())["last_version"] if state.exists() else 0
    latest_version = dataset.info(allow_missing=True)
    if latest_version < previous:
        raise ValueError("Dataset version history moved backwards")
    if latest_version == previous or not dataset.ready():
        return []
    installed = []
    for version in range(previous + 1, latest_version + 1):
        with tempfile.TemporaryDirectory(prefix=".download-", dir=destination) as temporary:
            dataset.download(version, Path(temporary))
            record = install_download(temporary, destination, job_id=job_id,
                                      dataset_ref=dataset.reference, version=version)
        installed.append(record)
        if on_checkpoint is not None:
            on_checkpoint(record)
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--current-job", required=True, type=Path)
    parser.add_argument("--poll-seconds", default=30, type=float)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    destination = initialize_destination(args.destination, job_id=args.job_id,
                  dataset_ref=args.dataset, current_job=args.current_job)
    dataset = KaggleDataset(args.dataset)
    with watcher_lock(destination):
        while True:
            records = poll_once(dataset, destination, job_id=args.job_id)
            for record in records:
                print(json.dumps({key: record[key] for key in ("attempt_id", "step", "dataset_version", "checkpoint_path")}), flush=True)
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    # Signed download URLs and authentication responses must never enter logs.
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Checkpoint relay stopped: {type(error).__name__}; details omitted to protect credentials") from None
