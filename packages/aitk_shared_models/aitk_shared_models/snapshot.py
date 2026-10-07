"""Atomic small factorized adapter generations; immutable ID and content hash."""
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

from .protocol import SharedModelError


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(metadata):
    for field in ("snapshot_id", "content_hash", "store_identity", "variant", "base_content_digest", "path"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise SharedModelError(f"Snapshot lacks {field}")
    if metadata["variant"] != "instruct" or len(metadata["content_hash"]) != 64 or type(metadata.get("step")) is not int or metadata["step"] < 0:
        raise SharedModelError("Invalid snapshot variant/hash/step")
    if type(metadata.get("created_ns")) is not int or metadata.get("preset") not in ("attention", "attention_shared_mlp"):
        raise SharedModelError("Invalid snapshot generation/target preset")
    if (type(metadata.get("rank")) is not int or metadata["rank"] <= 0
            or not isinstance(metadata.get("alpha"), (int, float))
            or not math.isfinite(metadata['alpha']) or metadata['alpha'] <= 0):
        raise SharedModelError("Snapshot requires factorized LoRA rank/alpha")
    path = Path(metadata["path"])
    if not path.is_absolute() or not path.is_file() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o222:
        raise SharedModelError("Snapshot file must be owner-controlled and immutable")


def publish_snapshot(directory, step, metadata, export):
    """export(path) reuses the application's normal safetensors adapter exporter."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
        raise SharedModelError("Snapshot directory must be owner-only")
    generation = str(uuid.uuid4())
    temporary = directory / f".{generation}.safetensors"
    destination = directory / f"{generation}.safetensors"
    try:
        export(str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        content_hash = file_hash(temporary)
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        result = dict(metadata, step=int(step), snapshot_id=generation, path=str(destination),
                      content_hash=content_hash, created_ns=time.time_ns())
        validate_snapshot(result)
        sidecar = destination.with_suffix(".json")
        sidecar.write_text(json.dumps(result, sort_keys=True))
        os.chmod(sidecar, 0o400)
        return result
    except BaseException:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def verify_snapshot(metadata, manifest):
    validate_snapshot(metadata)
    if metadata["base_content_digest"] != manifest["content_digest"] or metadata["variant"] != manifest["variant"]:
        raise SharedModelError("Snapshot belongs to another immutable backbone")
    if file_hash(metadata["path"]) != metadata["content_hash"]:
        raise SharedModelError("Snapshot content hash changed")
