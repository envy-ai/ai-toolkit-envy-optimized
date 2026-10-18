"""Strict streamed extraction of the paired-cache ZIP used by Kaggle inputs."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


def extract_cache_bundle(archive, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Cache ZIP contains duplicate members")
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or chr(92) in member.filename or str(path) != member.filename:
                raise ValueError("Cache ZIP contains an unsafe path")
            if member.is_dir() or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError("Cache ZIP must contain regular files only")
        manifests = [name for name in names if name == "manifest.json" or name.endswith("/manifest.json")]
        if len(manifests) != 1:
            raise ValueError("Cache ZIP must have exactly one manifest")
        manifest_name = manifests[0]
        if bundle.getinfo(manifest_name).file_size > 16 * 1024 * 1024:
            raise ValueError("Cache manifest is unexpectedly large")
        manifest_bytes = bundle.read(manifest_name)
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != 1 or manifest.get("kind") != "qwen21_fizgig_paired_cache":
            raise ValueError("Cache ZIP has an incompatible manifest")
        prefix = manifest_name.removesuffix("manifest.json")
        expected = {prefix + name for name in manifest["files"]} | {manifest_name}
        if set(names) != expected:
            raise ValueError("Cache ZIP members differ from the manifest allowlist")
        with tempfile.TemporaryDirectory(prefix=".paired-cache-", dir=output.parent) as temporary:
            stage = Path(temporary) / "cache"
            stage.mkdir()
            (stage / "manifest.json").write_bytes(manifest_bytes)
            for name, record in manifest["files"].items():
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or chr(92) in name or str(path) != name:
                    raise ValueError("Manifest contains an unsafe cache path")
                member = bundle.getinfo(prefix + name)
                if member.file_size != record["bytes"]:
                    raise ValueError("Cache member size differs from manifest")
                destination = stage / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with bundle.open(member) as source, destination.open("xb") as target:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        target.write(chunk)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError("Cache member checksum differs from manifest")
            stage.rename(output)
    return output
