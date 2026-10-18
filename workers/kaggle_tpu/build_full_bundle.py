"""Package only code and toolkit JSON/YAML resources for the full TPU gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def build(output):
    files = {ROOT / "info.py", ROOT / "version.py"}
    for directory in ("toolkit", "jobs", "extensions_built_in", "workers/kaggle_tpu"):
        files.update(path for path in (ROOT / directory).rglob("*.py") if "__pycache__" not in path.parts)
    for directory in ("toolkit/keymaps", "toolkit/orig_configs", "toolkit/diffusers_configs"):
        files.update(path for path in (ROOT / directory).rglob("*") if path.suffix in (".json", ".yaml", ".yml"))
    files.add(ROOT / "workers/kaggle_tpu/requirements_smoke.txt")
    manifest = {"schema": 1, "kind": "qwen21_full_model_source", "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "files": {}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Source must be a regular file: {path}")
            name, content = str(path.relative_to(ROOT)), path.read_bytes()
            manifest["files"][name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/kaggle_train/source/kaggle_tpu_full_bundle.zip")
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps({"output": str(args.output), "files": len(manifest["files"]), "size": args.output.stat().st_size}))
