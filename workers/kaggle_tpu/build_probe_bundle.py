"""Create the small, explicit Kaggle probe bundle from this worktree.

No repository archive, model, cache, credential, or unlisted file is included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "probe.py": ROOT / "workers/kaggle_tpu/probe.py",
    "convrot_xla.py": ROOT / "toolkit/util/convrot_xla.py",
}


def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema": 1, "source_commit": commit, "files": {}}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, path in SOURCES.items():
            content = path.read_bytes()
            manifest["files"][name] = hashlib.sha256(content).hexdigest()
            bundle.writestr(name, content)
        bundle.writestr("source_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/kaggle_tpu_probe_bundle.zip")
    args = parser.parse_args()
    print(build(args.output))
