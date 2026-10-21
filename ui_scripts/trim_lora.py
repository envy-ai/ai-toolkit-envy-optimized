"""UI scripts bridge: preview first, then explicitly save a new copy."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.trim_lora import main

if __name__ == '__main__':
    raise SystemExit(main(require_preview=True, json_output=True))
