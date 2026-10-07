"""Lazy bridge to the separately installable shared-model protocol package."""
import importlib
import sys
from pathlib import Path


def shared_package(module):
    try:
        return importlib.import_module("aitk_shared_models." + module)
    except ModuleNotFoundError as exc:
        if exc.name != "aitk_shared_models":
            raise
        # Source-tree development does not require modifying either live environment.
        package_root = Path(__file__).resolve().parents[1] / "packages" / "aitk_shared_models"
        if not package_root.is_dir():
            raise RuntimeError("Install aitk-shared-models in this Python environment") from exc
        sys.path.insert(0, str(package_root))
        return importlib.import_module("aitk_shared_models." + module)


def capture_shared_masters(module):
    return shared_package('ownership').capture_masters(module)


def normalize_config(value):
    if value in (None, False):
        return {"enabled": False}
    if not isinstance(value, dict):
        raise ValueError("model.shared_weights must be a mapping")
    result = dict(value)
    result.setdefault("enabled", True)
    result.setdefault("strict", True)
    result.setdefault("reserve_gib", 12)
    if result["enabled"]:
        if result["strict"] is not True:
            raise ValueError("Shared weights require strict attachment; no private-backbone fallback")
        if not result.get("device_uuid"):
            raise ValueError("Shared weights require the local GPU device_uuid")
        result['device_uuid'] = shared_package('protocol').normalize_device_uuid(result['device_uuid'])
        if float(result["reserve_gib"]) < 12:
            raise ValueError("Shared weights require at least 12 GiB MemAvailable reserve")
    return result
