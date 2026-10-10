"""Validate cached tensor headers and publish completed cache writes atomically."""
import logging
import os
import tempfile

from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file


class CachedTensorError(RuntimeError):
    def __init__(self, path, cause):
        self.path = os.fspath(path)
        super().__init__(f"Cannot read cached tensors from {self.path}: {cause}")


def cache_metadata(path):
    try:
        with safe_open(path, framework="pt", device="cpu") as cache:
            return cache.metadata()
    except (SafetensorError, OSError) as error:
        raise CachedTensorError(path, error) from error


def load_cached_file(path, device="cpu"):
    try:
        return load_file(path, device=device)
    except (SafetensorError, OSError) as error:
        raise CachedTensorError(path, error) from error


def is_valid_cache(path, required_keys=()):
    """Check only the header and file extent, without loading tensor payloads."""
    try:
        with safe_open(path, framework="pt", device="cpu") as cache:
            keys = set(cache.keys())
            if not keys or not set(required_keys).issubset(keys):
                raise ValueError(f"Missing cache tensors: {tuple(required_keys)}")
        return True
    except FileNotFoundError:
        return False
    except (SafetensorError, OSError, ValueError) as error:
        logging.getLogger(__name__).warning(
            "Invalid tensor cache %s; regenerating from source: %s", path, error)
        return False


def atomic_save_file(tensors, filename, metadata=None):
    """Keep readers and interrupted writes from exposing a partial cache file."""
    filename = os.path.abspath(os.fspath(filename))
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(filename)}.", suffix=".tmp")
    os.close(descriptor)
    try:
        save_file(tensors, temporary, metadata=metadata)
        # Flush file data before publishing its name, including on abrupt exits.
        with open(temporary, "rb+") as written:
            os.fsync(written.fileno())
        os.replace(temporary, filename)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
