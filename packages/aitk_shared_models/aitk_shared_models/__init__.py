"""Experimental protocol v1. Importing this package never initializes CUDA."""
from .protocol import PROTOCOL_VERSION, SharedModelError

__all__ = ["PROTOCOL_VERSION", "SharedModelError"]
