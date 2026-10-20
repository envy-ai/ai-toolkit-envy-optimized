"""Select PyTorch-only optional backends before importing training libraries."""
from __future__ import annotations

import os
import sys


def configure_torch_only():
    # Diffusers reads USE_FLAX/USE_TF/USE_TORCH once at its first import.
    # Kaggle's unrelated preinstalled Flax/JAX stack need not match our NumPy.
    os.environ.update(USE_TORCH="1", USE_FLAX="0", USE_TF="0")


def check_training_imports():
    """Import the real model/adapter dependencies without loading any weights."""
    configure_torch_only()
    from diffusers.utils import is_flax_available
    from toolkit.lora_special import LoRASpecialNetwork  # noqa: F401 - importability is the check
    from toolkit.assistant_lora import load_assistant_lora_from_path  # noqa: F401
    from workers.kaggle_tpu.export_paired_cache import qwen_module
    from workers.kaggle_tpu.static_qwen import StaticQwen  # noqa: F401

    qwen_module("qwen_image_2").QwenImage2Model
    qwen_module("src.transformer").QwenImage21Transformer2DModel
    forbidden = sorted(name for name in ("flax", "jax", "tensorflow") if name in sys.modules)
    if is_flax_available() or forbidden:
        raise RuntimeError(f"PyTorch-only import preflight loaded optional backends: {forbidden}")
    return {"status": "pass", "optional_backends": "torch_only"}


if __name__ == "__main__":
    import json

    print(json.dumps(check_training_imports()), flush=True)
