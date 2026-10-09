"""Run Iris diagnostics without initializing unrelated CUDA-only extensions."""
import os
import sys
import types
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
# The diffusion-model registry eagerly imports an external OmniGen Triton
# kernel that initializes CUDA even when only Iris CPU tests are requested.
# Import the real Iris modules through their package path without that registry.
package = types.ModuleType("extensions_built_in.diffusion_models")
package.__path__ = [str(root / "extensions_built_in/diffusion_models")]
sys.modules[package.__name__] = package
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromName("testing.test_iris"))
sys.exit(0 if result.wasSuccessful() else 1)
