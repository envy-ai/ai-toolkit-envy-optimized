import importlib
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

import torch


def _load_krea_modules():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    package_path = repo_root / "extensions_built_in/diffusion_models/krea2"
    package_name = "krea2_kv_cache_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package
    mmdit = importlib.import_module(f"{package_name}.src.mmdit")
    pipeline = importlib.import_module(f"{package_name}.src.pipeline")
    return mmdit, pipeline


class Krea2KVCacheTests(unittest.TestCase):
    def test_reference_cache_reuse_matches_capture_pass(self):
        mmdit, pipeline = _load_krea_modules()
        torch.manual_seed(1234)
        config = mmdit.SingleMMDiTConfig(
            features=32,
            tdim=16,
            txtdim=8,
            heads=4,
            multiplier=2,
            layers=1,
            patch=2,
            channels=4,
            kvheads=2,
            txtlayers=2,
            txtheads=2,
            txtkvheads=2,
        )
        model = mmdit.SingleStreamDiT(config).eval()
        latents = torch.randn(1, 4, 4, 4)
        context = torch.randn(1, 3, 16)
        text_mask = torch.ones(1, 3, dtype=torch.bool)
        references = [[torch.randn(4, 4, 4)]]
        timestep = torch.tensor([0.7])
        cache = {"kv": None, "mask": None}

        with torch.no_grad():
            capture_output = pipeline.predict_velocity(
                model,
                latents,
                timestep,
                context,
                text_mask,
                references,
                isolate_refs=True,
                ref_kv_cache=cache,
            )
            reuse_output = pipeline.predict_velocity(
                model,
                latents,
                timestep,
                context,
                text_mask,
                references,
                isolate_refs=True,
                ref_kv_cache=cache,
            )

        self.assertEqual(len(cache["kv"]), 1)
        self.assertTrue(
            torch.allclose(capture_output, reuse_output, atol=1e-5, rtol=1e-5)
        )

    def test_reference_cache_requires_isolated_attention(self):
        _, pipeline = _load_krea_modules()

        with self.assertRaisesRegex(ValueError, "requires isolate_refs"):
            pipeline.predict_velocity(
                model=SimpleNamespace(config=SimpleNamespace(patch=2)),
                latents=torch.zeros(1, 4, 4, 4),
                t=torch.tensor([0.5]),
                context=torch.zeros(1, 1, 16),
                text_mask=torch.ones(1, 1, dtype=torch.bool),
                ref_kv_cache={"kv": None, "mask": None},
            )


if __name__ == "__main__":
    unittest.main()
