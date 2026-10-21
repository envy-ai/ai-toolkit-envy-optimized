"""Component-specific FIBO2 cache invalidation contracts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.fibo2.fibo2 import Fibo2Model, resolve_sources


def cache_model(config):
    model = object.__new__(Fibo2Model)
    model.model_config = config
    model.font_paths = ()
    model.torch_dtype = torch.float32
    model.vae_torch_dtype = torch.float32
    model.max_text_length = 4096
    model.latent_space_version = "fibo2_flux2_packed_bn_v1"
    model._sources = resolve_sources(config)
    model._cache_identity = model._component_identity()
    return model


class Fibo2CacheIdentityTests(unittest.TestCase):
    def test_encoder_changes_invalidate_text_cache_and_vae_changes_invalidate_latents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer = root / "transformer.safetensors"
            encoder = root / "qwen3vl.safetensors"
            processor = root / "processor.json"
            vae = root / "flux2-vae.safetensors"
            transformer.write_bytes(b"transformer-v1")
            encoder.write_bytes(b"encoder-v1")
            processor.write_bytes(b"processor-v1")
            vae.write_bytes(b"vae-v1")
            config = SimpleNamespace(
                name_or_path=str(transformer),
                te_name_or_path=None,
                text_encoder_path=str(encoder),
                vae_path=str(vae),
                model_kwargs={"processor_path": str(processor)},
                quantize_te=False,
                qtype_te=None,
            )
            model = cache_model(config)
            text_v1 = model.get_text_embedding_space_version()
            latent_v1 = model.get_latent_space_version()

            encoder.write_bytes(b"encoder-v2-with-different-content")
            model._cache_identity = model._component_identity()
            text_v2 = model.get_text_embedding_space_version()
            self.assertNotEqual(text_v1, text_v2)

            vae.write_bytes(b"vae-v2-with-different-content")
            latent_v2 = model.get_latent_space_version()
            self.assertNotEqual(latent_v1, latent_v2)

    def test_encoder_source_precedence_and_latent_identity_excludes_transformer_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("primary-te", "legacy-te", "transformer", "vae"):
                (root / name).write_bytes(name.encode())
            config = SimpleNamespace(
                name_or_path=str(root / "transformer"),
                te_name_or_path=str(root / "primary-te"),
                text_encoder_path=str(root / "legacy-te"),
                vae_path=str(root / "vae"),
                model_kwargs={},
                quantize_te=False,
                qtype_te=None,
            )

            sources = resolve_sources(config)
            model = cache_model(config)
            latent_v1 = model.get_latent_space_version()
            config.name_or_path = str(root / "legacy-te")
            model._sources = resolve_sources(config)
            model._cache_identity = model._component_identity()

            self.assertEqual(sources["text_encoder"], str(root / "primary-te"))
            self.assertEqual(model._sources["text_encoder"], str(root / "primary-te"))
            self.assertEqual(model.get_latent_space_version(), latent_v1)


if __name__ == "__main__":
    unittest.main()
