import os
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class Ideogram4LowVramTests(unittest.TestCase):
    def test_transformer_accepts_text_only_llm_features_without_changing_output(self):
        from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
            Ideogram4Config,
            Ideogram4Transformer2DModel,
            LLM_TOKEN_INDICATOR,
            OUTPUT_IMAGE_INDICATOR,
            SEQUENCE_PADDING_INDICATOR,
        )

        torch.manual_seed(0)
        config = Ideogram4Config(
            emb_dim=24,
            num_layers=1,
            num_heads=3,
            intermediate_size=32,
            adanln_dim=8,
            in_channels=4,
            llm_features_dim=6,
            mrope_section=(1, 1, 1),
        )
        transformer = Ideogram4Transformer2DModel(config).eval()

        batch_size = 1
        text_tokens = 3
        image_tokens = 2
        seq_len = text_tokens + image_tokens

        llm_text = torch.randn(batch_size, text_tokens, config.llm_features_dim)
        llm_full = torch.cat(
            [
                llm_text,
                torch.zeros(batch_size, image_tokens, config.llm_features_dim),
            ],
            dim=1,
        )
        x = torch.randn(batch_size, seq_len, config.in_channels)
        x[:, :text_tokens] = 0
        t = torch.tensor([0.25])
        position_ids = torch.zeros(batch_size, seq_len, 3, dtype=torch.long)
        segment_ids = torch.tensor([[1, 1, SEQUENCE_PADDING_INDICATOR, 1, 1]])
        indicator = torch.tensor(
            [
                [
                    LLM_TOKEN_INDICATOR,
                    LLM_TOKEN_INDICATOR,
                    0,
                    OUTPUT_IMAGE_INDICATOR,
                    OUTPUT_IMAGE_INDICATOR,
                ]
            ]
        )

        with torch.no_grad():
            full_output = transformer(
                llm_features=llm_full,
                x=x,
                t=t,
                position_ids=position_ids,
                segment_ids=segment_ids,
                indicator=indicator,
            )
            text_only_output = transformer(
                llm_features=llm_text,
                x=x,
                t=t,
                position_ids=position_ids,
                segment_ids=segment_ids,
                indicator=indicator,
            )

        torch.testing.assert_close(text_only_output, full_output)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_low_vram_prompt_encoding_returns_text_encoder_to_cpu(self):
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model

        class FakeTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                return messages[0]["content"][0]["text"]

            def __call__(self, text, add_special_tokens, truncation, max_length):
                return {"input_ids": [1, 2]}

        class FakeTextEncoder:
            def __init__(self):
                self.device = torch.device("cpu")

            def to(self, device):
                self.device = torch.device(device)
                return self

        model = Ideogram4Model.__new__(Ideogram4Model)
        model.text_encoder = FakeTextEncoder()
        model.tokenizer = FakeTokenizer()
        model.device_torch = torch.device("cuda")
        model.torch_dtype = torch.float32
        model.max_text_length = 8
        model.model_config = SimpleNamespace(low_vram=True)

        def fake_get_features(text_encoder, token_ids, attention_mask, pos_2d):
            self.assertEqual(text_encoder.device, torch.device("cuda"))
            return torch.ones(1, token_ids.shape[1], 4, device="cpu")

        with patch(
            "extensions_built_in.diffusion_models.ideogram4.ideogram4.get_qwen3_vl_features",
            fake_get_features,
        ):
            embeds = model.get_prompt_embeds("test prompt")

        self.assertEqual(model.text_encoder.device, torch.device("cpu"))
        self.assertEqual(embeds.text_embeds[0].device, torch.device("cpu"))

    def test_low_vram_vae_load_stays_on_cpu(self):
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model

        class FakeVAE:
            def __init__(self, params):
                self.device = None
                self.dtype = None

            def load_state_dict(self, state_dict):
                self.state_dict_loaded = state_dict

            def to(self, device, dtype=None):
                self.device = torch.device(device)
                self.dtype = dtype
                return self

            def eval(self):
                return self

            def requires_grad_(self, value):
                self.requires_grad = value
                return self

        model = Ideogram4Model.__new__(Ideogram4Model)
        model.torch_dtype = torch.bfloat16
        model.vae_device_torch = torch.device("cuda")
        model.model_config = SimpleNamespace(low_vram=True)
        model._status_update_hooks = []

        with patch(
            "extensions_built_in.diffusion_models.ideogram4.ideogram4._load_component_state_dict",
            return_value={"dummy": torch.tensor(1)},
        ), patch(
            "extensions_built_in.diffusion_models.ideogram4.ideogram4.convert_diffusers_state_dict",
            return_value={"converted": torch.tensor(1)},
        ), patch(
            "extensions_built_in.diffusion_models.ideogram4.ideogram4.AutoEncoder",
            FakeVAE,
        ):
            vae = model._load_vae("unused")

        self.assertEqual(vae.device, torch.device("cpu"))
        self.assertEqual(vae.dtype, torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_low_vram_decode_returns_vae_to_cpu(self):
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model

        class FakeVAE:
            def __init__(self):
                self.device = torch.device("cpu")

            def to(self, device, dtype=None):
                self.device = torch.device(device)
                return self

            def decoder(self, z):
                assert self.device == torch.device("cuda")
                return torch.zeros(1, 3, 8, 8, device=z.device, dtype=z.dtype)

        model = Ideogram4Model.__new__(Ideogram4Model)
        model.vae = FakeVAE()
        model.vae_device_torch = torch.device("cuda")
        model.vae_torch_dtype = torch.float32
        model.model_config = SimpleNamespace(low_vram=True)
        model.patch_size = 2
        model._latent_shift = torch.zeros(1, 128, 1, 1)
        model._latent_scale = torch.ones(1, 128, 1, 1)

        latents = torch.zeros(1, 128, 1, 1)
        images = model.decode_latents(latents)

        self.assertEqual(model.vae.device, torch.device("cpu"))
        self.assertEqual(images.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_low_vram_encode_returns_vae_to_cpu(self):
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model

        class FakeParams:
            z_channels = 32

        class FakeVAE:
            def __init__(self):
                self.device = torch.device("cpu")
                self.params = FakeParams()

            def to(self, device, dtype=None):
                self.device = torch.device(device)
                return self

            def eval(self):
                return self

            def requires_grad_(self, value):
                return self

            def encoder(self, images):
                assert self.device == torch.device("cuda")
                return torch.zeros(images.shape[0], 64, 2, 2, device=images.device)

        model = Ideogram4Model.__new__(Ideogram4Model)
        model.vae = FakeVAE()
        model.vae_device_torch = torch.device("cuda")
        model.vae_torch_dtype = torch.float32
        model.model_config = SimpleNamespace(low_vram=True)
        model.patch_size = 2
        model._latent_shift = torch.zeros(1, 128, 1, 1)
        model._latent_scale = torch.ones(1, 128, 1, 1)

        image = torch.zeros(3, 16, 16)
        latents = model.encode_images([image])

        self.assertEqual(model.vae.device, torch.device("cpu"))
        self.assertEqual(latents.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
