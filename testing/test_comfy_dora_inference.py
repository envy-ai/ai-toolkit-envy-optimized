import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from toolkit.comfy_lora import prepare_qwen_image_2_inference_dora


class ComfyInferenceDoRATests(unittest.TestCase):
    def test_converts_legacy_ai_toolkit_keys_and_magnitude_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_root = Path(tmp_dir) / "loras"
            cache_root = Path(tmp_dir) / "cache"
            source = lora_root / "qwen" / "legacy.safetensors"
            source.parent.mkdir(parents=True)
            prefix = "transformer.transformer_blocks.0.attn.to_q"
            save_file(
                {
                    prefix + ".lora_A.weight": torch.ones(2, 4),
                    prefix + ".lora_B.weight": torch.ones(4, 2),
                    prefix + ".magnitude": torch.arange(4, dtype=torch.float32),
                },
                str(source),
            )

            prepared = prepare_qwen_image_2_inference_dora(
                "qwen/legacy.safetensors",
                str(cache_root),
                search_paths=[lora_root],
            )

            self.assertIsNotNone(prepared)
            self.assertNotEqual(Path(prepared), source)
            state = load_file(prepared)
            comfy_prefix = "diffusion_model.transformer_blocks.0.attn.to_q"
            self.assertEqual(
                set(state),
                {
                    comfy_prefix + ".lora_A.weight",
                    comfy_prefix + ".lora_B.weight",
                    comfy_prefix + ".dora_scale",
                },
            )
            self.assertEqual(state[comfy_prefix + ".dora_scale"].shape, (4, 1))

    def test_uses_already_compatible_comfy_dora_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_root = Path(tmp_dir) / "loras"
            source = lora_root / "ready.safetensors"
            lora_root.mkdir()
            prefix = "diffusion_model.transformer_blocks.0.attn.to_q"
            save_file(
                {
                    prefix + ".lora_A.weight": torch.ones(2, 4),
                    prefix + ".lora_B.weight": torch.ones(4, 2),
                    prefix + ".dora_scale": torch.ones(4, 1),
                },
                str(source),
            )

            prepared = prepare_qwen_image_2_inference_dora(
                "ready.safetensors",
                str(Path(tmp_dir) / "cache"),
                search_paths=[lora_root],
            )

            self.assertEqual(Path(prepared), source.resolve())

    def test_plain_lora_stays_on_vanilla_comfy_loader(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_root = Path(tmp_dir) / "loras"
            source = lora_root / "plain.safetensors"
            lora_root.mkdir()
            save_file(
                {
                    "diffusion_model.block.lora_A.weight": torch.ones(2, 4),
                    "diffusion_model.block.lora_B.weight": torch.ones(4, 2),
                },
                str(source),
            )

            prepared = prepare_qwen_image_2_inference_dora(
                "plain.safetensors",
                str(Path(tmp_dir) / "cache"),
                search_paths=[lora_root],
            )

            self.assertIsNone(prepared)


if __name__ == "__main__":
    unittest.main()
