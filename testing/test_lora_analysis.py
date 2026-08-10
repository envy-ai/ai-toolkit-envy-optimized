import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from safetensors.torch import save_file

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.lora_analysis import analyze_lora, classify_layer, format_analysis_text


class LoraAnalysisTests(unittest.TestCase):
    def test_groups_layers_and_ranks_relative_changes_against_quantized_base(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.safetensors"
            lora_path = root / "adapter.safetensors"
            save_file(
                {
                    "blocks.0.attn.q_proj.weight": torch.ones((2, 2), dtype=torch.int8),
                    "blocks.0.attn.q_proj.weight_scale": torch.full((2, 1), 0.5),
                    "blocks.1.attn.q_proj.weight": torch.ones((2, 2), dtype=torch.int8),
                    "blocks.1.attn.q_proj.weight_scale": torch.full((2, 1), 0.5),
                    "blocks.0.mlp.fc2.weight": torch.eye(2),
                },
                base_path,
            )
            save_file(
                {
                    "diffusion_model.blocks.0.attn.q_proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
                    "diffusion_model.blocks.0.attn.q_proj.lora_B.weight": torch.tensor([[1.0], [2.0]]),
                    "diffusion_model.blocks.1.attn.q_proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
                    "diffusion_model.blocks.1.attn.q_proj.lora_B.weight": torch.tensor([[0.25], [0.5]]),
                    "diffusion_model.blocks.0.mlp.fc2.lora_A.weight": torch.tensor([[0.0, 1.0]]),
                    "diffusion_model.blocks.0.mlp.fc2.lora_B.weight": torch.tensor([[0.5], [0.0]]),
                },
                lora_path,
                metadata={"ss_base_model": str(base_path)},
            )

            result = analyze_lora(lora_path)

            self.assertEqual(result["base_model"]["matched_layers"], 3)
            self.assertEqual(set(result["groups"]), {"attention_query", "mlp_output"})
            queries = result["groups"]["attention_query"]
            self.assertTrue(queries[0]["layer"].startswith("diffusion_model.blocks.0"))
            self.assertAlmostEqual(queries[0]["delta_fro_norm"], math.sqrt(5), places=6)
            self.assertAlmostEqual(queries[0]["base_fro_norm"], 1.0, places=6)
            self.assertAlmostEqual(queries[0]["relative_change"], math.sqrt(5), places=6)

    def test_uses_adjacent_training_config_for_base_and_lora_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.safetensors"
            lora_path = root / "adapter.safetensors"
            save_file({"blocks.0.attn.out_proj.weight": torch.eye(2)}, base_path)
            save_file(
                {
                    "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": torch.tensor([
                        [1.0, 0.0],
                        [0.0, 0.0],
                    ]),
                    "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": torch.tensor([
                        [2.0, 0.0],
                        [0.0, 0.0],
                    ]),
                },
                lora_path,
                metadata={"ss_base_model_version": "synthetic"},
            )
            (root / "config.yaml").write_text(yaml.safe_dump({
                "config": {
                    "process": [{
                        "type": "diffusion_trainer",
                        "model": {"name_or_path": "base.safetensors", "arch": "synthetic"},
                        "network": {"type": "lora", "linear": 2, "linear_alpha": 1},
                    }]
                }
            }))

            result = analyze_lora(lora_path)
            record = result["groups"]["attention_output"][0]

            self.assertEqual(result["base_model"]["reference"], "base.safetensors")
            self.assertEqual(record["scale"], 0.5)
            self.assertAlmostEqual(record["delta_fro_norm"], 1.0, places=6)

    def test_dora_uses_magnitude_vector_when_base_is_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.safetensors"
            lora_path = root / "adapter.safetensors"
            save_file({"blocks.0.mlp.fc1.weight": torch.eye(2)}, base_path)
            save_file(
                {
                    "diffusion_model.blocks.0.mlp.fc1.lora_A.weight": torch.tensor([[1.0, 0.0]]),
                    "diffusion_model.blocks.0.mlp.fc1.lora_B.weight": torch.zeros((2, 1)),
                    "diffusion_model.blocks.0.mlp.fc1.magnitude": torch.tensor([2.0, 1.0]),
                },
                lora_path,
                metadata={"ss_base_model": str(base_path)},
            )

            record = analyze_lora(lora_path)["groups"]["mlp_input_or_gate"][0]

            self.assertTrue(record["is_dora"])
            self.assertEqual(record["delta_fro_norm"], 0.0)
            self.assertAlmostEqual(record["effective_delta_fro_norm"], 1.0, places=6)
            self.assertAlmostEqual(record["relative_change"], 1 / math.sqrt(2), places=6)

    def test_no_base_uses_update_rms_and_formats_grouped_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_path = Path(temp_dir) / "adapter.safetensors"
            save_file(
                {
                    "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.ones((1, 2)),
                    "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.ones((3, 1)),
                },
                lora_path,
            )

            result = analyze_lora(lora_path, use_base=False)
            report = format_analysis_text(result)
            encoded = json.dumps(result)

            self.assertIn("[attention_qkv]", report)
            self.assertIn("Base comparison: unavailable", report)
            self.assertIn("effective_delta_rms", encoded)

    def test_layer_classifier_keeps_functionally_different_layers_separate(self):
        self.assertEqual(classify_layer("x.attn.q_proj", (4, 8)), "attention_query")
        self.assertEqual(classify_layer("x.attn.out_proj", (4, 8)), "attention_output")
        self.assertEqual(classify_layer("x.mlp.fc1", (4, 8)), "mlp_input_or_gate")
        self.assertEqual(classify_layer("x.mlp.fc2", (4, 8)), "mlp_output")
        self.assertEqual(classify_layer("x.txt_mlp.net.0", (4, 8)), "mlp_input_or_gate")
        self.assertEqual(classify_layer("x.img_mlp.net.2", (4, 8)), "mlp_output")
        self.assertEqual(
            classify_layer("x.transformer.img_mod.1", (4, 8)),
            "conditioning_modulation",
        )
        self.assertEqual(classify_layer("x.proj", (4, 8, 3, 3)), "convolution")

    def test_future_training_metadata_records_exact_base_model(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.meta = {}
        process.model_config = SimpleNamespace(name_or_path="Comfy-Org/MiniMax-H3")
        process.sd = SimpleNamespace(get_base_model_version=lambda: "minimax_h3")
        process.job = SimpleNamespace(name="test")
        process.trigger_word = None
        process.step_num = 10
        process.epoch_num = 2

        process.update_training_metadata()

        self.assertEqual(process.meta["ss_base_model"], "Comfy-Org/MiniMax-H3")
        self.assertEqual(process.meta["ss_base_model_version"], "minimax_h3")


if __name__ == "__main__":
    unittest.main()
