import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolkit.util.quantize import (
    move_nucleus_moe_quantized_weights,
    quantize_nucleus_moe_experts,
)


class SwiGLUExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 3
        self.gate_up_proj = nn.Parameter(torch.randn(3, 4, 10, dtype=torch.float32))
        self.down_proj = nn.Parameter(torch.randn(3, 5, 4, dtype=torch.float32))
        self.use_grouped_mm = True

    def _run_experts_for_loop(self, x, num_tokens_per_expert):
        chunks = torch.split(x, num_tokens_per_expert.tolist(), dim=0)
        outputs = []
        for expert_idx, chunk in enumerate(chunks):
            gate_up = torch.matmul(chunk, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            outputs.append(torch.matmul(F.silu(gate) * up, self.down_proj[expert_idx]))
        return torch.cat(outputs, dim=0)

    def forward(self, x, num_tokens_per_expert):
        return self._run_experts_for_loop(x, num_tokens_per_expert)


class NucleusMoEQuantizeTests(unittest.TestCase):
    def test_quantizes_raw_expert_parameters_without_cuda_residency(self):
        experts = SwiGLUExperts()
        x = torch.randn(9, 4)
        tokens_per_expert = torch.tensor([2, 3, 4], dtype=torch.long)
        expected = experts(x, tokens_per_expert)
        original_bytes = (
            experts.gate_up_proj.numel() * experts.gate_up_proj.element_size()
            + experts.down_proj.numel() * experts.down_proj.element_size()
        )

        quantized_count = quantize_nucleus_moe_experts(experts)
        actual = experts(x, tokens_per_expert)

        self.assertEqual(quantized_count, 1)
        self.assertFalse(isinstance(experts.gate_up_proj, nn.Parameter))
        self.assertFalse(isinstance(experts.down_proj, nn.Parameter))
        self.assertNotIn("gate_up_proj", dict(experts.named_parameters()))
        self.assertNotIn("down_proj", dict(experts.named_parameters()))
        self.assertLess(
            experts.gate_up_proj.storage_nbytes() + experts.down_proj.storage_nbytes(),
            original_bytes,
        )
        self.assertFalse(experts.use_grouped_mm)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=0.12, atol=0.12)

    def test_equal_token_experts_use_batched_quantized_path(self):
        experts = SwiGLUExperts()
        x = torch.randn(9, 4)
        tokens_per_expert = torch.tensor([3, 3, 3], dtype=torch.long)
        expected = experts(x, tokens_per_expert)

        quantize_nucleus_moe_experts(experts)

        dequantize_calls = {"gate_up": 0, "down": 0}
        orig_gate_up_dequantize = experts.gate_up_proj.dequantize
        orig_down_dequantize = experts.down_proj.dequantize

        def gate_up_dequantize(*args, **kwargs):
            dequantize_calls["gate_up"] += 1
            return orig_gate_up_dequantize(*args, **kwargs)

        def down_dequantize(*args, **kwargs):
            dequantize_calls["down"] += 1
            return orig_down_dequantize(*args, **kwargs)

        experts.gate_up_proj.dequantize = gate_up_dequantize
        experts.down_proj.dequantize = down_dequantize

        actual = experts(x, tokens_per_expert)

        self.assertEqual(dequantize_calls, {"gate_up": 1, "down": 1})
        torch.testing.assert_close(actual, expected, rtol=0.12, atol=0.12)

    def test_quantized_expert_storage_follows_module_to_without_expanding(self):
        root = nn.Sequential(SwiGLUExperts())
        experts = root[0]

        quantize_nucleus_moe_experts(root)
        root.to(dtype=torch.float16)

        self.assertFalse(experts.gate_up_proj.keep_on_cpu)
        self.assertFalse(experts.down_proj.keep_on_cpu)
        self.assertEqual(experts.gate_up_proj.qweight.dtype, torch.int8)
        self.assertEqual(experts.down_proj.qweight.dtype, torch.int8)
        self.assertEqual(experts.gate_up_proj.scale.dtype, torch.float16)
        self.assertEqual(experts.down_proj.scale.dtype, torch.float16)

        if torch.cuda.is_available():
            root.to("cuda")
            self.assertEqual(experts.gate_up_proj.qweight.device.type, "cuda")
            self.assertEqual(experts.down_proj.qweight.device.type, "cuda")

    def test_low_vram_quantized_expert_storage_stays_cpu_backed(self):
        root = nn.Sequential(SwiGLUExperts())
        experts = root[0]

        quantize_nucleus_moe_experts(root, keep_on_cpu=True)
        root.to(dtype=torch.float16)

        self.assertTrue(experts.gate_up_proj.keep_on_cpu)
        self.assertTrue(experts.down_proj.keep_on_cpu)
        self.assertEqual(experts.gate_up_proj.qweight.dtype, torch.int8)
        self.assertEqual(experts.down_proj.qweight.dtype, torch.int8)
        self.assertEqual(experts.gate_up_proj.qweight.device.type, "cpu")
        self.assertEqual(experts.down_proj.qweight.device.type, "cpu")
        self.assertEqual(experts.gate_up_proj.scale.dtype, torch.float16)
        self.assertEqual(experts.down_proj.scale.dtype, torch.float16)
        self.assertEqual(experts.gate_up_proj.scale.device.type, "cpu")
        self.assertEqual(experts.down_proj.scale.device.type, "cpu")

        if torch.cuda.is_available():
            root.to("cuda")
            self.assertEqual(experts.gate_up_proj.qweight.device.type, "cpu")
            self.assertEqual(experts.down_proj.qweight.device.type, "cpu")
            self.assertEqual(experts.gate_up_proj.scale.device.type, "cpu")
            self.assertEqual(experts.down_proj.scale.device.type, "cpu")

            x = torch.randn(9, 4, device="cuda", dtype=torch.float16)
            tokens_per_expert = torch.tensor([2, 3, 4], device="cuda", dtype=torch.long)
            actual = experts(x, tokens_per_expert)
            self.assertEqual(actual.device.type, "cuda")
            self.assertEqual(actual.shape, (9, 4))

    def test_cpu_backed_expert_storage_can_be_promoted_after_setup(self):
        root = nn.Sequential(SwiGLUExperts())
        experts = root[0]

        quantize_nucleus_moe_experts(root, keep_on_cpu=True)
        root.to(dtype=torch.float16)

        moved = move_nucleus_moe_quantized_weights(
            root,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(moved, 2)
        self.assertFalse(experts.gate_up_proj.keep_on_cpu)
        self.assertFalse(experts.down_proj.keep_on_cpu)
        self.assertEqual(experts.gate_up_proj.qweight.device.type, "cpu")
        self.assertEqual(experts.down_proj.qweight.device.type, "cpu")
        self.assertEqual(experts.gate_up_proj.scale.dtype, torch.float32)
        self.assertEqual(experts.down_proj.scale.dtype, torch.float32)

        root.to(dtype=torch.float16)
        self.assertEqual(experts.gate_up_proj.scale.dtype, torch.float16)
        self.assertEqual(experts.down_proj.scale.dtype, torch.float16)

        if torch.cuda.is_available():
            moved = move_nucleus_moe_quantized_weights(root, torch.device("cuda"))
            self.assertEqual(moved, 2)
            self.assertEqual(experts.gate_up_proj.qweight.device.type, "cuda")
            self.assertEqual(experts.down_proj.qweight.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
