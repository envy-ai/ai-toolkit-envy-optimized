"""Focused CPU checks for quantized checkpoint weight descriptors/decoders."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from toolkit.util.checkpoint_weights import (
    checkpoint_input_scale,
    decode_checkpoint_weight,
    describe_checkpoint_weight,
)


class CheckpointWeightCodecTests(unittest.TestCase):
    def setUp(self):
        fixture_root = Path(__file__).resolve().parents[1] / "tmp" / "lora-quant-bake"
        fixture_root.mkdir(parents=True, exist_ok=True)
        self._temp_root = tempfile.TemporaryDirectory(
            prefix="checkpoint-weights-", dir=fixture_root
        )
        self.root = Path(self._temp_root.name)

    def tearDown(self):
        self._temp_root.cleanup()

    @staticmethod
    def _module_with_layer(layer):
        model = torch.nn.Module()
        block = torch.nn.Module()
        model.add_module("block", block)
        block.add_module("proj", layer)
        return model

    @staticmethod
    def _write_marker(entries, config, key="block.proj.comfy_quant"):
        entries[key] = torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)

    def _decode(self, entries, *, metadata=None, expected=None, key="block.proj.weight"):
        path = self.root / f"fixture-{len(list(self.root.iterdir()))}.safetensors"
        save_file(entries, path, metadata=metadata)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            descriptor = describe_checkpoint_weight(handle, key)
            decoded = decode_checkpoint_weight(handle, descriptor)
            input_scale = checkpoint_input_scale(handle, descriptor)
        if expected is not None:
            torch.testing.assert_close(decoded, expected.float(), rtol=1e-5, atol=1e-5)
        return descriptor, decoded, input_scale, path

    def test_w4a8_decoder_matches_quantizer_across_optional_payload_variants(self):
        from toolkit.util.comfy_quant_export import export_comfy_quantized_layers
        from toolkit.util.comfy_w4a8_quant import ComfyW4A8Quantizer
        from toolkit.util.ostris_quant import OstrisLinear

        torch.manual_seed(4201)
        cases = [
            # W4A8 supports FP8 relative scales without a codebook and no correction.
            (torch.float8_e4m3fn, False, False),
            # Also exercise the FP32 relative-scale path without optional tensors.
            (torch.float32, False, False),
            # Published codebook and group correction payloads.
            (torch.float32, True, True),
        ]
        for case_index, (relative_dtype, with_codebook, with_correction) in enumerate(cases):
            with self.subTest(relative_dtype=relative_dtype, codebook=with_codebook,
                              correction=with_correction):
                module = torch.nn.Linear(64, 8, bias=False)
                module.__class__ = OstrisLinear
                module.ostris_orig_dtype = torch.float32
                quantizer = ComfyW4A8Quantizer()
                quantizer.qtype = "comfy_w4a8"
                packed = torch.randint(-128, 127, (8, 32), dtype=torch.int8)
                if relative_dtype == torch.float8_e4m3fn:
                    relative = (torch.rand(8, 4) * 0.2 + 0.05).to(relative_dtype)
                else:
                    relative = torch.rand(8, 4) * 0.2 + 0.05
                channel = torch.rand(8) * 0.4 + 0.1
                codebook = torch.linspace(-1.0, 1.0, 16) if with_codebook else None
                correction = torch.randn(4, 8) * 0.01 if with_correction else None
                quantizer.attach_(module, packed, relative, channel, codebook, correction,
                                  group_size=16, rotation=1)
                module.ostris_quantizer = quantizer
                exported, names, unsupported = export_comfy_quantized_layers(
                    self._module_with_layer(module)
                )
                self.assertEqual(names, ["block.proj"])
                self.assertEqual(unsupported, [])
                expected = quantizer.dequantize(module)
                descriptor, decoded, _, _ = self._decode(exported, expected=expected)
                self.assertEqual(descriptor.quant_format, "asym_w4a8_int8")
                self.assertEqual(descriptor.group_size, 16)
                self.assertEqual(descriptor.rotation, 1)
                self.assertEqual("block.proj.weight_codebook" in descriptor.auxiliary_keys,
                                 with_codebook)
                self.assertEqual("block.proj.weight_correction" in descriptor.auxiliary_keys,
                                 with_correction)
                self.assertEqual(tuple(decoded.shape), (8, 64))

    def test_awq_native_and_comfy_bake_deltas_follow_their_consumer_bases(self):
        from toolkit.util.comfy_quant_export import export_comfy_quantized_layers
        from toolkit.util.nvfp4_quant import Nvfp4Quantizer
        from toolkit.util.ostris_quant import convert_linear_to_ostris
        from toolkit.lora_trimming import _bake_dora_factors

        torch.manual_seed(4202)
        layer = torch.nn.Linear(16, 8, bias=False)
        quantizer = Nvfp4Quantizer()
        quantizer.qtype = "nvfp4"
        self.assertTrue(convert_linear_to_ostris(layer, quantizer))
        pre_scale = torch.linspace(0.65, 1.35, 16)
        layer.register_buffer("nv4_pre_scale", pre_scale.contiguous().view(torch.uint8), persistent=False)
        entries, names, unsupported = export_comfy_quantized_layers(self._module_with_layer(layer))
        self.assertEqual(names, ["block.proj"])
        self.assertEqual(unsupported, [])
        base_path = self.root / "awq-nvfp4.safetensors"
        save_file(entries, base_path)

        # More source rank than the full effective matrix rank, with alpha != rank.
        rank, alpha = 16, 7.0
        down = torch.randn(rank, 16) * 0.04
        up = torch.randn(8, rank) * 0.05
        magnitude = torch.linspace(0.7, 1.25, 8).reshape(8, 1)
        scale = alpha / rank
        unscaled_direction = up.double() @ down.double()
        stored_weight = quantizer.dequantize(layer).double()
        input_scale = pre_scale.double()
        raw_forward_weight = stored_weight * input_scale[None]

        for convention in ("toolkit", "comfy"):
            with self.subTest(convention=convention):
                with safe_open(str(base_path), framework="pt", device="cpu") as handle:
                    descriptor = describe_checkpoint_weight(handle, "block.proj.weight")
                    decoded_scale = checkpoint_input_scale(handle, descriptor)
                torch.testing.assert_close(decoded_scale, pre_scale)
                baked_down, baked_up, _, _, info, _ = _bake_dora_factors(
                    base_path, "block.proj.weight", down, up, magnitude, alpha,
                    convention, 16, descriptor, "float32"
                )
                norm_basis = stored_weight if convention == "comfy" else stored_weight
                norm_update = (scale * unscaled_direction if convention == "comfy"
                               else unscaled_direction)
                denominator = ((norm_basis + norm_update).norm(dim=1) if convention == "toolkit"
                               else norm_basis.norm(dim=1) + torch.finfo(torch.float32).eps)
                row_ratio = magnitude.double().reshape(-1) / denominator
                consumer_basis = stored_weight if convention == "comfy" else raw_forward_weight
                expected_delta = (consumer_basis + scale * unscaled_direction) * row_ratio[:, None] - consumer_basis
                output_scale = alpha / min(up.shape[0], down.shape[1])
                actual_delta = (baked_up.double() @ baked_down.double()) * output_scale
                torch.testing.assert_close(actual_delta, expected_delta, rtol=1e-8, atol=1e-8)
                self.assertEqual(int(torch.linalg.matrix_rank(expected_delta)), 8)
                self.assertEqual(info["output_basis"],
                                 "awq_stored_weight" if convention == "comfy" else "raw_input")
                self.assertEqual(info["normalization_update"],
                                 "scaled_update" if convention == "comfy" else "unscaled_BA")

    def test_invalid_packed_layouts_unknown_marker_fields_and_nonfinite_scales_fail(self):
        # Twelve stored columns represent 24 logical inputs, invalid for
        # 16-wide NVFP4 blocks.
        invalid_layout = {
            "block.proj.weight": torch.zeros(8, 12, dtype=torch.uint8),
            "block.proj.weight_scale": torch.zeros(512, dtype=torch.float8_e4m3fn),
            "block.proj.weight_scale_2": torch.tensor(1.0),
        }
        self._write_marker(invalid_layout, {"format": "nvfp4"})
        path = self.root / "invalid-layout.safetensors"
        save_file(invalid_layout, path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            with self.assertRaisesRegex(ValueError, "divisible by 16"):
                describe_checkpoint_weight(handle, "block.proj.weight")

        unknown = {
            "block.proj.weight": torch.zeros(8, 16, dtype=torch.float8_e4m3fn),
            "block.proj.weight_scale": torch.tensor(1.0),
        }
        self._write_marker(unknown, {"format": "float8_e4m3fn", "guessed_layout": True})
        path = self.root / "unknown-marker-field.safetensors"
        save_file(unknown, path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            with self.assertRaisesRegex(ValueError, "unsupported quantization format/config"):
                describe_checkpoint_weight(handle, "block.proj.weight")

        nonfinite = {
            "block.proj.weight": torch.zeros(8, 16, dtype=torch.float8_e4m3fn),
            "block.proj.weight_scale": torch.tensor(float("nan")),
        }
        self._write_marker(nonfinite, {"format": "float8_e4m3fn"})
        path = self.root / "nonfinite-scale.safetensors"
        save_file(nonfinite, path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            descriptor = describe_checkpoint_weight(handle, "block.proj.weight")
            with self.assertRaisesRegex(ValueError, "scales must be finite and strictly positive"):
                decode_checkpoint_weight(handle, descriptor)

    def test_awq_input_scale_must_be_finite_and_positive(self):
        entries = {
            "block.proj.weight": torch.zeros(8, 8, dtype=torch.uint8),
            "block.proj.weight_scale": torch.zeros(512, dtype=torch.float8_e4m3fn),
            "block.proj.weight_scale_2": torch.tensor(1.0),
            "block.proj.pre_quant_scale": torch.tensor(
                [1.0] * 7 + [0.0] + [1.0] * 8
            ),
        }
        self._write_marker(entries, {"format": "nvfp4"})
        path = self.root / "bad-awq-scale.safetensors"
        save_file(entries, path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            descriptor = describe_checkpoint_weight(handle, "block.proj.weight")
            with self.assertRaisesRegex(ValueError, "AWQ input scales must be finite and positive"):
                checkpoint_input_scale(handle, descriptor)


if __name__ == "__main__":
    unittest.main()
