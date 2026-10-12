"""Focused CPU contract checks for the isolated ConvRot8 XLA backend."""

import builtins
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from toolkit.util import convrot_xla
from toolkit.util.convrot_quant import ConvRotInt8Quantizer, quantize_int8_rows, rotate
from workers.kaggle_tpu import probe as kaggle_tpu_probe


class ConvRotXlaContractTests(unittest.TestCase):
    def _quantized_linear(self, in_features=16, out_features=8, bias=True):
        module = torch.nn.Linear(in_features, out_features, bias=bias)
        quantizer = ConvRotInt8Quantizer(rot_size=16)
        quantizer.quantize_(module, module.weight.detach().clone())
        return module, quantizer

    def test_scale_preparation_decodes_cpu_byte_view_without_changing_checkpoint_buffer(self):
        module, _ = self._quantized_linear()
        original_bytes = module.cr8_scales.clone()
        expected = original_bytes.contiguous().view(torch.float32).clone()
        root = torch.nn.Sequential(module)

        self.assertEqual(convrot_xla.prepare_convrot8_scales_for_xla_(root), 1)
        torch.testing.assert_close(module.cr8_xla_scales, expected)
        self.assertEqual(module.cr8_xla_scales.dtype, torch.float32)
        self.assertEqual(module.cr8_xla_scales.device.type, "cpu")
        self.assertEqual(module.cr8_scales.dtype, torch.uint8)
        torch.testing.assert_close(module.cr8_scales, original_bytes)
        self.assertNotIn("0.cr8_xla_scales", root.state_dict())

    def test_scale_preparation_rejects_invalid_scale_values_and_layout(self):
        for values in (
            torch.tensor([0.0, 1.0]).view(torch.uint8),
            torch.tensor([-1.0, 1.0]).view(torch.uint8),
            torch.tensor([float("nan"), 1.0]).view(torch.uint8),
            torch.tensor([float("inf"), 1.0]).view(torch.uint8),
        ):
            with self.subTest(values=values):
                module = torch.nn.Linear(16, 2)
                module.register_buffer("cr8_qdata", torch.zeros(2, 16, dtype=torch.int8))
                module.register_buffer("cr8_scales", values.clone())
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    convrot_xla.prepare_convrot8_scales_for_xla_(module)

        module = torch.nn.Linear(16, 2)
        module.register_buffer("cr8_qdata", torch.zeros(2, 16, dtype=torch.int8))
        module.register_buffer("cr8_scales", torch.zeros(7, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "do not match"):
            convrot_xla.prepare_convrot8_scales_for_xla_(module)

    def test_scale_preparation_requires_unpacked_int8_codes_and_cpu_byte_view(self):
        module = torch.nn.Linear(16, 2)
        module.register_buffer("cr8_qdata", torch.zeros(2, 16, dtype=torch.uint8))
        module.register_buffer("cr8_scales", torch.ones(2, dtype=torch.float32).view(torch.uint8))
        with self.assertRaisesRegex(TypeError, "unpacked int8"):
            convrot_xla.prepare_convrot8_scales_for_xla_(module)

        module.cr8_qdata = torch.zeros(2, 16, dtype=torch.int8)
        module.cr8_scales = torch.ones(2, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "CPU byte-view"):
            convrot_xla.prepare_convrot8_scales_for_xla_(module)

    def test_import_does_not_attempt_to_load_torch_xla(self):
        real_import = builtins.__import__

        def reject_torch_xla(name, *args, **kwargs):
            if name == "torch_xla" or name.startswith("torch_xla."):
                raise AssertionError("torch_xla imported during module import")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_torch_xla):
            importlib.reload(convrot_xla)

    def test_int8_dot_calls_xla_adapter_with_int32_accumulation(self):
        calls = []

        def dot(a, b, dims, *, preferred_element_type):
            calls.append((a, b, dims, preferred_element_type))
            return torch.ones((2, 3), dtype=torch.int32)

        fake_xla = types.SimpleNamespace(_XLAC=types.SimpleNamespace(_xla_dot_general=dot))
        with mock.patch.dict(sys.modules, {"torch_xla": fake_xla}):
            result = convrot_xla._int8_dot(
                torch.zeros(2, 16, dtype=torch.int8),
                torch.zeros(3, 16, dtype=torch.int8),
            )
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], (([-1], [-1]), ()))
        self.assertIs(calls[0][3], torch.int32)

    def test_int8_dot_reports_missing_adapter_and_wrong_result_dtype(self):
        a = torch.zeros(2, 16, dtype=torch.int8)
        b = torch.zeros(3, 16, dtype=torch.int8)
        with mock.patch.dict(sys.modules, {"torch_xla": None}):
            with self.assertRaisesRegex(RuntimeError, "matching torch_xla wheel"):
                convrot_xla._int8_dot(a, b)

        with mock.patch.dict(sys.modules, {"torch_xla": types.SimpleNamespace(_XLAC=object())}):
            with self.assertRaisesRegex(RuntimeError, "lacks _XLAC._xla_dot_general"):
                convrot_xla._int8_dot(a, b)

        wrong_dtype = types.SimpleNamespace(
            _XLAC=types.SimpleNamespace(_xla_dot_general=lambda *args, **kwargs: torch.ones(2, 3))
        )
        with mock.patch.dict(sys.modules, {"torch_xla": wrong_dtype}):
            with self.assertRaisesRegex(RuntimeError, "returned torch.float32, expected torch.int32"):
                convrot_xla._int8_dot(a, b)

    def test_ste_backward_uses_dequantized_codes_and_row_scales(self):
        qdata = torch.tensor([[1, -2, 3], [-4, 5, -6]], dtype=torch.int8)
        scales = torch.tensor([0.25, 0.5], dtype=torch.float32)
        grad = torch.tensor([[2.0, -1.0], [0.5, 3.0]])

        class FakeContext:
            def save_for_backward(self, *tensors):
                self.saved_tensors = tensors

        ctx = FakeContext()
        ctx.saved_tensors = (qdata, scales)
        result = convrot_xla._ConvRot8LinearSTE.backward(ctx, grad)
        expected_weight = qdata.float() * scales.unsqueeze(1)
        torch.testing.assert_close(result[0], grad @ expected_weight)
        self.assertEqual(result[1:], (None, None, None, None, None))

    def test_xla_backend_rejects_global_bf16_downcast_environment(self):
        for name in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16"):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: "1"}):
                with self.assertRaisesRegex(RuntimeError, name):
                    convrot_xla._reject_global_bf16_downcast()

    def test_probe_aggregates_all_rank_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "probe"

            def launch(_rank_fn, args):
                out = Path(args[0])
                for rank in range(2):
                    (out / f"rank-{rank}.json").write_text(
                        json.dumps({
                            "rank": rank,
                            "status": "pass",
                            "world_size": 2,
                            "global_logical_device_count": 2,
                        }),
                        encoding="utf-8",
                    )

            fake_xla = types.SimpleNamespace(launch=launch)
            with (
                mock.patch.object(sys, "argv", ["probe.py", "--output", str(report_dir)]),
                mock.patch.object(kaggle_tpu_probe, "preflight", return_value={"pjrt_device": "TPU"}),
                mock.patch.dict(sys.modules, {"torch_xla": fake_xla}),
            ):
                kaggle_tpu_probe.main()

            report = json.loads((report_dir / "diagnostic.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["expected_logical_devices"], 2)
            self.assertEqual(report["completed_logical_devices"], 2)
            self.assertEqual({rank["rank"] for rank in report["ranks"]}, {0, 1})

    def test_probe_rejects_partial_rank_reports_and_nonempty_rerun_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "partial"

            def launch(_rank_fn, args):
                out = Path(args[0])
                (out / "rank-0.json").write_text(
                    json.dumps({
                        "rank": 0,
                        "status": "pass",
                        "world_size": 2,
                        "global_logical_device_count": 2,
                    }),
                    encoding="utf-8",
                )

            fake_xla = types.SimpleNamespace(launch=launch)
            with (
                mock.patch.object(sys, "argv", ["probe.py", "--output", str(report_dir)]),
                mock.patch.object(kaggle_tpu_probe, "preflight", return_value={"pjrt_device": "TPU"}),
                mock.patch.dict(sys.modules, {"torch_xla": fake_xla}),
            ):
                with self.assertRaisesRegex(RuntimeError, "Only 1/2 logical devices"):
                    kaggle_tpu_probe.main()

            report = json.loads((report_dir / "diagnostic.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["completed_logical_devices"], 1)

            with (
                mock.patch.object(sys, "argv", ["probe.py", "--output", str(report_dir)]),
                mock.patch.object(kaggle_tpu_probe, "preflight", side_effect=AssertionError("must not rerun")),
                mock.patch.dict(sys.modules, {"torch_xla": fake_xla}),
            ):
                with self.assertRaisesRegex(RuntimeError, "must be new and empty"):
                    kaggle_tpu_probe.main()

    def test_xla_tensor_dispatch_is_limited_to_exact_int8_quantizer(self):
        class DeviceProxy:
            type = "xla"

        class TensorProxy:
            device = DeviceProxy()

            def numel(self):
                return 16

        module, quantizer = self._quantized_linear(in_features=16, out_features=8)
        x = TensorProxy()
        marker = torch.empty(0)
        with mock.patch.object(convrot_xla, "convrot8_linear_xla", return_value=marker) as dispatch:
            self.assertIs(quantizer.forward(module, x), marker)
        args = dispatch.call_args.args
        self.assertEqual(args[:4], (module, x, 16, quantizer.act_qmax))
        self.assertIs(args[4], rotate)
        self.assertIs(args[5], quantize_int8_rows)

        from toolkit.util.convrot_quant import ConvRotIntNQuantizer

        packed_quantizer = ConvRotIntNQuantizer(4, rot_size=16)
        module.crn_rot_size = 16
        with self.assertRaisesRegex(NotImplementedError, "frozen ConvRot8"):
            packed_quantizer.forward(module, x)

    def test_cpu_inference_path_keeps_existing_fake_quant_linear_behavior(self):
        module, quantizer = self._quantized_linear(in_features=16, out_features=8)
        x = torch.randn(3, 16)
        rot = module.cr8_rot_size
        x_rot = rotate(x, rot).reshape(-1, 16)
        weight_rot = module.cr8_qdata.float() * module.cr8_scales.view(torch.float32).unsqueeze(1)
        # CPU lacks the int8 GEMM path; preserve its existing W8A16 fallback.
        expected = torch.nn.functional.linear(x_rot, weight_rot.to(x.dtype), module.bias)
        expected = expected.reshape(3, 8)

        with torch.no_grad():
            actual = quantizer.forward(module, x)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
