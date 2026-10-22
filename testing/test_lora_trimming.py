import hashlib
import json
import tempfile
import threading
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save, save_file

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.lora_trimming import trim_lora, trim_lora_factors
from toolkit.models.DoRA import DoRAModule
from toolkit.models.loha import LoHaModule
from toolkit.metadata import add_model_hash_to_meta
from toolkit.util.comfy_quant_export import comfy_quant_marker, export_comfy_quantized_layers
from toolkit.util.convrot_quant import quantize_int8_rows
from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "tmp"


class FakeTransformer(torch.nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"proj": torch.nn.Linear(width, width, bias=False)})]
        )


class FakeBaseModel:
    arch = "fake_transformer"
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return ["transformer_blocks"]

    @staticmethod
    def convert_lora_weights_before_save(state_dict):
        return {
            key.replace("transformer.", "diffusion_model."): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def convert_lora_weights_before_load(state_dict):
        return {
            key.replace("diffusion_model.", "transformer."): value
            for key, value in state_dict.items()
        }


def make_network(rank, alpha=6, width=64, peft_format=True, network_type="lora"):
    base_model = FakeBaseModel()
    network = LoRASpecialNetwork(
        text_encoder=[],
        unet=FakeTransformer(width),
        lora_dim=rank,
        alpha=alpha,
        train_text_encoder=False,
        train_unet=True,
        target_lin_modules=["FakeTransformer"],
        network_config=NetworkConfig(linear=rank, linear_alpha=alpha),
        is_transformer=True,
        peft_format=peft_format,
        base_model=base_model,
        network_type=network_type,
    )
    # The network stores its base model through a weak reference.
    network._test_base_model = base_model
    network.apply_to(None, None, False, True)
    return network


def make_spectral_factors(dtype=torch.float32, rank=8, width=64):
    # The exact update has spectrum [9, 4, 0.25, ...], so rank two meets a
    # 0.07 relative Frobenius error while rank one does not. Larger dimensions
    # also make parameter savings dominate provenance metadata in the file.
    singular_values = torch.tensor([9.0, 4.0] + [0.25] + [0.01] * (rank - 3), dtype=torch.float64)
    roots = singular_values.sqrt().to(dtype)
    down = torch.zeros((rank, width), dtype=dtype)
    up = torch.zeros((width, rank), dtype=dtype)
    for index, value in enumerate(roots):
        down[index, index] = value
        up[index, index] = value
    return down, up


class LoraTrimmingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=ARTIFACT_ROOT)

    @staticmethod
    def _active_network():
        class ActiveNetwork:
            is_active = True
            is_merged_in = False
            is_lorm = False
            _multiplier = 1.0

            def __init__(self):
                self.torch_multiplier = torch.ones(1)

        return ActiveNetwork()

    def test_factorized_trim_meets_error_and_preserves_alpha_scaled_update(self):
        down, up = make_spectral_factors(rank=3, width=4)
        alpha = 6.0
        new_down, new_up, metrics = trim_lora_factors(
            down,
            up,
            alpha=alpha,
            relative_error=0.07,
            min_rank=1,
            max_rank=2,
        )

        old_scaled = (up @ down) * (alpha / down.shape[0])
        new_scaled = (new_up @ new_down) * (alpha / new_down.shape[0])
        measured_error = torch.linalg.vector_norm(old_scaled - new_scaled) / torch.linalg.vector_norm(old_scaled)

        self.assertEqual(new_down.shape[0], 2)
        self.assertEqual(new_up.shape[1], 2)
        self.assertLessEqual(float(measured_error), 0.07)
        self.assertAlmostEqual(metrics["relative_error"], float(measured_error), places=6)
        self.assertAlmostEqual(metrics["old_scale"], alpha / 3, places=7)
        self.assertAlmostEqual(metrics["new_scale"], alpha / 2, places=7)
        self.assertEqual(new_down.dtype, down.dtype)
        self.assertEqual(new_up.dtype, up.dtype)

    def test_supported_storage_dtypes_and_mixed_factor_dtypes_measure_written_values(self):
        cases = [
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float32),
            (torch.float64, torch.float64),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float64),
        ]
        for down_dtype, up_dtype in cases:
            with self.subTest(down=down_dtype, up=up_dtype):
                down, up = make_spectral_factors(dtype=torch.float32, rank=4, width=8)
                down = down.to(down_dtype)
                up = up.to(up_dtype)
                alpha = 5.25
                new_down, new_up, metrics = trim_lora_factors(
                    down, up, alpha=alpha, relative_error=0.04, max_rank=2
                )

                old_update = (up.double() @ down.double()) * (alpha / down.shape[0])
                actual_new_update = (new_up.double() @ new_down.double()) * (alpha / new_down.shape[0])
                measured_error = torch.linalg.vector_norm(old_update - actual_new_update) / torch.linalg.vector_norm(old_update)
                self.assertEqual(new_down.dtype, down_dtype)
                self.assertEqual(new_up.dtype, up_dtype)
                self.assertEqual(new_down.shape[0], 2)
                self.assertLessEqual(float(measured_error), 0.04)
                self.assertAlmostEqual(metrics["relative_error"], float(measured_error), places=7)
                self.assertAlmostEqual(metrics["old_scale"], alpha / 4, places=7)
                self.assertAlmostEqual(metrics["new_scale"], alpha / 2, places=7)

    def test_rank_limit_that_cannot_meet_requested_error_is_rejected(self):
        down, up = make_spectral_factors()
        with self.assertRaises(ValueError):
            trim_lora_factors(
                down,
                up,
                relative_error=0.01,
                max_rank=1,
            )

    def test_invalid_thresholds_ranks_and_scaling_are_rejected(self):
        down, up = make_spectral_factors()
        invalid_options = [
            {"retained_energy": 0.9, "relative_error": 0.1},
            {"retained_energy": 0.0},
            {"retained_energy": 1.1},
            {"relative_error": -0.1},
            {"relative_error": 1.1},
            {"min_rank": 0},
            {"max_rank": 9},
            {"min_rank": 3, "max_rank": 2},
            {"alpha": 0.0},
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                trim_lora_factors(down, up, **options)

    def test_zero_error_and_retained_energy_select_ranks_without_losing_scale(self):
        down, up = make_spectral_factors(rank=3, width=4)
        exact_down, exact_up, exact_metrics = trim_lora_factors(
            down,
            up,
            alpha=6.0,
            relative_error=0.0,
        )
        self.assertEqual(exact_down.shape[0], 3)
        self.assertTrue(torch.equal(exact_down, down))
        self.assertTrue(torch.equal(exact_up, up))
        self.assertEqual(exact_metrics["relative_error"], 0.0)

        energy_down, energy_up, energy_metrics = trim_lora_factors(
            down,
            up,
            alpha=None,
            retained_energy=0.98,
            min_rank=1,
            max_rank=3,
        )
        self.assertEqual(energy_down.shape[0], 2)
        self.assertEqual(energy_up.shape[1], 2)
        self.assertGreaterEqual(energy_metrics["retained_energy"], 0.98)
        self.assertEqual(energy_metrics["old_scale"], 1.0)
        self.assertEqual(energy_metrics["new_scale"], 1.0)

    def test_explicit_rank_policies_select_expected_spectrum_ranks(self):
        # The update has known singular values [9, 4, 1, .25]. These cases
        # distinguish kohya's sum-of-singular-values and Frobenius policies.
        down = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5]))
        up = down.clone()
        cases = [
            ("fixed", 2, 2),
            ("sv_ratio", 2.25, 1),  # strict > largest / ratio
            ("sv_ratio", 1.0, 1),
            ("sv_cumulative", 0.9, 2),  # 13 / 14.25 >= .9
            ("sv_fro", 0.98, 2),  # sqrt(97 / 98.0625) >= .98
            ("energy", 0.98, 2),
            ("relative_error", 0.11, 2),
        ]
        for policy, value, expected_rank in cases:
            with self.subTest(policy=policy, value=value):
                reduced_down, reduced_up, metrics = trim_lora_factors(
                    down, up, rank_policy=policy, rank_value=value
                )
                self.assertEqual(reduced_down.shape[0], expected_rank)
                self.assertEqual(reduced_up.shape[1], expected_rank)
                self.assertEqual(metrics["rank_policy"], policy)
                self.assertEqual(metrics["rank_value"], value)
                original = up.double() @ down.double()
                actual = reduced_up.double() @ reduced_down.double()
                measured_error = torch.linalg.vector_norm(original - actual) / torch.linalg.vector_norm(original)
                self.assertAlmostEqual(metrics["relative_error"], float(measured_error), places=6)
                self.assertLessEqual(metrics["new_bytes"], metrics["old_bytes"])

    def test_explicit_rank_policy_caps_are_not_promoted_to_error_guarantees(self):
        down = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5]))
        up = down.clone()
        reduced_down, _, metrics = trim_lora_factors(
            down, up, rank_policy="sv_ratio", rank_value=1.1, max_rank=1
        )
        self.assertEqual(reduced_down.shape[0], 1)
        self.assertGreater(metrics["relative_error"], 0.1)

        with self.assertRaisesRegex(ValueError, "not achievable"):
            trim_lora_factors(
                down, up, rank_policy="relative_error", rank_value=0.01, max_rank=1
            )

    def test_explicit_rank_policy_rejects_invalid_or_mixed_options(self):
        down = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5]))
        invalid = [
            {"rank_policy": "unknown", "rank_value": 0.5},
            {"rank_policy": "fixed", "rank_value": 1.5},
            {"rank_policy": "fixed", "rank_value": 0},
            {"rank_policy": "sv_ratio", "rank_value": 0.99},
            {"rank_policy": "sv_cumulative", "rank_value": 0},
            {"rank_policy": "sv_fro", "rank_value": 1.01},
            {"rank_policy": "relative_error", "rank_value": -0.1},
            {"rank_policy": "energy", "rank_value": 0.9, "retained_energy": 0.8},
            {"rank_policy": "fixed", "rank_value": 1, "relative_error": 0.1},
            {"rank_policy": "fixed", "rank_value": True},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                trim_lora_factors(down, down, **options)

    def test_file_api_keeps_default_and_legacy_threshold_paths_and_binds_explicit_policy(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "policy-source.safetensors"
            down, up = make_spectral_factors()
            save_file({
                "diffusion_model.block.proj.lora_down.weight": down,
                "diffusion_model.block.proj.lora_up.weight": up,
            }, source)

            default = trim_lora(source)
            legacy = trim_lora(source, relative_error=0.07, max_rank=2)
            explicit = trim_lora(source, rank_policy="fixed", rank_value=2)
            self.assertEqual(default["layers"][0]["new_rank"], 3)
            self.assertEqual(legacy["layers"][0]["new_rank"], 2)
            self.assertEqual(explicit["layers"][0]["new_rank"], 2)
            self.assertNotEqual(explicit["preview_token"], legacy["preview_token"])
            with self.assertRaises(ValueError):
                trim_lora(source, rank_policy="fixed", rank_value=2, relative_error=0.07)

    def test_dora_nonbake_preserves_magnitude_and_labels_direction_only_metrics(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dora.safetensors"
            destination = root / "dora-reduced.safetensors"
            source_network = make_network(rank=4, alpha=4, width=6, network_type="dora")
            down, up = make_spectral_factors(rank=4, width=6)
            magnitude = torch.tensor([1.2, 0.8, 1.1, 0.9, 1.3, 0.7], dtype=torch.float32)
            source_layer = source_network.unet_loras[0]
            with torch.no_grad():
                source_layer.lora_down.weight.copy_(down)
                source_layer.lora_up.weight.copy_(up)
                source_layer.magnitude.copy_(magnitude)
            state = source_network.get_state_dict(dtype=torch.float32)
            name = next(key.removesuffix(".lora_down.weight").removesuffix(".lora_A.weight")
                        for key in state if key.endswith(".lora_down.weight") or key.endswith(".lora_A.weight"))
            save_file(state, source, metadata={"network_module": "networks.dora", "use_dora": "true"})
            original = source.read_bytes()

            preview = trim_lora(source, rank_policy="fixed", rank_value=2)
            layer = preview["layers"][0]
            self.assertEqual(layer["input_format"], "dora")
            self.assertEqual(layer["output_format"], "dora")
            self.assertEqual(layer["metric_scope"], "pre_normalization_direction")
            self.assertTrue(any("final base-dependent DoRA" in warning for warning in preview["warnings"]))
            written = trim_lora(source, rank_policy="fixed", rank_value=2,
                                output=destination, expected_preview=preview["preview_token"])
            self.assertTrue(written["written"])
            self.assertEqual(source.read_bytes(), original)
            output = load_file(destination)
            magnitude_key = next(key for key in output if key.endswith(".dora_scale"))
            self.assertTrue(torch.equal(output[magnitude_key].reshape(-1), magnitude))
            down_key = next(key for key in output
                            if key.endswith(".lora_down.weight") or key.endswith(".lora_A.weight"))
            self.assertEqual(output[down_key].shape[0], 2)
            dora_loader = make_network(rank=2, alpha=4, width=6, network_type="dora")
            dora_loader.load_weights(str(destination))
            torch.testing.assert_close(dora_loader.unet_loras[0].magnitude, magnitude)

    def test_loha_conversion_matches_independent_hadamard_update_and_loads_as_lora(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "loha.safetensors"
            destination = root / "loha-as-lora.safetensors"
            source_network = make_network(rank=2, alpha=2.0)
            source_factor = next(key for key in source_network.get_state_dict(dtype=torch.float32)
                                 if key.endswith(".lora_A.weight"))
            name = source_factor.removesuffix(".lora_A.weight")
            torch.manual_seed(2026)
            w1a, w1b = torch.randn(64, 2), torch.randn(2, 64)
            w2a, w2b = torch.randn(64, 2), torch.randn(2, 64)
            alpha = 2.0
            state = {
                name + ".hada_w1_a": w1a,
                name + ".hada_w1_b": w1b,
                name + ".hada_w2_a": w2a,
                name + ".hada_w2_b": w2b,
                name + ".alpha": torch.tensor(alpha),
            }
            save_file(state, source, metadata={"ss_network_module": "lycoris.kohya"})
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "explicit convert_to_lora"):
                trim_lora(source, rank_policy="fixed", rank_value=4)
            preview = trim_lora(source, rank_policy="fixed", rank_value=4, convert_to_lora=True)
            layer = preview["layers"][0]
            self.assertEqual(layer["input_format"], "loha")
            self.assertEqual(layer["output_format"], "lora")
            self.assertEqual(layer["metric_scope"], "effective_update")
            self.assertEqual(layer["original_stored_rank"], 2)
            self.assertEqual(layer["expanded_rank"], 4)
            self.assertEqual(layer["new_rank"], 4)
            written = trim_lora(source, rank_policy="fixed", rank_value=4, convert_to_lora=True,
                                output=destination, expected_preview=preview["preview_token"])
            self.assertEqual(source.read_bytes(), original)
            output = load_file(destination)
            down, up = output[name + ".lora_down.weight"], output[name + ".lora_up.weight"]
            original_update = (w1a @ w1b) * (w2a @ w2b) * (alpha / 2)
            converted_update = (up @ down) * (alpha / 4)
            torch.testing.assert_close(converted_update, original_update, rtol=1e-5, atol=1e-5)
            target_network = make_network(rank=4, alpha=alpha)
            target_network.load_weights(str(destination))
            loaded = target_network.get_state_dict(dtype=torch.float32)
            self.assertTrue(any(torch.equal(value, down) for key, value in loaded.items()
                                if key.endswith("lora_A.weight")))

    def test_lokr_factored_and_full_components_convert_and_memory_budget_rejects_early(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            name = "diffusion_model.block.proj"
            torch.manual_seed(2027)
            a1, b1 = torch.randn(3, 2), torch.randn(2, 4)
            a2, b2 = torch.randn(2, 2), torch.randn(2, 3)
            alpha = 2.0
            factored = {
                name + ".lokr_w1_a": a1, name + ".lokr_w1_b": b1,
                name + ".lokr_w2_a": a2, name + ".lokr_w2_b": b2,
                name + ".alpha": torch.tensor(alpha),
            }
            expected = torch.kron(a1 @ b1, a2 @ b2) * (alpha / 2)
            for label, state in (("factored", factored),
                                 ("full", {name + ".lokr_w1": a1 @ b1,
                                           name + ".lokr_w2": a2 @ b2,
                                           name + ".alpha": torch.tensor(alpha)})):
                source = root / f"{label}.safetensors"
                destination = root / f"{label}-lora.safetensors"
                save_file(state, source)
                original = source.read_bytes()
                with self.assertRaisesRegex(ValueError, "working-memory estimate"):
                    trim_lora(source, convert_to_lora=True, max_working_mb=0.0001)
                self.assertFalse(destination.exists())
                preview = trim_lora(source, rank_policy="fixed", rank_value=4, convert_to_lora=True)
                self.assertEqual(preview["layers"][0]["input_format"], "lokr")
                self.assertEqual(preview["layers"][0]["metric_scope"], "effective_update")
                trim_lora(source, rank_policy="fixed", rank_value=4, convert_to_lora=True,
                          output=destination, expected_preview=preview["preview_token"])
                output = load_file(destination)
                actual = (output[name + ".lora_up.weight"] @ output[name + ".lora_down.weight"]) * (alpha / 4)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                self.assertEqual(source.read_bytes(), original)

    def test_all_full_lokr_without_alpha_gets_explicit_unit_scale_after_conversion_and_retrims(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            name = "diffusion_model.block.proj"
            source, converted, retrimmed = (
                root / "full-lokr.safetensors",
                root / "converted.safetensors",
                root / "retrimmed.safetensors",
            )
            torch.manual_seed(2028)
            w1, w2 = torch.randn(4, 3), torch.randn(2, 3)
            save_file({name + ".lokr_w1": w1, name + ".lokr_w2": w2}, source,
                      metadata={"ss_network_module": "lycoris.kohya", "ss_network_alpha": "1"})
            preview = trim_lora(source, rank_policy="fixed", rank_value=2, convert_to_lora=True)
            layer = preview["layers"][0]
            self.assertIsNone(layer["original_alpha"])
            self.assertEqual(layer["alpha"], 2.0)
            self.assertEqual(layer["new_scale"], 1.0)
            trim_lora(source, rank_policy="fixed", rank_value=2, convert_to_lora=True,
                      output=converted, expected_preview=preview["preview_token"])
            with safe_open(str(converted), framework="pt", device="cpu") as handle:
                output_metadata = handle.metadata() or {}
            output = load_file(converted)
            self.assertEqual(float(output[name + ".alpha"]), 2.0)
            self.assertEqual(output_metadata["ss_network_alpha"], "1")

            retrim_preview = trim_lora(converted, rank_policy="fixed", rank_value=2)
            self.assertEqual(retrim_preview["layers"][0]["alpha"], 2.0)
            self.assertEqual(retrim_preview["layers"][0]["old_rank"], 2)
            trim_lora(converted, rank_policy="fixed", rank_value=2, output=retrimmed,
                      expected_preview=retrim_preview["preview_token"])
            self.assertTrue(retrimmed.exists())

    def test_unsupported_adapter_shapes_and_mixed_conversion_are_not_written(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            name = "diffusion_model.block.proj"
            cases = [
                ("doha", {name + ".hada_w1_a": torch.ones(2, 2),
                          name + ".hada_w1_b": torch.ones(2, 2),
                          name + ".hada_w2_a": torch.ones(2, 2),
                          name + ".hada_w2_b": torch.ones(2, 2),
                          name + ".hada_t1": torch.ones(2, 2)}),
                ("mixed", {name + ".lora_down.weight": torch.ones(2, 4),
                           name + ".lora_up.weight": torch.ones(4, 2),
                           "other.hada_w1_a": torch.ones(2, 2),
                           "other.hada_w1_b": torch.ones(2, 2),
                           "other.hada_w2_a": torch.ones(2, 2),
                           "other.hada_w2_b": torch.ones(2, 2)}),
            ]
            for label, state in cases:
                source, destination = root / f"{label}.safetensors", root / f"{label}-out.safetensors"
                save_file(state, source)
                original = source.read_bytes()
                with self.subTest(label=label), self.assertRaises(ValueError):
                    trim_lora(source, convert_to_lora=True, output=destination)
                self.assertEqual(source.read_bytes(), original)
                self.assertFalse(destination.exists())

    def test_dora_bake_matches_each_selected_convention_and_loads_as_ordinary_lora(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source_network = make_network(rank=2, alpha=2.0)
            factor_key = next(key for key in source_network.get_state_dict(dtype=torch.float32)
                              if key.endswith(".lora_A.weight"))
            name = factor_key.removesuffix(".lora_A.weight")
            base_key = name + ".weight"
            down, up = make_spectral_factors(rank=3, width=64)
            magnitude = torch.linspace(0.7, 1.3, 64, dtype=torch.float32)
            torch.manual_seed(432)
            base = torch.randn(64, 64, dtype=torch.float32)
            source = root / "dora-source.safetensors"
            base_path = root / "matching-base.safetensors"
            save_file({
                name + ".lora_down.weight": down,
                name + ".lora_up.weight": up,
                name + ".alpha": torch.tensor(2.0),
                name + ".dora_scale": magnitude.reshape(64, 1),
            }, source, metadata={"network_module": "networks.dora", "use_dora": "true"})
            save_file({base_key: base}, base_path)
            source_before, base_before = source.read_bytes(), base_path.read_bytes()

            direction = (up.double() @ down.double()) * (2.0 / 3)
            expected_effective_deltas = {
                "comfy": (base.double() + direction) * (
                    magnitude.double() / (base.double().norm(dim=1) + torch.finfo(torch.float32).eps)
                )[:, None] - base.double(),
                # Toolkit DoRA normalizes using unscaled B@A, while its forward
                # applies alpha/rank to the adapter output.
                "toolkit": (base.double() + direction) * (
                    magnitude.double() / (base.double() + (up.double() @ down.double())).norm(dim=1)
                )[:, None] - base.double(),
            }

            for convention in ("comfy", "toolkit"):
                destination = root / f"baked-{convention}.safetensors"
                preview = trim_lora(source, bake_dora_to_lora=True, base_model_path=base_path,
                                    dora_convention=convention, rank_policy="fixed", rank_value=64)
                layer = preview["layers"][0]
                self.assertTrue(layer["baked"])
                self.assertEqual(layer["input_format"], "dora")
                self.assertEqual(layer["output_format"], "lora")
                self.assertEqual(layer["metric_scope"], "effective_update")
                self.assertEqual(layer["base_weight_key"], base_key)
                self.assertEqual(layer["dora_convention"], convention)
                self.assertEqual(preview["base_model"]["sha256"], hashlib.sha256(base_before).hexdigest())
                written = trim_lora(source, bake_dora_to_lora=True, base_model_path=base_path,
                                    dora_convention=convention, rank_policy="fixed", rank_value=64,
                                    output=destination, expected_preview=preview["preview_token"])
                self.assertTrue(written["written"])
                self.assertEqual(source.read_bytes(), source_before)
                self.assertEqual(base_path.read_bytes(), base_before)
                output = load_file(destination)
                self.assertNotIn(name + ".dora_scale", output)
                baked_update = (output[name + ".lora_up.weight"].double() @
                                output[name + ".lora_down.weight"].double()) * (2.0 / 64)
                torch.testing.assert_close(baked_update, expected_effective_deltas[convention],
                                           rtol=1e-5, atol=1e-5)

                target_network = make_network(rank=64, alpha=2.0)
                target_network.load_weights(str(destination))
                loaded_module = target_network.unet_loras[0]
                loader_update = (loaded_module.lora_up.weight.double() @
                                 loaded_module.lora_down.weight.double()) * (2.0 / 64)
                torch.testing.assert_close(loader_update, expected_effective_deltas[convention],
                                           rtol=1e-5, atol=1e-5)

    def test_native_dora_bake_matches_training_forward_with_alpha_and_bias(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            torch.manual_seed(1201)
            base = torch.nn.Linear(4, 3, bias=True)
            network = self._active_network()
            module = DoRAModule("native", base, lora_dim=2, alpha=3.0, network=network)
            module.org_forward = base.forward
            with torch.no_grad():
                module.lora_down.weight.copy_(torch.tensor([[0.4, -0.2, 0.1, 0.3],
                                                            [-0.3, 0.5, 0.2, -0.1]]))
                module.lora_up.weight.copy_(torch.tensor([[0.2, -0.4], [0.3, 0.1], [-0.2, 0.5]]))
                module.magnitude.copy_(torch.tensor([0.8, 1.4, 0.65]))
            x = torch.tensor([[0.1, 0.7, -0.5, 0.2], [-0.4, 0.3, 0.6, 0.9]])
            actual_training = module.forward(x)
            weight, bias = base.weight.detach(), base.bias.detach()
            direction = module.lora_up.weight.detach() @ module.lora_down.weight.detach()
            ratio = module.magnitude.detach() / torch.linalg.vector_norm(weight + direction, dim=1)
            expected_training = (
                torch.nn.functional.linear(x, weight, bias)
                + torch.nn.functional.linear(x, direction * (3.0 / 2), None)
            ) * ratio
            torch.testing.assert_close(actual_training, expected_training)

            adapter_name = "lora_transformer_layers.0.proj"
            base_name = "layers.0.proj"
            source, base_path, output = (root / name for name in ("native.safetensors", "base.safetensors", "baked.safetensors"))
            save_file({
                adapter_name + ".lora_down.weight": module.lora_down.weight.detach().clone(),
                adapter_name + ".lora_up.weight": module.lora_up.weight.detach().clone(),
                adapter_name + ".alpha": torch.tensor(3.0),
                adapter_name + ".magnitude": module.magnitude.detach().clone(),
            }, source, metadata={"network_module": "networks.dora", "use_dora": "true"})
            save_file({base_name + ".weight": weight.clone(), base_name + ".bias": bias.clone()}, base_path)
            preview = trim_lora(source, bake_dora_to_lora=True, base_model_path=base_path,
                                dora_convention="toolkit", rank_policy="fixed", rank_value=3)
            trim_lora(source, output=output, expected_preview=preview["preview_token"],
                      bake_dora_to_lora=True, base_model_path=base_path, dora_convention="toolkit",
                      rank_policy="fixed", rank_value=3)
            result = load_file(output)
            effective_weight = weight + result[adapter_name + ".lora_up.weight"] @ result[adapter_name + ".lora_down.weight"] * (3.0 / 3)
            effective_bias = bias + result[adapter_name + ".diff_b"]
            actual_baked = torch.nn.functional.linear(x.double(), effective_weight.double(), effective_bias.double())
            torch.testing.assert_close(actual_baked, actual_training.double(), rtol=1e-5, atol=1e-5)
            self.assertNotIn(adapter_name + ".magnitude", result)
            torch.testing.assert_close(result[adapter_name + ".diff_b"].double(), ((ratio - 1) * bias).double())

            # Verify the ordinary inference loader consumes diff_b alongside
            # the reduced factors on the matching biased target module.
            from toolkit.inference_lora import InferenceLoRA
            holder = type("Holder", (), {})()
            holder.model = torch.nn.Module()
            layers = torch.nn.ModuleList()
            target_block = torch.nn.Module()
            target = torch.nn.Linear(4, 3, bias=True)
            with torch.no_grad():
                target.weight.copy_(weight)
                target.bias.copy_(bias)
            target_block.add_module("proj", target)
            layers.append(target_block)
            holder.model.add_module("layers", layers)
            inference_adapter = InferenceLoRA(str(output)).load(holder).attach()
            try:
                loaded_output = target(x)
            finally:
                inference_adapter.detach()
            torch.testing.assert_close(loaded_output.double(), actual_training.double(), rtol=1e-5, atol=1e-5)

    def test_moss_fused_qkv_magnitude_bakes_to_loadable_split_projections(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            torch.manual_seed(9201)
            prefix = "lora_transformer_layers.0.proj."
            base_key = "layers.0.proj.qkv.weight"
            down = torch.randn(2, 4)
            ups = [torch.randn(2, 2) for _ in range(3)]
            base = torch.randn(6, 4)
            magnitude = torch.linspace(0.5, 1.4, 6).reshape(6, 1)
            tensors = {}
            for name, up in zip(("q", "k", "v"), ups):
                tensors[prefix + name + ".lora_A.weight"] = down.clone()
                tensors[prefix + name + ".lora_B.weight"] = up
                tensors[prefix + name + ".alpha"] = torch.tensor(3.0)
            tensors[prefix + "qkv.dora_scale"] = magnitude
            source, base_path, output = (root / name for name in ("moss.safetensors", "moss-base.safetensors", "moss-baked.safetensors"))
            save_file(tensors, source, metadata={"network_module": "networks.dora", "use_dora": "true"})
            save_file({base_key: base}, base_path)
            direction = torch.cat(ups, dim=0).double() @ down.double() * (3.0 / 2)
            expected = (base.double() + direction) * (
                magnitude.double().flatten() / (base.double().norm(dim=1) + torch.finfo(torch.float32).eps)
            )[:, None] - base.double()
            preview = trim_lora(source, bake_dora_to_lora=True, base_model_path=base_path,
                                dora_convention="comfy", rank_policy="fixed", rank_value=4)
            record = preview["layers"][0]
            self.assertEqual(record["split_qkv_outputs"], [prefix + name for name in ("q", "k", "v")])
            self.assertEqual(record["input_format"], "dora")
            trim_lora(source, output=output, expected_preview=preview["preview_token"],
                      bake_dora_to_lora=True, base_model_path=base_path, dora_convention="comfy",
                      rank_policy="fixed", rank_value=4)
            result = load_file(output)
            self.assertNotIn(prefix + "qkv.dora_scale", result)
            combined = torch.cat([result[prefix + name + ".lora_B.weight"] for name in ("q", "k", "v")], dim=0)
            shared_down = result[prefix + "q.lora_A.weight"]
            rebuilt = combined.double() @ shared_down.double() * (3.0 / 4)
            torch.testing.assert_close(rebuilt, expected, rtol=1e-5, atol=1e-5)
            self.assertTrue(torch.equal(shared_down, result[prefix + "k.lora_A.weight"]))
            self.assertTrue(torch.equal(shared_down, result[prefix + "v.lora_A.weight"]))

    def test_doha_linear_and_convolution_bake_match_training_module_updates(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            torch.manual_seed(4231)
            for label, base in (
                ("linear", torch.nn.Linear(4, 3, bias=True)),
                ("conv", torch.nn.Conv2d(1, 3, kernel_size=2, bias=True)),
            ):
                with self.subTest(layout=label):
                    network = self._active_network()
                    module = LoHaModule("doha", base, lora_dim=2, alpha=3.0, network=network, use_dora=True)
                    module.org_forward = base.forward
                    with torch.no_grad():
                        module.hada_w1_a.copy_(torch.randn_like(module.hada_w1_a) * 0.2)
                        module.hada_w1_b.copy_(torch.randn_like(module.hada_w1_b) * 0.2)
                        module.hada_w2_a.copy_(torch.randn_like(module.hada_w2_a) * 0.2)
                        module.hada_w2_b.copy_(torch.randn_like(module.hada_w2_b) * 0.2)
                        module.magnitude.copy_(module._weight_norms(adapted=True) * 0.83)
                    x = torch.randn(2, 4) if label == "linear" else torch.randn(2, 1, 5, 5)
                    expected = module.forward(x)
                    adapter_name = f"lora_transformer_layers.0.{label}"
                    base_name = f"layers.0.{label}"
                    source, base_path, output = (root / f"doha-{label}-{suffix}.safetensors"
                                                 for suffix in ("source", "base", "out"))
                    factors = {
                        ".hada_w1_a": module.hada_w1_a.detach().clone(),
                        ".hada_w1_b": module.hada_w1_b.detach().clone(),
                        ".hada_w2_a": module.hada_w2_a.detach().clone(),
                        ".hada_w2_b": module.hada_w2_b.detach().clone(),
                    }
                    save_file({**{adapter_name + suffix: value for suffix, value in factors.items()},
                               adapter_name + ".alpha": torch.tensor(3.0),
                               adapter_name + ".magnitude": module.magnitude.detach().clone()}, source)
                    save_file({base_name + ".weight": base.weight.detach().clone(),
                               base_name + ".bias": base.bias.detach().clone()}, base_path)
                    rank = min(base.weight.shape[0], base.weight[0].numel())
                    preview = trim_lora(source, bake_dora_to_lora=True, base_model_path=base_path,
                                        dora_convention="toolkit", rank_policy="fixed", rank_value=rank)
                    self.assertEqual(preview["layers"][0]["input_format"], "doha")
                    trim_lora(source, output=output, expected_preview=preview["preview_token"],
                              bake_dora_to_lora=True, base_model_path=base_path, dora_convention="toolkit",
                              rank_policy="fixed", rank_value=rank)
                    result = load_file(output)
                    new_down = result[adapter_name + ".lora_down.weight"]
                    new_up = result[adapter_name + ".lora_up.weight"]
                    flat_down = new_down.reshape(new_down.shape[0], -1)
                    flat_up = new_up.reshape(new_up.shape[0], new_up.shape[1])
                    delta = (flat_up.double() @ flat_down.double()).reshape_as(base.weight)
                    if label == "linear":
                        actual = torch.nn.functional.linear(x.double(), base.weight.double() + delta,
                                                            base.bias.double())
                    else:
                        actual = torch.nn.functional.conv2d(
                            x.double(), base.weight.double() + delta, base.bias.double(),
                            stride=base.stride, padding=base.padding, dilation=base.dilation,
                        )
                    torch.testing.assert_close(actual, expected.double(), rtol=2e-5, atol=2e-5)
                    self.assertNotIn(adapter_name + ".magnitude", result)

    def test_dora_bake_requires_matching_float_base_and_preview_rechecks_base_hash(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            name = "diffusion_model.block.proj"
            down, up = make_spectral_factors(rank=3, width=4)
            source = root / "dora.safetensors"
            save_file({name + ".lora_down.weight": down,
                       name + ".lora_up.weight": up,
                       name + ".dora_scale": torch.ones((4, 1)),
                       name + ".alpha": torch.tensor(2.0)}, source)
            missing_layer = root / "wrong-layer.safetensors"
            save_file({"other.layer.weight": torch.eye(4)}, missing_layer)
            wrong_shape = root / "wrong-shape.safetensors"
            save_file({name + ".weight": torch.eye(3)}, wrong_shape)
            quantized = root / "quantized.safetensors"
            save_file({name + ".weight": torch.eye(4), name + ".weight_scale": torch.ones(4)}, quantized)
            for bad_base in (missing_layer, wrong_shape, quantized):
                with self.subTest(base=bad_base.name), self.assertRaises(ValueError):
                    trim_lora(source, bake_dora_to_lora=True, base_model_path=bad_base,
                              dora_convention="comfy", rank_policy="fixed", rank_value=4)

            matching = root / "matching.safetensors"
            save_file({name + ".weight": torch.eye(4)}, matching)
            with self.assertRaisesRegex(ValueError, "working-memory estimate"):
                trim_lora(source, bake_dora_to_lora=True, base_model_path=matching,
                          dora_convention="toolkit", rank_policy="fixed", rank_value=4,
                          max_working_mb=0.0001)
            with self.assertRaisesRegex(ValueError, "requires base_model_path"):
                trim_lora(source, bake_dora_to_lora=True, dora_convention="toolkit")
            preview = trim_lora(source, bake_dora_to_lora=True, base_model_path=matching,
                                dora_convention="toolkit", rank_policy="fixed", rank_value=4)
            replacement = torch.eye(4) * 2
            save_file({name + ".weight": replacement}, matching)
            destination = root / "stale-preview.safetensors"
            with self.assertRaisesRegex(ValueError, "Preview is stale"):
                trim_lora(source, bake_dora_to_lora=True, base_model_path=matching,
                          dora_convention="toolkit", rank_policy="fixed", rank_value=4,
                          output=destination, expected_preview=preview["preview_token"])
            self.assertFalse(destination.exists())

    def test_dora_bake_decodes_real_comfy_convrot8_base_and_matches_training_weight_view(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source_network = make_network(rank=2, alpha=2.0, width=64)
            factor_key = next(key for key in source_network.get_state_dict(dtype=torch.float32)
                              if key.endswith(".lora_A.weight"))
            name = factor_key.removesuffix(".lora_A.weight")
            base_key = name + ".weight"

            torch.manual_seed(812)
            base_module = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            quantizer = get_ostris_quantizer("convrot8")
            self.assertTrue(quantizer.can_quantize(base_module))
            self.assertTrue(convert_linear_to_ostris(base_module, quantizer))
            # This is the training-time OstrisLinear.weight view: the existing
            # codec reconstructs the logical weight, then restores orig_dtype.
            training_weight = base_module.weight.detach().clone()
            self.assertEqual(training_weight.dtype, torch.bfloat16)

            model = torch.nn.Module()
            current = model
            for component in name.split(".")[:-1]:
                child = torch.nn.Module()
                current.add_module(component, child)
                current = child
            current.add_module(name.split(".")[-1], base_module)
            entries, exported, unexportable = export_comfy_quantized_layers(model)
            self.assertIn(name, exported)
            self.assertEqual(unexportable, [])
            base_path = root / "convrot8-base.safetensors"
            save_file(entries, base_path)
            base_before = base_path.read_bytes()

            # Exercise the same descriptor and decoder used by baking, and
            # compare it with the actual OstrisLinear training view.
            from toolkit.util.checkpoint_weights import (
                decode_checkpoint_weight,
                describe_checkpoint_weight,
            )

            with safe_open(str(base_path), framework="pt", device="cpu") as handle:
                descriptor = describe_checkpoint_weight(handle, base_key)
                decoded = decode_checkpoint_weight(handle, descriptor)
            torch.testing.assert_close(decoded.to(torch.bfloat16), training_weight)
            self.assertEqual(descriptor.quant_format, "int8_tensorwise")
            self.assertEqual(descriptor.storage_dtype, "I8")
            self.assertEqual(descriptor.shape, (64, 64))

            down, up = make_spectral_factors(rank=3, width=64)
            magnitude = torch.linspace(0.7, 1.3, 64, dtype=torch.float32).reshape(64, 1)
            source = root / "convrot8-dora.safetensors"
            save_file({
                name + ".lora_down.weight": down,
                name + ".lora_up.weight": up,
                name + ".alpha": torch.tensor(2.0),
                name + ".dora_scale": magnitude,
            }, source, metadata={"network_module": "networks.dora", "use_dora": "true"})
            source_before = source.read_bytes()

            tokens = set()
            for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
                base = decoded.to(dtype).double()
                direction = (up.double() @ down.double()) * (2.0 / 3.0)
                native_norm_weight = base + (up.double() @ down.double())
                expected = {
                    "comfy": (base + direction) * (
                        magnitude.double().reshape(-1) /
                        (base.norm(dim=1) + torch.finfo(torch.float32).eps)
                    )[:, None] - base,
                    "toolkit": (base + direction) * (
                        magnitude.double().reshape(-1) / native_norm_weight.norm(dim=1)
                    )[:, None] - base,
                }
                for convention in ("comfy", "toolkit"):
                    destination = root / f"baked-{dtype_name}-{convention}.safetensors"
                    options = dict(
                        bake_dora_to_lora=True,
                        base_model_path=base_path,
                        dora_convention=convention,
                        base_dequant_dtype=dtype_name,
                        rank_policy="fixed",
                        rank_value=64,
                    )
                    preview = trim_lora(source, **options)
                    self.assertEqual(preview["base_model"]["weight_decoder"], "comfy_logical_weight_v2")
                    record = preview["layers"][0]
                    self.assertEqual(record["base_weight"]["quant_format"], "int8_tensorwise")
                    self.assertEqual(record["base_weight"]["requested_dequant_dtype"], dtype_name)
                    self.assertEqual(record["base_weight"]["reconstruction_dtype"], dtype_name)
                    tokens.add(preview["preview_token"])
                    written = trim_lora(
                        source, output=destination,
                        expected_preview=preview["preview_token"], **options
                    )
                    self.assertTrue(written["written"])
                    result = load_file(destination)
                    self.assertNotIn(name + ".dora_scale", result)
                    baked = (result[name + ".lora_up.weight"].double() @
                             result[name + ".lora_down.weight"].double()) * (2.0 / 64)
                    torch.testing.assert_close(baked, expected[convention], rtol=1e-5, atol=1e-5)
                    target_network = make_network(rank=64, alpha=2.0, width=64)
                    target_network.load_weights(str(destination))
                    loaded_module = target_network.unet_loras[0]
                    loaded_update = (loaded_module.lora_up.weight.double() @
                                     loaded_module.lora_down.weight.double()) * (2.0 / 64)
                    torch.testing.assert_close(loaded_update, expected[convention],
                                               rtol=1e-5, atol=1e-5)
            self.assertEqual(len(tokens), 4, "dtype and convention must both bind the preview token")
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(base_path.read_bytes(), base_before)

    def test_bake_base_codecs_decode_like_their_quantizers_and_feed_the_reducer(self):
        from toolkit.util.checkpoint_weights import decode_checkpoint_weight, describe_checkpoint_weight
        from toolkit.util.ostris_quant import OstrisLinear

        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            cases = []

            def export_quantized(label, qtype, width, *, awq=False):
                linear = torch.nn.Linear(width, 8, bias=False)
                quantizer = get_ostris_quantizer(qtype)
                self.assertTrue(convert_linear_to_ostris(linear, quantizer))
                if awq:
                    pre_scale = torch.linspace(0.7, 1.3, width)
                    linear.register_buffer("nv4_pre_scale", pre_scale.contiguous().view(torch.uint8), persistent=False)
                model = torch.nn.Module()
                block = torch.nn.Module()
                model.add_module("block", block)
                block.add_module("proj", linear)
                entries, exported, unsupported = export_comfy_quantized_layers(model)
                self.assertEqual(exported, ["block.proj"])
                self.assertEqual(unsupported, [])
                if label == "nvfp4":
                    from toolkit.util.comfy_quant_export import comfy_quant_marker
                    entries["block.proj.comfy_quant"] = comfy_quant_marker(
                        {"format": "nvfp4", "full_precision_matrix_mult": True}
                    )
                cases.append((label, entries, quantizer.dequantize(linear), width, awq))

            torch.manual_seed(707)
            export_quantized("nvfp4", "nvfp4", 64, awq=True)
            export_quantized("w4a4", "convrotcomfyw4a4", 256)
            export_quantized("fp8-marker", "float8_e4m3fn", 64)

            # W4A8 has a published import/dequantize path rather than a fresh
            # quantize_ method. Exercise its real buffer attachment and exporter.
            from toolkit.util.comfy_w4a8_quant import ComfyW4A8Quantizer
            qmodule = torch.nn.Linear(64, 8, bias=False)
            qmodule.__class__ = OstrisLinear
            qmodule.ostris_orig_dtype = torch.float32
            w4a8 = ComfyW4A8Quantizer()
            w4a8.qtype = "comfy_w4a8"
            packed = torch.randint(-128, 127, (8, 32), dtype=torch.int8)
            relative = torch.rand(8, 4) * 0.2 + 0.05
            channel = torch.rand(8) * 0.4 + 0.1
            codebook = torch.linspace(-1.0, 1.0, 16)
            correction = torch.zeros(4, 8)
            w4a8.attach_(qmodule, packed, relative, channel, codebook, correction, group_size=16, rotation=1)
            qmodule.ostris_quantizer = w4a8
            model = torch.nn.Module()
            block = torch.nn.Module()
            model.add_module("block", block)
            block.add_module("proj", qmodule)
            entries, exported, unsupported = export_comfy_quantized_layers(model)
            self.assertEqual(exported, ["block.proj"])
            self.assertEqual(unsupported, [])
            cases.append(("w4a8", entries, w4a8.dequantize(qmodule), 64, False))

            fp8_entries = next(entries for label, entries, _, _, _ in cases if label == "fp8-marker")
            fp8_header = {k: v for k, v in fp8_entries.items() if not k.endswith(".comfy_quant")}
            fp8_legacy = {k: v for k, v in fp8_header.items() if k != "block.proj.weight_scale"}
            fp8_legacy["block.proj.scale_weight"] = fp8_header["block.proj.weight_scale"]
            fp8_legacy["scaled_fp8"] = torch.ones(1, dtype=torch.uint8)
            cases += [
                ("fp8-header", fp8_header, next(w for label, _, w, _, _ in cases if label == "fp8-marker"), 64, False,
                 {"_quantization_metadata": json.dumps({"layers": {"block.proj": {
                     "format": "float8_e4m3fn", "full_precision_matrix_mult": False}}})}),
                ("fp8-legacy", fp8_legacy, next(w for label, _, w, _, _ in cases if label == "fp8-marker"), 64, False, {}),
            ]

            for case in cases:
                label, entries, expected, width, awq = case[:5]
                metadata = case[5] if len(case) > 5 else None
                with self.subTest(codec=label):
                    base_path = root / f"{label}.safetensors"
                    save_file(entries, base_path, metadata=metadata)
                    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
                        descriptor = describe_checkpoint_weight(handle, "block.proj.weight")
                        decoded = decode_checkpoint_weight(handle, descriptor)
                        pre_scale = handle.get_tensor("block.proj.pre_quant_scale") if awq else None
                    self.assertEqual(tuple(decoded.shape), (8, width))
                    torch.testing.assert_close(decoded, expected.float(), rtol=1e-5, atol=1e-5)
                    if label == "nvfp4":
                        self.assertEqual(descriptor.quant_format, "nvfp4")
                        self.assertEqual(descriptor.marker_key, "block.proj.comfy_quant")
                        torch.testing.assert_close(pre_scale, torch.linspace(0.7, 1.3, width))
                    if label == "w4a4":
                        self.assertEqual(descriptor.quant_format, "convrot_w4a4")
                    if label == "w4a8":
                        self.assertEqual(descriptor.quant_format, "asym_w4a8_int8")
                        self.assertIn("block.proj.weight_correction", descriptor.auxiliary_keys)
                    if label == "fp8-header":
                        self.assertEqual(descriptor.marker_key, "_quantization_metadata:block.proj")
                    if label == "fp8-legacy":
                        self.assertEqual(descriptor.marker_key, "scaled_fp8")
                    if not awq:
                        adapter_name = "lora_transformer_block.proj"
                        torch.manual_seed(991)
                        down, up = torch.randn(2, width) * 0.03, torch.randn(8, 2) * 0.04
                        magnitude = torch.linspace(0.65, 1.2, 8).reshape(8, 1)
                        source = root / f"{label}-dora.safetensors"
                        output = root / f"{label}-baked.safetensors"
                        save_file({adapter_name + ".lora_down.weight": down,
                                   adapter_name + ".lora_up.weight": up,
                                   adapter_name + ".alpha": torch.tensor(2.5),
                                   adapter_name + ".dora_scale": magnitude}, source)
                        preview = trim_lora(source, bake_dora_to_lora=True, base_model_path=base_path,
                                            dora_convention="comfy", rank_policy="fixed", rank_value=8)
                        self.assertEqual(preview["layers"][0]["base_weight"]["quant_format"], descriptor.quant_format)
                        trim_lora(source, output=output, expected_preview=preview["preview_token"],
                                  bake_dora_to_lora=True, base_model_path=base_path, dora_convention="comfy",
                                  rank_policy="fixed", rank_value=8)
                        self.assertTrue(output.exists())

    def test_dora_bake_rejects_bare_integer_and_malformed_or_unknown_markers(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            name = "diffusion_model.block.proj"
            source = root / "dora.safetensors"
            down, up = make_spectral_factors(rank=3, width=4)
            save_file({
                name + ".lora_down.weight": down,
                name + ".lora_up.weight": up,
                name + ".dora_scale": torch.ones((4, 1)),
            }, source)
            invalid_bases = {
                "bare-int8": {
                    name + ".weight": torch.ones((4, 4), dtype=torch.int8),
                },
                "unknown-format": {
                    name + ".weight": torch.ones((4, 4), dtype=torch.int8),
                    name + ".weight_scale": torch.ones((4, 1)),
                    name + ".comfy_quant": torch.tensor(
                        list(b'{"format":"unknown_codec"}'), dtype=torch.uint8),
                },
                "malformed-nvfp4-layout": {
                    name + ".weight": torch.ones((4, 4), dtype=torch.int8),
                    name + ".weight_scale": torch.ones((4, 1)),
                    name + ".comfy_quant": torch.tensor(
                        list(b'{"format":"nvfp4"}'), dtype=torch.uint8),
                },
                "malformed-marker": {
                    name + ".weight": torch.ones((4, 4), dtype=torch.int8),
                    name + ".weight_scale": torch.ones((4, 1)),
                    name + ".comfy_quant": torch.tensor(list(b"not-json"), dtype=torch.uint8),
                },
                "missing-scale": {
                    name + ".weight": torch.ones((4, 4), dtype=torch.int8),
                    name + ".comfy_quant": torch.tensor(
                        list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8),
                },
            }
            for label, tensors in invalid_bases.items():
                with self.subTest(base=label):
                    base = root / f"{label}.safetensors"
                    save_file(tensors, base)
                    original = base.read_bytes()
                    with self.assertRaises(ValueError):
                        trim_lora(source, bake_dora_to_lora=True, base_model_path=base,
                                  dora_convention="toolkit", rank_policy="fixed", rank_value=4)
                    self.assertEqual(base.read_bytes(), original)

    def test_dora_bake_accepts_plain_tensorwise_int8_and_explicit_fp8_bases(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            name = "diffusion_model.block.proj"
            source = root / "dora.safetensors"
            down, up = make_spectral_factors(rank=3, width=64)
            magnitude = torch.linspace(0.75, 1.2, 64).reshape(64, 1)
            alpha = 2.0
            save_file({
                name + ".lora_down.weight": down,
                name + ".lora_up.weight": up,
                name + ".alpha": torch.tensor(alpha),
                name + ".dora_scale": magnitude,
            }, source)

            torch.manual_seed(913)
            original = torch.randn((64, 64), dtype=torch.float32)
            int8_codes, row_scales = quantize_int8_rows(original)
            int8_path = root / "plain-int8.safetensors"
            save_file({
                name + ".weight": int8_codes,
                name + ".weight_scale": row_scales.reshape(-1, 1),
                name + ".comfy_quant": comfy_quant_marker({"format": "int8_tensorwise"}),
            }, int8_path)

            fp8_linear = torch.nn.Linear(64, 64, bias=False)
            fp8_linear.weight.data.copy_(original)
            fp8_quantizer = get_ostris_quantizer("float8_e4m3fn")
            self.assertTrue(convert_linear_to_ostris(fp8_linear, fp8_quantizer))
            fp8_root = torch.nn.Module()
            parent = torch.nn.Module()
            block = torch.nn.Module()
            fp8_root.add_module("diffusion_model", parent)
            parent.add_module("block", block)
            block.add_module("proj", fp8_linear)
            fp8_entries, fp8_exported, fp8_unexportable = export_comfy_quantized_layers(fp8_root)
            self.assertEqual(fp8_exported, ["diffusion_model.block.proj"])
            self.assertEqual(fp8_unexportable, [])
            fp8_path = root / "fp8.safetensors"
            save_file(fp8_entries, fp8_path)

            from toolkit.util.checkpoint_weights import decode_checkpoint_weight, describe_checkpoint_weight

            for label, base_path, expected_format in (
                ("plain-int8", int8_path, "int8_tensorwise"),
                ("fp8", fp8_path, "float8_e4m3fn"),
            ):
                with self.subTest(codec=label):
                    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
                        descriptor = describe_checkpoint_weight(handle, name + ".weight")
                        decoded = decode_checkpoint_weight(handle, descriptor)
                    self.assertEqual(descriptor.quant_format, expected_format)
                    self.assertEqual(tuple(decoded.shape), (64, 64))
                    self.assertTrue(torch.isfinite(decoded).all())
                    preview = trim_lora(
                        source, bake_dora_to_lora=True, base_model_path=base_path,
                        dora_convention="toolkit", base_dequant_dtype="float32",
                        rank_policy="fixed", rank_value=64,
                    )
                    self.assertEqual(preview["layers"][0]["base_weight"]["quant_format"], expected_format)
                    destination = root / f"{label}-baked.safetensors"
                    trim_lora(
                        source, output=destination, expected_preview=preview["preview_token"],
                        bake_dora_to_lora=True, base_model_path=base_path,
                        dora_convention="toolkit", base_dequant_dtype="float32",
                        rank_policy="fixed", rank_value=64,
                    )
                    result = load_file(destination)
                    actual = (result[name + ".lora_up.weight"].double() @
                              result[name + ".lora_down.weight"].double()) * (alpha / 64)
                    direction = (up.double() @ down.double()) * (alpha / 3)
                    adapted = decoded.double() + direction
                    native_norm_weight = decoded.double() + (up.double() @ down.double())
                    expected = adapted * (magnitude.double().reshape(-1) / native_norm_weight.norm(dim=1))[:, None] - decoded.double()
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_quantized_base_working_budget_rejects_before_decoding_payload(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            name = "diffusion_model.block.proj"
            base_path = root / "marked.safetensors"
            # Valid-looking marked layout is enough: the budget check must be
            # performed from header dimensions before reading codes/scales.
            save_file({
                name + ".weight": torch.zeros((64, 64), dtype=torch.int8),
                name + ".weight_scale": torch.ones((64, 1)),
                name + ".comfy_quant": torch.tensor(
                    list(b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":64}'),
                    dtype=torch.uint8,
                ),
            }, base_path)
            source = root / "dora.safetensors"
            down, up = make_spectral_factors(rank=3, width=64)
            save_file({
                name + ".lora_down.weight": down,
                name + ".lora_up.weight": up,
                name + ".dora_scale": torch.ones((64, 1)),
            }, source)
            from unittest.mock import patch

            with patch("toolkit.lora_trimming.decode_checkpoint_weight",
                       side_effect=AssertionError("decoded before budget admission")) as decode:
                with self.assertRaisesRegex(ValueError, "working-memory estimate"):
                    trim_lora(source, bake_dora_to_lora=True, base_model_path=base_path,
                              dora_convention="toolkit", max_working_mb=0.01)
                decode.assert_not_called()

    def test_zero_update_is_finite_and_keeps_a_nonzero_rank(self):
        down = torch.zeros((3, 4), dtype=torch.float32)
        up = torch.zeros((5, 3), dtype=torch.float32)
        new_down, new_up, metrics = trim_lora_factors(
            down,
            up,
            alpha=3.0,
            relative_error=0.0,
        )
        self.assertEqual(new_down.shape[0], 1)
        self.assertEqual(new_up.shape[1], 1)
        self.assertTrue(torch.isfinite(new_down).all())
        self.assertTrue(torch.isfinite(new_up).all())
        self.assertEqual(metrics["relative_error"], 0.0)

    def test_finite_but_unsupported_fp64_ranges_fail_before_output(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            for label, magnitude in (("overflow", 1e150), ("underflow", 1e-100)):
                with self.subTest(range=label):
                    down = torch.eye(2, dtype=torch.float64) * magnitude
                    up = torch.eye(2, dtype=torch.float64) * magnitude
                    with self.assertRaises(ValueError):
                        trim_lora_factors(down, up, relative_error=0.1)

                    source = root / f"{label}.safetensors"
                    destination = root / f"{label}-trimmed.safetensors"
                    save_file(
                        {
                            "diffusion_model.block.proj.lora_down.weight": down,
                            "diffusion_model.block.proj.lora_up.weight": up,
                        },
                        source,
                    )
                    original_source = source.read_bytes()
                    with self.assertRaises(ValueError):
                        trim_lora(source, relative_error=0.1, output=destination)
                    self.assertEqual(source.read_bytes(), original_source)
                    self.assertFalse(destination.exists())

    def test_convolution_factors_reduce_rank_and_preserve_kernel_shapes(self):
        down = torch.zeros((3, 2, 3, 3), dtype=torch.float32)
        up = torch.zeros((4, 3, 1, 1), dtype=torch.float32)
        down[0, 0, 0, 0] = 5.0
        down[1, 1, 1, 1] = 2.0
        down[2, 0, 2, 2] = 0.1
        up[0, 0, 0, 0] = 1.0
        up[1, 1, 0, 0] = 1.0
        up[2, 2, 0, 0] = 1.0

        new_down, new_up, metrics = trim_lora_factors(
            down,
            up,
            relative_error=0.03,
            max_rank=2,
            alpha=2.0,
        )

        self.assertEqual(new_down.shape, (2, 2, 3, 3))
        self.assertEqual(new_up.shape, (4, 2, 1, 1))
        self.assertLessEqual(metrics["relative_error"], 0.03)
        original_update = (up[:, :, 0, 0].double() @ down.flatten(1).double()) * (2 / 3)
        trimmed_update = (new_up[:, :, 0, 0].double() @ new_down.flatten(1).double()) * (2 / 2)
        measured_error = torch.linalg.vector_norm(original_update - trimmed_update) / torch.linalg.vector_norm(original_update)
        self.assertAlmostEqual(float(measured_error), metrics["relative_error"], places=6)

    def test_large_unicode_archive_size_and_hashes_match_toolkit_helpers(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "unicode-source.safetensors"
            destination = root / "unicode-trimmed.safetensors"
            # Rank two output is about 1.1 MiB, enough for the legacy hash's
            # 1 MiB window without materializing the dense output-by-input map.
            rank, width = 8, 72000
            down = torch.zeros((rank, width), dtype=torch.float32)
            up = torch.zeros((width, rank), dtype=torch.float32)
            singular = torch.tensor([10.0, 5.0] + [0.1] * (rank - 2))
            roots = singular.sqrt()
            for index, value in enumerate(roots):
                down[index, index] = value
                up[index, index] = value
            metadata = {
                "ss_base_model": "synthetic/模型-Δ",
                "ss_training_comment": "naïve café 🐈 — проверка",
                "custom_note": "metadata with non-ASCII: 東京",
                "sshs_model_hash": "old-model-hash",
                "sshs_legacy_hash": "old-legacy",
            }
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                source,
                metadata=metadata,
            )
            preview = trim_lora(source, relative_error=0.04, max_rank=2)
            written = trim_lora(
                source,
                relative_error=0.04,
                max_rank=2,
                output=destination,
                expected_preview=preview["preview_token"],
            )
            self.assertEqual(written["estimated_output_bytes"], destination.stat().st_size)
            self.assertGreater(destination.stat().st_size, 0x110000)

            output_state = load_file(destination)
            with safe_open(str(destination), framework="pt", device="cpu") as handle:
                output_metadata = handle.metadata() or {}
            self.assertEqual(output_metadata["ss_training_comment"], metadata["ss_training_comment"])
            self.assertEqual(output_metadata["custom_note"], metadata["custom_note"])
            # This uses toolkit.metadata's real Safetensors serialization plus
            # the Additional Networks full-data and legacy-window hash helpers.
            expected_metadata = OrderedDict(
                (key, value) for key, value in metadata.items() if key.startswith("ss_")
            )
            add_model_hash_to_meta(output_state, expected_metadata)
            self.assertEqual(output_metadata["sshs_model_hash"], expected_metadata["sshs_model_hash"])
            self.assertEqual(output_metadata["sshs_legacy_hash"], expected_metadata["sshs_legacy_hash"])
            serialized = save(output_state, expected_metadata)
            self.assertGreater(len(serialized), 0x110000)

    def test_preview_write_preserves_metadata_size_and_loads_in_toolkit_network(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.safetensors"
            destination = root / "trimmed.safetensors"

            source_network = make_network(rank=8, alpha=6)
            down, up = make_spectral_factors()
            layer = source_network.unet_loras[0]
            with torch.no_grad():
                layer.lora_down.weight.copy_(down)
                layer.lora_up.weight.copy_(up)
                layer.alpha.fill_(6)
            state = source_network.get_state_dict(dtype=torch.float32)
            metadata = {
                "ss_base_model": "synthetic/local-base",
                "custom_note": "keep this metadata",
                "sshs_model_hash": "stale-derived-hash",
            }
            save_file(state, source, metadata=metadata)
            original_bytes = source.read_bytes()
            original_hash = hashlib.sha256(original_bytes).hexdigest()

            preview = trim_lora(
                source,
                relative_error=0.07,
                min_rank=1,
                max_rank=2,
            )
            self.assertFalse(preview["written"])
            self.assertEqual(preview["source_sha256"], original_hash)
            self.assertTrue(preview["preview_token"])
            self.assertEqual(preview["layers"][0]["old_rank"], 8)
            self.assertEqual(preview["layers"][0]["new_rank"], 2)
            self.assertLess(preview["estimated_output_bytes"], preview["input_bytes"])

            written = trim_lora(
                source,
                relative_error=0.07,
                min_rank=1,
                max_rank=2,
                output=destination,
                expected_preview=preview["preview_token"],
            )
            self.assertTrue(written["written"])
            self.assertEqual(written["estimated_output_bytes"], destination.stat().st_size)
            self.assertEqual(source.read_bytes(), original_bytes)

            with safe_open(str(destination), framework="pt", device="cpu") as handle:
                result_metadata = handle.metadata() or {}
            self.assertEqual(result_metadata["ss_base_model"], metadata["ss_base_model"])
            self.assertEqual(result_metadata["custom_note"], metadata["custom_note"])
            self.assertNotEqual(result_metadata["sshs_model_hash"], metadata["sshs_model_hash"])
            provenance = json.loads(result_metadata["aitk_lora_trimming"])
            self.assertEqual(
                provenance["original_hashes"]["sshs_model_hash"],
                metadata["sshs_model_hash"],
            )

            target_network = make_network(rank=2, alpha=6)
            target_network.load_weights(str(destination))
            target_state = target_network.get_state_dict(dtype=torch.float32)
            output_state = load_file(destination)
            factor_keys = [
                key for key in output_state
                if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")
            ]
            self.assertEqual(len(factor_keys), 2)
            for key in factor_keys:
                torch.testing.assert_close(target_state[key], output_state[key])

    def test_preview_token_binds_source_and_options_without_touching_destination(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.safetensors"
            destination = root / "out.safetensors"
            down, up = make_spectral_factors()
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                source,
            )
            preview = trim_lora(source, relative_error=0.07, max_rank=2)

            with self.assertRaises((ValueError, RuntimeError)):
                trim_lora(
                    source,
                    relative_error=0.06,
                    max_rank=2,
                    output=destination,
                    expected_preview=preview["preview_token"],
                )
            self.assertFalse(destination.exists())

            destination.write_bytes(b"pre-existing destination")
            original_destination = destination.read_bytes()
            with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
                trim_lora(
                    source,
                    relative_error=0.07,
                    max_rank=2,
                    output=destination,
                    expected_preview=preview["preview_token"],
                )
            self.assertEqual(destination.read_bytes(), original_destination)

    def test_source_path_cannot_be_output_and_symlink_destination_is_not_replaced(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.safetensors"
            target = root / "target.safetensors"
            link = root / "link.safetensors"
            down, up = make_spectral_factors()
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                source,
            )
            preview = trim_lora(source, relative_error=0.07, max_rank=2)
            original_source = source.read_bytes()

            with self.assertRaises((ValueError, FileExistsError, RuntimeError)):
                trim_lora(
                    source,
                    relative_error=0.07,
                    max_rank=2,
                    output=source,
                    expected_preview=preview["preview_token"],
                )
            self.assertEqual(source.read_bytes(), original_source)

            target.write_bytes(b"symlink target unchanged")
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
                trim_lora(
                    source,
                    relative_error=0.07,
                    max_rank=2,
                    output=link,
                    expected_preview=preview["preview_token"],
                )
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_bytes(), b"symlink target unchanged")

    def test_concurrent_writers_publish_exactly_one_complete_output(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.safetensors"
            destination = root / "out.safetensors"
            down, up = make_spectral_factors()
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                source,
            )
            preview = trim_lora(source, relative_error=0.07, max_rank=2)
            both_ready = threading.Barrier(2)

            def write_after_both_preflighted():
                return trim_lora(
                    source,
                    relative_error=0.07,
                    max_rank=2,
                    output=destination,
                    expected_preview=preview["preview_token"],
                    progress=lambda _message: both_ready.wait(timeout=20),
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(write_after_both_preflighted) for _ in range(2)]
                outcomes = []
                for future in futures:
                    try:
                        outcomes.append(future.result(timeout=60))
                    except Exception as error:
                        outcomes.append(error)

            self.assertEqual(sum(isinstance(item, dict) and item["written"] for item in outcomes), 1)
            self.assertEqual(sum(isinstance(item, FileExistsError) for item in outcomes), 1)
            with safe_open(str(destination), framework="pt", device="cpu") as handle:
                self.assertEqual(len(handle.keys()), 2)

    def test_changed_source_invalidates_preview_and_publishes_nothing(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.safetensors"
            destination = root / "out.safetensors"
            down, up = make_spectral_factors()
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                source,
            )
            preview = trim_lora(source, relative_error=0.07, max_rank=2)
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down * 1.01,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                source,
            )
            with self.assertRaises((ValueError, RuntimeError)):
                trim_lora(
                    source,
                    relative_error=0.07,
                    max_rank=2,
                    output=destination,
                    expected_preview=preview["preview_token"],
                )
            self.assertFalse(destination.exists())

    def test_rejects_dora_loha_lokr_and_unknown_tensor_payloads(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            down, up = make_spectral_factors()
            unsupported = {
                "dora": {
                    "diffusion_model.block.proj.lora_A.weight": down,
                    "diffusion_model.block.proj.lora_B.weight": up,
                    "diffusion_model.block.proj.dora_scale": torch.ones(5),
                },
                "loha": {
                    "diffusion_model.block.proj.hada_w1_a": torch.ones((2, 4)),
                    "diffusion_model.block.proj.hada_w1_b": torch.ones((5, 2)),
                    "diffusion_model.block.proj.hada_w2_a": torch.ones((2, 4)),
                    "diffusion_model.block.proj.hada_w2_b": torch.ones((5, 2)),
                },
                "lokr": {
                    "diffusion_model.block.proj.lokr_w1": torch.ones((2, 2)),
                    "diffusion_model.block.proj.lokr_w2": torch.ones((2, 2)),
                },
                "unknown": {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                    "unrecognized_payload": torch.ones(1),
                },
            }
            for label, tensors in unsupported.items():
                with self.subTest(adapter=label):
                    source = root / f"{label}.safetensors"
                    save_file(tensors, source)
                    with self.assertRaises(ValueError):
                        trim_lora(source, relative_error=0.07)

            mixed = root / "mixed-conventions.safetensors"
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                    "diffusion_model.block.proj.lora_A.weight": down.clone(),
                    "diffusion_model.block.proj.lora_B.weight": up.clone(),
                },
                mixed,
            )
            with self.assertRaisesRegex(ValueError, "Ambiguous factor conventions"):
                trim_lora(mixed)

            metadata_dora = root / "metadata-dora.safetensors"
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                metadata_dora,
                metadata={"ss_network_args": json.dumps({"use_dora": True})},
            )
            with self.assertRaisesRegex(ValueError, "DoRA"):
                trim_lora(metadata_dora)

            metadata_rslora = root / "metadata-rslora.safetensors"
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                metadata_rslora,
                metadata={"ss_network_args": json.dumps({"use_rslora": True})},
            )
            with self.assertRaisesRegex(ValueError, "rsLoRA"):
                trim_lora(metadata_rslora)

    def test_peft_sidecar_is_rejected_before_trimming(self):
        with self.temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source = root / "peft-adapter.safetensors"
            down, up = make_spectral_factors()
            save_file(
                {
                    "diffusion_model.block.proj.lora_down.weight": down,
                    "diffusion_model.block.proj.lora_up.weight": up,
                },
                source,
            )
            (root / "adapter_config.json").write_text(
                json.dumps({"r": 8, "lora_alpha": 8}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "sidecar"):
                trim_lora(source, relative_error=0.1)


if __name__ == "__main__":
    unittest.main()
