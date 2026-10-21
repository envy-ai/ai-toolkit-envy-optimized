import hashlib
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.lora_trimming import trim_lora, trim_lora_factors


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


def make_network(rank, alpha=6, width=64, peft_format=True):
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

    def test_convolution_factors_keep_supported_kernel_shapes(self):
        down = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
        up = torch.zeros((4, 2, 1, 1), dtype=torch.float32)
        down[0, 0, 0, 0] = 3.0
        down[1, 1, 1, 1] = 2.0
        up[0, 0, 0, 0] = 3.0
        up[1, 1, 0, 0] = 2.0

        new_down, new_up, metrics = trim_lora_factors(
            down,
            up,
            relative_error=0.0,
            alpha=2.0,
        )

        self.assertEqual(new_down.shape, down.shape)
        self.assertEqual(new_up.shape, up.shape)
        self.assertEqual(metrics["relative_error"], 0.0)
        torch.testing.assert_close(
            new_up[:, :, 0, 0] @ new_down.flatten(1),
            up[:, :, 0, 0] @ down.flatten(1),
        )

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


if __name__ == "__main__":
    unittest.main()
