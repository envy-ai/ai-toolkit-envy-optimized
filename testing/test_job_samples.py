import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

from toolkit.job_samples import (
    checkpoint_step,
    read_checkpoint_metadata,
    render_saved_samples,
    select_checkpoint,
    validate_render_config,
)


def write_checkpoint(path, step):
    header = {
        "__metadata__": {"training_info": json.dumps({"step": step})},
        "lora.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
    }
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"data")
    return path


def valid_config():
    return {
        "config": {
            "name": "sample-job",
            "process": [
                {
                    "type": "diffusion_trainer",
                    "network": {"type": "lora", "linear": 4},
                    "model": {"name_or_path": "model", "arch": "test"},
                    "train": {"steps": 30, "dtype": "bf16"},
                    "sample": {
                        "comfy": {
                            "enabled": True,
                            "provider": "ordinary",
                            "api_url": "http://127.0.0.1:8188",
                            "timeout": 1,
                        },
                        "samples": [{"prompt": "configured prompt", "seed": 12}],
                    },
                    "datasets": [],
                }
            ],
        }
    }


class JobSampleCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="job-samples-test-")
        self.folder = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_selects_latest_checkpoint_and_ignores_unrelated_files(self):
        older = write_checkpoint(self.folder / "sample-job_000000020.safetensors", 20)
        latest = write_checkpoint(self.folder / "sample-job_000000120.safetensors", 120)
        os.utime(older, ns=(1_000_000_000, 1_000_000_000))
        os.utime(latest, ns=(2_000_000_000, 2_000_000_000))
        write_checkpoint(self.folder / "another-job_000000999.safetensors", 999)
        (self.folder / "sample-job_000000999.safetensors.tmp").write_bytes(b"partial")

        selected = select_checkpoint(self.folder, "sample-job")

        self.assertEqual(selected, latest)
        self.assertEqual(checkpoint_step(selected), 120)
        self.assertEqual(read_checkpoint_metadata(older)["step"], 20)

    def test_final_checkpoint_without_step_suffix_can_be_selected_as_latest(self):
        older = write_checkpoint(self.folder / "sample-job_000000020.safetensors", 20)
        final = write_checkpoint(self.folder / "sample-job.safetensors", 25)
        os.utime(older, ns=(1_000_000_000, 1_000_000_000))
        os.utime(final, ns=(2_000_000_000, 2_000_000_000))

        self.assertEqual(checkpoint_step(final), 25)
        self.assertEqual(select_checkpoint(self.folder, "sample-job"), final)

    def test_checkpoint_metadata_must_match_step_suffix(self):
        checkpoint = write_checkpoint(self.folder / "sample-job_000000900.safetensors", 125)

        with self.assertRaisesRegex(ValueError, "filename and metadata disagree"):
            checkpoint_step(checkpoint)

    def test_missing_checkpoint_has_explicit_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "No saved LoRA checkpoint"):
            select_checkpoint(self.folder, "sample-job")

    def test_corrupt_latest_candidate_is_not_silently_skipped(self):
        write_checkpoint(self.folder / "sample-job_000000020.safetensors", 20)
        (self.folder / "sample-job_000000120.safetensors").write_bytes(b"broken")

        with self.assertRaisesRegex(ValueError, "Incomplete safetensors checkpoint"):
            select_checkpoint(self.folder, "sample-job")


class JobSampleConfigTests(unittest.TestCase):
    def test_accepts_configured_comfy_workflow_samples(self):
        config = valid_config()

        process = validate_render_config(config)

        self.assertEqual(process["sample"]["samples"][0]["prompt"], "configured prompt")

    def test_rejects_unsupported_job_shapes_and_sample_modes(self):
        cases = []
        no_process = valid_config()
        no_process["config"]["process"] = []
        cases.append((no_process, "exactly one"))

        specialized = valid_config()
        specialized["config"]["process"][0]["type"] = "slider_trainer"
        cases.append((specialized, "ordinary diffusion trainers"))

        no_network = valid_config()
        no_network["config"]["process"][0].pop("network")
        cases.append((no_network, "unmerged LoRA"))

        internal = valid_config()
        internal["config"]["process"][0]["sample"]["comfy"]["enabled"] = False
        cases.append((internal, "requires the ComfyUI workflow renderer"))

        shared = valid_config()
        shared["config"]["process"][0]["sample"]["comfy"]["provider"] = "aitk_shared_models"
        cases.append((shared, "shared models require a live trainer"))

        no_samples = valid_config()
        no_samples["config"]["process"][0]["sample"].pop("samples")
        cases.append((no_samples, "no configured samples"))

        for config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_render_config(config)


class SavedSampleRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="job-sample-render-test-")
        self.folder = Path(self.tempdir.name)
        self.checkpoint = write_checkpoint(self.folder / "sample-job_000000007.safetensors", 7)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_renders_configured_samples_without_constructing_or_advancing_a_trainer(self):
        observed = {}

        class FakeRendererBase:
            def sample(self, step):
                observed["step"] = step
                observed["renderer_step"] = self.step_num
                observed["samples"] = self.sample_config.samples

            def run(self):
                observed["trainer_run_called"] = True

        class FakeConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeSampleConfig(FakeConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.comfy = FakeConfig(**kwargs["comfy"])

        class FakeComfyClient:
            def __init__(self, *args, **kwargs):
                pass

            def _request_json(self, method, path):
                observed["queue_request"] = (method, path)
                return {"queue_running": [], "queue_pending": []}

        def inject_trigger(prompt, *_args, **_kwargs):
            return prompt

        base_module = types.ModuleType("jobs.process.BaseSDTrainProcess")
        base_module.BaseSDTrainProcess = FakeRendererBase
        config_module = types.ModuleType("toolkit.config_modules")
        config_module.ModelConfig = FakeConfig
        config_module.SampleConfig = FakeSampleConfig
        config_module.TrainConfig = FakeConfig
        comfy_module = types.ModuleType("toolkit.comfy_sample")
        comfy_module.ComfyApiClient = FakeComfyClient
        prompt_module = types.ModuleType("toolkit.prompt_utils")
        prompt_module.inject_trigger_into_prompt = inject_trigger

        with mock.patch.dict(
            sys.modules,
            {
                "jobs.process.BaseSDTrainProcess": base_module,
                "toolkit.config_modules": config_module,
                "toolkit.comfy_sample": comfy_module,
                "toolkit.prompt_utils": prompt_module,
            },
        ):
            render_saved_samples(valid_config(), self.folder, self.checkpoint, lambda _message: None)

        self.assertEqual(observed["step"], 7)
        self.assertEqual(observed["renderer_step"], 7)
        self.assertEqual(observed["samples"], [{"prompt": "configured prompt", "seed": 12}])
        self.assertEqual(observed["queue_request"], ("GET", "/api/queue"))
        self.assertNotIn("trainer_run_called", observed)
        self.assertTrue((self.folder / "samples").is_dir())

    def test_active_comfy_queue_error_is_propagated_before_sample_call(self):
        called = []

        class FakeRendererBase:
            def sample(self, step):
                called.append(step)

        class FakeConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeSampleConfig(FakeConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.comfy = FakeConfig(**kwargs["comfy"])

        class BusyComfyClient:
            def __init__(self, *args, **kwargs):
                pass

            def _request_json(self, method, path):
                return {"queue_running": [{"prompt_id": "active"}], "queue_pending": []}

        base_module = types.ModuleType("jobs.process.BaseSDTrainProcess")
        base_module.BaseSDTrainProcess = FakeRendererBase
        config_module = types.ModuleType("toolkit.config_modules")
        config_module.ModelConfig = config_module.TrainConfig = FakeConfig
        config_module.SampleConfig = FakeSampleConfig
        comfy_module = types.ModuleType("toolkit.comfy_sample")
        comfy_module.ComfyApiClient = BusyComfyClient
        prompt_module = types.ModuleType("toolkit.prompt_utils")
        prompt_module.inject_trigger_into_prompt = lambda prompt, *_args, **_kwargs: prompt

        with mock.patch.dict(
            sys.modules,
            {
                "jobs.process.BaseSDTrainProcess": base_module,
                "toolkit.config_modules": config_module,
                "toolkit.comfy_sample": comfy_module,
                "toolkit.prompt_utils": prompt_module,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "active or pending work"):
                render_saved_samples(valid_config(), self.folder, self.checkpoint, lambda _message: None)

        self.assertEqual(called, [])


class QwenSavedSampleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="qwen-sample-render-test-")
        self.folder = Path(self.tempdir.name)
        self.checkpoint = write_checkpoint(self.folder / "sample-job_000000007.safetensors", 7)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_real_training_sample_builder_renders_configured_qwen_prompts_without_gpu(self):
        config = valid_config()
        process = config["config"]["process"][0]
        process["model"].update({
            "arch": "qwen_image_2",
            "name_or_path": "Qwen/Qwen-Image-2.1",
        })
        process["sample"].update({
            "comfy": {
                "enabled": True,
                "provider": "ordinary",
                "api_url": "http://comfy.invalid:8188",
                "workflow_path": "config/comfy_templates/qwen_image_2_lora_sample.json.njk",
                "model": "qwen_image_2.1.safetensors",
                "vae": "qwen_image_2.1_vae.safetensors",
                "text_encoder": "qwen_2.5_vl_7b.safetensors",
            },
            "samples": [
                {"prompt": "first configured Qwen prompt", "seed": 41},
                {"prompt": "second configured Qwen prompt", "seed": 42},
            ],
        })
        observations = {"prompts": [], "outputs": []}

        class FakeComfyClient:
            def __init__(self, *args, **kwargs):
                pass

            def _request_json(self, method, route):
                self_route = (method, route)
                observations.setdefault("requests", []).append(self_route)
                return {"queue_running": [], "queue_pending": []}

            def begin_live_progress(self):
                pass

            def post_prompt(self, workflow):
                observations["prompts"].append(json.dumps(workflow))
                return f"prompt-{len(observations['prompts'])}"

            def wait_for_images(self, _prompt_id, **_kwargs):
                return [{"filename": "rendered.png", "subfolder": "", "type": "output"}]

            def download_image(self, _image, output_path):
                observations["outputs"].append(output_path)

            def release_vram(self):
                pass

            def wait_for_vram_release(self, **_kwargs):
                return True

        import toolkit.comfy_sample
        import importlib
        base_module = importlib.import_module("jobs.process.BaseSDTrainProcess")
        from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

        with mock.patch.object(toolkit.comfy_sample, "ComfyApiClient", FakeComfyClient), mock.patch.object(
            base_module, "ComfyApiClient", FakeComfyClient
        ), mock.patch.object(
            BaseSDTrainProcess, "_create_comfy_image_progress_bar", return_value=None
        ), mock.patch.object(BaseSDTrainProcess, "_update_comfy_image_progress_bar"), mock.patch.object(
            BaseSDTrainProcess, "_close_comfy_image_progress_bar"
        ):
            render_saved_samples(config, self.folder, self.checkpoint, lambda _message: None)

        self.assertEqual(len(observations["prompts"]), 2)
        self.assertIn("first configured Qwen prompt", observations["prompts"][0])
        self.assertIn("second configured Qwen prompt", observations["prompts"][1])
        self.assertTrue(all("_000000007_" in path for path in observations["outputs"]))
        self.assertEqual(observations["requests"], [("GET", "/api/queue")])
        self.assertTrue((self.folder / "samples").is_dir())


if __name__ == "__main__":
    unittest.main()
