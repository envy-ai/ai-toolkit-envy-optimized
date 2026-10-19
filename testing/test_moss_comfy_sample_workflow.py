"""Focused MOSS-Image ComfyUI preview workflow coverage."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.comfy_sample import (
    ComfySampleRequest,
    DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH,
    get_workflow_for_sample,
    get_workflow_for_samples,
)


ROOT = Path(__file__).resolve().parents[1]


class MossComfySampleWorkflowTests(unittest.TestCase):
    def request(self, **overrides):
        values = dict(
            prompt="a precise prompt",
            negative_prompt="blur and noise",
            width=768,
            height=1024,
            steps=23,
            cfg=2.75,
            seed=2**64 - 1,
            model="models/moss.safetensors",
            vae="models/vae.safetensors",
            text_encoder="models/qwen35.safetensors",
            sampler="euler_ancestral",
            scheduler="simple",
            inference_lora="helpers/style.safetensors",
            inference_lora_strength=0.35,
            output_format="png",
            output_quality="high",
            training_lora_path="/trainer/output/current.safetensors",
            training_lora_filename="current.safetensors",
            filename_prefix="aitk/current",
            training_lora_strength=-0.6,
            model_arch="moss_image",
        )
        values.update(overrides)
        return ComfySampleRequest(**values)

    def test_template_uses_moss_encoder_standard_vae_and_dynamic_sampling(self):
        request = self.request()
        graph = get_workflow_for_sample(DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH, request)

        self.assertEqual(graph["1"]["inputs"]["unet_name"], request.model)
        self.assertEqual(graph["2"]["class_type"], "CLIPLoader")
        self.assertEqual(graph["2"]["inputs"]["clip_name"], request.text_encoder)
        self.assertEqual(graph["2"]["inputs"]["type"], "moss_image")
        self.assertEqual(graph["3"], {"class_type": "VAELoader", "inputs": {"vae_name": request.vae}})
        self.assertEqual(graph["8"]["class_type"], "EmptyLatentImage")
        self.assertEqual(graph["8"]["inputs"], {"width": 768, "height": 1024, "batch_size": 1})
        self.assertEqual(graph["7"]["class_type"], "CLIPTextEncode")
        self.assertEqual(graph["7"]["inputs"]["text"], "blur and noise")
        self.assertEqual(graph["9"]["inputs"], {
            "model": ["5", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["8", 0], "seed": 2**64 - 1, "steps": 23,
            "cfg": 2.75, "sampler_name": "euler_ancestral", "scheduler": "simple", "denoise": 1,
        })
        self.assertEqual(graph["10"]["inputs"]["vae"], ["3", 0])
        self.assertEqual(graph["11"]["class_type"], "SaveImage")
        self.assertEqual(graph["11"]["inputs"]["filename_prefix"], "aitk/current")

    def test_training_lora_and_inference_lora_have_separate_roles(self):
        request = self.request()
        graph = get_workflow_for_sample(DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH, request)

        self.assertEqual(graph["4"]["class_type"], "load_lora_from_absolute_path")
        self.assertEqual(graph["4"]["inputs"]["absolute_path"], request.training_lora_path)
        self.assertEqual(graph["4"]["inputs"]["lora_strength"], -0.6)
        self.assertEqual(graph["4"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["4"]["inputs"]["clip"], ["2", 0])
        self.assertEqual(graph["5"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["5"]["inputs"], {
            "lora_name": "helpers/style.safetensors", "strength_model": 0.35, "model": ["4", 0],
        })
        self.assertEqual(graph["6"]["inputs"]["clip"], ["4", 1])
        self.assertEqual(graph["9"]["inputs"]["model"], ["5", 0])

        no_inference_lora = get_workflow_for_sample(
            DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH,
            self.request(inference_lora="", inference_lora_strength=0),
        )
        self.assertNotIn("5", no_inference_lora)
        self.assertEqual(no_inference_lora["9"]["inputs"]["model"], ["4", 0])
        self.assertEqual(no_inference_lora["4"]["inputs"]["lora_strength"], -0.6)

        zero_strength_inference_lora = get_workflow_for_sample(
            DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH,
            self.request(inference_lora_strength=0),
        )
        self.assertEqual(zero_strength_inference_lora["5"]["inputs"]["lora_name"], "helpers/style.safetensors")
        self.assertEqual(zero_strength_inference_lora["5"]["inputs"]["strength_model"], 0)
        self.assertEqual(zero_strength_inference_lora["9"]["inputs"]["model"], ["5", 0])

    def test_empty_negative_prompt_uses_zeroed_positive_conditioning(self):
        graph = get_workflow_for_sample(
            DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH, self.request(negative_prompt="")
        )

        self.assertEqual(graph["7"]["class_type"], "ConditioningZeroOut")
        self.assertEqual(graph["7"]["inputs"]["conditioning"], ["6", 0])

    def test_template_rejects_other_architectures_edit_references_and_specialized_modes(self):
        base_variant = get_workflow_for_sample(
            DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH, self.request(model_arch="moss_image:base")
        )
        self.assertEqual(base_variant["2"]["inputs"]["type"], "moss_image")

        with self.assertRaisesRegex(ValueError, "requires a MOSS-Image model"):
            get_workflow_for_sample(
                DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH, self.request(model_arch="krea2")
            )
        with self.assertRaisesRegex(ValueError, "text-to-image only"):
            get_workflow_for_sample(
                DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH, self.request(control_image="reference.png")
            )
        with self.assertRaisesRegex(ValueError, "require native previews"):
            get_workflow_for_sample(
                DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH, self.request(specialized_flow=True)
            )

    def test_individual_workflow_rejects_batch_api(self):
        with self.assertRaisesRegex(ValueError, "individual requests"):
            get_workflow_for_samples(DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH, object())

    def test_batch_dispatch_routes_moss_prompts_individually_even_when_batching_enabled(self):
        for arch in ("moss_image", "moss_image:base"):
            with self.subTest(arch=arch):
                process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
                process.model_config = SimpleNamespace(arch=arch)
                process._render_comfy_samples = Mock(return_value="individual")
                samples = [object(), object()]
                sample_config = SimpleNamespace(comfy=SimpleNamespace(send_prompts_as_batch=True))

                with patch("jobs.process.BaseSDTrainProcess.print_acc") as log:
                    result = process._render_comfy_sample_batch(samples, sample_config, step=12)

                self.assertEqual(result, "individual")
                process._render_comfy_samples.assert_called_once_with(samples, sample_config, step=12)
                self.assertIn("individual prompts", log.call_args.args[0])

    def test_workflow_selection_preserves_moss_template_for_batch_setting(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.model_config = SimpleNamespace(arch="moss_image:base")
        comfy = SimpleNamespace(workflow_path=DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH)

        self.assertEqual(process._get_comfy_workflow_path(comfy), DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH)
        self.assertEqual(
            process._get_comfy_workflow_path(comfy, batch=True), DEFAULT_COMFY_MOSS_IMAGE_WORKFLOW_PATH
        )

        comfy.workflow_path = "config/comfy_templates/krea2_lora_sample.json.njk"
        with self.assertRaisesRegex(ValueError, "not moss_image"):
            process._get_comfy_workflow_path(comfy, batch=True)

    def test_ui_exposes_the_dedicated_workflow_and_accepts_it_for_ordinary_moss_previews(self):
        simple_job = (ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        capabilities = (ROOT / "ui/src/app/jobs/new/trainingCapabilities.ts").read_text()

        self.assertIn(
            "value: 'config/comfy_templates/moss_image_lora_sample.json.njk'",
            simple_job,
        )
        self.assertIn(
            "Select MOSS-Image LoRA image for ordinary text-to-image previews",
            capabilities,
        )
        self.assertIn("MOSS-Image specialized modes require native previews", capabilities)


if __name__ == "__main__":
    unittest.main()
