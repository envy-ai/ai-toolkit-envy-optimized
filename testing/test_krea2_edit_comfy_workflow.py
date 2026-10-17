"""Focused Krea 2 edit Comfy coverage; no Comfy server or model weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.comfy_sample import (
    ComfySampleRequest,
    DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH,
    get_workflow_for_sample,
    get_workflow_for_samples,
)
from toolkit.config_modules import GenerateImageConfig


class Krea2EditComfyWorkflowTests(unittest.TestCase):
    def request(self, **overrides):
        values = dict(
            prompt="edit prompt",
            negative_prompt="avoid blur",
            width=1024,
            height=512,
            steps=17,
            cfg=1.25,
            seed=2**64 - 1,
            model="models/krea2.safetensors",
            vae="models/qwen_vae.safetensors",
            text_encoder="models/qwen3vl.safetensors",
            sampler="euler",
            scheduler="simple",
            inference_lora="",
            inference_lora_strength=0.4,
            output_format="png",
            output_quality="high",
            training_lora_path="/trainer/output/current.safetensors",
            training_lora_filename="current.safetensors",
            filename_prefix="ai-toolkit/current",
            control_image="ComfyUI/reference.png",
            training_lora_strength=-0.65,
            model_arch="krea2",
        )
        values.update(overrides)
        return ComfySampleRequest(**values)

    def test_graph_propagates_request_and_preserves_edit_contract(self):
        request = self.request(prompt="change the coat", negative_prompt="extra limbs")
        graph = get_workflow_for_sample(DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH, request)

        self.assertEqual(graph["55"]["inputs"]["unet_name"], request.model)
        self.assertEqual(graph["56"]["inputs"]["clip_name"], request.text_encoder)
        self.assertEqual(graph["57"]["inputs"]["vae_name"], request.vae)
        self.assertEqual(graph["72"]["inputs"]["image"], request.control_image)
        self.assertEqual(graph["82"]["inputs"], {"width": 1024, "height": 512, "batch_size": 1})
        self.assertEqual(graph["53"]["inputs"]["seed"], 2**64 - 1)
        self.assertEqual(graph["53"]["inputs"]["steps"], 17)
        self.assertEqual(graph["53"]["inputs"]["cfg"], 1.25)
        self.assertEqual(graph["53"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(graph["53"]["inputs"]["scheduler"], "simple")
        self.assertEqual(graph["53"]["inputs"]["denoise"], 1)
        self.assertEqual(graph["84"]["inputs"]["prompt"], "change the coat")
        self.assertEqual(graph["84"]["inputs"]["grounding_px"], 0)
        self.assertEqual(graph["84"]["inputs"]["system_prompt"], "")
        self.assertEqual(graph["85"]["inputs"]["prompt"], "extra limbs")
        self.assertEqual(graph["85"]["inputs"]["grounding_px"], 768)
        self.assertEqual(graph["85"]["inputs"]["system_prompt"], "")
        self.assertEqual(graph["79"]["inputs"]["ref_boost"], 2)
        self.assertEqual(graph["79"]["inputs"]["ref_boost_a"], 2)
        self.assertEqual(graph["79"]["inputs"]["fit_mode"], "fit")
        self.assertEqual(graph["111"]["inputs"]["absolute_path"], request.training_lora_path)
        self.assertEqual(graph["111"]["inputs"]["lora_strength"], -0.65)
        self.assertEqual(graph["29"]["inputs"]["filename_prefix"], request.filename_prefix)
        self.assertNotIn("71", graph)
        self.assertEqual(graph["111"]["inputs"]["model"], ["55", 0])

    def test_optional_relative_helper_is_applied_before_training_snapshot(self):
        graph = get_workflow_for_sample(
            DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH,
            self.request(inference_lora="helpers/identity.safetensors", inference_lora_strength=0.3),
        )

        self.assertEqual(graph["71"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["71"]["inputs"]["lora_name"], "helpers/identity.safetensors")
        self.assertEqual(graph["71"]["inputs"]["strength_model"], 0.3)
        self.assertEqual(graph["111"]["inputs"]["model"], ["71", 0])

    def test_single_request_rejects_bad_arch_flow_and_control_cardinality(self):
        for overrides, message in (
            ({"model_arch": "qwen_image_2"}, "requires a Krea 2 model"),
            ({"specialized_flow": True}, "for diffusion_trainer"),
            ({"control_image": None}, "exactly one control image"),
            ({"control_image_2": "second.png"}, "exactly one control image"),
            ({"control_image_3": "third.png"}, "exactly one control image"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                get_workflow_for_sample(DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH, self.request(**overrides))

    def test_template_cannot_be_used_for_batch_requests(self):
        with self.assertRaisesRegex(ValueError, "individual requests"):
            get_workflow_for_samples(DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH, object())

    def trainer(self, root):
        trainer = object.__new__(BaseSDTrainProcess)
        trainer.save_root = root
        trainer.model_config = SimpleNamespace(arch="krea2", model_kwargs={})
        trainer.flow_profile = None
        for name in (
            "_cleanup_legacy_comfy_sample_loras", "_log_comfy_sample_generation_start",
            "_log_comfy_prompt_submitted", "_create_comfy_image_progress_bar",
            "_close_comfy_image_progress_bar", "_update_comfy_sample_status",
            "_log_comfy_sample_generation_time", "sample_step_hook",
            "_should_cancel_comfy_prompt_wait",
        ):
            setattr(trainer, name, Mock())
        trainer._save_current_network_for_comfy = Mock(return_value="/trainer/current.safetensors")
        trainer._get_comfy_lora_display_filename = Mock(return_value="current.safetensors")
        trainer._get_comfy_training_lora_path = Mock(return_value="/comfy/current.safetensors")
        trainer._prepare_comfy_inference_lora_path = Mock(return_value="")
        trainer._run_with_models_offloaded_for_comfy = lambda render: render(False, False)
        return trainer

    def sample(self, root, name, **kwargs):
        sample = GenerateImageConfig(
            prompt=f"prompt {name}", seed=42, width=1024, height=512,
            num_inference_steps=9, guidance_scale=1, network_multiplier=-0.5,
            output_path=str(Path(root) / f"{name}.png"), **kwargs,
        )
        return sample

    def comfy(self):
        return SimpleNamespace(
            workflow_path=DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH,
            api_url="http://unused", timeout=1, run_in_background=False,
            negative_prompt="", model="krea.safetensors", vae="vae.safetensors",
            audio_vae="", text_encoder="te.safetensors", sampler="euler",
            scheduler="simple", inference_lora="", inference_lora_strength=1,
            output_format="png", output_quality="high",
        )

    def test_batch_route_uploads_each_reference_and_renders_individual_prompts(self):
        with TemporaryDirectory() as root:
            trainer = self.trainer(root)
            samples = [
                self.sample(root, "first", ctrl_img="source-a.png"),
                self.sample(root, "second", ctrl_img_1="source-b.png"),
            ]
            config = SimpleNamespace(comfy=self.comfy())
            graphs = []
            client = Mock()
            client.upload_image.side_effect = lambda path: f"uploaded:{path}"
            client.post_prompt.side_effect = lambda graph: graphs.append(graph) or str(len(graphs))
            client.wait_for_images.side_effect = lambda *args, **kwargs: ["output"]

            with patch("jobs.process.BaseSDTrainProcess.ComfyApiClient", return_value=client), patch(
                "jobs.process.BaseSDTrainProcess.flush"
            ):
                trainer._render_comfy_sample_batch(samples, SimpleNamespace(comfy=config.comfy), step=10)

            self.assertEqual(client.upload_image.call_args_list, [
                unittest.mock.call("source-a.png"), unittest.mock.call("source-b.png")
            ])
            self.assertEqual(len(graphs), 2)
            self.assertEqual([graph["84"]["inputs"]["prompt"] for graph in graphs], ["prompt first", "prompt second"])
            self.assertEqual([graph["72"]["inputs"]["image"] for graph in graphs], [
                "uploaded:source-a.png", "uploaded:source-b.png"
            ])
            self.assertEqual([graph["111"]["inputs"]["lora_strength"] for graph in graphs], [-0.5, -0.5])

    def test_batch_route_rejects_missing_or_extra_references_before_side_effects(self):
        cases = (
            ("missing", {}),
            ("second", {"ctrl_img": "source.png", "ctrl_img_2": "second.png"}),
            ("third", {"ctrl_img": "source.png", "ctrl_img_3": "third.png"}),
        )
        for name, control_fields in cases:
            with self.subTest(name=name), TemporaryDirectory() as root:
                trainer = self.trainer(root)
                sample = self.sample(root, name, **control_fields)
                with patch("jobs.process.BaseSDTrainProcess.ComfyApiClient") as client:
                    with self.assertRaisesRegex(ValueError, "missing ctrl_img|exactly one control"):
                        trainer._render_comfy_sample_batch(
                            [sample], SimpleNamespace(comfy=self.comfy()), step=10
                        )
                client.assert_not_called()
                trainer._save_current_network_for_comfy.assert_not_called()
                trainer._cleanup_legacy_comfy_sample_loras.assert_not_called()

    def test_empty_sample_batch_has_no_side_effects(self):
        with TemporaryDirectory() as root:
            trainer = self.trainer(root)
            with patch("jobs.process.BaseSDTrainProcess.ComfyApiClient") as client:
                result = trainer._render_comfy_sample_batch([], SimpleNamespace(comfy=self.comfy()), step=10)
            self.assertIsNone(result)
            client.assert_not_called()
            trainer._save_current_network_for_comfy.assert_not_called()
            trainer._cleanup_legacy_comfy_sample_loras.assert_not_called()


if __name__ == "__main__":
    unittest.main()
