"""Focused coverage for the experimental Ostris Krea 2 edit Comfy workflow."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.comfy_sample import (
    ComfySampleRequest,
    DEFAULT_COMFY_KREA2_OSTRIS_EDIT_WORKFLOW_PATH,
    get_workflow_for_sample,
    get_workflow_for_samples,
)
from toolkit.config_modules import GenerateImageConfig, ModelConfig


class Krea2OstrisEditComfyWorkflowTests(unittest.TestCase):
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
            control_image="ComfyUI/reference-1.png",
            training_lora_strength=-0.65,
            model_arch="krea2",
            model_kwargs={
                "edit": True,
                "match_target_res": True,
                "kv_cache": True,
                "vlm_max_pixels": 56**2,
                "control_image_max_pixels": 64**2,
            },
            model_dtype="float16",
            control_background=[19, 71, 139],
        )
        values.update(overrides)
        return ComfySampleRequest(**values)

    def test_saved_arch_variant_normalizes_and_keeps_edit_training_settings(self):
        config = ModelConfig(
            name_or_path="krea/Krea-2-Raw",
            arch="krea2:o_edit",
            model_kwargs={"edit": True, "match_target_res": True, "kv_cache": True},
        )

        self.assertEqual(config.arch, "krea2")
        self.assertEqual(
            config.model_kwargs,
            {"edit": True, "match_target_res": True, "kv_cache": True},
        )

    def test_graph_maps_reference_preprocessing_patch_and_training_lora(self):
        request = self.request(
            control_image_2="ComfyUI/reference-2.png",
            control_image_3="ComfyUI/reference-3.png",
        )
        graph = get_workflow_for_sample(DEFAULT_COMFY_KREA2_OSTRIS_EDIT_WORKFLOW_PATH, request)

        self.assertEqual(graph["55"]["inputs"]["unet_name"], request.model)
        self.assertEqual(graph["56"]["inputs"]["clip_name"], request.text_encoder)
        self.assertEqual(graph["57"]["inputs"]["vae_name"], request.vae)
        self.assertEqual(graph["53"]["class_type"], "KSampler")
        self.assertEqual(graph["53"]["inputs"]["seed"], 2**64 - 1)
        self.assertEqual(graph["53"]["inputs"]["steps"], request.steps)
        self.assertEqual(graph["53"]["inputs"]["cfg"], request.cfg)
        self.assertEqual(graph["53"]["inputs"]["sampler_name"], request.sampler)
        self.assertEqual(graph["53"]["inputs"]["scheduler"], request.scheduler)

        conditioning = graph["84"]
        self.assertEqual(conditioning["class_type"], "AIToolkitKreaEditConditioning")
        self.assertEqual(conditioning["inputs"]["prompt"], request.prompt)
        self.assertEqual(conditioning["inputs"]["negative_prompt"], request.negative_prompt)
        self.assertEqual(conditioning["inputs"]["match_target_res"], True)
        self.assertEqual(conditioning["inputs"]["vlm_max_pixels"], 56**2)
        self.assertEqual(conditioning["inputs"]["control_image_max_pixels"], 64**2)
        self.assertEqual(conditioning["inputs"]["dtype"], "float16")
        self.assertEqual(conditioning["inputs"]["background"], "[19, 71, 139]")
        self.assertEqual(conditioning["inputs"]["image1"], ["72", 0])
        self.assertEqual(conditioning["inputs"]["mask1"], ["72", 1])
        self.assertEqual(conditioning["inputs"]["image2"], ["74", 0])
        self.assertEqual(conditioning["inputs"]["mask2"], ["74", 1])
        self.assertEqual(conditioning["inputs"]["image3"], ["75", 0])
        self.assertEqual(conditioning["inputs"]["mask3"], ["75", 1])
        for node_id, path in (("72", request.control_image), ("74", request.control_image_2),
                              ("75", request.control_image_3)):
            self.assertEqual(graph[node_id]["inputs"]["image"], path)

        self.assertEqual(graph["79"]["class_type"], "Krea2OstrisEditModelPatch")
        self.assertEqual(graph["79"]["inputs"]["kv_cache"], True)
        self.assertEqual(graph["79"]["inputs"]["model"], ["111", 0])
        self.assertEqual(graph["111"]["inputs"]["absolute_path"], request.training_lora_path)
        self.assertEqual(graph["111"]["inputs"]["lora_strength"], -0.65)
        self.assertEqual(graph["111"]["inputs"]["model"], ["55", 0])
        self.assertEqual(graph["29"]["inputs"]["filename_prefix"], request.filename_prefix)

    def test_optional_inference_lora_is_upstream_of_training_lora(self):
        graph = get_workflow_for_sample(
            DEFAULT_COMFY_KREA2_OSTRIS_EDIT_WORKFLOW_PATH,
            self.request(inference_lora="helpers/identity.safetensors", inference_lora_strength=0.3),
        )

        self.assertEqual(graph["71"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["71"]["inputs"]["lora_name"], "helpers/identity.safetensors")
        self.assertEqual(graph["71"]["inputs"]["strength_model"], 0.3)
        self.assertEqual(graph["111"]["inputs"]["model"], ["71", 0])

    def test_single_reference_graph_omits_unrequested_reference_nodes(self):
        graph = get_workflow_for_sample(
            DEFAULT_COMFY_KREA2_OSTRIS_EDIT_WORKFLOW_PATH, self.request()
        )

        self.assertEqual(graph["84"]["inputs"]["image1"], ["72", 0])
        self.assertNotIn("image2", graph["84"]["inputs"])
        self.assertNotIn("image3", graph["84"]["inputs"])
        self.assertNotIn("74", graph)
        self.assertNotIn("75", graph)

    def test_workflow_guards_reject_wrong_models_specialized_mode_and_batch(self):
        for overrides, message in (
            ({"model_arch": "qwen_image_2"}, "requires a Krea 2 edit-training model"),
            ({"model_kwargs": {"edit": False}}, "requires a Krea 2 edit-training model"),
            ({"specialized_flow": True}, "for diffusion_trainer"),
            ({"control_image": None}, "at least one control image"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                get_workflow_for_sample(
                    DEFAULT_COMFY_KREA2_OSTRIS_EDIT_WORKFLOW_PATH,
                    self.request(**overrides),
                )
        with self.assertRaisesRegex(ValueError, "individual requests"):
            get_workflow_for_samples(
                DEFAULT_COMFY_KREA2_OSTRIS_EDIT_WORKFLOW_PATH, object()
            )

    def test_specialized_krea_edit_preview_path_remains_native(self):
        from dataclasses import replace
        from testing.test_cross_model_comfy_templates import CrossModelComfyTemplateTests

        request = replace(
            CrossModelComfyTemplateTests().request("krea2"),
            specialized_flow=True,
            model_kwargs={"edit": True, "kv_cache": True, "match_target_res": True},
            control_image="reference.png",
        )
        graph = get_workflow_for_sample(
            "config/comfy_templates/krea2_lora_sample.json.njk", request
        )

        self.assertEqual(graph["17"]["class_type"], "SamplerCustomAdvanced")
        self.assertEqual(graph["1"]["class_type"], "AIToolkitKreaEditConditioning")
        self.assertEqual(graph["85"]["class_type"], "Krea2OstrisEditModelPatch")
        self.assertEqual(graph["85"]["inputs"]["kv_cache"], True)

    def trainer(self, root):
        trainer = object.__new__(BaseSDTrainProcess)
        trainer.save_root = root
        trainer.model_config = ModelConfig(
            name_or_path="krea/Krea-2-Raw",
            arch="krea2:o_edit",
            model_kwargs={"edit": True, "match_target_res": True, "kv_cache": True},
        )
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

    def sample(self, root, **references):
        return GenerateImageConfig(
            prompt="edit this", seed=42, width=1024, height=512,
            num_inference_steps=9, guidance_scale=1, output_path=str(Path(root) / "sample.png"),
            **references,
        )

    def comfy(self):
        return SimpleNamespace(
            workflow_path=DEFAULT_COMFY_KREA2_OSTRIS_EDIT_WORKFLOW_PATH,
            api_url="http://unused", timeout=1, run_in_background=False,
            negative_prompt="avoid blur", model="krea.safetensors", vae="vae.safetensors",
            audio_vae="", text_encoder="te.safetensors", sampler="euler",
            scheduler="simple", inference_lora="", inference_lora_strength=1,
            output_format="png", output_quality="high",
        )

    def test_invalid_references_fail_before_checkpoint_cleanup_or_client_upload(self):
        cases = (
            ("missing first", {}, ValueError),
            ("missing file", {"ctrl_img_1": "/does/not/exist.png"}, FileNotFoundError),
            ("nonconsecutive third", {"ctrl_img_1": "reference-1.png", "ctrl_img_3": "reference-3.png"}, ValueError),
            ("conflicting alias", {"ctrl_img": "legacy.png", "ctrl_img_1": "explicit.png"}, ValueError),
        )
        for name, references, error_type in cases:
            with self.subTest(name=name), TemporaryDirectory() as root:
                trainer = self.trainer(root)
                with patch("jobs.process.BaseSDTrainProcess.ComfyApiClient") as client:
                    with self.assertRaises(error_type):
                        trainer._render_comfy_samples(
                            [self.sample(root, **references)],
                            SimpleNamespace(comfy=self.comfy()), step=10,
                        )
                client.assert_not_called()
                trainer._save_current_network_for_comfy.assert_not_called()
                trainer._cleanup_legacy_comfy_sample_loras.assert_not_called()


if __name__ == "__main__":
    unittest.main()
