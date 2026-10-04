import copy
import json
import pathlib
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def install_config_module_import_stubs():
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object
        sys.modules["torch"] = torch_stub
    if "torchaudio" not in sys.modules:
        torchaudio_stub = types.ModuleType("torchaudio")
        torchaudio_stub.save = lambda *args, **kwargs: None
        sys.modules["torchaudio"] = torchaudio_stub
    if "torchao" not in sys.modules:
        sys.modules["torchao"] = types.ModuleType("torchao")
        sys.modules["torchao.quantization"] = types.ModuleType("torchao.quantization")
        quant_primitives_stub = types.ModuleType("torchao.quantization.quant_primitives")
        quant_primitives_stub._DTYPE_TO_BIT_WIDTH = {}
        sys.modules["torchao.quantization.quant_primitives"] = quant_primitives_stub
    if "toolkit.audio.album_artwork" not in sys.modules:
        album_artwork_stub = types.ModuleType("toolkit.audio.album_artwork")
        album_artwork_stub.add_album_artwork = lambda *args, **kwargs: None
        sys.modules["toolkit.audio.album_artwork"] = album_artwork_stub
    if "toolkit.prompt_utils" not in sys.modules:
        prompt_utils_stub = types.ModuleType("toolkit.prompt_utils")
        prompt_utils_stub.PromptEmbeds = object
        sys.modules["toolkit.prompt_utils"] = prompt_utils_stub


class ComfySampleWorkflowTests(unittest.TestCase):
    def setUp(self):
        from toolkit.comfy_sample import ComfySampleRequest

        self.workflow = {
            "2": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 4}},
            "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "old_model.safetensors"}},
            "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "old_clip.safetensors"}},
            "6": {"class_type": "VAELoader", "inputs": {"vae_name": "old_vae.safetensors"}},
            "60": {"class_type": "VAEUtils_CustomVAELoader", "inputs": {"vae_name": "old_custom_vae.safetensors", "disable_offload": True}},
            "17": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": ["26", 3],
                    "steps": 30,
                    "cfg": 4,
                    "sampler_name": "old_sampler",
                    "scheduler": "old_scheduler",
                    "model": ["37", 0],
                    "positive": ["1", 0],
                    "negative": ["7", 0],
                    "latent_image": ["2", 0],
                },
            },
            "23": {
                "class_type": "SaveImageWithMetaData",
                "inputs": {
                    "filename_prefix": "old_prefix",
                    "output_format": "png",
                    "quality": "low",
                    "images": ["35", 0],
                },
            },
            "22": {
                "class_type": "JWDatetimeString",
                "inputs": {"format": "%Y-%m-%d/ai-toolkit_[FILENAME_WITHOUT_.safetensors]"},
            },
            "26": {"class_type": "Seed", "inputs": {"seed": 0}},
            "35": {"class_type": "AddLabel", "inputs": {"text": "old_label", "image": ["3", 0]}},
            "36": {
                "class_type": "load_lora_from_absolute_path",
                "inputs": {
                    "absolute_path": "",
                    "lora_strength": 1,
                    "model": ["4", 0],
                    "clip": ["5", 0],
                },
            },
            "37": {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": "old_inference.safetensors",
                    "strength_model": 1,
                    "strength_clip": 1,
                    "model": ["36", 0],
                    "clip": ["36", 1],
                },
            },
            "39": {"class_type": "Text Multiline", "inputs": {"text": "old prompt"}},
        }
        self.request = ComfySampleRequest(
            prompt="new prompt",
            width=904,
            height=1464,
            steps=8,
            cfg=1.5,
            seed=123,
            model="krea2_raw.safetensors",
            vae="wan_vae.safetensors",
            text_encoder="qwen_clip.safetensors",
            sampler="euler",
            scheduler="simple",
            inference_lora="krea2_turbo.safetensors",
            inference_lora_strength=0.35,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/sample_0001",
        )

    def test_patch_workflow_sets_comfy_sample_inputs(self):
        from toolkit.comfy_sample import patch_workflow_for_sample

        patched = patch_workflow_for_sample(copy.deepcopy(self.workflow), self.request)

        self.assertEqual(patched["4"]["inputs"]["unet_name"], "krea2_raw.safetensors")
        self.assertEqual(patched["5"]["inputs"]["clip_name"], "qwen_clip.safetensors")
        self.assertEqual(patched["6"]["inputs"]["vae_name"], "wan_vae.safetensors")
        self.assertEqual(patched["60"]["inputs"]["vae_name"], "wan_vae.safetensors")
        self.assertEqual(patched["2"]["inputs"]["width"], 904)
        self.assertEqual(patched["2"]["inputs"]["height"], 1464)
        self.assertEqual(patched["2"]["inputs"]["batch_size"], 1)
        self.assertEqual(patched["17"]["inputs"]["steps"], 8)
        self.assertEqual(patched["17"]["inputs"]["cfg"], 1.5)
        self.assertEqual(patched["17"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(patched["17"]["inputs"]["scheduler"], "simple")
        self.assertEqual(patched["26"]["inputs"]["seed"], 123)
        self.assertEqual(patched["36"]["inputs"]["absolute_path"], "/tmp/current_lora.safetensors")
        self.assertEqual(patched["37"]["inputs"]["lora_name"], "krea2_turbo.safetensors")
        self.assertEqual(patched["37"]["inputs"]["strength_model"], 0.35)
        self.assertEqual(patched["37"]["inputs"]["strength_clip"], 0.35)
        self.assertEqual(patched["23"]["inputs"]["output_format"], "webp_with_json")
        self.assertEqual(patched["23"]["inputs"]["quality"], "high")
        self.assertEqual(patched["23"]["inputs"]["filename_prefix"], "ai-toolkit/sample_0001")
        self.assertEqual(patched["22"]["inputs"]["format"], "%Y-%m-%d/ai-toolkit_[current_lora]")
        self.assertEqual(patched["35"]["inputs"]["text"], "current_lora")
        self.assertEqual(patched["39"]["inputs"]["text"], "new prompt")

    def test_patch_workflow_disables_regular_lora_loader_when_no_inference_lora(self):
        from toolkit.comfy_sample import patch_workflow_for_sample

        request = replace(self.request, inference_lora="")
        patched = patch_workflow_for_sample(copy.deepcopy(self.workflow), request)

        self.assertEqual(patched["17"]["inputs"]["model"], ["36", 0])
        self.assertNotIn("37", patched)

    def test_patch_workflow_replaces_zeroed_negative_conditioning(self):
        from toolkit.comfy_sample import patch_workflow_for_sample

        workflow = copy.deepcopy(self.workflow)
        workflow["1"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old prompt", "clip": ["37", 1]},
        }
        workflow["7"] = {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["1", 0]},
        }
        patched = patch_workflow_for_sample(
            workflow, replace(self.request, negative_prompt="blurry, text")
        )

        self.assertEqual(patched["7"]["class_type"], "CLIPTextEncode")
        self.assertEqual(patched["7"]["inputs"]["text"], "blurry, text")
        self.assertEqual(patched["7"]["inputs"]["clip"], ["37", 1])
        self.assertEqual(patched["17"]["inputs"]["negative"], ["7", 0])

    def test_renders_nunjucks_template_workflow(self):
        from toolkit.comfy_sample import render_nunjucks_workflow

        rendered = render_nunjucks_workflow(
            "config/comfy_templates/krea2_lora_sample.json.njk",
            self.request,
        )

        self.assertEqual(rendered["4"]["inputs"]["unet_name"], "krea2_raw.safetensors")
        self.assertEqual(rendered["5"]["inputs"]["clip_name"], "qwen_clip.safetensors")
        self.assertEqual(rendered["3"]["class_type"], "VAEUtils_VAEDecodeTiled")
        self.assertEqual(rendered["3"]["inputs"]["upscale"], -1)
        self.assertFalse(rendered["3"]["inputs"]["tile"])
        self.assertEqual(rendered["3"]["inputs"]["tile_size"], 512)
        self.assertEqual(rendered["3"]["inputs"]["overlap"], 64)
        self.assertEqual(rendered["3"]["inputs"]["temporal_size"], 4096)
        self.assertEqual(rendered["3"]["inputs"]["temporal_overlap"], 64)
        self.assertEqual(rendered["6"]["class_type"], "VAEUtils_CustomVAELoader")
        self.assertEqual(rendered["6"]["inputs"]["vae_name"], "wan_vae.safetensors")
        self.assertTrue(rendered["6"]["inputs"]["disable_offload"])
        self.assertEqual(rendered["2"]["inputs"]["width"], 904)
        self.assertEqual(rendered["2"]["inputs"]["height"], 1464)
        self.assertEqual(rendered["2"]["inputs"]["batch_size"], 1)
        self.assertEqual(rendered["17"]["inputs"]["steps"], 8)
        self.assertEqual(rendered["17"]["inputs"]["cfg"], 1.5)
        self.assertEqual(rendered["17"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(rendered["17"]["inputs"]["scheduler"], "simple")
        self.assertEqual(rendered["26"]["inputs"]["seed"], 123)
        self.assertEqual(rendered["36"]["inputs"]["absolute_path"], "/tmp/current_lora.safetensors")
        self.assertEqual(rendered["37"]["inputs"]["lora_name"], "krea2_turbo.safetensors")
        self.assertEqual(rendered["37"]["inputs"]["strength_model"], 0.35)
        self.assertEqual(rendered["37"]["inputs"]["strength_clip"], 0.35)
        self.assertEqual(rendered["23"]["inputs"]["output_format"], "webp_with_json")
        self.assertEqual(rendered["23"]["inputs"]["quality"], "high")
        self.assertEqual(rendered["23"]["inputs"]["metadata_scope"], "full")
        self.assertTrue(rendered["23"]["inputs"]["prefer_nearest"])
        self.assertNotIn("file_format", rendered["23"]["inputs"])
        self.assertEqual(rendered["23"]["inputs"]["filename_prefix"], "ai-toolkit/sample_0001")
        self.assertEqual(rendered["23"]["inputs"]["images"], ["3", 0])
        self.assertNotIn("22", rendered)
        self.assertNotIn("35", rendered)
        self.assertEqual(rendered["39"]["inputs"]["text"], "new prompt")

    def test_nunjucks_template_disables_regular_lora_loader_when_no_inference_lora(self):
        from toolkit.comfy_sample import render_nunjucks_workflow

        request = replace(self.request, inference_lora="")
        rendered = render_nunjucks_workflow(
            "config/comfy_templates/krea2_lora_sample.json.njk",
            request,
        )

        self.assertEqual(rendered["1"]["inputs"]["clip"], ["36", 1])
        self.assertEqual(rendered["17"]["inputs"]["model"], ["36", 0])
        self.assertNotIn("37", rendered)

    def test_image_templates_encode_nonempty_negative_prompt(self):
        from toolkit.comfy_sample import (
            ComfyBatchSampleRequest,
            render_nunjucks_workflow,
        )

        negative_prompt = 'blurry, "watermark"'
        batch_request = ComfyBatchSampleRequest(
            prompts=["first", "second"],
            width=1024,
            height=1024,
            steps=8,
            cfg=2.5,
            seeds=[123, 124],
            model="model.safetensors",
            vae="vae.safetensors",
            text_encoder="clip.safetensors",
            sampler="euler",
            scheduler="simple",
            inference_lora="",
            inference_lora_strength=1,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/negative_batch",
            control_images=["first.png", "second.png"],
            negative_prompt=negative_prompt,
        )
        single_request = replace(
            self.request,
            negative_prompt=negative_prompt,
            control_image="first.png",
        )
        for path, request, negative_node, field in (
            ("krea2_lora_sample.json.njk", single_request, "7", "text"),
            ("krea2_lora_sample_batch_easy_use.json.njk", batch_request, "7", "text"),
            ("qwen_image_2_lora_sample.json.njk", single_request, "1", "negative_prompt"),
            ("qwen_image_2_lora_sample_batch_easy_use.json.njk", batch_request, "71", "negative_prompt"),
            ("qwen_image_edit_lora_sample.json.njk", single_request, "72", "prompt"),
            ("qwen_image_edit_lora_sample_batch_easy_use.json.njk", batch_request, "72", "prompt"),
            ("qwen_image_edit_plus_lora_sample.json.njk", single_request, "72", "prompt"),
            ("qwen_image_edit_plus_lora_sample_batch_easy_use.json.njk", batch_request, "72", "prompt"),
        ):
            with self.subTest(path=path):
                rendered = render_nunjucks_workflow(f"config/comfy_templates/{path}", request)
                self.assertEqual(rendered[negative_node]["inputs"][field], negative_prompt)
                if path.startswith("krea2"):
                    self.assertEqual(rendered[negative_node]["class_type"], "CLIPTextEncode")
                    self.assertEqual(rendered["17"]["inputs"]["negative"], [negative_node, 0])
                elif path.startswith("qwen_image_edit"):
                    self.assertEqual(rendered["17"]["inputs"]["negative"], [negative_node, 0])
                else:
                    self.assertEqual(rendered["17"]["inputs"]["negative"], [negative_node, 1])

        empty_krea = render_nunjucks_workflow(
            "config/comfy_templates/krea2_lora_sample.json.njk", self.request
        )
        self.assertEqual(empty_krea["7"]["class_type"], "ConditioningZeroOut")

    def test_renders_qwen_image_2_single_template_with_native_vae_nodes(self):
        import toolkit.comfy_sample as comfy_sample

        request = replace(
            self.request,
            model="qwen_image_2.1_int8_convrot.safetensors",
            vae="qwen_image_2.1_vae_bf16.safetensors",
            text_encoder="qwen3vl_8b_int8_convrot.safetensors",
            inference_lora="",
        )
        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_2_WORKFLOW_PATH,
            request,
        )

        self.assertEqual(rendered["1"]["class_type"], "TextEncodeQwenImage21")
        self.assertEqual(rendered["1"]["inputs"]["prompt"], "new prompt")
        self.assertEqual(rendered["1"]["inputs"]["negative_prompt"], "")
        self.assertEqual(rendered["5"]["inputs"]["type"], "qwen_image")
        self.assertEqual(rendered["6"]["class_type"], "VAELoader")
        self.assertEqual(rendered["3"]["class_type"], "VAEDecodeTiled")
        self.assertEqual(rendered["3"]["inputs"]["tile_size"], 512)
        self.assertEqual(rendered["17"]["inputs"]["positive"], ["1", 0])
        self.assertEqual(rendered["17"]["inputs"]["negative"], ["1", 1])
        self.assertNotIn("vae", rendered["1"]["inputs"])
        self.assertNotIn("images.image_1", rendered["1"]["inputs"])
        self.assertNotIn("80", rendered)
        self.assertFalse(any(
            node["class_type"].startswith("VAEUtils_")
            for node in rendered.values()
        ))

    def test_qwen_image_2_single_template_accepts_three_reference_images(self):
        import toolkit.comfy_sample as comfy_sample

        request = replace(
            self.request,
            model="qwen_image_2.1_int8_convrot.safetensors",
            vae="qwen_image_2.1_vae_bf16.safetensors",
            text_encoder="qwen3vl_8b_int8_convrot.safetensors",
            inference_lora="",
            control_image="ai-toolkit/reference_1.png",
            control_image_2="ai-toolkit/reference_2.png",
            control_image_3="ai-toolkit/reference_3.png",
        )
        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_2_WORKFLOW_PATH,
            request,
        )

        self.assertEqual(rendered["1"]["inputs"]["vae"], ["6", 0])
        self.assertEqual(rendered["1"]["inputs"]["images.image_1"], ["80", 0])
        self.assertEqual(rendered["1"]["inputs"]["images.image_2"], ["86", 0])
        self.assertEqual(rendered["1"]["inputs"]["images.image_3"], ["88", 0])
        self.assertEqual(rendered["80"]["inputs"]["image"], request.control_image)
        self.assertEqual(rendered["86"]["inputs"]["image"], request.control_image_2)
        self.assertEqual(rendered["88"]["inputs"]["image"], request.control_image_3)
        self.assertEqual(rendered["17"]["inputs"]["latent_image"], ["2", 0])

    def test_qwen_image_2_template_uses_absolute_loader_for_prepared_dora(self):
        import toolkit.comfy_sample as comfy_sample

        request = replace(
            self.request,
            inference_lora="qwen2.1/legacy-dora.safetensors",
            inference_lora_absolute_path=(
                "/tmp/.comfy_inference_lora_cache/legacy-comfy-dora.safetensors"
            ),
        )
        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_2_WORKFLOW_PATH,
            request,
        )

        self.assertEqual(rendered["37"]["class_type"], "load_lora_from_absolute_path")
        self.assertEqual(
            rendered["37"]["inputs"]["absolute_path"],
            request.inference_lora_absolute_path,
        )
        self.assertEqual(rendered["37"]["inputs"]["lora_strength"], 0.35)
        self.assertNotIn("lora_name", rendered["37"]["inputs"])

    def test_qwen_image_2_reference_slots_must_be_contiguous(self):
        import toolkit.comfy_sample as comfy_sample

        request = replace(
            self.request,
            control_image="",
            control_image_2="ai-toolkit/reference_2.png",
        )
        with self.assertRaisesRegex(ValueError, "control image 2 requires control image 1"):
            comfy_sample.render_nunjucks_workflow(
                comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_2_WORKFLOW_PATH,
                request,
            )

    def test_renders_qwen_image_2_batch_template(self):
        import toolkit.comfy_sample as comfy_sample

        batch_request = comfy_sample.ComfyBatchSampleRequest(
            prompts=["first prompt", "second prompt"],
            width=1440,
            height=1440,
            steps=20,
            cfg=3,
            seeds=[42, 43],
            model="qwen_image_2.1_int8_convrot.safetensors",
            vae="qwen_image_2.1_vae_bf16.safetensors",
            text_encoder="qwen3vl_8b_int8_convrot.safetensors",
            sampler="seeds_2",
            scheduler="simple",
            inference_lora="",
            inference_lora_strength=1,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/qwen21_batch",
        )
        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_2_BATCH_WORKFLOW_PATH,
            batch_request,
        )

        self.assertEqual(rendered["41"]["inputs"]["total"], 2)
        self.assertEqual(rendered["71"]["class_type"], "TextEncodeQwenImage21")
        self.assertEqual(rendered["71"]["inputs"]["prompt"], ["70", 0])
        self.assertEqual(rendered["17"]["inputs"]["negative"], ["71", 1])
        self.assertEqual(rendered["6"]["class_type"], "VAELoader")
        self.assertEqual(rendered["3"]["class_type"], "VAEDecodeTiled")
        self.assertEqual(rendered["23"]["inputs"]["images"], ["50", 0])
        self.assertNotIn("80", rendered)
        self.assertNotIn("86", rendered)
        self.assertNotIn("88", rendered)

    def test_qwen_image_2_batch_omits_unused_reference_slots(self):
        import toolkit.comfy_sample as comfy_sample

        batch_request = comfy_sample.ComfyBatchSampleRequest(
            prompts=["first", "second", "third"],
            width=1376,
            height=1376,
            steps=20,
            cfg=3,
            seeds=[42, 42, 42],
            model="qwen_image_2.1_int8_convrot.safetensors",
            vae="qwen_image_2.1_vae_bf16.safetensors",
            text_encoder="qwen3vl_8b_int8_convrot.safetensors",
            sampler="euler",
            scheduler="sgm_uniform",
            inference_lora="",
            inference_lora_strength=1,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/qwen21_batch",
            control_images=["ref_1.png", "ref_2.png", "ref_3.png"],
        )
        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_2_BATCH_WORKFLOW_PATH,
            batch_request,
        )

        self.assertEqual(rendered["71"]["inputs"]["images.image_1"], ["80", 0])
        self.assertNotIn("images.image_2", rendered["71"]["inputs"])
        self.assertNotIn("images.image_3", rendered["71"]["inputs"])
        self.assertNotIn("86", rendered)
        self.assertNotIn("88", rendered)
        for node in rendered.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], rendered)

    def test_qwen_image_2_batch_template_accepts_reference_images(self):
        import toolkit.comfy_sample as comfy_sample

        batch_request = comfy_sample.ComfyBatchSampleRequest(
            prompts=["first prompt", "second prompt"],
            width=1440,
            height=1440,
            steps=20,
            cfg=3,
            seeds=[42, 43],
            model="qwen_image_2.1_int8_convrot.safetensors",
            vae="qwen_image_2.1_vae_bf16.safetensors",
            text_encoder="qwen3vl_8b_int8_convrot.safetensors",
            sampler="seeds_2",
            scheduler="simple",
            inference_lora="",
            inference_lora_strength=1,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/qwen21_batch",
            control_images=["ai-toolkit/ref_1.png", "ai-toolkit/ref_2.png"],
            control_images_2=["ai-toolkit/ref_1b.png", "ai-toolkit/ref_2b.png"],
        )
        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_2_BATCH_WORKFLOW_PATH,
            batch_request,
        )

        self.assertEqual(rendered["1400"]["inputs"]["image"], "ai-toolkit/ref_1.png")
        self.assertEqual(rendered["1401"]["inputs"]["image"], "ai-toolkit/ref_2.png")
        self.assertEqual(rendered["1600"]["inputs"]["image"], "ai-toolkit/ref_1b.png")
        self.assertEqual(rendered["1601"]["inputs"]["image"], "ai-toolkit/ref_2b.png")
        self.assertEqual(rendered["71"]["inputs"]["vae"], ["6", 0])
        self.assertEqual(rendered["71"]["inputs"]["images.image_1"], ["80", 0])
        self.assertEqual(rendered["71"]["inputs"]["images.image_2"], ["86", 0])
        self.assertEqual(rendered["80"]["class_type"], "easy indexAnything")
        self.assertEqual(rendered["86"]["class_type"], "easy indexAnything")

    def test_renders_minimax_h3_fl2v_video_template(self):
        from toolkit.comfy_sample import render_nunjucks_workflow

        request = replace(
            self.request,
            model="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            vae="minimax_h3_video_vae_fp16.safetensors",
            audio_vae="minimax_h3_audio_vae_fp32.safetensors",
            text_encoder="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            sampler="res_multistep",
            scheduler="simple",
            control_image="ai-toolkit/uploaded_first_frame.png",
            num_frames=100,
            fps=12,
            training_lora_strength=0.7,
        )
        rendered = render_nunjucks_workflow(
            "config/comfy_templates/minimax_h3_fl2v_lora_sample.json.njk",
            request,
        )

        self.assertEqual(rendered["10"]["inputs"]["image"], "ai-toolkit/uploaded_first_frame.png")
        self.assertEqual(rendered["20"]["inputs"]["unet_name"], request.model)
        self.assertEqual(rendered["21"]["inputs"]["clip_name"], request.text_encoder)
        self.assertEqual(rendered["22"]["inputs"]["vae_name"], request.vae)
        self.assertEqual(rendered["23"]["inputs"]["vae_name"], request.audio_vae)
        self.assertEqual(rendered["30"]["inputs"]["absolute_path"], request.training_lora_path)
        self.assertEqual(rendered["30"]["inputs"]["lora_strength"], 0.7)
        self.assertEqual(rendered["40"]["inputs"]["prompt"], request.prompt)
        self.assertEqual(rendered["40"]["inputs"]["width"], 896)
        self.assertEqual(rendered["40"]["inputs"]["height"], 1440)
        self.assertEqual(rendered["40"]["inputs"]["length"], 107)
        self.assertEqual(rendered["40"]["inputs"]["first_frame"], ["10", 0])
        self.assertEqual(rendered["50"]["inputs"]["noise_seed"], request.seed)
        self.assertEqual(rendered["51"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(rendered["52"]["inputs"]["steps"], request.steps)
        self.assertEqual(rendered["52"]["inputs"]["model"], ["31", 0])
        self.assertEqual(rendered["62"]["inputs"]["fps"], 24)
        self.assertEqual(rendered["63"]["class_type"], "SaveVideo")
        self.assertEqual(rendered["63"]["inputs"]["filename_prefix"], request.filename_prefix)

    def test_minimax_h3_fl2v_template_can_skip_optional_inference_lora(self):
        from toolkit.comfy_sample import render_nunjucks_workflow

        request = replace(
            self.request,
            inference_lora="",
            audio_vae="minimax_h3_audio_vae_fp32.safetensors",
            control_image="ai-toolkit/uploaded_first_frame.png",
            num_frames=107,
            fps=24,
        )
        rendered = render_nunjucks_workflow(
            "config/comfy_templates/minimax_h3_fl2v_lora_sample.json.njk",
            request,
        )

        self.assertNotIn("31", rendered)
        self.assertEqual(rendered["40"]["inputs"]["clip"], ["30", 1])
        self.assertEqual(rendered["52"]["inputs"]["model"], ["30", 0])

    def test_minimax_h3_fl2v_template_supports_text_only_video(self):
        from toolkit.comfy_sample import render_nunjucks_workflow

        request = replace(
            self.request,
            audio_vae="minimax_h3_audio_vae_fp32.safetensors",
            control_image=None,
            control_image_2=None,
            num_frames=107,
            fps=24,
        )
        rendered = render_nunjucks_workflow(
            "config/comfy_templates/minimax_h3_fl2v_lora_sample.json.njk",
            request,
        )

        self.assertNotIn("10", rendered)
        self.assertNotIn("11", rendered)
        self.assertNotIn("first_frame", rendered["40"]["inputs"])
        self.assertNotIn("last_frame", rendered["40"]["inputs"])

    def test_minimax_h3_fl2v_template_supports_last_frame(self):
        from toolkit.comfy_sample import render_nunjucks_workflow

        request = replace(
            self.request,
            audio_vae="minimax_h3_audio_vae_fp32.safetensors",
            control_image="ai-toolkit/uploaded_first_frame.png",
            control_image_2="ai-toolkit/uploaded_last_frame.png",
            num_frames=107,
            fps=24,
        )
        rendered = render_nunjucks_workflow(
            "config/comfy_templates/minimax_h3_fl2v_lora_sample.json.njk",
            request,
        )

        self.assertEqual(rendered["11"]["inputs"]["image"], request.control_image_2)
        self.assertEqual(rendered["40"]["inputs"]["last_frame"], ["11", 0])

    def test_renders_easy_use_batch_template_workflow(self):
        import toolkit.comfy_sample as comfy_sample

        self.assertTrue(hasattr(comfy_sample, "ComfyBatchSampleRequest"))
        batch_request = comfy_sample.ComfyBatchSampleRequest(
            prompts=["first prompt", "second prompt"],
            width=904,
            height=1464,
            steps=8,
            cfg=1.5,
            seeds=[123, 124],
            model="krea2_raw.safetensors",
            vae="wan_vae.safetensors",
            text_encoder="qwen_clip.safetensors",
            sampler="euler",
            scheduler="simple",
            inference_lora="krea2_turbo.safetensors",
            inference_lora_strength=0.35,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/sample_batch",
        )

        rendered = comfy_sample.render_nunjucks_workflow(
            "config/comfy_templates/krea2_lora_sample_batch_easy_use.json.njk",
            batch_request,
        )

        self.assertEqual(rendered["41"]["class_type"], "easy forLoopStart")
        self.assertEqual(rendered["41"]["inputs"]["total"], 2)
        self.assertEqual(rendered["50"]["class_type"], "easy forLoopEnd")
        self.assertEqual(rendered["50"]["inputs"]["flow"], ["41", 0])
        self.assertEqual(rendered["50"]["inputs"]["initial_value1"], ["74", 0])
        self.assertEqual(rendered["70"]["class_type"], "easy indexAnything")
        self.assertEqual(rendered["70"]["inputs"]["index"], ["41", 1])
        self.assertEqual(rendered["71"]["class_type"], "CLIPTextEncode")
        self.assertEqual(rendered["71"]["inputs"]["text"], ["70", 0])
        self.assertEqual(rendered["73"]["class_type"], "easy indexAnything")
        self.assertEqual(rendered["17"]["inputs"]["seed"], ["73", 0])
        self.assertEqual(rendered["74"]["class_type"], "easy batchAnything")
        self.assertEqual(rendered["74"]["inputs"]["any_1"], ["41", 2])
        self.assertEqual(rendered["74"]["inputs"]["any_2"], ["3", 0])
        self.assertEqual(rendered["23"]["inputs"]["images"], ["50", 0])
        self.assertEqual(rendered["23"]["inputs"]["filename_prefix"], "ai-toolkit/sample_batch")
        self.assertEqual(rendered["23"]["inputs"]["output_format"], "webp_with_json")
        self.assertEqual(rendered["23"]["inputs"]["quality"], "high")
        self.assertEqual(rendered["23"]["inputs"]["metadata_scope"], "full")
        self.assertTrue(rendered["23"]["inputs"]["prefer_nearest"])
        self.assertNotIn("file_format", rendered["23"]["inputs"])
        self.assertEqual(rendered["1000"]["inputs"]["value"], "first prompt")
        self.assertEqual(rendered["1001"]["inputs"]["value"], "second prompt")
        self.assertEqual(rendered["1100"]["inputs"]["value"], 123)
        self.assertEqual(rendered["1101"]["inputs"]["value"], 124)

    def test_renders_qwen_image_edit_easy_use_batch_template(self):
        import toolkit.comfy_sample as comfy_sample

        batch_request = comfy_sample.ComfyBatchSampleRequest(
            prompts=["make it winter", "make it sunset"],
            width=1024,
            height=1024,
            steps=20,
            cfg=2.5,
            seeds=[123, 124],
            model="qwen_image_edit_fp8_e4m3fn.safetensors",
            vae="qwen_image_vae.safetensors",
            text_encoder="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            sampler="euler",
            scheduler="simple",
            inference_lora="",
            inference_lora_strength=1,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/qwen_edit_batch",
            control_images=[
                "ai-toolkit/first.png",
                "ai-toolkit/second.png",
            ],
        )

        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_EDIT_BATCH_WORKFLOW_PATH,
            batch_request,
        )

        self.assertEqual(rendered["4"]["inputs"]["unet_name"], "qwen_image_edit_fp8_e4m3fn.safetensors")
        self.assertEqual(rendered["5"]["inputs"]["type"], "qwen_image")
        self.assertEqual(rendered["6"]["inputs"]["vae_name"], "qwen_image_vae.safetensors")
        self.assertEqual(rendered["1400"]["class_type"], "LoadImage")
        self.assertEqual(rendered["1400"]["inputs"]["image"], "ai-toolkit/first.png")
        self.assertEqual(rendered["1401"]["inputs"]["image"], "ai-toolkit/second.png")
        self.assertEqual(rendered["80"]["inputs"]["any"], ["1500", 0])
        self.assertNotIn("81", rendered)
        self.assertEqual(rendered["82"]["class_type"], "ImageScaleToTotalPixels")
        self.assertEqual(rendered["82"]["inputs"]["megapixels"], 1)
        self.assertEqual(rendered["82"]["inputs"]["image"], ["80", 0])
        self.assertEqual(rendered["71"]["class_type"], "TextEncodeQwenImageEdit")
        self.assertEqual(rendered["71"]["inputs"]["prompt"], ["70", 0])
        self.assertEqual(rendered["71"]["inputs"]["image"], ["82", 0])
        self.assertEqual(rendered["72"]["class_type"], "TextEncodeQwenImageEdit")
        self.assertEqual(rendered["72"]["inputs"]["prompt"], "")
        self.assertEqual(rendered["83"]["class_type"], "VAEEncode")
        self.assertEqual(rendered["83"]["inputs"]["pixels"], ["82", 0])
        self.assertEqual(rendered["84"]["class_type"], "ModelSamplingAuraFlow")
        self.assertEqual(rendered["84"]["inputs"]["shift"], 3)
        self.assertEqual(rendered["84"]["inputs"]["model"], ["36", 0])
        self.assertEqual(rendered["85"]["class_type"], "CFGNorm")
        self.assertEqual(rendered["17"]["inputs"]["model"], ["85", 0])
        self.assertEqual(rendered["17"]["inputs"]["latent_image"], ["83", 0])
        self.assertNotIn("37", rendered)

    def test_renders_qwen_image_edit_single_template(self):
        import toolkit.comfy_sample as comfy_sample

        request = replace(
            self.request,
            model="qwen_image_edit_fp8_e4m3fn.safetensors",
            control_image="ai-toolkit/reference.png",
        )
        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_EDIT_WORKFLOW_PATH,
            request,
        )

        self.assertEqual(rendered["17"]["inputs"]["seed"], 123)
        self.assertEqual(rendered["71"]["inputs"]["prompt"], "new prompt")
        self.assertEqual(rendered["80"]["inputs"]["image"], "ai-toolkit/reference.png")
        self.assertEqual(rendered["83"]["class_type"], "VAEEncode")
        self.assertEqual(rendered["23"]["inputs"]["images"], ["3", 0])
        self.assertFalse(any(node["class_type"].startswith("easy ") for node in rendered.values()))

    def test_renders_qwen_image_edit_plus_easy_use_batch_template(self):
        import toolkit.comfy_sample as comfy_sample

        batch_request = comfy_sample.ComfyBatchSampleRequest(
            prompts=["combine the subjects", "swap the materials"],
            width=1024,
            height=1024,
            steps=20,
            cfg=2.5,
            seeds=[123, 124],
            model="qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            vae="qwen_image_vae.safetensors",
            text_encoder="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            sampler="euler",
            scheduler="simple",
            inference_lora="",
            inference_lora_strength=1,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/qwen_edit_plus_batch",
            control_images=["ai-toolkit/first-a.png", "ai-toolkit/first-b.png"],
            control_images_2=["ai-toolkit/second-a.png", "ai-toolkit/second-b.png"],
            control_images_3=["ai-toolkit/third-a.png", "ai-toolkit/third-b.png"],
        )

        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_BATCH_WORKFLOW_PATH,
            batch_request,
        )

        self.assertEqual(rendered["5"]["inputs"]["type"], "qwen_image")
        self.assertEqual(rendered["1600"]["class_type"], "LoadImage")
        self.assertEqual(rendered["1600"]["inputs"]["image"], "ai-toolkit/second-a.png")
        self.assertEqual(rendered["1801"]["inputs"]["image"], "ai-toolkit/third-b.png")
        self.assertEqual(rendered["71"]["class_type"], "TextEncodeQwenImageEditPlus")
        self.assertEqual(rendered["71"]["inputs"]["image1"], ["80", 0])
        self.assertEqual(rendered["71"]["inputs"]["image2"], ["86", 0])
        self.assertEqual(rendered["71"]["inputs"]["image3"], ["88", 0])
        self.assertEqual(rendered["72"]["class_type"], "TextEncodeQwenImageEdit")
        self.assertEqual(rendered["72"]["inputs"]["prompt"], "")
        self.assertEqual(rendered["72"]["inputs"]["image"], ["80", 0])
        self.assertNotIn("image1", rendered["72"]["inputs"])
        self.assertNotIn("90", rendered)
        self.assertNotIn("91", rendered)
        self.assertEqual(rendered["17"]["inputs"]["positive"], ["71", 0])
        self.assertEqual(rendered["17"]["inputs"]["negative"], ["72", 0])
        self.assertEqual(rendered["83"]["class_type"], "EmptyLatentImage")
        self.assertEqual(rendered["83"]["inputs"]["width"], 1024)
        self.assertEqual(rendered["83"]["inputs"]["height"], 1024)
        self.assertEqual(rendered["83"]["inputs"]["batch_size"], 1)
        self.assertEqual(rendered["84"]["inputs"]["shift"], 3)
        self.assertEqual(rendered["17"]["inputs"]["latent_image"], ["83", 0])

        rendered_2511 = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_BATCH_WORKFLOW_PATH,
            replace(batch_request, model="qwen_image_edit_2511_bf16.safetensors"),
        )
        self.assertEqual(rendered_2511["84"]["inputs"]["shift"], 3.1)

    def test_renders_qwen_image_edit_plus_single_template_with_sample_resolution(self):
        import toolkit.comfy_sample as comfy_sample

        request = replace(
            self.request,
            model="qwen_image_edit_2511_bf16.safetensors",
            control_image="ai-toolkit/character.png",
            control_image_2="ai-toolkit/pose.png",
        )
        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_WORKFLOW_PATH,
            request,
        )

        self.assertEqual(rendered["17"]["inputs"]["seed"], 123)
        self.assertEqual(rendered["71"]["inputs"]["prompt"], "new prompt")
        self.assertEqual(rendered["71"]["inputs"]["image1"], ["80", 0])
        self.assertEqual(rendered["71"]["inputs"]["image2"], ["86", 0])
        self.assertNotIn("image3", rendered["71"]["inputs"])
        self.assertEqual(rendered["80"]["inputs"]["image"], "ai-toolkit/character.png")
        self.assertEqual(rendered["86"]["inputs"]["image"], "ai-toolkit/pose.png")
        self.assertNotIn("88", rendered)
        self.assertEqual(rendered["83"]["inputs"]["width"], 904)
        self.assertEqual(rendered["83"]["inputs"]["height"], 1464)
        self.assertEqual(rendered["84"]["inputs"]["shift"], 3.1)
        self.assertEqual(rendered["23"]["inputs"]["images"], ["3", 0])
        self.assertFalse(any(node["class_type"].startswith("easy ") for node in rendered.values()))

    def test_qwen_image_edit_plus_template_omits_unused_optional_images(self):
        import toolkit.comfy_sample as comfy_sample

        batch_request = comfy_sample.ComfyBatchSampleRequest(
            prompts=["first", "second"],
            width=1024,
            height=1024,
            steps=20,
            cfg=2.5,
            seeds=[123, 124],
            model="qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            vae="qwen_image_vae.safetensors",
            text_encoder="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            sampler="euler",
            scheduler="simple",
            inference_lora="",
            inference_lora_strength=1,
            output_format="webp_with_json",
            output_quality="high",
            training_lora_path="/tmp/current_lora.safetensors",
            training_lora_filename="current_lora.safetensors",
            filename_prefix="ai-toolkit/qwen_edit_plus_batch",
            control_images=["ai-toolkit/first.png", "ai-toolkit/second.png"],
        )

        rendered = comfy_sample.render_nunjucks_workflow(
            comfy_sample.DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_BATCH_WORKFLOW_PATH,
            batch_request,
        )

        self.assertNotIn("image2", rendered["71"]["inputs"])
        self.assertNotIn("image3", rendered["71"]["inputs"])
        self.assertNotIn("86", rendered)
        self.assertNotIn("87", rendered)
        self.assertNotIn("88", rendered)
        self.assertNotIn("89", rendered)

    def test_extracts_combo_options_from_comfy_object_info(self):
        from toolkit.comfy_sample import extract_input_options

        object_info = {
            "KSampler": {
                "input": {
                    "required": {
                        "sampler_name": [["euler", "dpmpp_2m"], {"tooltip": "Sampler"}],
                        "scheduler": ["COMBO", {"options": ["simple", "normal"]}],
                    }
                }
            }
        }

        self.assertEqual(extract_input_options(object_info, "KSampler", "sampler_name"), ["euler", "dpmpp_2m"])
        self.assertEqual(extract_input_options(object_info, "KSampler", "scheduler"), ["simple", "normal"])

    def test_selects_image_output_from_history(self):
        from toolkit.comfy_sample import get_history_output_images

        history = {
            "abc": {
                "outputs": {
                    "23": {
                        "images": [
                            {
                                "filename": "sample.webp",
                                "subfolder": "ai-toolkit",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        }

        self.assertEqual(
            get_history_output_images(history, "abc"),
            [{"filename": "sample.webp", "subfolder": "ai-toolkit", "type": "output"}],
        )


class ComfySampleConfigTests(unittest.TestCase):
    def test_sample_config_parses_comfy_settings(self):
        install_config_module_import_stubs()

        from toolkit.config_modules import SampleConfig

        sample = SampleConfig(
            comfy={
                "enabled": True,
                "api_url": "http://127.0.0.1:8188",
                "negative_prompt": "blurry, watermark",
                "model": "krea2_raw.safetensors",
                "vae": "wan_vae.safetensors",
                "text_encoder": "qwen_clip.safetensors",
                "sampler": "euler",
                "scheduler": "simple",
                "inference_lora": "turbo.safetensors",
                "inference_lora_strength": 0.42,
                "send_prompts_as_batch": True,
                "run_in_background": True,
                "training_lora_path_replace_from": "/mnt/train",
                "training_lora_path_replace_to": "R:/train",
                "output_format": "webp_with_json",
                "output_quality": "high",
            }
        )

        self.assertTrue(sample.comfy.enabled)
        self.assertEqual(sample.comfy.negative_prompt, "blurry, watermark")
        self.assertEqual(sample.comfy.workflow_path, "config/comfy_templates/krea2_lora_sample.json.njk")
        self.assertEqual(sample.comfy.model, "krea2_raw.safetensors")
        self.assertEqual(sample.comfy.inference_lora, "turbo.safetensors")
        self.assertEqual(sample.comfy.inference_lora_strength, 0.42)
        self.assertTrue(sample.comfy.send_prompts_as_batch)
        self.assertTrue(sample.comfy.run_in_background)
        self.assertEqual(sample.comfy.training_lora_path_replace_from, "/mnt/train")
        self.assertEqual(sample.comfy.training_lora_path_replace_to, "R:/train")

    def test_comfy_background_sampling_defaults_to_disabled(self):
        install_config_module_import_stubs()

        from toolkit.config_modules import ComfySampleConfig

        comfy = ComfySampleConfig()

        self.assertFalse(comfy.run_in_background)
        self.assertEqual(comfy.negative_prompt, "")
        self.assertEqual(comfy.timeout, 30 * 60)
        self.assertEqual(comfy.training_lora_path_replace_from, "")
        self.assertEqual(comfy.training_lora_path_replace_to, "")


class ComfyApiClientTests(unittest.TestCase):
    def test_prompt_wait_defaults_to_thirty_minutes(self):
        from toolkit.comfy_sample import ComfyApiClient

        self.assertEqual(ComfyApiClient().timeout, 30 * 60)

    def test_prompt_wait_can_be_cancelled_when_training_stops(self):
        from toolkit.comfy_sample import ComfyApiClient, ComfyPromptWaitCancelled

        client = ComfyApiClient()
        client.get_history = mock.Mock(return_value={})

        with self.assertRaises(ComfyPromptWaitCancelled):
            client.wait_for_images("abc", cancel_check=lambda: True)

        client.get_history.assert_not_called()

    def test_post_prompt_includes_workflow_metadata_for_save_nodes(self):
        from toolkit.comfy_sample import ComfyApiClient

        payloads = []
        client = ComfyApiClient()
        client._request_json = lambda method, path, payload=None: payloads.append((method, path, payload)) or {"prompt_id": "abc"}
        workflow = {
            "23": {
                "class_type": "SaveImageWithMetaData",
                "inputs": {"output_format": "webp_with_json"},
            }
        }

        prompt_id = client.post_prompt(workflow)

        self.assertEqual(prompt_id, "abc")
        self.assertEqual(payloads[0][0], "POST")
        self.assertEqual(payloads[0][1], "/api/prompt")
        self.assertIs(payloads[0][2]["prompt"], workflow)
        self.assertEqual(payloads[0][2]["client_id"], client.client_id)
        self.assertIs(payloads[0][2]["extra_data"]["extra_pnginfo"]["workflow"], workflow)

    def test_live_progress_websocket_uses_comfy_client_id(self):
        from toolkit.comfy_sample import ComfyApiClient

        client = ComfyApiClient(api_url="https://comfy.example.test:8188/api")

        self.assertEqual(
            client._progress_websocket_url(),
            f"wss://comfy.example.test:8188/api/ws?clientId={client.client_id}",
        )

    def test_wait_for_images_reports_live_websocket_progress(self):
        from toolkit.comfy_sample import ComfyApiClient

        client = ComfyApiClient(poll_interval=0.01)
        websocket = mock.Mock()
        websocket.recv.return_value = json.dumps({
            "type": "progress",
            "data": {"prompt_id": "abc", "value": 3, "max": 10},
        })
        client._progress_websocket = websocket
        client.get_history = mock.Mock(side_effect=[
            {},
            {
                "abc": {
                    "outputs": {
                        "9": {
                            "images": [{"filename": "sample.png", "type": "output"}],
                        },
                    },
                },
            },
        ])
        progress = []

        images = client.wait_for_images(
            "abc",
            progress_callback=lambda value, maximum: progress.append((value, maximum)),
        )

        self.assertEqual(progress, [(3, 10)])
        self.assertEqual(images[0]["filename"], "sample.png")
        websocket.close.assert_called_once_with()

    def test_post_prompt_reports_comfy_validation_response(self):
        import io
        import urllib.error
        from toolkit.comfy_sample import ComfyApiClient

        client = ComfyApiClient()
        client._request_json = mock.Mock(side_effect=urllib.error.HTTPError(
            "http://localhost:8188/api/prompt", 400, "Bad Request", {},
            io.BytesIO(b'{"error":"missing node 1701"}'),
        ))

        with self.assertRaisesRegex(RuntimeError, "missing node 1701"):
            client.post_prompt({})

    def test_unload_models_can_clear_comfy_model_cache_when_ram_is_low(self):
        from toolkit.comfy_sample import ComfyApiClient

        payloads = []
        client = ComfyApiClient()
        client._request_json = lambda method, path, payload=None: payloads.append((method, path, payload))

        client.unload_models(free_memory=True)

        self.assertEqual(payloads, [
            ("POST", "/api/free", {"unload_models": True, "free_memory": True})
        ])

    def test_unload_models_preserves_comfy_execution_cache_by_default(self):
        from toolkit.comfy_sample import ComfyApiClient

        payloads = []
        client = ComfyApiClient()
        client._request_json = lambda method, path, payload=None: payloads.append((method, path, payload))

        client.unload_models()

        self.assertEqual(payloads, [
            ("POST", "/api/free", {"unload_models": True})
        ])

    def test_release_vram_requests_cache_preserving_custom_endpoint(self):
        from toolkit.comfy_sample import ComfyApiClient

        payloads = []
        client = ComfyApiClient(api_url="http://comfy.test:8188")

        def request_json(method, path, payload=None):
            payloads.append((method, path, payload))
            return {"released": True, "released_bytes": 123}

        client._request_json = request_json

        with mock.patch("builtins.print") as print_mock:
            result = client.release_vram()

        self.assertEqual(payloads, [
            ("POST", "/comfyui-cache-monitor/release_vram", None)
        ])
        self.assertEqual(result["released_bytes"], 123)
        self.assertIn(
            "http://comfy.test:8188/comfyui-cache-monitor/release_vram",
            print_mock.call_args.args[0],
        )

    def test_release_vram_warns_and_uses_legacy_free_endpoint_when_custom_endpoint_is_missing(self):
        from toolkit.comfy_sample import ComfyApiClient

        import urllib.error

        payloads = []
        client = ComfyApiClient(api_url="http://comfy.test:8188")

        def request_json(method, path, payload=None):
            payloads.append((method, path, payload))
            if path == "/comfyui-cache-monitor/release_vram":
                raise urllib.error.HTTPError(
                    "http://comfy.test:8188/comfyui-cache-monitor/release_vram",
                    404,
                    "Not Found",
                    {},
                    None,
                )

        client._request_json = request_json

        with mock.patch("builtins.print") as print_mock:
            result = client.release_vram()

        self.assertEqual(payloads, [
            ("POST", "/comfyui-cache-monitor/release_vram", None),
            ("POST", "/api/free", {"unload_models": True}),
        ])
        self.assertFalse(result["released"])
        self.assertTrue(result["fallback_requested"])
        self.assertTrue(result["fallback_succeeded"])
        warning = print_mock.call_args_list[-1].args[0]
        self.assertIn("WARNING", warning)
        self.assertIn("download/install the comfyui-cache-monitor", warning.lower())
        self.assertIn("http://comfy.test:8188/api/free", warning)

    def test_release_vram_uses_legacy_free_endpoint_on_unconfirmed_response(self):
        from toolkit.comfy_sample import ComfyApiClient

        payloads = []
        client = ComfyApiClient()

        def request_json(method, path, payload=None):
            payloads.append((method, path, payload))
            return {} if path == "/comfyui-cache-monitor/release_vram" else None

        client._request_json = request_json

        with mock.patch("builtins.print"):
            result = client.release_vram()

        self.assertEqual(payloads, [
            ("POST", "/comfyui-cache-monitor/release_vram", None),
            ("POST", "/api/free", {"unload_models": True}),
        ])
        self.assertTrue(result["fallback_requested"])

    def test_release_vram_does_not_raise_when_both_endpoints_fail(self):
        from toolkit.comfy_sample import ComfyApiClient

        import urllib.error

        client = ComfyApiClient(api_url="http://comfy.test:8188")
        client._request_json = mock.Mock(side_effect=urllib.error.URLError("offline"))

        with mock.patch("builtins.print") as print_mock:
            result = client.release_vram()

        self.assertEqual(client._request_json.call_count, 2)
        self.assertFalse(result["fallback_succeeded"])
        self.assertEqual(result["fallback_error"], "<urlopen error offline>")
        self.assertIn(
            "Legacy ComfyUI VRAM release also failed",
            print_mock.call_args_list[-1].args[0],
        )

    def test_upload_image_posts_multipart_to_comfy_input_storage(self):
        from toolkit.comfy_sample import ComfyApiClient

        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"name":"uploaded.png","subfolder":"ai-toolkit","type":"input"}'
        )
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image_file.write(b"png bytes")
            image_file.flush()
            with mock.patch("toolkit.comfy_sample.urllib.request.urlopen", return_value=response) as urlopen:
                client = ComfyApiClient()
                remote_name = client.upload_image(image_file.name)

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8188/api/upload/image")
        self.assertEqual(request.method, "POST")
        self.assertIn("multipart/form-data", request.headers["Content-type"])
        self.assertIn(b'name="type"', request.data)
        self.assertIn(b'name="subfolder"', request.data)
        self.assertIn(b'name="image"', request.data)
        self.assertIn(b"png bytes", request.data)
        self.assertEqual(remote_name, "ai-toolkit/uploaded.png")
        self.assertEqual(client._uploaded_images, {remote_name})


class ComfySampleTrainProcessTests(unittest.TestCase):
    def test_normal_comfy_handoff_preserves_execution_cache(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        single_start = source.index("def _render_comfy_samples")
        batch_start = source.index("def _render_comfy_sample_batch", single_start)
        sample_start = source.index("def sample", batch_start)

        single_source = source[single_start:batch_start]
        batch_source = source[batch_start:sample_start]

        self.assertIn("client.release_vram()", single_source)
        self.assertNotIn("client.unload_models(free_memory=True)", single_source)
        self.assertIn("client.release_vram()", batch_source)
        self.assertNotIn("client.unload_models(free_memory=True)", batch_source)

    def test_train_process_offloads_models_once_before_comfy_sample_loop(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        render_start = source.index("def _render_comfy_samples")
        render_end = source.index("def _render_comfy_sample_batch", render_start)
        render_source = source[render_start:render_end]
        loop_start = source.index("for i, gen_config in enumerate(gen_img_config_list):", render_start)
        patch_start = source.index("patched_workflow = get_workflow_for_sample(workflow_path, request, workflow)", render_start)
        prompt_start = source.index("prompt_id = client.post_prompt(patched_workflow)", render_start)
        before_loop = source[render_start:loop_start]
        between_patch_and_prompt = source[patch_start:prompt_start]
        method_start = source.index("def _ensure_models_offloaded_for_comfy")
        next_method = source.index("\n    def ", method_start + 1)
        method_source = source[method_start:next_method]

        self.assertEqual(render_source.count("self._ensure_models_offloaded_for_comfy(client)"), 1)
        self.assertIn("self._ensure_models_offloaded_for_comfy(client)", before_loop)
        self.assertNotIn("self._ensure_models_offloaded_for_comfy(client)", between_patch_and_prompt)
        self.assertIn("self.sd.set_device_state(copy.deepcopy(empty_preset))", method_source)
        self.assertIn("self._offload_comfy_auxiliary_modules()", method_source)
        self.assertIn("free_memory=True", method_source)

    def test_train_process_restores_auxiliary_networks_after_comfy_sample(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        iterator_start = source.index("def _iter_comfy_auxiliary_offload_modules")
        iterator_end = source.index("\n    def ", iterator_start + 1)
        iterator_source = source[iterator_start:iterator_end]
        wrapper_start = source.index("def _run_with_models_offloaded_for_comfy")
        wrapper_end = source.index("\n    def ", wrapper_start + 1)
        wrapper_source = source[wrapper_start:wrapper_end]

        self.assertIn("getattr(self, 'network', None)", iterator_source)
        self.assertIn("'assistant_lora'", iterator_source)
        self.assertIn("'accuracy_recovery_adapter'", iterator_source)
        self.assertIn("_capture_comfy_auxiliary_device_state()", wrapper_source)
        self.assertIn(
            "_restore_comfy_auxiliary_modules(auxiliary_device_state)",
            wrapper_source,
        )

    def test_comfy_sample_generation_time_is_logged(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        render_start = source.index("def _render_comfy_samples")
        render_end = source.index("def sample", render_start)
        render_source = source[render_start:render_end]

        self.assertIn("time.perf_counter()", render_source)
        self.assertIn("_log_comfy_sample_generation_start", render_source)
        self.assertIn("_log_comfy_prompt_submitted", render_source)
        self.assertIn("Starting ComfyUI", source)
        self.assertIn("Submitted ComfyUI", source)
        self.assertIn("_update_comfy_sample_status", source)
        self.assertIn("_log_comfy_sample_generation_time", render_source)
        self.assertIn("ComfyUI sample generation completed in", source)
        self.assertIn("comfy_sample_generation_seconds", source)

    def test_comfy_uses_saved_output_lora_path_for_training_lora(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        self.assertIn("def _get_comfy_lora_output_path", source)
        save_start = source.index("def _save_current_network_for_comfy")
        save_end = source.index("\n    def ", save_start + 1)
        save_source = source[save_start:save_end]
        path_start = source.index("def _get_comfy_lora_output_path")
        path_end = source.index("\n    def ", path_start + 1)
        path_source = source[path_start:path_end]
        render_start = source.index("def _render_comfy_samples")
        render_end = source.index("def sample", render_start)
        render_source = source[render_start:render_end]

        self.assertIn("os.path.abspath(os.path.join(self.save_root, self._get_comfy_lora_display_filename(step)))", path_source)
        self.assertIn("def _save_current_network_for_comfy(self, step=None)", save_source)
        self.assertIn("file_path = self._get_comfy_lora_output_path(step)", save_source)
        self.assertIn("_save_current_network_for_comfy(step=step)", render_source)
        self.assertNotIn("_comfy_current", save_source)
        self.assertNotIn("time.time_ns()", save_source)

    def test_train_process_can_submit_comfy_prompts_as_one_batch_workflow(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()

        self.assertIn("def _render_comfy_sample_batch", source)
        self.assertIn("get_workflow_for_samples", source)
        self.assertIn("send_prompts_as_batch", source)
        self.assertIn("_render_comfy_sample_batch(gen_img_config_list, sample_config, step=step)", source)

    def test_comfy_batch_dispatch_allows_per_sample_scalar_overrides(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()

        self.assertNotIn("def _can_render_comfy_sample_batch", source)
        self.assertNotIn("ComfyUI prompt batching requires matching width", source)
        self.assertIn(
            "if sample_config.comfy.send_prompts_as_batch and len(gen_img_config_list) > 1:",
            source,
        )

    def test_qwen_image_edit_batch_uploads_control_images_before_rendering(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        batch_start = source.index("def _render_comfy_sample_batch")
        sample_start = source.index("def sample", batch_start)
        batch_source = source[batch_start:sample_start]

        self.assertIn("DEFAULT_COMFY_QWEN_IMAGE_EDIT_BATCH_WORKFLOW_PATH", source)
        self.assertIn("gen_config.ctrl_img_1 or gen_config.ctrl_img", batch_source)
        self.assertIn("client.upload_image(control_image_path)", batch_source)
        self.assertIn("control_images=uploaded_control_images", batch_source)

    def test_qwen_image_edit_plus_splits_batches_by_control_image_count(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        batch_start = source.index("def _render_comfy_sample_batch")
        sample_start = source.index("def sample", batch_start)
        batch_source = source[batch_start:sample_start]

        self.assertIn("DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_BATCH_WORKFLOW_PATH", source)
        self.assertIn("groups_by_reference_count = OrderedDict()", batch_source)
        self.assertIn(
            "groups_by_reference_count.setdefault(group_key, []).append((i, gen_config))",
            batch_source,
        )
        self.assertIn("gen_config.width", batch_source)
        self.assertIn("gen_config.height", batch_source)
        self.assertIn("indexed_batch_groups = list(groups_by_reference_count.values())", batch_source)
        self.assertIn("control_image_paths_2", batch_source)
        self.assertIn("control_image_paths_3", batch_source)
        self.assertIn("control_images_2=uploaded_control_images_2", batch_source)
        self.assertIn("control_images_3=uploaded_control_images_3", batch_source)
        self.assertNotIn("same number of control images", batch_source)

    def test_qwen_image_edit_plus_sub_batches_preserve_outputs_and_support_singletons(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        batch_start = source.index("def _render_comfy_sample_batch")
        sample_start = source.index("def sample", batch_start)
        batch_source = source[batch_start:sample_start]

        self.assertIn(
            "output_paths = [gen_config.get_image_path(i) for i, gen_config in enumerate(gen_img_config_list)]",
            batch_source,
        )
        self.assertIn("is_batch_group = len(group_configs) > 1", batch_source)
        self.assertIn("single_workflow_path = self._get_comfy_workflow_path(comfy_config, batch=False)", batch_source)
        self.assertIn("patched_workflow = get_workflow_for_sample(", batch_source)
        self.assertIn(
            "client.download_image(images[group_index], output_paths[original_index])",
            batch_source,
        )
        self.assertIn(
            "self.sample_step_hook(completed_samples - 1, len(gen_img_config_list))",
            batch_source,
        )

    def test_workflow_config_selects_batch_and_single_templates(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        workflow_start = source.index("def _get_comfy_workflow_path")
        workflow_end = source.index("\n    def ", workflow_start + 1)
        workflow_source = source[workflow_start:workflow_end]

        self.assertIn("(DEFAULT_COMFY_WORKFLOW_PATH, DEFAULT_COMFY_BATCH_WORKFLOW_PATH)", workflow_source)
        self.assertIn("model_arch == \"qwen_image_2\"", workflow_source)
        self.assertIn(
            "DEFAULT_COMFY_QWEN_IMAGE_2_WORKFLOW_PATH,\n"
            "                    DEFAULT_COMFY_QWEN_IMAGE_2_BATCH_WORKFLOW_PATH",
            workflow_source,
        )
        self.assertIn(
            "DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_WORKFLOW_PATH,\n"
            "                    DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_BATCH_WORKFLOW_PATH",
            workflow_source,
        )
        self.assertIn(
            "DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_BATCH_WORKFLOW_PATH,\n"
            "                    DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_WORKFLOW_PATH",
            workflow_source,
        )

    def test_qwen_single_workflows_upload_control_images_per_sample(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        single_start = source.index("def _render_comfy_samples")
        batch_start = source.index("def _render_comfy_sample_batch", single_start)
        single_source = source[single_start:batch_start]

        self.assertIn("DEFAULT_COMFY_QWEN_IMAGE_2_WORKFLOW_PATH", single_source)
        self.assertIn("DEFAULT_COMFY_QWEN_IMAGE_EDIT_PLUS_WORKFLOW_PATH", single_source)
        self.assertIn("is_qwen_image_2_workflow", single_source)
        self.assertIn("return client.upload_image(path)", single_source)
        self.assertIn("upload_control(control_image_path)", single_source)
        self.assertIn("upload_control(control_image_path_2)", single_source)
        self.assertIn("upload_control(control_image_path_3)", single_source)
        self.assertIn("control_image=uploaded_control_image", single_source)
        self.assertIn("control_image_2=uploaded_control_image_2", single_source)
        self.assertIn("control_image_3=uploaded_control_image_3", single_source)

    def test_train_process_can_run_comfy_sampling_in_background(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        single_start = source.index("def _render_comfy_samples")
        batch_start = source.index("def _render_comfy_sample_batch")
        sample_start = source.index("def sample", batch_start)
        single_source = source[single_start:batch_start]
        batch_source = source[batch_start:sample_start]
        startup_release_start = source.index("def _get_comfy_config_for_startup_release")
        startup_release_end = source.index("\n    def ", startup_release_start + 1)
        startup_release_source = source[startup_release_start:startup_release_end]

        self.assertIn("import threading", source)
        self.assertIn("def _run_comfy_background_task", source)
        self.assertIn("threading.Thread", source)
        self.assertIn("if comfy_config.run_in_background:", single_source)
        self.assertIn("if comfy_config.run_in_background:", batch_source)
        self.assertIn("offload_models=False", single_source)
        self.assertIn("unload_models=False", single_source)
        self.assertIn("offload_models=False", batch_source)
        self.assertIn("unload_models=False", batch_source)
        self.assertNotIn("run_in_background", startup_release_source)

    def test_comfy_prompt_waits_are_cancelled_when_training_stops(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        render_start = source.index("def _render_comfy_samples")
        render_end = source.index("def sample", render_start)
        render_source = source[render_start:render_end]
        error_start = source.index("def on_error")
        error_end = source.index("\n    def ", error_start + 1)
        error_source = source[error_start:error_end]

        self.assertEqual(
            render_source.count("cancel_check=self._should_cancel_comfy_prompt_wait"),
            2,
        )
        self.assertIn("self._comfy_prompt_wait_cancel_event.set()", error_source)
        self.assertIn("should_stop = getattr(self, 'should_stop', None)", source)
        self.assertIn("daemon=True", source)

    def test_train_process_tracks_and_waits_for_background_comfy_samples_at_shutdown(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        init_start = source.index("def __init__")
        init_end = source.index("\n    def ", init_start + 1)
        init_source = source[init_start:init_end]
        task_start = source.index("def _run_comfy_background_task")
        task_end = source.index("\n    def ", task_start + 1)
        task_source = source[task_start:task_end]
        run_end_start = source.index("##  END TRAIN LOOP")
        run_end_end = source.index("\n    def push_to_hub", run_end_start)
        run_end_source = source[run_end_start:run_end_end]

        self.assertIn("self._comfy_background_threads = []", init_source)
        self.assertIn("self._comfy_background_errors = []", init_source)
        self.assertIn("self._comfy_background_threads.append(thread)", task_source)
        self.assertIn("def _wait_for_comfy_background_tasks", source)
        self.assertIn("def _release_training_memory_before_comfy_wait", source)

        release_index = run_end_source.index("_release_training_memory_before_comfy_wait")
        wait_index = run_end_source.index("_wait_for_comfy_background_tasks")
        logger_finish_index = run_end_source.index("self.logger.finish()")
        done_hook_index = run_end_source.index("self.done_hook()")

        self.assertLess(release_index, wait_index)
        self.assertLess(wait_index, logger_finish_index)
        self.assertLess(logger_finish_index, done_hook_index)

    def test_train_process_maps_comfy_training_lora_path_for_remote_api(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        method_start = source.index("def _get_comfy_training_lora_path")
        method_end = source.index("\n    def ", method_start + 1)
        method_source = source[method_start:method_end]
        render_start = source.index("def _render_comfy_samples")
        render_end = source.index("def sample", render_start)
        render_source = source[render_start:render_end]

        self.assertIn("training_lora_path_replace_from", method_source)
        self.assertIn("training_lora_path_replace_to", method_source)
        self.assertIn("replace_to.endswith(('/', '\\\\'))", method_source)
        self.assertIn("return replace_to + suffix", method_source)
        self.assertIn("comfy_training_lora_path = self._get_comfy_training_lora_path", render_source)
        self.assertIn("training_lora_path=comfy_training_lora_path", render_source)


class StableDiffusionModelTests(unittest.TestCase):
    def test_inactive_inference_lora_is_offloaded_after_model_load(self):
        source = (REPO_ROOT / "toolkit/stable_diffusion_model.py").read_text()
        load_start = source.index("self.is_loaded = True")
        inference_start = source.index("if self.model_config.inference_lora_path is not None:", load_start)
        inference_end = source.index("if self.is_pixart", inference_start)
        inference_source = source[inference_start:inference_end]

        self.assertIn("load_assistant_lora_from_path", inference_source)
        self.assertIn("self.assistant_lora.is_active = False", inference_source)
        self.assertIn("self.assistant_lora.force_to('cpu', self.torch_dtype)", inference_source)


class ComfySampleUITests(unittest.TestCase):
    def test_sample_card_exposes_comfy_controls(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        sample_start = source.index('<Card title="Sample">')
        sample_prompts_start = source.index("Sample Prompts", sample_start)
        sample_section = source[sample_start:sample_prompts_start]

        self.assertIn('label="Use ComfyUI Renderer"', sample_section)
        self.assertIn('label="Negative Prompt"', sample_section)
        self.assertIn("config.process[0].sample.comfy.negative_prompt", sample_section)
        self.assertIn("config.process[0].sample.comfy.enabled", sample_section)
        self.assertIn('label="Comfy Model"', sample_section)
        self.assertIn('label="Comfy VAE"', sample_section)
        self.assertIn('label="Comfy Text Encoder"', sample_section)
        self.assertIn('label="Comfy Sampler"', sample_section)
        self.assertIn('label="Comfy Scheduler"', sample_section)
        self.assertIn('label="Inference LoRA Strength"', sample_section)
        self.assertIn('label="Comfy Inference LoRA"', sample_section)
        self.assertIn('label="Send Prompts as Batch"', sample_section)
        self.assertIn('label="Run ComfyUI in Background"', sample_section)
        self.assertIn("config.process[0].sample.comfy.run_in_background", sample_section)
        self.assertIn('label="LoRA Path Replace From"', sample_section)
        self.assertIn('label="LoRA Path Replace To"', sample_section)
        self.assertIn("config.process[0].sample.comfy.training_lora_path_replace_from", sample_section)
        self.assertIn("config.process[0].sample.comfy.training_lora_path_replace_to", sample_section)
        self.assertIn('label="Output Format"', sample_section)
        self.assertIn('label="Output Quality"', sample_section)


if __name__ == "__main__":
    unittest.main()
