import copy
import pathlib
import sys
import types
import unittest
from dataclasses import replace


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


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
        self.assertEqual(rendered["1000"]["inputs"]["value"], "first prompt")
        self.assertEqual(rendered["1001"]["inputs"]["value"], "second prompt")
        self.assertEqual(rendered["1100"]["inputs"]["value"], 123)
        self.assertEqual(rendered["1101"]["inputs"]["value"], 124)

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

        from toolkit.config_modules import SampleConfig

        sample = SampleConfig(
            comfy={
                "enabled": True,
                "api_url": "http://127.0.0.1:8188",
                "model": "krea2_raw.safetensors",
                "vae": "wan_vae.safetensors",
                "text_encoder": "qwen_clip.safetensors",
                "sampler": "euler",
                "scheduler": "simple",
                "inference_lora": "turbo.safetensors",
                "inference_lora_strength": 0.42,
                "send_prompts_as_batch": True,
                "output_format": "webp_with_json",
                "output_quality": "high",
            }
        )

        self.assertTrue(sample.comfy.enabled)
        self.assertEqual(sample.comfy.workflow_path, "config/comfy_templates/krea2_lora_sample.json.njk")
        self.assertEqual(sample.comfy.model, "krea2_raw.safetensors")
        self.assertEqual(sample.comfy.inference_lora, "turbo.safetensors")
        self.assertEqual(sample.comfy.inference_lora_strength, 0.42)
        self.assertTrue(sample.comfy.send_prompts_as_batch)


class ComfyApiClientTests(unittest.TestCase):
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
        self.assertIs(payloads[0][2]["extra_data"]["extra_pnginfo"]["workflow"], workflow)

    def test_unload_models_can_clear_comfy_model_cache_when_ram_is_low(self):
        from toolkit.comfy_sample import ComfyApiClient

        payloads = []
        client = ComfyApiClient()
        client._request_json = lambda method, path, payload=None: payloads.append((method, path, payload))

        client.unload_models(free_memory=True)

        self.assertEqual(payloads, [
            ("POST", "/api/free", {"unload_models": True, "free_memory": True})
        ])


class ComfySampleTrainProcessTests(unittest.TestCase):
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
        self.assertIn("free_memory=True", method_source)

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

    def test_batch_workflow_config_falls_back_to_single_template_for_individual_samples(self):
        source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
        workflow_start = source.index("def _get_comfy_workflow_path")
        workflow_end = source.index("\n    def ", workflow_start + 1)
        workflow_source = source[workflow_start:workflow_end]

        self.assertIn("if batch and workflow_path == DEFAULT_COMFY_WORKFLOW_PATH", workflow_source)
        self.assertIn("return DEFAULT_COMFY_BATCH_WORKFLOW_PATH", workflow_source)
        self.assertIn("if not batch and workflow_path == DEFAULT_COMFY_BATCH_WORKFLOW_PATH", workflow_source)
        self.assertIn("return DEFAULT_COMFY_WORKFLOW_PATH", workflow_source)


class ComfySampleUITests(unittest.TestCase):
    def test_sample_card_exposes_comfy_controls(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        sample_start = source.index('<Card title="Sample">')
        sample_prompts_start = source.index("Sample Prompts", sample_start)
        sample_section = source[sample_start:sample_prompts_start]

        self.assertIn('label="Use ComfyUI Renderer"', sample_section)
        self.assertIn("config.process[0].sample.comfy.enabled", sample_section)
        self.assertIn('label="Comfy Model"', sample_section)
        self.assertIn('label="Comfy VAE"', sample_section)
        self.assertIn('label="Comfy Text Encoder"', sample_section)
        self.assertIn('label="Comfy Sampler"', sample_section)
        self.assertIn('label="Comfy Scheduler"', sample_section)
        self.assertIn('label="Inference LoRA Strength"', sample_section)
        self.assertIn('label="Comfy Inference LoRA"', sample_section)
        self.assertIn('label="Send Prompts as Batch"', sample_section)
        self.assertIn('label="Output Format"', sample_section)
        self.assertIn('label="Output Quality"', sample_section)


if __name__ == "__main__":
    unittest.main()
