import copy
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from toolkit.paths import get_path


DEFAULT_COMFY_API_URL = "http://127.0.0.1:8188"
DEFAULT_COMFY_WORKFLOW_PATH = "config/comfy_templates/krea2_lora_sample.json.njk"
DEFAULT_COMFY_BATCH_WORKFLOW_PATH = "config/comfy_templates/krea2_lora_sample_batch_easy_use.json.njk"
NUNJUCKS_RENDERER_PATH = "ui/scripts/render_comfy_template.mjs"


@dataclass
class ComfySampleRequest:
    prompt: str
    width: int
    height: int
    steps: int
    cfg: float
    seed: int
    model: str
    vae: str
    text_encoder: str
    sampler: str
    scheduler: str
    inference_lora: str
    inference_lora_strength: float
    output_format: str
    output_quality: str
    training_lora_path: str
    training_lora_filename: str
    filename_prefix: str


@dataclass
class ComfyBatchSampleRequest:
    prompts: List[str]
    width: int
    height: int
    steps: int
    cfg: float
    seeds: List[int]
    model: str
    vae: str
    text_encoder: str
    sampler: str
    scheduler: str
    inference_lora: str
    inference_lora_strength: float
    output_format: str
    output_quality: str
    training_lora_path: str
    training_lora_filename: str
    filename_prefix: str


def _find_node_id(workflow: Dict[str, Any], class_type: str) -> Optional[str]:
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id
    return None


def _node_inputs(workflow: Dict[str, Any], class_type: str) -> Optional[Dict[str, Any]]:
    node_id = _find_node_id(workflow, class_type)
    if node_id is None:
        return None
    return workflow[node_id].setdefault("inputs", {})


def _set_if_present(inputs: Optional[Dict[str, Any]], key: str, value: Any):
    if inputs is None:
        return
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    inputs[key] = value


def _replace_node_links(workflow: Dict[str, Any], old_node_id: str, new_node_id: str):
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) == 2 and value[0] == old_node_id:
                inputs[key] = [new_node_id, value[1]]


def _remove_node(workflow: Dict[str, Any], node_id: str):
    workflow.pop(node_id, None)


def _strip_safetensors(filename: str) -> str:
    stem = os.path.basename(filename)
    if stem.endswith(".safetensors"):
        return stem[:-len(".safetensors")]
    return os.path.splitext(stem)[0]


def workflow_path_is_template(path: str) -> bool:
    return os.path.splitext(path or "")[1].lower() in (".njk", ".nunjucks")


def _training_lora_context(training_lora_filename: str, training_lora_path: str) -> Dict[str, str]:
    training_lora_filename = training_lora_filename or os.path.basename(training_lora_path)
    training_lora_stem = _strip_safetensors(training_lora_filename)
    return {
        "training_lora_path": training_lora_path,
        "training_lora_filename": training_lora_filename,
        "training_lora_stem": training_lora_stem,
    }


def build_template_context(request: Any) -> Dict[str, Any]:
    if isinstance(request, ComfyBatchSampleRequest):
        if len(request.prompts) < 2:
            raise ValueError("ComfyUI batch sample requests require at least two prompts")
        lora_context = _training_lora_context(request.training_lora_filename, request.training_lora_path)
        return {
            "prompts": request.prompts,
            "sample_count": len(request.prompts),
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg": request.cfg,
            "seeds": request.seeds,
            "model": request.model,
            "vae": request.vae,
            "text_encoder": request.text_encoder,
            "sampler": request.sampler,
            "scheduler": request.scheduler,
            "inference_lora": request.inference_lora,
            "inference_lora_enabled": bool(request.inference_lora),
            "inference_lora_strength": request.inference_lora_strength,
            "output_format": request.output_format,
            "output_quality": request.output_quality,
            "filename_prefix": request.filename_prefix,
            **lora_context,
        }

    training_lora_filename = request.training_lora_filename or os.path.basename(request.training_lora_path)
    training_lora_stem = _strip_safetensors(training_lora_filename)
    return {
        "prompt": request.prompt,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg": request.cfg,
        "seed": request.seed,
        "model": request.model,
        "vae": request.vae,
        "text_encoder": request.text_encoder,
        "sampler": request.sampler,
        "scheduler": request.scheduler,
        "inference_lora": request.inference_lora,
        "inference_lora_enabled": bool(request.inference_lora),
        "inference_lora_strength": request.inference_lora_strength,
        "output_format": request.output_format,
        "output_quality": request.output_quality,
        "training_lora_path": request.training_lora_path,
        "training_lora_filename": training_lora_filename,
        "training_lora_stem": training_lora_stem,
        "filename_prefix": request.filename_prefix,
    }


def _fill_lora_filename_placeholders(value: str, lora_filename: str) -> str:
    lora_stem = _strip_safetensors(lora_filename)
    return (
        value
        .replace("FILENAME_WITHOUT_.safetensors", lora_stem)
        .replace("FILENAME WITHOUT .safetensors", lora_stem)
    )


def patch_workflow_for_sample(workflow: Dict[str, Any], request: ComfySampleRequest) -> Dict[str, Any]:
    workflow = copy.deepcopy(workflow)

    _set_if_present(_node_inputs(workflow, "UNETLoader"), "unet_name", request.model)
    _set_if_present(_node_inputs(workflow, "CLIPLoader"), "clip_name", request.text_encoder)
    _set_if_present(_node_inputs(workflow, "VAELoader"), "vae_name", request.vae)
    _set_if_present(_node_inputs(workflow, "VAEUtils_CustomVAELoader"), "vae_name", request.vae)

    latent_inputs = _node_inputs(workflow, "EmptyLatentImage")
    _set_if_present(latent_inputs, "width", request.width)
    _set_if_present(latent_inputs, "height", request.height)
    _set_if_present(latent_inputs, "batch_size", 1)

    sampler_inputs = _node_inputs(workflow, "KSampler")
    _set_if_present(sampler_inputs, "steps", request.steps)
    _set_if_present(sampler_inputs, "cfg", request.cfg)
    _set_if_present(sampler_inputs, "sampler_name", request.sampler)
    _set_if_present(sampler_inputs, "scheduler", request.scheduler)

    _set_if_present(_node_inputs(workflow, "Seed"), "seed", request.seed)
    _set_if_present(_node_inputs(workflow, "Text Multiline"), "text", request.prompt)

    absolute_lora_node_id = _find_node_id(workflow, "load_lora_from_absolute_path")
    regular_lora_node_id = _find_node_id(workflow, "LoraLoader")

    absolute_lora_inputs = _node_inputs(workflow, "load_lora_from_absolute_path")
    _set_if_present(absolute_lora_inputs, "absolute_path", request.training_lora_path)

    regular_lora_inputs = _node_inputs(workflow, "LoraLoader")
    if request.inference_lora:
        _set_if_present(regular_lora_inputs, "lora_name", request.inference_lora)
        _set_if_present(regular_lora_inputs, "strength_model", request.inference_lora_strength)
        _set_if_present(regular_lora_inputs, "strength_clip", request.inference_lora_strength)
    elif absolute_lora_node_id is not None and regular_lora_node_id is not None:
        _replace_node_links(workflow, regular_lora_node_id, absolute_lora_node_id)
        _remove_node(workflow, regular_lora_node_id)

    save_inputs = _node_inputs(workflow, "SaveImageWithMetaData")
    _set_if_present(save_inputs, "filename_prefix", request.filename_prefix)
    _set_if_present(save_inputs, "output_format", request.output_format)
    _set_if_present(save_inputs, "quality", request.output_quality)

    datetime_inputs = _node_inputs(workflow, "JWDatetimeString")
    if datetime_inputs is not None and isinstance(datetime_inputs.get("format"), str):
        datetime_inputs["format"] = _fill_lora_filename_placeholders(
            datetime_inputs["format"],
            request.training_lora_filename,
        )

    _set_if_present(_node_inputs(workflow, "AddLabel"), "text", _strip_safetensors(request.training_lora_filename))
    return workflow


def render_nunjucks_workflow(path: str, request: Any) -> Dict[str, Any]:
    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError("Rendering ComfyUI .njk workflows requires node to be available on PATH")

    template_path = get_path(path)
    renderer_path = get_path(NUNJUCKS_RENDERER_PATH)
    context_json = json.dumps(build_template_context(request))
    result = subprocess.run(
        [node_path, renderer_path, template_path],
        input=context_json,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=get_path("."),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to render ComfyUI workflow template {path}: {result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Rendered ComfyUI workflow template {path} was not valid JSON: {e}") from e


def get_workflow_for_sample(
    workflow_path: str,
    request: ComfySampleRequest,
    base_workflow: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if workflow_path_is_template(workflow_path):
        return render_nunjucks_workflow(workflow_path, request)

    workflow = base_workflow if base_workflow is not None else load_workflow(workflow_path)
    return patch_workflow_for_sample(workflow, request)


def get_workflow_for_samples(
    workflow_path: str,
    request: ComfyBatchSampleRequest,
    base_workflow: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if workflow_path_is_template(workflow_path):
        return render_nunjucks_workflow(workflow_path, request)

    raise RuntimeError("ComfyUI prompt batching requires a .njk workflow template")


def extract_input_options(object_info: Dict[str, Any], node_class: str, input_name: str) -> List[str]:
    node_info = object_info.get(node_class)
    if node_info is None and len(object_info) == 1:
        node_info = next(iter(object_info.values()))
    if not isinstance(node_info, dict):
        return []

    input_sections = node_info.get("input", {})
    for section_name in ("required", "optional"):
        section = input_sections.get(section_name, {})
        spec = section.get(input_name)
        if not isinstance(spec, list) or len(spec) == 0:
            continue
        first = spec[0]
        if isinstance(first, list):
            return [str(item) for item in first]
        if first == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            options = spec[1].get("options", [])
            if isinstance(options, list):
                return [str(item) for item in options]
    return []


def get_history_output_images(history: Dict[str, Any], prompt_id: str) -> List[Dict[str, str]]:
    prompt_history = history.get(prompt_id, history)
    outputs = prompt_history.get("outputs", {}) if isinstance(prompt_history, dict) else {}
    images: List[Dict[str, str]] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for image in output.get("images", []) or []:
            if isinstance(image, dict) and image.get("filename"):
                images.append(image)
    return images


class ComfyApiClient:
    def __init__(self, api_url: str = DEFAULT_COMFY_API_URL, timeout: int = 600, poll_interval: float = 1.0):
        self.api_url = (api_url or DEFAULT_COMFY_API_URL).rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _url(self, path: str) -> str:
        return self.api_url + path

    def _request_json(self, method: str, path: str, payload: Optional[dict] = None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self._url(path), data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def post_prompt(self, workflow: Dict[str, Any]) -> str:
        response = self._request_json(
            "POST",
            "/api/prompt",
            {
                "prompt": workflow,
                "extra_data": {
                    "extra_pnginfo": {
                        "workflow": workflow,
                    },
                },
            },
        )
        if not response or "prompt_id" not in response:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {response}")
        return response["prompt_id"]

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        return self._request_json("GET", f"/api/history/{urllib.parse.quote(prompt_id)}") or {}

    def wait_for_images(self, prompt_id: str) -> List[Dict[str, str]]:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            history = self.get_history(prompt_id)
            images = get_history_output_images(history, prompt_id)
            if images:
                return images
            prompt_history = history.get(prompt_id, {})
            status = prompt_history.get("status", {}) if isinstance(prompt_history, dict) else {}
            if isinstance(status, dict) and status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI prompt failed: {status}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

    def download_image(self, image: Dict[str, str], output_path: str) -> str:
        params = {
            "filename": image["filename"],
            "type": image.get("type", "output"),
            "subfolder": image.get("subfolder", ""),
        }
        source_ext = os.path.splitext(image["filename"])[1]
        if source_ext:
            output_path = os.path.splitext(output_path)[0] + source_ext
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = self._url("/api/view?" + urllib.parse.urlencode(params))
        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            data = response.read()
        with open(output_path, "wb") as f:
            f.write(data)
        return output_path

    def unload_models(self, free_memory: bool = False, ignore_errors: bool = False):
        try:
            payload = {"unload_models": True}
            if free_memory:
                payload["free_memory"] = True
            self._request_json("POST", "/api/free", payload)
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            if not ignore_errors:
                raise


def load_workflow(path: str) -> Dict[str, Any]:
    with open(get_path(path), "r") as f:
        return json.load(f)
