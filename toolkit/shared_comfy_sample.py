"""Synchronous shared Hunyuan preview provider using normal Comfy HTTP outputs."""
import copy
import json
import time
from pathlib import Path

from toolkit.comfy_sample import ComfyApiClient

DEFAULT_WORKFLOW = Path(__file__).resolve().parents[1] / 'integrations' / 'ComfyUI-AITK-SharedModels' / 'workflows' / 'hunyuan_instruct_shared_edit.json'


def build_workflow(template, coordinator, generation, config, gen, references, filename_prefix):
    workflow = copy.deepcopy(template)
    classes = {node['class_type']: key for key, node in workflow.items()}
    required = ('AITKSharedHunyuanImage3Loader', 'AITKSharedLoRASnapshot', 'HunyuanImage3VAELoader',
                'HunyuanImage3ImageEncode', 'HunyuanImage3EmptyLatent', 'KSampler', 'SaveImage')
    if any(name not in classes for name in required):
        raise ValueError('Shared Hunyuan workflow lacks required loader/snapshot/conditioning/sampler nodes')
    def inputs(name):
        return workflow[classes[name]]['inputs']
    inputs('AITKSharedHunyuanImage3Loader').update(store_identity=coordinator.store_identity,
        socket=coordinator.client.socket.getpeername())
    inputs('AITKSharedLoRASnapshot').update(snapshot_id=generation, strength=gen.network_multiplier)
    inputs('HunyuanImage3VAELoader')['vae_name'] = config.vae
    if 'CLIPVisionLoader' in classes:
        inputs('CLIPVisionLoader')['clip_name'] = config.clip_vision or config.text_encoder
    encode = inputs('HunyuanImage3ImageEncode')
    encode.update(prompt=gen.prompt, width=gen.width, height=gen.height)
    for name in list(encode):
        if name.startswith('images.'):
            del encode[name]
    if references:
        if not config.vae or not (config.clip_vision or config.text_encoder):
            raise ValueError('Shared edit previews require sample.comfy.vae and clip_vision model names')
        loader_id = classes.get('LoadImage')
        if loader_id is None:
            raise ValueError('Shared edit workflow lacks LoadImage')
        for index, reference in enumerate(references):
            node_id = loader_id if index == 0 else f'aitk_reference_{index + 1}'
            workflow[node_id] = {'class_type': 'LoadImage', 'inputs': {'image': reference}}
            encode[f'images.image_{index + 1}'] = [node_id, 0]
    else:
        workflow[classes['HunyuanImage3ImageEncode']]['class_type'] = 'HunyuanImage3TextEncode'
        for name in ('vae', 'clip_vision', 'vit_strength', 'latent_strength', 'reference_vit_padding'):
            encode.pop(name, None)
        for name in ('LoadImage', 'CLIPVisionLoader'):
            if name in classes:
                workflow.pop(classes[name], None)
    inputs('HunyuanImage3EmptyLatent').update(width=gen.width, height=gen.height)
    inputs('KSampler').update(seed=gen.seed, steps=gen.num_inference_steps, cfg=gen.guidance_scale,
                             sampler_name=config.sampler, scheduler=config.scheduler)
    inputs('SaveImage')['filename_prefix'] = filename_prefix
    return workflow


def render_shared_samples(process, configs, sample_config, step=None):
    coordinator = process.sd.shared_coordinator
    if coordinator.state != 'ACTIVE':
        raise RuntimeError('Shared previews require the trainer to own the GPU before publishing')
    client = ComfyApiClient(sample_config.comfy.api_url, sample_config.comfy.timeout)
    if sample_config.comfy.sampler != 'euler' or sample_config.comfy.scheduler != 'simple':
        raise ValueError('Shared Hunyuan previews currently require the non-distilled euler/simple sampling contract')
    if sample_config.comfy.negative_prompt:
        raise ValueError('Hunyuan shared encoders construct unconditional conditioning; separate negative prompts are unsupported')
    client.require_shared_provider(process.model_config.shared_weights)
    configured = sample_config.comfy.workflow_path
    workflow_path = Path(configured) if configured and configured.endswith('.json') else DEFAULT_WORKFLOW
    template = json.loads(workflow_path.read_text())
    generation = coordinator.publish_adapter(process.step_num)
    # Pin generation while the request is prepared/queued too; Comfy pins again
    # for the whole execution and cache fingerprint resolves latest only once.
    coordinator.client.open_adapter(generation, coordinator.store_identity)
    coordinator.suspend()
    success = False
    prompt_id = None
    start = time.perf_counter()
    completed = 0
    try:
        process._log_comfy_sample_generation_start(len(configs), batch=False)
        process._update_comfy_sample_status('Waiting for ComfyUI shared preview')
        for index, gen in enumerate(configs):
            paths = [gen.ctrl_img_1 or gen.ctrl_img, gen.ctrl_img_2, gen.ctrl_img_3]
            if any(paths[i] is not None and paths[i - 1] is None for i in (1, 2)):
                raise ValueError('Shared preview reference slots must be consecutive')
            references = [client.upload_image(path) for path in paths if path]
            output = gen.get_image_path(index)
            workflow = build_workflow(template, coordinator, generation, sample_config.comfy, gen,
                                      references, 'ai-toolkit/' + Path(output).stem)
            client.validate_workflow_schema(workflow)
            client.begin_live_progress()
            prompt_id = client.post_prompt(workflow)
            process._log_comfy_prompt_submitted(prompt_id, len(configs), sample_index=index + 1)
            progress = process._create_comfy_image_progress_bar([index], len(configs), gen.num_inference_steps)
            image_completed = False
            try:
                images = client.wait_for_images(prompt_id, cancel_check=process._should_cancel_comfy_prompt_wait,
                    progress_callback=lambda value, maximum: process._update_comfy_image_progress_bar(progress, value, maximum))
                image_completed = True
            finally:
                process._close_comfy_image_progress_bar(progress, image_completed)
            if not images:
                raise RuntimeError('Shared Comfy prompt completed without images')
            client.download_image(images[0], output)
            process.sample_step_hook(index, len(configs))
            completed += 1
            process._update_comfy_sample_status(f'ComfyUI shared sampling - {completed}/{len(configs)}')
            prompt_id = None
        success = True
    finally:
        process._log_comfy_sample_generation_time(time.perf_counter() - start, completed, len(configs), step=step)
        if prompt_id is not None:
            client.cancel_shared_prompt(prompt_id)
        coordinator.client.release_adapter(generation)
        if success:
            # Broker reacquisition waits for the whole-prompt finally cleanup,
            # even if image output reached history before GPU teardown finished.
            coordinator.resume()
            process._update_comfy_sample_status('Training')
        # Errors/cancel leave trainer parked. Job cleanup may close its CPU owner;
        # it never resumes under an unverified still-running Comfy prompt.
