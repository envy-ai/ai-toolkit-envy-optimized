"""Explicit live Comfy smoke renders of a saved training adapter, signed strengths.

Does not install nodes or restart services. Uses the provided model template,
registered components, one render at a time, and releases GPU models on exit.
No numerical/native image parity or long-run quality is claimed.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolkit.comfy_sample import (COMFY_CACHE_MONITOR_RELEASE_PATH, ComfyApiClient,
                                 ComfySampleRequest, get_workflow_for_sample)


def release_smoke_vram(client, report):
    """Skip occupied queues; never fall back to evicting the shared CPU cache."""
    try:
        queue = client._request_json('GET', '/queue', timeout=10)
        if queue['queue_running'] or queue['queue_pending']:
            report['vram_cleanup'] = {'skipped': 'Comfy queue is occupied'}
            return
        response = client._request_json('POST', COMFY_CACHE_MONITOR_RELEASE_PATH, timeout=15)
        if not response or response.get('released') is not True:
            raise RuntimeError('Cache-preserving endpoint did not confirm VRAM release')
        report['vram_cleanup'] = response
    except Exception as error:
        # The ordinary client's compatibility fallback can discard RAM caches.
        # Opt-in smoke tests must instead report the cleanup problem explicitly.
        report['vram_cleanup'] = {'error': str(error), 'cache_eviction_attempted': False}


def evaluate_identity_renders(renders):
    """Check a deliberately zero-delta adapter, not a learned-quality threshold.

    A successful loader/render is insufficient: quantized patch/requantization
    can change the base even when the LoRA update is mathematically zero.
    """
    import numpy as np
    from PIL import Image

    if len({item['strength'] for item in renders}) != len(renders):
        raise ValueError('Identity check requires unique strengths')
    baselines = [item for item in renders if item['strength'] == 0]
    if len(baselines) != 1 or len(renders) < 2:
        raise ValueError('Identity check requires zero and nonzero strengths')

    def pixels(path):
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB'), dtype=np.int16)

    baseline = pixels(baselines[0]['image'])
    comparisons = []
    for item in renders:
        if item['strength'] == 0:
            continue
        current = pixels(item['image'])
        if current.shape != baseline.shape:
            raise ValueError('Identity render dimensions differ')
        difference = np.abs(current - baseline)
        comparisons.append(dict(strength=item['strength'],
            identical=bool(np.array_equal(current, baseline)),
            mean_absolute_channel_difference=float(difference.mean()),
            max_absolute_channel_difference=int(difference.max())))
    return dict(passed=all(item['identical'] for item in comparisons),
                comparisons=comparisons)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['qwen_image_2', 'krea2', 'anima', 'ideogram4'], required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--api-url', default='http://127.0.0.1:8188')
    parser.add_argument('--model')
    parser.add_argument('--text-encoder')
    parser.add_argument('--vae')
    parser.add_argument('--cfg', type=float, default=1.)
    parser.add_argument('--negative-prompt', default='')
    parser.add_argument('--control-image', action='append', type=Path, default=[],
                        help='Krea edit reference; repeat for up to three consecutive image slots')
    parser.add_argument('--expect-identity', action='store_true',
                        help='For a zero-delta fixture: require nonzero strengths to equal the base pixels')
    args = parser.parse_args()
    if not math.isfinite(args.cfg) or args.cfg < 1:
        parser.error('Smoke CFG must be finite and at least 1')
    if args.control_image and (args.arch != 'krea2' or len(args.control_image) > 3):
        parser.error('Edit preview smoke controls are limited to Krea and three image slots')
    if any(not path.is_file() for path in args.control_image):
        parser.error('Every control image must be an existing file')
    adapter, root = args.adapter.resolve(), args.root.resolve()
    if not adapter.is_file():
        parser.error('Adapter must be an existing exported safetensors file')
    if not str(root).startswith('/tmp/'):
        parser.error('Use an isolated output root under /tmp')
    root.mkdir(parents=True, exist_ok=True)
    if (root / f'{args.arch}-comfy-report.json').exists():
        parser.error('Use a fresh root; do not overwrite earlier evidence')
    queue = json.load(urllib.request.urlopen(args.api_url.rstrip('/') + '/queue', timeout=10))
    if queue['queue_running'] or queue['queue_pending']:
        parser.error('ComfyUI is occupied; defer rather than queue behind another user')
    stats = json.load(urllib.request.urlopen(args.api_url.rstrip('/') + '/system_stats', timeout=10))
    if not stats['devices'] or stats['devices'][0]['vram_free'] < 18 * 1024 ** 3:
        parser.error('GPU is occupied: wait until the training test exits')
    components = {
        'qwen_image_2': ('d/diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
                         'd/qwen3vl_8b_int8_convrot.safetensors', 'd/vae/qwen_image_2.1_vae_bf16.safetensors'),
        'krea2': ('krea2_raw.safetensors', 'qwen3vl_4b_bf16.safetensors', 'qwen_image_vae.safetensors'),
        'anima': ('anima_baseV10.safetensors', 'qwen_3_06b_base.safetensors', 'qwen_image_vae.safetensors'),
        'ideogram4': ('ideogram4_fp8_scaled.safetensors', 'qwen3vl_8b_bf16.safetensors', 'flux2-vae.safetensors'),
    }
    model, text, vae = components[args.arch]
    client = ComfyApiClient(args.api_url, timeout=300)
    report = {'arch': args.arch, 'adapter': str(adapter), 'quality_validation': False,
              'verification': 'live loader and preview execution only', 'renders': [], 'status': 'starting'}
    report_path = root / f'{args.arch}-comfy-report.json'
    try:
        controls = [None] * 3
        if args.control_image:
            from toolkit.control_image import load_control_rgb
            report['control_sources'] = [str(path.resolve()) for path in args.control_image]
            for index, path in enumerate(args.control_image):
                controls[index] = client.upload_image(str(path), subfolder=f'ai-toolkit/{root.name}',
                                                      prepared_image=load_control_rgb(path))
        for strength in (-1., 0., .5, 1.):
            request = ComfySampleRequest(prompt='A cat sitting beside a sunny window, detailed illustration',
                negative_prompt=args.negative_prompt, width=256, height=256, steps=2, cfg=args.cfg, seed=71,
                model=args.model or model, text_encoder=args.text_encoder or text, vae=args.vae or vae,
                sampler='euler', scheduler='simple', inference_lora='', inference_lora_strength=0.,
                output_format='png', output_quality='high', training_lora_path=str(adapter),
                training_lora_filename=adapter.name, filename_prefix=f'ai-toolkit-gpu-smoke/{root.name}/{args.arch}_{strength}',
                training_lora_strength=strength, model_arch=args.arch, specialized_flow=True,
                control_image=controls[0], control_image_2=controls[1], control_image_3=controls[2],
                model_kwargs={'edit': True, 'kv_cache': True} if args.control_image else {})
            graph = get_workflow_for_sample(f'config/comfy_templates/{args.arch}_lora_sample.json.njk', request)
            client.validate_workflow_schema(graph)
            started = time.perf_counter()
            prompt_id = client.post_prompt(graph)
            images = client.wait_for_images(prompt_id)
            assert images, 'Comfy finished without an image'
            path = root / f'{args.arch}_{strength}.png'
            client.download_image(images[0], str(path))
            report['renders'].append({'strength': strength, 'prompt_id': prompt_id,
                'seconds': time.perf_counter() - started, 'image': str(path), 'request': asdict(request)})
            report_path.write_text(json.dumps(report, indent=2))
            print('COMFY_GPU_SMOKE ' + json.dumps(report['renders'][-1]), flush=True)
        if args.expect_identity:
            report['identity_check'] = evaluate_identity_renders(report['renders'])
            if not report['identity_check']['passed']:
                raise AssertionError('Zero-delta adapter changes the base output; see identity_check')
        report['status'] = 'passed'
    except BaseException:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        # No full cache eviction or service restart; existing CPU weight caches
        # are preserved. Only the smoke test's active GPU models are offloaded.
        release_smoke_vram(client, report)
        report_path.write_text(json.dumps(report, indent=2))
        print('COMFY_GPU_SMOKE_REPORT ' + str(report_path), flush=True)


if __name__ == '__main__':
    main()
