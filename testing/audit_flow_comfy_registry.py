"""Audit provided preview graphs via GET /object_info only; never queue renders.

Run from the project with the ai-toolkit Conda environment. File-choice checks
establish registration, not file contents or architecture compatibility. The
training adapter path is a placeholder: this command never loads an adapter.
"""
import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import sys
import urllib.request

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit.comfy_sample import (ComfySampleRequest, ComfyBatchSampleRequest,
    get_workflow_for_sample, get_workflow_for_samples)
from toolkit.comfy_schema import workflow_schema_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-url', default='http://127.0.0.1:8188')
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--seed', type=int, default=2 ** 32 - 1, help='Include realistic large-seed widget ranges in the audit.')
    parser.add_argument('--verbose', action='store_true', help='Include successful graphs, not just failures.')
    parser.add_argument('--krea-model', default='krea2_raw.safetensors')
    parser.add_argument('--anima-model', default='anima_baseV10.safetensors')
    parser.add_argument('--ideogram-model', default='ideogram4_fp8_scaled.safetensors')
    parser.add_argument('--krea-text-encoder', default='qwen3vl_4b_bf16.safetensors')
    parser.add_argument('--anima-text-encoder', default='qwen_3_06b_base.safetensors')
    parser.add_argument('--ideogram-text-encoder', default='qwen3vl_4b_bf16.safetensors')
    parser.add_argument('--krea-vae', default='qwen_image_vae.safetensors')
    parser.add_argument('--anima-vae', default='qwen_image_vae.safetensors')
    parser.add_argument('--ideogram-vae', default='flux2-vae.safetensors')
    parser.add_argument('--inference-lora', default='', help='Optional registered LoRA name to audit that graph branch too; never loaded.')
    parser.add_argument('--krea-control-image', default='', help='Existing Comfy input image name: audit Krea edit graph too. Requires the optional preview helper to be installed; never uploaded or read by this command.')
    args = parser.parse_args()
    with urllib.request.urlopen(args.api_url.rstrip('/') + '/object_info', timeout=args.timeout) as response:
        registry = json.load(response)
    reports = []
    for arch, prefix in [('krea2', 'krea'), ('anima', 'anima'), ('ideogram4', 'ideogram')]:
        for cfg in (1., 3.):
            for negative in ('', 'haze'):
                for inference in (['', args.inference_lora] if args.inference_lora else ['']):
                    for policy in (['image_only', 'negative_prompt'] if arch == 'ideogram4' else ['image_only']):
                        request = ComfySampleRequest(prompt='A cat sitting on a table.', width=768, height=1024,
                            steps=20, cfg=cfg, seed=args.seed, model=getattr(args, prefix + '_model'),
                            vae=getattr(args, prefix + '_vae'), text_encoder=getattr(args, prefix + '_text_encoder'),
                            sampler='euler', scheduler='simple', inference_lora=inference,
                            inference_lora_strength=.75, output_format='png', output_quality='high',
                            training_lora_path='/schema-audit/not-loaded.safetensors',
                            training_lora_filename='not-loaded.safetensors', filename_prefix='schema-audit/not-queued',
                            training_lora_strength=-.5, negative_prompt=negative, model_arch=arch,
                            specialized_flow=True, model_kwargs={'ideogram_cfg_reference': policy})
                        if arch == 'krea2' and args.krea_control_image:
                            request = replace(request, control_image=args.krea_control_image, model_kwargs={'edit': True})
                        variants = [(f'config/comfy_templates/{arch}_lora_sample.json.njk', request)]
                        if arch == 'krea2' and not args.krea_control_image:
                            values = {key: value for key, value in vars(request).items()
                                if key in {field.name for field in fields(ComfyBatchSampleRequest)}}
                            batch = ComfyBatchSampleRequest(**values, prompts=[request.prompt, 'A dog.'], seeds=[args.seed, args.seed + 1])
                            variants.append(('config/comfy_templates/krea2_lora_sample_batch_easy_use.json.njk', batch))
                        for path, sample in variants:
                            graph = (get_workflow_for_samples(path, sample) if isinstance(sample, ComfyBatchSampleRequest)
                                else get_workflow_for_sample(path, sample))
                            errors = workflow_schema_errors(graph, registry)
                            reports.append({'arch': arch, 'batch': isinstance(sample, ComfyBatchSampleRequest),
                                'cfg': cfg, 'negative': negative, 'cfg_reference': policy,
                                'inference_adapter_branch': bool(inference), 'nodes': len(graph), 'errors': errors})
    print(json.dumps({'verification': 'static registered-schema only; no renders or weight loads',
        'graphs': len(reports), 'passed': sum(not report['errors'] for report in reports),
        'results': reports if args.verbose else [report for report in reports if report['errors']]}, indent=2))
    return int(any(report['errors'] for report in reports))


if __name__ == '__main__':
    raise SystemExit(main())
