"""Explicitly invoked, bounded full-checkpoint smoke test (never in CPU discovery).

Uses isolated synthetic data/output, existing local weights, and the real job
startup/cache/train/save/sample paths. Does not edit the UI database or services.
Network downloads are disabled. The resulting report is runtime evidence, NOT a
quality benchmark. Run separately for each architecture and resume phase.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import yaml


def fixture(root):
    for label, phase in [('liked', 0), ('disliked', 25)]:
        folder = root / 'data' / label
        folder.mkdir(parents=True, exist_ok=True)
        for index, (width, height) in enumerate([(256, 256), (320, 256)]):
            path = folder / f'{index}.png'
            if not path.exists():
                y, x = np.mgrid[:height, :width]
                pixels = np.stack([(x + phase) % 256, (y + phase) % 256,
                                   ((x + y) // 2 + index * 20) % 256], axis=-1).astype(np.uint8)
                Image.fromarray(pixels).save(path)
                path.with_suffix('.txt').write_text('An abstract red and blue gradient, smooth colors.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['qwen_image_2', 'krea2', 'anima', 'ideogram4'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--mode', choices=['diffusion_kto', 'flow_dpo', 'guidance_distillation',
                                         'fizgig_image_slider', 'fizgig_prompt_slider', 'loha', 'doha', 'sliderspace'],
                        default='diffusion_kto')
    parser.add_argument('--discovery-mode', choices=['provided', 'generated', 'both'], default='provided')
    parser.add_argument('--adapter-type', choices=['lora', 'dora'], default='lora')
    parser.add_argument('--edit-references', type=int, choices=[0, 1, 2], default=0,
                        help='Exercise actual Krea edit sources in DPO/distillation and native previews')
    parser.add_argument('--quant-cache', type=Path, help='Reuse an existing isolated smoke cache across modes')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-samples', action='store_true')
    parser.add_argument('--model-path')
    args = parser.parse_args()
    if not 1 <= args.steps <= 8:
        parser.error('Smoke tests are bounded to 1–8 total optimizer steps')
    if args.edit_references and (args.arch != 'krea2' or args.mode not in ('flow_dpo', 'guidance_distillation')):
        parser.error('Edit reference smoke tests are limited to Krea DPO/distillation')
    root = args.root.resolve()
    if not (str(root).startswith('/tmp/') or str(root).startswith(str(Path(__file__).resolve().parents[1] / 'tmp') + '/')):
        parser.error('Use an isolated root under /tmp or the repository tmp folder')
    root.mkdir(parents=True, exist_ok=True)
    suffix_name = 'kto' if args.mode == 'diffusion_kto' else args.mode
    name = f'{args.arch}_{suffix_name}_smoke'
    output = root / 'output' / name
    if output.exists() and not args.resume:
        parser.error('Output already exists: use --resume or a fresh root')
    queue = json.load(urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=10))
    if queue['queue_running'] or queue['queue_pending']:
        parser.error('ComfyUI is occupied; defer the GPU smoke test')
    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    free, total = torch.cuda.mem_get_info()
    if free < 18 * 1024 ** 3:
        parser.error('Less than 18 GiB free: defer rather than evict another GPU user')
    fixture(root)
    examples = Path(__file__).resolve().parents[1] / 'config/examples'
    suffix = 'qwen_image_21' if args.arch == 'qwen_image_2' else args.arch
    if args.mode == 'diffusion_kto':
        example = examples / f'train_diffusion_kto_{suffix}.yaml'
    else:
        # The shared ports use these same process/config schemas on Qwen too.
        template_arch = 'krea2' if args.arch == 'qwen_image_2' else args.arch
        template_mode = 'guidance_distillation' if args.mode in ('loha', 'doha') else args.mode
        example = examples / f'train_{template_mode}_{template_arch}.yaml'
    config = yaml.safe_load(example.read_text())
    config['config']['name'] = name
    process = config['config']['process'][0]
    process['training_folder'] = str(root / 'output')
    process['sqlite_db_path'] = str(root / 'unused-ui.db')
    os.environ.pop('AITK_JOB_ID', None)
    process['performance_log_every'] = 1
    model = process['model']
    model['arch'] = args.arch
    paths = {'qwen_image_2': 'Comfy-Org/Qwen-Image-2.1',
             'krea2': '/home/bart/ComfyUI/models/diffusion_models/krea2_raw.safetensors'}
    if not args.model_path and args.arch not in paths:
        parser.error('This architecture needs an explicit locally complete --model-path')
    model['name_or_path'] = args.model_path or paths[args.arch]
    model.update(qtype='convrot8', qtype_te='convrot8', layer_offloading=True,
                 layer_offloading_transformer_percent=.75, layer_offloading_text_encoder_percent=1.,
                 low_vram=True, compile=False)
    model['quantized_model_cache_dir'] = str((args.quant_cache or root / 'quantized-cache').resolve())
    if args.arch == 'krea2':
        model['model_kwargs'].update(text_encoder_path='Qwen/Qwen3-VL-4B-Instruct',
                                     vae_path='Qwen/Qwen-Image', max_text_length=128)
        if args.edit_references:
            model['model_kwargs'].update(edit=True, kv_cache=True)
    elif args.arch == 'qwen_image_2':
        model['text_encoder_path'] = '/d/comfy_models/text_encoders/qwen3vl_8b_int8_convrot.safetensors'
    process['network'].update(linear=2, linear_alpha=1)
    process['network']['type'] = args.mode if args.mode in ('loha', 'doha') else args.adapter_type
    if args.mode == 'diffusion_kto':
        process['diffusion_kto'].update(reference_estimator='score_window', score_window_size=2)
    else:
        if args.mode in ('loha', 'doha'):
            process['type'] = 'sd_trainer'
            process.pop('guidance_distillation')
        if args.mode == 'guidance_distillation':
            process['guidance_distillation']['teacher_cfg_scale'] = 2
        if args.mode in ('fizgig_image_slider', 'fizgig_prompt_slider'):
            process['fizgig_slider'].update(cfg_scale=2, bank_size=2, bank_steps=2, bank_resolution=256,
                anchor_prompts=[{'prompt': 'A dog sitting beside a window.', 'negative_prompt': 'cat'}])
        if args.mode == 'sliderspace':
            process['sliderspace'].update(discovery_mode=args.discovery_mode, num_directions=2,
                discovery_samples=3, resolution=256, discovery_steps=2, cfg_scale=1,
                concept_prompts=['An abstract red and blue gradient, smooth colors.'],
                discovery_datasets=[{'folder_path': str(root / 'data' / label)} for label in ('liked', 'disliked')],
                discovery_buckets=True, seed=71)
        if args.mode != 'fizgig_prompt_slider':
            process['datasets'] = process['datasets'][:1]
    for data, label in zip(process['datasets'], ['liked', 'disliked']):
        data.update(folder_path=str(root / 'data' / label), resolution=[256],
                    num_workers=0, cache_latents_num_workers=0)
        if args.mode != 'diffusion_kto':
            data.pop('kto_label', None)
            if args.mode in ('flow_dpo', 'fizgig_image_slider'):
                data['control_path_1'] = str(root / 'data' / 'disliked')
            if args.mode == 'fizgig_image_slider':
                data['anchor_path'] = str(root / 'data' / 'disliked')
        if args.edit_references:
            first_slot = 2 if args.mode == 'flow_dpo' else 1
            for offset, source_label in enumerate(('liked', 'disliked')[:args.edit_references]):
                data[f'control_path_{first_slot + offset}'] = str(root / 'data' / source_label)
    process['train'].update(steps=args.steps, gradient_accumulation=2,
                            optimizer='adamw', lr=0.0001, skip_first_sample=True,
                            disable_sampling=args.no_samples, seed=71)
    if args.mode == 'sliderspace':
        process['train']['gradient_accumulation'] = 1
    process['save'].update(save_every=1, max_step_saves_to_keep=20, dtype='float32')
    process['sample'].update(sample_every=100, sample_steps=2, width=256, height=256,
                             guidance_scale=1, seed=71)
    if args.edit_references:
        for sample in process['sample']['samples']:
            for slot, source_label in enumerate(('liked', 'disliked')[:args.edit_references], start=1):
                sample[f'ctrl_img_{slot}'] = str(root / 'data' / source_label / '0.png')
    config_path = root / f'{args.arch}-{args.mode}-steps-{args.steps}.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(arch=args.arch, mode=args.mode, configuration=str(config_path), resume=args.resume,
                  edit_references=args.edit_references,
                  free_vram_before=free, total_vram=total, gpu=torch.cuda.get_device_name(),
                  synthetic_data=True, quality_validation=False, updates=[], status='starting')
    started = time.perf_counter()
    report_path = root / f'{args.arch}-{args.mode}-steps-{args.steps}-report.json'
    old_samples = set((output / 'samples').glob('*'))
    expected_optimizer = None
    if args.resume:
        optimizer_path = output / 'optimizer.pt'
        if args.mode == 'sliderspace':
            marker = json.loads((output / 'sliderspace_state/checkpoint_final.json').read_text())
            optimizer_path = output / 'sliderspace_state' / f"bank_{marker['step']:09d}.optimizer.pt"
        expected_optimizer = torch.load(optimizer_path, weights_only=True, map_location='cpu')['state']
    torch.manual_seed(71)
    torch.cuda.reset_peak_memory_stats()
    try:
        from toolkit.job import get_job
        job = get_job(str(config_path))
        trainer = job.process[0]
        original = trainer.train_single_accumulation
        original_model_hook = trainer.hook_after_model_load
        original_loop = trainer.hook_train_loop

        def loaded():
            original_model_hook()
            from toolkit.flow_cache_identity import resolved_flow_components
            report['loaded_components'] = resolved_flow_components(trainer.sd)

        def train_loop(batch):
            step_started = time.perf_counter()
            result = original_loop(batch)
            torch.cuda.synchronize()
            report.setdefault('optimizer_windows', []).append({'step': int(trainer.step_num),
                'seconds': time.perf_counter() - step_started})
            return result

        def audit(batch, accum_scale=1.):
            if not report['updates']:
                report['startup_cache_seconds'] = time.perf_counter() - started
            if 'loaded_components' not in report:
                from toolkit.flow_cache_identity import resolved_flow_components
                report['loaded_components'] = resolved_flow_components(trainer.sd)
            if expected_optimizer is not None and not report['updates']:
                restored = trainer.optimizer.state_dict()['state']
                assert restored.keys() == expected_optimizer.keys(), 'Optimizer state mapping changed on resume'
                for key, values in expected_optimizer.items():
                    for field, value in values.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(restored[key][field].detach().cpu(), value, atol=0, rtol=0)
                        else:
                            assert restored[key][field] == value, 'Optimizer state differs on resume'
                report['optimizer_resume_exact'] = True
            step_started = time.perf_counter()
            loss = original(batch, accum_scale)
            gradients = [p.grad for p in trainer.network.parameters() if p.requires_grad and p.grad is not None]
            assert gradients and all(torch.isfinite(g).all().item() for g in gradients), 'Missing/nonfinite adapter gradients'
            grad_norm = math.sqrt(sum(g.float().square().sum().item() for g in gradients))
            assert grad_norm > 0, 'Zero adapter gradients'
            assert all(p.grad is None for p in trainer.sd.unet.parameters() if not p.requires_grad), 'Frozen base received gradients'
            value = float(loss)
            assert math.isfinite(value), 'Nonfinite loss'
            torch.cuda.synchronize()
            report['updates'].append(dict(step=int(trainer.step_num), loss=value, grad_norm=grad_norm,
                                          seconds=time.perf_counter() - step_started))
            print('GPU_SMOKE_UPDATE ' + json.dumps(report['updates'][-1]), flush=True)
            return loss

        trainer.train_single_accumulation = audit
        trainer.hook_after_model_load = loaded
        trainer.hook_train_loop = train_loop
        job.run()
        assert report['updates'], 'No actual updates performed (resume may already be complete)'
        if args.resume:
            assert report['updates'][0]['step'] > 0, 'Resume restarted from step zero'
        if args.mode == 'sliderspace':
            marker = json.loads((output / 'sliderspace_state/checkpoint_final.json').read_text())
            bank = output / 'sliderspace_state' / f"bank_{marker['step']:09d}.safetensors"
            assert bank.is_file() and bank.with_suffix('.optimizer.pt').is_file(), 'Missing complete bank/optimizer'
            assert len(marker['exports']) == 2, 'Missing independent direction exports'
            report['bank_manifest'] = marker
        else:
            assert (output / 'optimizer.pt').is_file(), 'Missing optimizer save'
        assert list(output.glob('*.safetensors')), 'Missing adapter export'
        report['status'] = 'passed'
        report['final_step'] = int(trainer.step_num)
        from toolkit.flow_training import trainer_flow_profile
        report['flow_profile'] = trainer_flow_profile(trainer).identity()
        report['model_config'] = process['model']
        report['adapter_config'] = process['network']
        report['samples'] = [str(path) for path in (output / 'samples').glob('*') if path.is_file()]
        if not args.no_samples:
            assert set((output / 'samples').glob('*')) - old_samples, 'Missing new native final preview'
        job.cleanup()
    except BaseException:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report.update(seconds=time.perf_counter() - started,
                      peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
        report_path.write_text(json.dumps(report, indent=2))
        print('GPU_SMOKE_REPORT ' + str(report_path), flush=True)


if __name__ == '__main__':
    main()
