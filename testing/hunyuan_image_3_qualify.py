"""Run a real Hunyuan job through the normal trainer and record qualification.

No checkpoint downloads or synthetic model substitutions are performed here.
Example: conda run --no-capture-output -n ai-toolkit python -u
testing/hunyuan_image_3_qualify.py job.yaml --report /tmp/h3.json
--preview /tmp/base.png --preview-steps 50 --preview-strength 0
"""
import argparse
import json
from pathlib import Path
import resource
import random
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil
import torch
import yaml

from toolkit.config_modules import GenerateImageConfig
from toolkit.job import get_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Normal sd_trainer YAML, including real local assets/dataset')
    parser.add_argument('--report', required=True)
    parser.add_argument('--preview', help='Optional extra native sample output after training')
    parser.add_argument('--preview-steps', type=int, default=50)
    parser.add_argument('--preview-strength', type=float, default=0.)
    parser.add_argument('--preview-t2i', action='store_true', help='Extra preview excludes edit references')
    parser.add_argument('--seed', type=int, default=42, help='Seed initial adapters, noise and random offload selection')
    args = parser.parse_args()
    random.seed(args.seed);torch.manual_seed(args.seed)
    job = get_job(yaml.safe_load(Path(args.config).read_text()))
    trainer = job.process[0]
    if trainer.model_config.arch not in ('hunyuan_image_3_instruct', 'hunyuan_image_3_base'):
        parser.error('Configuration must select a HunyuanImage 3 architecture')
    started = time.perf_counter()
    process = psutil.Process()
    report = {'config': str(Path(args.config).resolve()), 'steps': [], 'geometry': [],
              'gradients': {'finite': True, 'nonzero_tensors': []}, 'status': 'running',
              'peaks': {'allocated_gib': 0., 'reserved_gib': 0., 'rss_gib': 0., 'process_vram_mib': 0}}
    stopped = threading.Event()
    seen_geometry, seen_gradients = set(), set()

    def persist():
        path = Path(args.report); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))

    def frozen_storage():
        seen = set()
        resident = pinned = meta = 0
        dtypes = {}
        for tensor in list(trainer.sd.model.parameters()) + list(trainer.sd.model.buffers()):
            if tensor.is_meta:
                meta += tensor.numel() * tensor.element_size()
                continue
            key = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if key in seen:
                continue
            seen.add(key)
            size = tensor.numel() * tensor.element_size()
            dtype = str(tensor.dtype)
            dtypes[dtype] = dtypes.get(dtype, 0) + size
            if tensor.is_cuda:
                resident += size
            elif tensor.is_pinned():
                pinned += size
        return dict(resident_gib=resident/2**30, pinned_tensor_gib=pinned/2**30,
                    meta_gib=meta/2**30, dtype_bytes=dtypes,
                    host_available_gib=psutil.virtual_memory().available/2**30,
                    offload_fraction=trainer.model_config.layer_offloading_transformer_percent,
                    seed=args.seed)

    def poll():
        while not stopped.wait(.5):
            peak = report['peaks']
            peak['rss_gib'] = max(peak['rss_gib'], process.memory_info().rss / 2**30)
            peak['allocated_gib'] = max(peak['allocated_gib'], torch.cuda.memory_allocated() / 2**30)
            peak['reserved_gib'] = max(peak['reserved_gib'], torch.cuda.memory_reserved() / 2**30)
            try:
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                                '--format=csv,noheader,nounits'], text=True)
                for row in rows.splitlines():
                    pid, used = row.split(',')
                    if int(pid) == process.pid:
                        peak['process_vram_mib'] = max(peak['process_vram_mib'], int(used))
            except (subprocess.SubprocessError, ValueError):
                pass

    before_loop = trainer.hook_before_train_loop
    def install_gradients():
        before_loop()
        report['frozen_storage_after_conditioning_unload'] = frozen_storage()
        # Offload selection consumes Python RNG draws. Reset the training RNG
        # after preparation so residency benchmarks use identical training inputs.
        random.seed(args.seed);torch.manual_seed(args.seed)
        for name, parameter in trainer.network.named_parameters():
            if not parameter.requires_grad:
                continue
            def observe(gradient, name=name):
                report['gradients']['finite'] &= bool(gradient.isfinite().all())
                if name not in seen_gradients and bool(gradient.count_nonzero()):
                    seen_gradients.add(name)
                    report['gradients']['nonzero_tensors'].append(name)
            parameter.register_hook(observe)
    trainer.hook_before_train_loop = install_gradients

    after_load = trainer.hook_after_model_load
    def inspect_geometry():
        after_load()
        report['frozen_storage'] = frozen_storage()
        def sample_progress(index, total, latents):
            print(f'HUNYUAN SAMPLE {index + 1}/{total} elapsed={time.perf_counter() - started:.2f}s', flush=True)
        trainer.sd.sample_step_hook = sample_progress
        predict = trainer.sd.get_noise_prediction
        def observed(latent_model_input, timestep, text_embeddings, *rest, **kwargs):
            latents, embeds = latent_model_input, text_embeddings
            from extensions_built_in.diffusion_models.hunyuan_image_3.src.conditioning import rebuild_target_ids
            ids = rebuild_target_ids(embeds.ids[0], latents.shape[-2:], trainer.sd.tokenizer[0], trainer.sd.variant)
            shape = (len(ids), int(embeds.reference_count[0]), *latents.shape[-2:])
            if shape not in seen_geometry:
                seen_geometry.add(shape)
                report['geometry'].append(dict(sequence_tokens=shape[0], references=shape[1],
                                               latent_height=shape[2], latent_width=shape[3]))
            return predict(latents, timestep, embeds, *rest, **kwargs)
        trainer.sd.get_noise_prediction = observed
    trainer.hook_after_model_load = inspect_geometry

    end_step = trainer.end_step_hook
    def record_step():
        end_step()
        ups = [parameter.detach().float() for name, parameter in trainer.network.named_parameters()
               if 'lora_up' in name or 'lora_B' in name]
        row = dict(step=trainer.step_num, elapsed_s=time.perf_counter()-started,
                   allocated_gib=torch.cuda.memory_allocated()/2**30,
                   reserved_gib=torch.cuda.memory_reserved()/2**30,
                   rss_gib=process.memory_info().rss/2**30,
                   adapter_up_norm=float(torch.stack([p.square().sum() for p in ups]).sum().sqrt()) if ups else None)
        report['steps'].append(row);persist();print('HUNYUAN QUALIFICATION', row, flush=True)
    trainer.end_step_hook = record_step

    if args.preview:
        # The normal trainer releases its holder before job.run returns.
        # Observe its existing final phase, then preserve the standard release.
        release = trainer._release_training_memory_before_comfy_wait
        def preview_and_release(*release_args, **release_kwargs):
            try:
                sample = trainer.sample_config
                item = sample.samples[0]
                config = GenerateImageConfig(prompt=item.prompt, width=sample.width, height=sample.height,
                                             num_inference_steps=args.preview_steps,
                                             guidance_scale=sample.guidance_scale, seed=42,
                                             network_multiplier=args.preview_strength, output_path=args.preview,
                                             ctrl_img_1=None if args.preview_t2i else item.ctrl_img_1,
                                             ctrl_img_2=None if args.preview_t2i else item.ctrl_img_2,
                                             ctrl_img_3=None if args.preview_t2i else item.ctrl_img_3)
                trainer._run_with_optimizer_state_offload(lambda: trainer.sd.generate_images([config]))
            finally:
                release(*release_args, **release_kwargs)
        trainer._release_training_memory_before_comfy_wait = preview_and_release

    watcher = threading.Thread(target=poll, daemon=True);watcher.start()
    try:
        job.run()
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed';report['error'] = repr(error)
        raise
    finally:
        stopped.set();watcher.join(timeout=2)
        report['elapsed_s'] = time.perf_counter()-started
        report['maxrss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        persist()
        job.cleanup()


if __name__ == '__main__':
    main()
