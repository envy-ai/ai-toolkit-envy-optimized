"""Fixed, opt-in spatial prediction probes for ordinary image LoRA training.

This uses the same image item, latent, prompt cache, scheduler and model forward
as the scalar validation path. Probe inputs live on CPU between captures.
"""

import contextlib
import copy
import hashlib
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.train_tools import get_torch_dtype


LEVELS = (0.25, 0.5, 0.75)
MAX_SIDE = 48


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, allow_nan=False)
    os.replace(tmp, path)


def _scales_from_frames(directory, steps):
    scales = {}
    for step in steps:
        with (directory / f'step-{step:09d}.json').open(encoding='utf-8') as handle:
            frame = json.load(handle)
        for key, levels in frame['maps'].items():
            key_scales = scales.setdefault(key, {})
            for level, entry in levels.items():
                level_scales = key_scales.setdefault(level, {})
                for name, grid in entry.items():
                    value = max((abs(v) for v in grid['v']), default=0.0)
                    level_scales[name] = max(level_scales.get(name, 0.0), value)
    return scales


def _capture_signature(trainer):
    model_config = trainer.model_config
    train_config = trainer.train_config
    scheduler_config = getattr(trainer.sd.noise_scheduler, 'config', {})
    source = getattr(model_config, 'name_or_path', None)
    source_stat = None
    if source and os.path.isfile(str(source)):
        stat = os.stat(source)
        source_stat = (stat.st_size, stat.st_mtime_ns)
    payload = {
        'model_arch': model_config.arch,
        'model_source': source,
        'model_source_stat': source_stat,
        'vae_path': getattr(model_config, 'vae_path', None),
        'text_encoder_path': getattr(model_config, 'text_encoder_path', None),
        'model_kwargs': model_config.model_kwargs,
        'network': {key: getattr(getattr(trainer, 'network_config', None), key, None) for key in
                    ('type', 'rank', 'linear', 'conv', 'alpha', 'linear_alpha',
                     'conv_alpha', 'network_kwargs', 'transformer_only')},
        'scheduler_config': dict(scheduler_config),
        'train_scheduler': train_config.noise_scheduler,
        'timestep_type': train_config.timestep_type,
        'num_train_timesteps': train_config.num_train_timesteps,
        'min_denoising_steps': train_config.min_denoising_steps,
        'max_denoising_steps': train_config.max_denoising_steps,
        'linear_timesteps': train_config.linear_timesteps,
        'linear_timesteps2': train_config.linear_timesteps2,
        'cfg_scale': train_config.cfg_scale,
        'bypass_guidance_embedding': train_config.bypass_guidance_embedding,
        'dtype': train_config.dtype,
    }
    helper = getattr(model_config, 'assistant_lora_path', None)
    payload['training_helper'] = helper
    if helper and os.path.isfile(str(helper)):
        helper_stat = os.stat(helper)
        payload['training_helper_stat'] = (helper_stat.st_size, helper_stat.st_mtime_ns)
    raw = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _grid(values):
    values = values.float().detach().cpu()
    height, width = values.shape
    factor = max(1, (max(height, width) + MAX_SIDE - 1) // MAX_SIDE)
    if factor > 1:
        values = torch.nn.functional.avg_pool2d(values[None, None], factor, ceil_mode=True)[0, 0]
    height, width = values.shape
    return {'w': width, 'h': height, 'v': [float(v) for v in values.flatten().tolist()]}


def _module_modes(module):
    return [(child, child.training) for child in module.modules()]


def _restore_modes(modes):
    if modes:
        modes[0][0].train(modes[0][1])
    for module, was_training in modes:
        module.training = was_training


def _digest_prompt(embeds, digest):
    fields = ({key: embeds[key] for key in embeds.keys()} if hasattr(embeds, 'keys')
              else {key: getattr(embeds, key, None) for key in
                    ('text_embeds', 'pooled_embeds', 'attention_mask', 'minimax_token_tags')})
    def visit(value):
        if value is None:
            digest.update(b'none')
        elif torch.is_tensor(value):
            data = value.detach().cpu().contiguous()
            digest.update(str((tuple(data.shape), data.dtype)).encode())
            digest.update(data.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                visit(value[key])
        else:
            raise TypeError(f'Unsupported prompt embedding value {type(value).__name__}')
    for key in sorted(fields):
        digest.update(key.encode())
        visit(fields[key])
    digest.update(json.dumps(getattr(embeds, 'frozen_dtype_keys', None), sort_keys=True).encode())


@contextlib.contextmanager
def _preserve_rng(device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    # torch.manual_seed seeds every CUDA device, so every device must be forked.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _check_supported(trainer):
    config = trainer.train_config
    model = trainer.sd
    arch = getattr(model, 'arch', None)
    if arch not in ('qwen_image', 'qwen_image_2', 'krea2', 'moss_image'):
        raise ValueError(f'Training maps support Qwen Image, Krea 2 and MOSS Image text-to-image; got {arch!r}')
    if not model.is_flow_matching or getattr(model, 'is_edit', False) or getattr(model, 'is_video', False):
        raise ValueError('Training maps require ordinary still-image text-to-image flow training')
    if trainer.network_config is None or trainer.network_config.type not in ('lora', 'dora'):
        raise ValueError('Training maps require an unmerged LoRA or DoRA network')
    if trainer.network is None or getattr(trainer.network, 'is_merged_in', False):
        raise ValueError('Training maps require an attached, unmerged trainable network')
    if trainer.adapter_config is not None or trainer.embed_config is not None or trainer.decorator_config is not None:
        raise ValueError('Training maps do not support adapters, embeddings or decorators')
    if trainer.train_config.train_text_encoder:
        raise ValueError('Training maps do not support text-encoder training')
    if config.unload_text_encoder and not trainer.is_caching_text_embeddings:
        raise ValueError('Training maps need cached text embeddings when the text encoder is unloaded')
    incompatible = {
        'loss_target': config.loss_target != 'noise',
        'loss_type': config.loss_type != 'mse',
        'train_turbo': config.train_turbo,
        'do_cfg': config.do_cfg or config.do_random_cfg,
        'do_guidance_loss': config.do_guidance_loss,
        'do_prior_divergence': config.do_prior_divergence,
        'do_signal_amplification': config.do_signal_amplification,
        'do_signal_correction_noise': config.do_signal_correction_noise,
        'do_batch_noise_correction': config.do_batch_noise_correction,
        'dynamic_noise_offset': config.dynamic_noise_offset,
        'noise_offset': config.noise_offset != 0.0,
        'blended_blur_noise': config.blended_blur_noise,
        'optimal_noise_pairing_samples': config.optimal_noise_pairing_samples != 1,
        'adaptive_scaling_factor': config.adaptive_scaling_factor,
        'standardize_images': config.standardize_images,
        'do_blank_stabilization': config.do_blank_stabilization,
        'noise_multiplier': config.noise_multiplier != 1.0,
        'target_noise_multiplier': config.target_noise_multiplier != 1.0,
        'random_noise_multiplier': config.random_noise_multiplier != 0.0,
        'random_noise_shift': config.random_noise_shift != 0.0,
        'noisy_latent_multiplier': config.noisy_latent_multiplier != 1.0,
        'img_multiplier': config.img_multiplier not in (None, 1.0),
        't0_loss_target': config.t0_loss_target,
    }
    enabled = [key for key, active in incompatible.items() if active]
    if enabled:
        raise ValueError('Training maps do not support these training options: ' + ', '.join(enabled))
    if config.noise_scheduler != 'flowmatch':
        raise ValueError('Training maps require the flowmatch training scheduler')


def validate_request(trainer):
    """Reject unsupported job shapes before model weights are allocated."""
    if trainer.__class__.__name__ not in ('SDTrainer', 'UITrainer', 'DiffusionTrainer'):
        raise ValueError('Training maps are available only in ordinary diffusion training')
    if trainer.accelerator.num_processes != 1:
        raise ValueError('Training maps currently require a single training process')
    arch = trainer.model_config.arch
    if arch not in ('qwen_image', 'qwen_image_2', 'krea2', 'moss_image'):
        raise ValueError(f'Training maps do not support model architecture {arch!r}')
    if trainer.network_config is None or trainer.network_config.type not in ('lora', 'dora'):
        raise ValueError('Training maps require a LoRA or DoRA network')
    if trainer.model_config.refiner_name_or_path is not None:
        raise ValueError('Training maps do not support refiner training')
    kwargs = trainer.model_config.model_kwargs or {}
    raw_arch = str((trainer.get_conf('model', {}) or {}).get('arch') or '')
    if (kwargs.get('edit') or kwargs.get('kv_cache') or raw_arch.endswith(':edit')
            or raw_arch.endswith(':base_edit')):
        raise ValueError('Training maps currently require text-to-image training without edit references')
    if trainer.train_config.noise_scheduler != 'flowmatch':
        raise ValueError('Training maps require a flowmatch training scheduler')
    config = trainer.train_config
    unsupported = {
        'train_text_encoder': config.train_text_encoder,
        'loss_target': config.loss_target != 'noise',
        'loss_type': config.loss_type != 'mse',
        'train_turbo': config.train_turbo,
        'do_cfg': config.do_cfg or config.do_random_cfg,
        'do_guidance_loss': config.do_guidance_loss,
        'do_prior_divergence': config.do_prior_divergence,
        'do_signal_amplification': config.do_signal_amplification,
        'do_signal_correction_noise': config.do_signal_correction_noise,
        'do_batch_noise_correction': config.do_batch_noise_correction,
        'dynamic_noise_offset': config.dynamic_noise_offset,
        'noise_offset': config.noise_offset != 0.0,
        'blended_blur_noise': config.blended_blur_noise,
        'optimal_noise_pairing_samples': config.optimal_noise_pairing_samples != 1,
        'adaptive_scaling_factor': config.adaptive_scaling_factor,
        'standardize_images': config.standardize_images,
        'do_blank_stabilization': config.do_blank_stabilization,
        'noise_multiplier': config.noise_multiplier != 1.0,
        'target_noise_multiplier': config.target_noise_multiplier != 1.0,
        'random_noise_multiplier': config.random_noise_multiplier != 0.0,
        'random_noise_shift': config.random_noise_shift != 0.0,
        'noisy_latent_multiplier': config.noisy_latent_multiplier != 1.0,
        'img_multiplier': config.img_multiplier not in (None, 1.0),
        't0_loss_target': config.t0_loss_target,
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError('Training maps do not support these training options: ' + ', '.join(enabled))


def _selected_items(trainer):
    from toolkit.data_loader import get_dataloader_datasets
    candidates = []
    for dataset in get_dataloader_datasets(trainer.data_loader):
        for index, file_item in enumerate(dataset.file_list):
            if getattr(file_item, 'is_reg', False) or getattr(file_item, 'is_video', False):
                continue
            candidates.append((str(file_item.path), dataset, index))
    candidates.sort(key=lambda entry: entry[0])
    if not candidates:
        raise ValueError('Training maps found no still images in training datasets')
    count = min(trainer.logging_config.training_maps_images, len(candidates))
    indices = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)] if count > 1 else [len(candidates) // 2]
    return [candidates[index] for index in indices]


class TrainingMaps:
    def __init__(self, trainer):
        self.directory = Path(trainer.save_root) / 'training_maps'
        self.manifest_path = self.directory / 'manifest.json'
        self.manifest = {'version': 1, 'status': 'ready', 'error': None,
                         'levels': list(LEVELS), 'probes': [], 'steps': [], 'scales': {},
                         'signature': _capture_signature(trainer)}
        self.probes = []
        self.baseline = {}
        self.baseline_step = trainer.start_step
        self.last_step = trainer.start_step
        self.failed = False

    def setup(self, trainer):
        _check_supported(trainer)
        from PIL import Image
        device = trainer.device_torch
        dtype = get_torch_dtype(trainer.train_config.dtype)
        with contextlib.ExitStack() as placements, _preserve_rng(device), torch.no_grad():
            torch.manual_seed(210819)
            random.seed(210819)
            np.random.seed(210819)
            vae_placement_saved = False
            text_placement_saved = False
            for number, (source_path, dataset, index) in enumerate(_selected_items(trainer)):
                original = dataset.file_list[index]
                ds_config = original.dataset_config
                if (getattr(original, 'has_augmentations', False) or original.augments
                        or ds_config.random_crop or getattr(ds_config, 'random_scale', False)):
                    raise ValueError('Training maps require images without random crops or augmentations')
                if getattr(ds_config, 'shuffle_tokens', False) or getattr(ds_config, 'random_triggers', None):
                    raise ValueError('Training maps require captions without token shuffling or random triggers')
                item = dataset._get_single_item(index)
                if item.path != source_path:
                    raise ValueError(f'Training maps could not load selected image {source_path}; dataset substituted {item.path}')
                batch = DataLoaderBatchDTO(file_items=[item])
                if batch.latents is None:
                    if batch.tensor is None or trainer.sd.vae is None:
                        raise ValueError(f'Training maps cannot encode probe image {source_path}')
                    if not vae_placement_saved:
                        vae = trainer.sd.vae
                        vae_device = vae.device
                        placements.callback(vae.to, vae_device)
                        placements.callback(_restore_modes, _module_modes(vae))
                        vae_placement_saved = True
                    latents = trainer.sd.encode_images(batch.tensor.to(device, dtype=dtype))
                else:
                    latents = batch.latents
                if latents.ndim != 4 or latents.shape[0] != 1:
                    raise ValueError(f'Training maps expected one 4-D image latent for {source_path}')
                latents_cpu = latents.detach().to('cpu', dtype=torch.float32, copy=True)

                caption = (item.get_dropout_caption() if getattr(item, '_caption_was_dropped', False)
                           else (item.caption or ''))
                if batch.prompt_embeds is not None:
                    embeds = batch.prompt_embeds.clone().detach().to('cpu')
                else:
                    if trainer.sd.text_encoder is None:
                        raise ValueError(f'Training maps need cached text embeddings for {source_path}')
                    if not text_placement_saved:
                        encoders = trainer.sd.text_encoder
                        encoders = encoders if isinstance(encoders, list) else [encoders]
                        for encoder in encoders:
                            encoder_device = next(encoder.parameters()).device
                            placements.callback(encoder.to, encoder_device)
                            placements.callback(_restore_modes, _module_modes(encoder))
                        trainer.sd.text_encoder_to(device)
                        text_placement_saved = True
                    prompt = caption
                    if trainer.trigger_word is not None:
                        prompt = trainer.sd.inject_trigger_into_prompt(
                            prompt, trigger=trainer.trigger_word, add_if_not_present=True)
                    embeds = trainer.sd.encode_prompt([prompt]).detach().to('cpu')

                # The display image follows the file item's own resize, flip and crop.
                # Bypass the latent cache only for this one image rendering pass.
                display = copy.deepcopy(item)
                display.is_latent_cached = False
                display.is_text_embedding_cached = False
                display.load_and_process_image(None, only_load_latents=True)
                image = display.tensor
                if not isinstance(image, Image.Image):
                    raise ValueError(f'Training maps could not render the processed probe image {source_path}')
                image = image.convert('RGB')
                image_name = f'probe-{number}.png'
                self.directory.mkdir(parents=True, exist_ok=True)
                image.save(self.directory / image_name)

                key = f'probe-{number}'
                generator = torch.Generator(device='cpu').manual_seed(210819 + number)
                noise = torch.randn(latents_cpu.shape, generator=generator, dtype=torch.float32)
                digest = hashlib.sha256()
                digest.update(latents_cpu.contiguous().numpy().tobytes())
                digest.update(noise.contiguous().numpy().tobytes())
                _digest_prompt(embeds, digest)
                digest.update(image.tobytes())
                self.probes.append({'key': key, 'latents': latents_cpu, 'noise': noise, 'embeds': embeds})
                self.manifest['probes'].append({
                    'key': key, 'image_path': image_name,
                    'source_path': source_path, 'caption': caption,
                    'crop': [int(item.crop_x), int(item.crop_y), int(item.crop_width), int(item.crop_height)],
                    'width': image.width, 'height': image.height,
                    'fingerprint': digest.hexdigest(),
                })
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding='utf-8') as handle:
                previous = json.load(handle)
            if (trainer.start_step > 0 and previous.get('version') == 1
                    and previous.get('probes') == self.manifest['probes']
                    and previous.get('levels') == self.manifest['levels']
                    and previous.get('signature') == self.manifest['signature']):
                self.manifest['steps'] = sorted({int(step) for step in previous.get('steps', [])
                                                 if int(step) <= trainer.start_step})
                for future in self.directory.glob('step-*.json'):
                    if int(future.stem.split('-')[-1]) > trainer.start_step:
                        future.unlink()
                self.manifest['scales'] = _scales_from_frames(self.directory, self.manifest['steps'])
                if self.manifest['steps']:
                    self.baseline_step = min(self.manifest['steps'])
                    first_frame = self.directory / f'step-{self.baseline_step:09d}.json'
                    with first_frame.open(encoding='utf-8') as handle:
                        baseline_maps = json.load(handle)['maps']
                    for key, levels in baseline_maps.items():
                        for level, entry in levels.items():
                            self.baseline[(key, level)] = entry['error']
            else:
                for old_frame in self.directory.glob('step-*.json'):
                    old_frame.unlink()
        _write_json(self.manifest_path, self.manifest)

    def error(self, error):
        self.failed = True
        self.manifest['status'] = 'error'
        self.manifest['error'] = {'message': str(error), 'stack': traceback.format_exc()}
        from toolkit.print import print_acc
        print_acc('Training map recording stopped:\n' + self.manifest['error']['stack'])
        _write_json(self.manifest_path, self.manifest)

    def capture(self, trainer, step):
        if self.failed:
            return
        try:
            maps = trainer._run_with_optimizer_state_offload(
                lambda: self._capture(trainer, step)
            )
            filename = f'step-{step:09d}.json'
            _write_json(self.directory / filename, {'step': step, 'maps': maps})
            self.manifest['steps'] = sorted(set(self.manifest['steps'] + [step]))
            scales = self.manifest['scales']
            for key, levels in maps.items():
                key_scales = scales.setdefault(key, {})
                for level, entry in levels.items():
                    level_scales = key_scales.setdefault(level, {})
                    for name, grid in entry.items():
                        value = max((abs(v) for v in grid['v']), default=0.0)
                        level_scales[name] = max(level_scales.get(name, 0.0), value)
            _write_json(self.manifest_path, self.manifest)
            self.last_step = step
        except Exception as error:
            self.error(error)

    def _capture(self, trainer, step):
        model = trainer.sd
        network = trainer.network
        device = trainer.device_torch
        dtype = get_torch_dtype(trainer.train_config.dtype)
        model_modes = _module_modes(model.unet)
        network_modes = _module_modes(network)
        was_active = network.is_active
        old_multiplier = network.multiplier
        old_scheduler = model.noise_scheduler
        ema = trainer.ema
        was_ema_training = ema._is_train_mode if ema is not None else False
        result = {}
        try:
            if was_active:
                raise RuntimeError('Training maps cannot capture while the network is already active')
            model.noise_scheduler = copy.deepcopy(old_scheduler)
            model.unet.eval()
            network.eval()
            if ema is not None and was_ema_training:
                ema.eval()
            with _preserve_rng(device), torch.no_grad(), network:
                torch.manual_seed(210819)
                for probe in self.probes:
                    latents = probe['latents'].to(device, dtype=dtype)
                    noise = probe['noise'].to(device, dtype=dtype)
                    embeds = probe['embeds'].clone().to(device, dtype=dtype)
                    scheduler = model.noise_scheduler
                    timestep_type = trainer.train_config.timestep_type
                    if trainer.train_config.linear_timesteps or trainer.train_config.linear_timesteps2 or timestep_type == 'linear':
                        timestep_type = 'linear'
                    scheduler.set_train_timesteps(
                        trainer.train_config.num_train_timesteps, device=device,
                        timestep_type=timestep_type, latents=latents,
                        patch_size=(getattr(getattr(model.unet, 'config', None), 'patch_size', None) or 1),
                    )
                    per_level = {}
                    for level in LEVELS:
                        index = (scheduler.sigmas[:-1].float() - level).abs().argmin()
                        timestep = scheduler.timesteps[index].reshape(1)
                        noisy = model.add_noise(latents, noise, timestep).detach()
                        target = noise.float() - latents.float()
                        network.multiplier = 1.0
                        prediction = model.predict_noise(
                            latents=noisy, conditional_embeddings=embeds, timestep=timestep,
                            guidance_scale=1.0,
                            guidance_embedding_scale=trainer.train_config.cfg_scale,
                            bypass_guidance_embedding=trainer.train_config.bypass_guidance_embedding,
                        ).float()
                        if prediction.shape != target.shape:
                            raise ValueError(f'Training map prediction shape {prediction.shape} differs from target {target.shape}')
                        error_map = (prediction - target).square().mean(dim=1)[0]
                        error_grid = _grid(error_map)
                        level_key = f'{level:g}'
                        base_key = (probe['key'], level_key)
                        if base_key not in self.baseline:
                            self.baseline[base_key] = error_grid
                        network.multiplier = 0.0
                        base_prediction = model.predict_noise(
                            latents=noisy, conditional_embeddings=embeds, timestep=timestep,
                            guidance_scale=1.0,
                            guidance_embedding_scale=trainer.train_config.cfg_scale,
                            bypass_guidance_embedding=trainer.train_config.bypass_guidance_embedding,
                        ).float()
                        effect_map = (prediction - base_prediction).abs().mean(dim=1)[0]
                        entry = {'error': error_grid, 'effect': _grid(effect_map)}
                        if step != self.baseline_step:
                            baseline = self.baseline[base_key]
                            if (baseline['w'], baseline['h']) != (error_grid['w'], error_grid['h']):
                                raise ValueError('Training map grid shape changed from the baseline')
                            entry['improvement'] = {
                                'w': error_grid['w'], 'h': error_grid['h'],
                                'v': [old - new for old, new in zip(baseline['v'], error_grid['v'])],
                            }
                        per_level[level_key] = entry
                    result[probe['key']] = per_level
        finally:
            network.multiplier = old_multiplier
            network.is_active = was_active
            _restore_modes(network_modes)
            _restore_modes(model_modes)
            if ema is not None and was_ema_training and not ema._is_train_mode:
                ema.train()
            model.noise_scheduler = old_scheduler
        return result
