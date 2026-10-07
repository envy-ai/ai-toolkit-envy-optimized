"""Detached per-image diagnostics. No image decoding or additional model passes."""

import base64
import os
from types import SimpleNamespace

import torch


def recording_enabled(trainer):
    config = getattr(trainer, 'logging_config', None)
    enabled = getattr(config, 'record_training_examples', True) if config is not None else getattr(trainer, 'record_training_examples', False)
    return enabled and getattr(getattr(trainer, 'accelerator', None), 'is_main_process', True) and hasattr(trainer, 'logger')


def source_item(path, caption='', dataset_path=None, **attributes):
    return SimpleNamespace(path=str(path), caption=caption,
        dataset_config=SimpleNamespace(folder_path=str(dataset_path or os.path.dirname(str(path)))), **attributes)


def remember_source(trainer, item):
    if recording_enabled(trainer):
        trainer._loss_report_sources = getattr(trainer, '_loss_report_sources', []) + [item]


def log_image_losses(trainer, batch, raw, weighted, timesteps=None, noise=None, scope='per_image', captions=None, extras=None):
    """Observe existing objective vectors after backward, without changing reductions."""
    if not recording_enabled(trainer) or not getattr(batch, 'file_items', None):
        return
    n = len(batch.file_items)
    if raw.numel() != n or weighted.numel() != n:
        return  # Expanded/mixed batches are recorded by the shared-input fallback.
    weights = getattr(batch, 'loss_multiplier_list', [1.] * n)
    raw_cpu, weighted_cpu = raw.detach().float().cpu().flatten().tolist(), weighted.detach().float().cpu().flatten().tolist()
    times = timesteps.detach().float().cpu().flatten().tolist() if timesteps is not None else []
    stats = noise.detach().flatten(1) if noise is not None and noise.shape[0] == n else None
    means = stats.mean(1, dtype=torch.float32).cpu().tolist() if stats is not None else [None] * n
    stds = stats.std(1, correction=0).float().cpu().tolist() if stats is not None else [None] * n
    values = [[raw_cpu[i], weights[i], times[i] if len(times) == n else None, None, means[i], stds[i]] for i in range(n)]
    records = training_example_records(trainer, batch, values)
    for i, record in enumerate(records):
        record['weighted_loss'] = weighted_cpu[i]
        record['presentation']['loss_attribution'] = scope
        if captions is not None and len(captions) == n:
            record['metadata']['caption'] = captions[i]
        if extras:
            record['presentation'].update(extras)
        record['presentation']['model_batch_size'] = n
    if scope == 'pair_objective':
        import copy
        paired = []
        for i, (item, record) in enumerate(zip(batch.file_items, records)):
            record['presentation']['pair_index'] = i
            record['metadata']['input_role'] = 'preferred'
            if getattr(item, 'unconditional_path', None):
                partner = copy.deepcopy(record)
                partner['metadata'].update(path=os.path.abspath(item.unconditional_path),
                    dataset_path=os.path.dirname(os.path.abspath(item.unconditional_path)), input_role='rejected')
                paired.append(partner)
        records.extend(paired)
    trainer.logger.log_training_examples(records, getattr(trainer, '_loss_report_step_rng', None))


def remember_bank_source(trainer, latent):
    if recording_enabled(trainer):
        item = getattr(trainer, '_loss_report_bank_sources', {}).get(id(latent))
        if item is not None:
            remember_source(trainer, item)


def save_practice_source(trainer, latent, image, name, caption):
    if not recording_enabled(trainer):
        return
    directory = os.path.join(trainer.save_root, 'loss_report_practice')
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, name + '.png')
    image.save(filename)
    if not hasattr(trainer, '_loss_report_bank_sources'):
        trainer._loss_report_bank_sources = {}
    trainer._loss_report_bank_sources[id(latent)] = source_item(filename, caption, directory, role='practice_image')


def record_batch_inputs(trainer, batch, loss=None, sources=None):
    """Shared objectives associate real inputs with a step, without inventing image losses."""
    if not recording_enabled(trainer):
        return
    items = list(sources or [])
    seen = {str(item.path) for item in items}
    items.extend(item for item in (getattr(batch, 'file_items', []) or []) if str(item.path) not in seen)
    originals = list(items)
    for item in originals:
        partner = getattr(item, 'unconditional_path', None)
        if partner and str(partner) not in {str(source.path) for source in items}:
            items.append(source_item(partner, getattr(item, 'caption', ''), role='paired_image'))
    if not items:
        items = [source_item('prompt://training-step', '', role='prompt_objective')]
    proxy = SimpleNamespace(file_items=items, latents=getattr(batch, 'latents', None))
    values = [[None, getattr(item, 'loss_multiplier', 1.), None, None, None, None] for item in items]
    records = training_example_records(trainer, proxy, values)
    for item, record in zip(items, records):
        record['presentation']['loss_attribution'] = 'shared_step'
        record['metadata']['input_role'] = getattr(item, 'role', 'training_input')
        record['metadata']['source_kind'] = 'prompt' if str(item.path).startswith('prompt://') else 'image'
    rng = getattr(trainer, '_loss_report_step_rng', None)
    trainer.logger.log_training_examples(records, rng)


def record_step_inputs(trainer, batches, loss_dict):
    if not recording_enabled(trainer) or not loss_dict:
        return
    if 'loss' in loss_dict:
        trainer.logger.log({'loss/loss': loss_dict['loss']})
    if trainer.logger.pending_training_examples():
        return
    sources = getattr(trainer, '_loss_report_sources', [])
    if sources:
        record_batch_inputs(trainer, None, sources=sources)
    else:
        for batch in batches if isinstance(batches, list) else [batches]:
            record_batch_inputs(trainer, batch)


def capture_training_rng(device):
    """Observe RNG state without drawing numbers or changing the training sequence."""
    encode = lambda state: base64.b64encode(state.cpu().numpy().tobytes()).decode('ascii')
    result = {'torch_version': str(torch.__version__), 'cpu': encode(torch.get_rng_state()),
              'device': str(device), 'cuda': None}
    if torch.device(device).type == 'cuda':
        result['cuda'] = encode(torch.cuda.get_rng_state(device))
    return result


def training_example_records(trainer, batch, values):
    """values: CPU rows of loss, weight, time, correction RMS, noise mean/std."""
    if len(batch.file_items) != len(values):
        raise ValueError('Per-image loss logging requires one file item per prediction')
    model_config = getattr(trainer, 'model_config', None)
    helper = getattr(model_config, 'assistant_lora_path', None)
    helper_stamp = None
    if helper:
        try:
            stat = os.stat(helper)
            helper_stamp = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        except OSError:
            pass
    settings = {
        'model_arch': getattr(trainer.sd, 'arch', None),
        'model_name_or_path': getattr(model_config, 'name_or_path', None),
        'trainer': type(trainer).__name__,
        'teacher_cfg_scale': getattr(trainer, 'teacher_cfg_scale', None),
        'negative_prompt': getattr(trainer, 'teacher_negative_prompt', None),
        'objective': getattr(trainer, 'distillation_objective', getattr(trainer.train_config, 'loss_type', None)),
        'cfg_reference': getattr(trainer, 'cfg_reference', 'negative_prompt'),
        'helper_lora_path': helper, 'helper_file': helper_stamp,
        'dtype': trainer.train_config.dtype,
        'min_denoising_steps': getattr(trainer.train_config, 'min_denoising_steps', None),
        'max_denoising_steps': getattr(trainer.train_config, 'max_denoising_steps', None),
        'latent_shape': list(batch.latents.shape[1:]) if getattr(batch, 'latents', None) is not None else None,
        'flow_profile': trainer.flow_profile.identity() if hasattr(trainer, 'flow_profile') else None,
        'noise_sampling': ('torch.randn float32; timestep draw precedes noise draw' if hasattr(trainer, 'distillation_objective')
                           else 'Trainer-specific; RNG snapshots are captured before the training hook'),
    }
    records = []
    for item, row in zip(batch.file_items, values):
        loss, weight, timestep, correction, noise_mean, noise_std = row
        dataset = getattr(item, 'dataset_config', None)
        path = str(item.path) if str(item.path).startswith('prompt://') else os.path.abspath(item.path)
        dataset_path = os.path.abspath(getattr(dataset, 'folder_path', None)
                                       or getattr(dataset, 'dataset_path', None) or os.path.dirname(path))
        metadata = {**settings, 'path': path, 'dataset_path': dataset_path,
                    'caption': getattr(item, 'caption', '') or '',
                    'original_width': getattr(item, 'width', None),
                    'original_height': getattr(item, 'height', None),
                    'control_paths': getattr(item, 'control_path', None),
                    'unconditional_path': getattr(item, 'unconditional_path', None),
                    'kto_label': getattr(dataset, 'kto_label', None),
                    'text_embedding_path': item.get_text_embedding_path() if hasattr(item, 'get_text_embedding_path') else None}
        presentation = {key: getattr(item, key, None) for key in
                        ('scale_to_width', 'scale_to_height', 'crop_x', 'crop_y',
                         'crop_width', 'crop_height', 'flip_x', 'flip_y')}
        records.append({'metadata': metadata, 'presentation': presentation, 'loss': loss,
                        'weighted_loss': loss * weight if loss is not None else None, 'loss_weight': weight, 'timestep': timestep,
                        'teacher_correction_rms': correction, 'noise_mean': noise_mean, 'noise_std': noise_std})
    if len(records) != len(values):
        raise ValueError('Per-image loss logging requires one file item per prediction')
    return records
