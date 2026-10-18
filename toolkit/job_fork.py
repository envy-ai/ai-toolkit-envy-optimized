"""Inspect and independently copy one complete generation of a UI training job.

This module intentionally avoids importing trainers/models: the UI invokes it
with its configured Python interpreter, and trainers reuse the resume guard.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


def _process(config):
    processes = config.get('config', {}).get('process', [])
    if len(processes) != 1:
        raise ValueError('Fork requires exactly one training process')
    return processes[0]


def resume_signature(config):
    process = _process(config)
    train = process.get('train', {})
    sliderspace_settings = process.get('sliderspace', {})
    if process.get('type') == 'sliderspace':
        from dataclasses import asdict
        from toolkit.sliderspace import SliderSpaceConfig
        sliderspace_settings = asdict(SliderSpaceConfig(**sliderspace_settings))
    sliderspace = {key: value for key, value in sliderspace_settings.items()
                   if key != 'loss_weight' and not key.startswith('preview_')}
    # Loss, LR, schedule, datasets and sample edits are allowed. Model/network
    # identity and the trainable parameter layout must continue to match.
    model = process.get('model', {})
    network = process.get('network')
    signature = {'type': 'diffusion_trainer' if process.get('type') == 'ui_trainer' else process.get('type'),
            'model': model if process.get('type') == 'sliderspace' else {key: value for key, value in model.items()
                      if key.endswith('name_or_path') or key.endswith('_path') and key != 'inference_lora_path'
                      or key.startswith('is_') or key in ('arch', 'model_kwargs', 'v_pred')},
            'network': {'type': network.get('type', 'lora'),
                        'linear': network.get('rank', network.get('linear', 4)),
                        'linear_alpha': network.get('linear_alpha', network.get('alpha', 1.0)),
                        **{key: value for key, value in (network or {}).items()
                        if key in ('conv', 'conv_alpha',
                                   'lokr_full_rank', 'lokr_factor', 'transformer_only',
                                   'network_kwargs', 'ignore_if_contains', 'only_if_contains',
                                   'old_lokr_format', 'loha_dora', 'all_layers', 'lorm',
                                   'split_multistage_loras', 'save_magnitude_less_lora')}}
                       if network is not None else None,
            'sliderspace': sliderspace,
            'train': {key: train.get(key, default) for key, default in
                      (('optimizer', 'adamw'), ('train_unet', True), ('train_text_encoder', False),
                       ('train_refiner', False), ('merge_network_on_save', False),
                       ('learnable_snr_gos', False), ('do_paramiter_swapping', False))},
            **{key: process.get(key) for key in ('adapter', 'embedding', 'decorator')}}
    if signature['train']['optimizer'] == 'prodigyopt':
        signature['train']['optimizer'] = 'prodigy'
    return signature


def _stamp(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Checkpoint file is missing or is a symlink: {path}')
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def _manifest_stamp(path):
    return _stamp(path)[:2]


def _metadata(path):
    if path.is_dir():
        import yaml
        _stamp(path / 'aitk_meta.yaml')
        with (path / 'aitk_meta.yaml').open() as handle:
            data = yaml.safe_load(handle)
    else:
        from safetensors import safe_open
        _stamp(path)
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            data = handle.metadata() or {}
    info = data.get('training_info')
    if isinstance(info, str):
        info = json.loads(info)
    if not isinstance(info, dict) or type(info.get('step')) is not int or info['step'] < 0:
        raise ValueError(f'Checkpoint has no valid training step metadata: {path}')
    return data, info['step']


def _optimizer(path):
    import torch
    state = torch.load(str(path), map_location='cpu', weights_only=True)
    if (not isinstance(state, dict) or not isinstance(state.get('state'), dict)
            or not isinstance(state.get('param_groups'), list) or not state['param_groups']
            or any(not isinstance(group, dict) or not isinstance(group.get('params'), list)
                   for group in state['param_groups'])):
        raise ValueError(f'Invalid optimizer resume state: {path}')
    # Internal per-parameter counters need not equal the trainer step after
    # prior resumes or conditional updates. The trainer restores these as-is.


def _tree_files(path):
    if path.is_symlink():
        raise ValueError(f'Checkpoint directories must not be symlinks: {path}')
    for child in sorted(path.rglob('*')):
        if child.is_symlink():
            raise ValueError(f'Checkpoint contains a symlink: {child}')
        if child.is_file():
            yield child


def inspect_job(source_folder, source_name, source_config):
    """Return latest complete resume generation; raise on incomplete state."""
    root = Path(source_folder)
    process = _process(source_config)
    if process.get('name', source_name) != source_name:
        raise ValueError('Fork does not support a process name different from its job name')
    if (any(process.get(key) for key in ('adapter', 'embedding', 'decorator'))
            or process.get('train', {}).get('train_refiner')
            or process.get('train', {}).get('merge_network_on_save')):
        raise ValueError('Fork does not support combined adapter/embedding/decorator, refiner, or merged-network checkpoints')
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Source training folder is missing or is a symlink')
    files = []
    warnings = []
    marker = None
    record = None
    if process.get('type') == 'sliderspace':
        candidates = []
        for path in (root / 'sliderspace_state').glob('checkpoint_*.json'):
            value = json.loads(path.read_text())
            step = value.get('step')
            if type(step) is not int or step < 0 or path.name not in ('checkpoint_final.json', f'checkpoint_{step:09d}.json'):
                raise ValueError(f'Invalid SliderSpace checkpoint marker: {path}')
            candidates.append((step, path.stat().st_mtime_ns, path, value))
        if not candidates:
            raise ValueError('No complete SliderSpace bank checkpoint exists')
        step, _, marker, record = max(candidates, key=lambda value: value[:2])
        if not isinstance(record.get('signature'), str) or Path(record['signature']).name != record['signature']:
            raise ValueError('Invalid SliderSpace checkpoint signature')
        if (root / 'sliderspace_state').is_symlink():
            raise ValueError('SliderSpace checkpoint directory must not be a symlink')
        checkpoint = root / 'sliderspace_state' / f'bank_{step:09d}.safetensors'
        optimizer = checkpoint.with_suffix('.optimizer.pt')
        exports = record.get('exports')
        directions = process.get('sliderspace', {}).get('num_directions', 4)
        if (not isinstance(exports, list) or len(exports) != directions or len(set(exports)) != len(exports)
                or any(not isinstance(name, str) or Path(name).name != name
                       or not name.startswith(source_name + '_direction_') for name in exports)):
            raise ValueError('Invalid SliderSpace checkpoint export manifest')
        files = [checkpoint, optimizer, *(root / name for name in exports)]
        if record.get('files') is not None:
            actual = {path.name: _manifest_stamp(path) for path in files}
            if actual != record['files']:
                raise ValueError('SliderSpace published checkpoint files changed or are incomplete')
        metadata, metadata_step = _metadata(checkpoint)
        if metadata_step != step or metadata.get('ss_sliderspace_signature') != record.get('signature'):
            raise ValueError('SliderSpace bank metadata does not match its checkpoint marker')
        # The discovery bank fixes PCA directions and generated training inputs.
        # Copy only the active signature/component cache identified by the bank.
        discovery = root / 'sliderspace_discovery' / record['signature']
        components = metadata.get('ss_sliderspace_components')
        if components is not None:
            if isinstance(components, str):
                components = json.loads(components)
            digest = hashlib.sha256(json.dumps(components, sort_keys=True, default=str).encode()).hexdigest()
            discovery = discovery / digest
        manifest = discovery / 'manifest.json'
        for directory in [root / 'sliderspace_discovery', root / 'sliderspace_discovery' / record['signature'], discovery]:
            if directory.is_symlink():
                raise ValueError('SliderSpace discovery cache must not contain symlink directories')
        if not manifest.is_file():
            raise ValueError('SliderSpace discovery cache is missing; cannot preserve its training bank')
        cache = json.loads(manifest.read_text())
        if cache.get('signature') != record['signature'] or (components is not None and cache.get('components') != components):
            raise ValueError('SliderSpace discovery cache identity does not match its bank')
        if (not isinstance(cache.get('entries'), list) or not isinstance(cache.get('files'), dict)
                or set(cache['files']) != {f'{index:06d}.{extension}' for index in range(len(cache['entries']))
                                           for extension in ('png', 'safetensors')}):
            raise ValueError('SliderSpace discovery cache input manifest is incomplete')
        for name, stamp in cache.get('files', {}).items():
            if Path(name).name != name or _manifest_stamp(discovery / name) != stamp:
                raise ValueError('SliderSpace discovery cache files changed or are incomplete')
            files.append(discovery / name)
        features = discovery / 'features.safetensors'
        if _manifest_stamp(features) != cache.get('feature_file'):
            raise ValueError('SliderSpace discovery features changed or are incomplete')
        files.extend([features, manifest, marker])
        layout = 'sliderspace'
    else:
        # Match canonical trainer names, excluding direction exports, adapters,
        # merged exports, temporary files, and unrelated prefix matches.
        pattern = re.compile(re.escape(source_name) + r'(?:_(\d{9}))?(?:\.safetensors)?$')
        candidates = []
        for path in root.iterdir():
            match = pattern.fullmatch(path.name)
            if not match or (not path.is_dir() and path.suffix != '.safetensors'):
                continue
            candidates.append((path.stat().st_ctime_ns, path, match[1]))
        if not candidates:
            raise ValueError('No supported resumable checkpoint with training metadata exists')
        # The ordinary trainer chooses the most recently changed checkpoint,
        # including a restored lower-step final checkpoint. Match its ctime
        # semantics; the fork contains only that selected generation.
        _, checkpoint, filename_step = max(candidates, key=lambda value: value[0])
        _, step = _metadata(checkpoint)
        if filename_step is not None and int(filename_step) != step:
            raise ValueError(f'Checkpoint filename and metadata steps disagree: {checkpoint}')
        if process.get('network') and checkpoint.is_dir():
            raise ValueError('Network training requires a safetensors checkpoint')
        files = list(_tree_files(checkpoint)) if checkpoint.is_dir() else [checkpoint]
        optimizer = root / 'optimizer.pt'
        optimizer_stamp = _stamp(optimizer)
        if optimizer_stamp[1] < max(_stamp(path)[1] for path in files):
            warnings.append('The source optimizer file predates the latest weights. This fork restores the same saved optimizer that the source job would use when resuming.')
        files.append(optimizer)
        if process.get('train', {}).get('learnable_snr_gos'):
            files.append(root / 'learnable_snr.json')
        layout = 'diffusers' if checkpoint.is_dir() else 'safetensors'
    _optimizer(optimizer)
    if process.get('type', '').startswith('fizgig'):
        cache = root / 'fizgig_practice_cache'
        if cache.is_dir():
            files.extend(path for path in _tree_files(cache)
                         if re.fullmatch(r'(?:prompt|anchor)_\d{6}\.safetensors', path.name))
    return {'step': step, 'checkpoint': str(checkpoint.relative_to(root)), 'layout': layout,
            'warnings': warnings,
            'optimizer': str(optimizer.relative_to(root)),
            'files': {str(path.relative_to(root)): _stamp(path) for path in files},
            'marker': str(marker.relative_to(root)) if marker else None, 'record': record}


def fork_job(source_folder, target_folder, source_name, target_name, source_config, target_config):
    """Copy a verified generation into a new folder, removing it on failure."""
    if resume_signature(source_config) != resume_signature(target_config):
        raise ValueError('Fork must preserve the source model, network, optimizer and trainable parameter layout')
    if _process(target_config).get('name', target_name) != target_name:
        raise ValueError('Fork process name must match its new job name')
    if _process(target_config).get('train', {}).get('start_step') is not None:
        raise ValueError('Fork must resume its checkpoint metadata step; remove train.start_step')
    snapshot = inspect_job(source_folder, source_name, source_config)
    source, target = Path(source_folder), Path(target_folder)
    target.mkdir(parents=False, exist_ok=False)
    try:
        mapped = {}
        for name in snapshot['files']:
            parts = Path(name).parts
            is_checkpoint = parts[0] == Path(snapshot['checkpoint']).parts[0]
            is_export = snapshot['record'] is not None and name in snapshot['record']['exports']
            renamed = target_name + parts[0][len(source_name):] if (is_checkpoint and snapshot['layout'] != 'sliderspace') or is_export else parts[0]
            destination_name = str(Path(renamed, *parts[1:]))
            mapped[name] = destination_name
            destination = target / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
        if snapshot['marker']:
            record = dict(snapshot['record'])
            record['exports'] = [mapped[name] for name in record['exports']]
            if 'files' in record:
                record['files'] = {Path(mapped[name]).name: _manifest_stamp(target / mapped[name])
                                   for name in snapshot['files']
                                   if Path(name).name in snapshot['record']['files']}
            (target / mapped[snapshot['marker']]).write_text(json.dumps(record, indent=2))
        # No symlinks/hardlinks: optimizer and every mutable cache belong to
        # the fork. Validate the source again after the complete copy.
        if inspect_job(source_folder, source_name, source_config) != snapshot:
            raise ValueError('Source checkpoint changed while forking; pause it before retrying')
        guard = {'source_name': source_name, 'minimum_step': snapshot['step'],
                 'signature': resume_signature(target_config)}
        (target / '.fork_resume.json').write_text(json.dumps(guard, indent=2))
        result = inspect_job(target, target_name, target_config)
        return {key: result[key] for key in ('step', 'checkpoint', 'layout', 'warnings')}
    except BaseException:
        shutil.rmtree(target)
        raise


def validate_fork_resume(save_root, config):
    """Keep forked runs strict, including later resumes after new saves."""
    marker = Path(save_root) / '.fork_resume.json'
    if not marker.exists():
        return None
    guard = json.loads(marker.read_text())
    if resume_signature(config) != guard['signature']:
        raise ValueError('Fork resume model/network/optimizer layout changed; clone into a fresh job instead')
    if _process(config).get('train', {}).get('start_step') is not None:
        raise ValueError('Fork resume requires the checkpoint metadata step, without train.start_step')
    snapshot = inspect_job(save_root, config['config']['name'], config)
    if snapshot['step'] < guard['minimum_step']:
        raise ValueError('Fork resume checkpoint predates its source snapshot')
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'copy'])
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    try:
        if args.action == 'inspect':
            result = inspect_job(payload['source_folder'], payload['source_name'], payload['source_config'])
            result = {key: result[key] for key in ('step', 'checkpoint', 'layout', 'warnings')}
        else:
            result = fork_job(**payload)
        print(json.dumps(result))
    except Exception as error:
        import traceback
        print(json.dumps({'error': str(error), 'stack': traceback.format_exc()}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
