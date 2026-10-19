"""Render saved job samples without constructing or running a trainer.

The public validation/checkpoint helpers use only the standard library so the
UI can reject invalid work before importing the model stack.
"""

import copy
import json
from pathlib import Path
import re
import struct


def validate_render_config(config):
    processes = config.get('config', {}).get('process', [])
    if len(processes) != 1:
        raise ValueError('Rendering saved samples requires exactly one training process.')
    process = processes[0]
    if process.get('type') not in ('diffusion_trainer', 'ui_trainer', 'sd_trainer'):
        raise ValueError('Saved sample rendering currently supports ordinary diffusion trainers; specialized trainers require their training preview state.')
    if not process.get('network') or process.get('train', {}).get('merge_network_on_save'):
        raise ValueError('Saved sample rendering requires an unmerged LoRA checkpoint.')
    if any(process.get(key) for key in ('adapter', 'embedding', 'decorator')):
        raise ValueError('Saved sample rendering cannot reconstruct trained adapters, embeddings or decorators.')
    sample = process.get('sample', {})
    comfy = sample.get('comfy', {})
    if not comfy.get('enabled'):
        raise ValueError('Saved sample rendering currently requires the ComfyUI workflow renderer. Internal sampling needs architecture-specific network attachment and memory setup owned by the trainer.')
    if comfy.get('provider', 'ordinary') != 'ordinary':
        raise ValueError('Saved sample rendering requires the ComfyUI workflow provider; shared models require a live trainer.')
    if not sample.get('samples') and not sample.get('prompts'):
        raise ValueError('The job has no configured samples.')
    samples = sample.get('samples', [{'prompt': prompt} for prompt in sample.get('prompts', [])])
    if any(not isinstance(item, dict) or not isinstance(item.get('prompt'), str) for item in samples):
        raise ValueError('Every configured sample must contain a text prompt (an empty string is allowed).')
    return process


def read_checkpoint_metadata(checkpoint):
    checkpoint = Path(checkpoint)
    with checkpoint.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f'Incomplete safetensors checkpoint: {checkpoint.name}')
        length = struct.unpack('<Q', prefix)[0]
        size = checkpoint.stat().st_size
        if length < 2 or length > 100_000_000 or length + 8 > size:
            raise ValueError(f'Invalid safetensors header: {checkpoint.name}')
        header = json.loads(stream.read(length))
    tensors = [value for key, value in header.items() if key != '__metadata__']
    if not tensors:
        raise ValueError(f'Checkpoint contains no tensors: {checkpoint.name}')
    for tensor in tensors:
        offsets = tensor.get('data_offsets', [])
        if (len(offsets) != 2 or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - length - 8):
            raise ValueError(f'Incomplete tensor data in checkpoint: {checkpoint.name}')
    metadata = header.get('__metadata__', {})
    training = metadata.get('training_info', '{}')
    if isinstance(training, str):
        training = json.loads(training)
    if not isinstance(training, dict):
        raise ValueError(f'Invalid training metadata in checkpoint: {checkpoint.name}')
    return training


def checkpoint_step(checkpoint):
    match = re.fullmatch(r'.+_(\d{9})\.safetensors', Path(checkpoint).name)
    metadata = read_checkpoint_metadata(checkpoint)
    if not match and 'step' not in metadata:
        raise ValueError(f'Checkpoint has no saved training step in its metadata: {Path(checkpoint).name}')
    step = metadata.get('step', int(match[1]) if match else 0)
    if type(step) is not int or step < 0:
        raise ValueError(f'Invalid checkpoint training step: {step!r}')
    if match and metadata.get('step') is not None and step != int(match[1]):
        raise ValueError(f'Checkpoint filename and metadata disagree on training step: {Path(checkpoint).name}')
    return step


def select_checkpoint(job_folder: Path, job_name: str) -> Path:
    folder = Path(job_folder).resolve()
    pattern = re.compile(re.escape(job_name) + r'(?:_\d{9})?\.safetensors')
    candidates = [item for item in folder.iterdir() if pattern.fullmatch(item.name)]
    if not candidates:
        raise FileNotFoundError(f'No saved LoRA checkpoint found for {job_name}. Save a checkpoint before rendering samples.')
    # Training's normal final save has no step suffix. Modification time also
    # covers that file without inspecting older, potentially damaged saves.
    selected = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    if selected.is_symlink() or not selected.is_file() or selected.resolve().parent != folder:
        raise ValueError(f'Checkpoint must be a regular file in the job folder: {selected.name}')
    checkpoint_step(selected)  # Never silently fall back from a corrupt latest save.
    return selected


def render_saved_samples(config, job_folder, checkpoint, update_status):
    """Reuse the exact training sample builder and workflow renderer, on disk weights.

    No trainer constructor, model loader, optimizer, dataset or training loop is
    entered. The small host exposes only the state these two shared methods use.
    """
    process = validate_render_config(config)
    from types import SimpleNamespace
    import threading
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
    from toolkit.config_modules import ModelConfig, SampleConfig, TrainConfig
    from toolkit.comfy_sample import ComfyApiClient
    from toolkit.prompt_utils import inject_trigger_into_prompt

    class SavedCheckpointRenderer(BaseSDTrainProcess):
        def __init__(self):
            self.job = SimpleNamespace(name=config['config']['name'])
            self.save_root = str(Path(job_folder).resolve())
            self.train_config = TrainConfig(**copy.deepcopy(process.get('train', {})))
            self.model_config = ModelConfig(**{**copy.deepcopy(process['model']), 'dtype': self.train_config.dtype})
            self.sample_config = SampleConfig(**copy.deepcopy(process['sample']))
            self.sample_config.comfy.run_in_background = False
            self.first_sample_config = self.sample_config
            self.accelerator = SimpleNamespace(is_main_process=True)
            self.sd = SimpleNamespace(inject_trigger_into_prompt=inject_trigger_into_prompt)
            self.adapter_config = self.adapter = self.embedding = self.ema = None
            self.trigger_word = process.get('trigger_word')
            self.logger = None
            self.step_num = checkpoint_step(checkpoint)
            self.dataset_configs = [SimpleNamespace(**item) for item in process.get('datasets', [])]
            self._comfy_prompt_wait_cancel_event = threading.Event()

        def _refresh_live_sample_config(self):
            # This request uses the immutable configuration accepted by the API.
            return

        def _save_current_network_for_comfy(self, step=None):
            return str(checkpoint)

        def _get_comfy_lora_display_filename(self, step=None):
            return Path(checkpoint).name

        def _cleanup_legacy_comfy_sample_loras(self, sample_folder):
            # An inference request never removes training artifacts.
            return

        def _run_with_models_offloaded_for_comfy(self, func):
            return func(offload_models=False)

        def _wait_for_comfy_vram_release(self, client, free_bytes_before_comfy=None):
            # There are no local models to restore and no local CUDA context.
            return True

        def _update_comfy_sample_status(self, message):
            update_status(message)

        def _log_comfy_sample_generation_time(self, elapsed_seconds, completed_samples, total_samples, step=None):
            update_status(f'Rendered {completed_samples}/{total_samples} samples in {elapsed_seconds:.1f}s')

    renderer = SavedCheckpointRenderer()
    client = ComfyApiClient(renderer.sample_config.comfy.api_url, timeout=renderer.sample_config.comfy.timeout)
    queue = client._request_json('GET', '/api/queue')
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise RuntimeError('ComfyUI has active or pending work. Wait for its queue to finish before rendering saved samples.')
    Path(job_folder, 'samples').mkdir(exist_ok=True)
    renderer.sample(step=renderer.step_num)
