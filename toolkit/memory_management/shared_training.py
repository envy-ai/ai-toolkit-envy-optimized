"""Cooperative trainer handoff at explicit successful optimizer boundaries."""
import gc
import os
import uuid
from pathlib import Path

from toolkit.shared_models import shared_package


class SharedTrainingCoordinator:
    def __init__(self, holder, config, process):
        import torch
        if holder.device_torch.type != 'cuda':
            raise ValueError('Shared trainer requires one local CUDA device')
        properties = torch.cuda.get_device_properties(holder.device_torch)
        normalize_uuid = shared_package('protocol').normalize_device_uuid
        actual = normalize_uuid(str(getattr(properties, 'uuid', '')))
        if actual != normalize_uuid(config['device_uuid']):
            raise ValueError(f'Shared lease device_uuid must match local torch device UUID ({actual})')
        self.holder, self.config = holder, config
        self.device_uuid = actual
        self.client = shared_package('client').Client(config.get('socket'), role='trainer')
        self.request_id = 'training-' + str(uuid.uuid4())
        self.state = 'WAITING'
        self.private_slots = []
        self.private_modules = []
        self.optimizer_slots = []
        self.process = process
        self.store_identity = None
        self.snapshot_id = None
        self.client.acquire_gpu(self.device_uuid, self.request_id,
                                cancel_check=process._should_cancel_comfy_prompt_wait)
        self.state = 'ACTIVE'

    def bind_process(self, process):
        self.process = process

    def _park_tensor_slots(self):
        import torch
        process = self.process
        modules = [getattr(self.holder, 'vae', None)] + list(getattr(self.holder, 'text_encoder', []) or [])
        modules += list(process._iter_comfy_auxiliary_offload_modules() or [])
        seen = set()
        self.private_slots = []
        self.private_modules = []
        for module in modules:
            if module is None:
                continue
            children = list(module.modules())
            get_all = getattr(module, 'get_all_modules', None)
            if callable(get_all):
                for adapter in get_all():
                    children.extend(adapter.modules())
            for child in children:
                if id(child) in seen:
                    continue
                seen.add(id(child))
                self.private_modules.append(child)
                for kind, slots in (('parameter', child._parameters), ('buffer', child._buffers)):
                    for name, value in list(slots.items()):
                        if value is None or value.is_meta or value.device.type == 'cpu':
                            continue
                        if kind == 'parameter' and value.grad is not None:
                            raise RuntimeError('GPU handoff requires cleared adapter gradients')
                        self.private_slots.append((slots, name, kind, value.device))
                        if kind == 'parameter':
                            value.data = value.detach().to('cpu')
                        else:
                            slots[name] = value.to('cpu')
        # Preserve every optimizer tensor's own device (including CPU step counters),
        # rather than moving an entire 8-bit optimizer back to an assumed device.
        self.optimizer_slots = []
        def park(container):
            if isinstance(container, dict):
                items = list(container.items())
            elif isinstance(container, list):
                items = list(enumerate(container))
            elif isinstance(container, tuple):
                if any(isinstance(v, torch.Tensor) and v.device.type != 'cpu' for v in container):
                    raise RuntimeError('Unsupported tuple GPU optimizer state; handoff refused')
                items = list(enumerate(container))
            else:
                return
            for key, value in items:
                if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
                    self.optimizer_slots.append((container, key, value.device))
                    container[key] = value.to('cpu')
                else:
                    park(value)
        park(process.optimizer.state if process.optimizer is not None else {})
        self.ema_slots = []
        if process.ema is not None:
            for attr in ('shadow_params', 'collected_params'):
                values = getattr(process.ema, attr, None)
                if values is not None:
                    for index, value in enumerate(values):
                        if value.device.type != 'cpu':
                            self.ema_slots.append((values, index, value.device))
                            values[index] = value.to('cpu')

    def suspend(self):
        import torch
        if self.state != 'ACTIVE':
            raise RuntimeError(f'Cannot suspend trainer in {self.state}')
        self.state = 'SUSPENDING'
        try:
            self.holder.model._memory_manager.suspend_for_external_gpu()
            self._park_tensor_slots()
            self._park_persistent_roots()
            # Deterministic backend tables are rebuilt lazily after resume. Keeping
            # tables keyed to CUDA with CPU values would give the next step wrong devices.
            import sys
            quant = sys.modules.get('toolkit.util.convrot_quant')
            if quant is not None:
                for name in ('_hadamard_cache', '_edges_cache'):
                    cache = getattr(quant, name, {})
                    for key, value in list(cache.items()):
                        if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                            del cache[key]
                value = None  # Do not retain the last removed CUDA cache entry locally.
            torch.cuda.synchronize(self.holder.device_torch)
            if self.holder.device_torch.type == 'cuda':
                clear_workspaces = getattr(torch._C, '_cuda_clearCublasWorkspaces', None)
                if callable(clear_workspaces):
                    clear_workspaces()
            gc.collect()
            torch.cuda.empty_cache()
            retained = torch.cuda.memory_allocated(self.holder.device_torch)
            if retained != 0:
                details = shared_package('diagnostics').live_cuda_tensor_summary()
                raise RuntimeError(f'Trainer retains {retained} CUDA allocator bytes after parking; GPU lease remains owned; tensors={details}')
            self.client.ack_quiescent(self.device_uuid)
            self.state = 'SUSPENDED'
        except BaseException:
            # No acknowledgment after partial parking. Failure cannot grant Comfy GPU.
            self.state = 'FAILED'
            raise

    def resume(self):
        if self.state != 'SUSPENDED':
            raise RuntimeError(f'Cannot resume trainer in {self.state}')
        self.client.acquire_gpu(self.device_uuid, self.request_id,
                                cancel_check=self.process._should_cancel_comfy_prompt_wait)
        self.state = 'RESUMING'
        try:
            self.holder.model._memory_manager.resume_from_external_gpu()
            for slots, name, kind, device in self.private_slots:
                if kind == 'parameter':
                    slots[name].data = slots[name].data.to(device)
                else:
                    slots[name] = slots[name].to(device)
            for slots, name, device in self.optimizer_slots + self.ema_slots:
                slots[name] = slots[name].to(device)
            self._restore_persistent_roots()
            self.private_slots.clear()
            self.optimizer_slots.clear()
            self.ema_slots.clear()
            self.state = 'ACTIVE'
        except BaseException:
            self.state = 'FAILED'
            raise

    def _park_persistent_roots(self):
        """Validation/latent/embedding caches stay private, with exact device restore."""
        import torch
        self.persistent_devices = []
        def park(value, path, devices):
            if isinstance(value, torch.Tensor):
                if value.device.type == 'cuda':
                    if isinstance(value, torch.nn.Parameter):
                        raise RuntimeError('Unmanaged persistent CUDA parameter prevents shared handoff')
                    devices[path] = value.device
                    return value.to('cpu')
                return value
            if isinstance(value, dict):
                for key, item in list(value.items()):
                    value[key] = park(item, path + (key,), devices)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    value[index] = park(item, path + (index,), devices)
            elif isinstance(value, tuple):
                return tuple(park(item, path + (index,), devices) for index, item in enumerate(value))
            elif type(value).__name__ in ('AdvancedPromptEmbeds', 'PromptEmbeds'):
                park(vars(value), path + ('__aitk_attrs__',), devices)
            return value
        owners = [self.process, self.holder, getattr(self.holder, 'pipeline', None)]
        owners += [getattr(self.holder, 'noise_scheduler', None),
                   getattr(getattr(self.holder, 'pipeline', None), 'scheduler', None),
                   getattr(self.process, 'lr_scheduler', None)]
        # bitsandbytes keeps GPU quantization maps on optimizer.name2qmap as well
        # as in state. EMA implementations can likewise own plain tensor caches.
        owners += [getattr(self.process, 'optimizer', None), getattr(self.process, 'ema', None)]
        owners += list(self.holder.model.modules()) + self.private_modules
        for owner in owners:
            if owner is None:
                continue
            for name, value in list(vars(owner).items()):
                devices = {}
                moved = park(value, (), devices)
                if devices:
                    setattr(owner, name, moved)
                    self.persistent_devices.append((owner, name, devices))

    def _restore_persistent_roots(self):
        def restore(value, path, devices):
            if path in devices:
                return value.to(devices[path])
            if isinstance(value, dict):
                for key, item in list(value.items()):
                    value[key] = restore(item, path + (key,), devices)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    value[index] = restore(item, path + (index,), devices)
            elif isinstance(value, tuple):
                return tuple(restore(item, path + (index,), devices) for index, item in enumerate(value))
            elif type(value).__name__ in ('AdvancedPromptEmbeds', 'PromptEmbeds'):
                restore(vars(value), path + ('__aitk_attrs__',), devices)
            return value
        for owner, name, devices in self.persistent_devices:
            setattr(owner, name, restore(getattr(owner, name), (), devices))
        self.persistent_devices.clear()

    def publish_adapter(self, step):
        process = self.process
        if process.network is None:
            raise ValueError('Shared generation requires a factorized training LoRA')
        adapter = self.holder.get_adapter_metadata(process.network)
        manifest = self.holder.model._shared_arena.manifest
        directory = self.config.get('snapshot_directory') or str(Path(self.client.socket.getpeername()).parent / 'snapshots')
        from toolkit.train_tools import get_torch_dtype
        def export(path):
            multiplier = process.network.multiplier
            process.network.multiplier = 1.0
            try:
                process.network.save_weights(path, dtype=get_torch_dtype(process.save_config.dtype),
                    metadata={'hunyuan_training': __import__('json').dumps(adapter)})
            finally:
                process.network.multiplier = multiplier
        metadata = shared_package('snapshot').publish_snapshot(
            directory, step,
            {'store_identity': self.store_identity, 'variant': manifest['variant'],
             'base_content_digest': manifest['content_digest'], 'preset': adapter['preset'],
             'rank': int(adapter['rank']), 'alpha': float(adapter['alpha'])},
            export)
        self.snapshot_id = self.client.publish_adapter(metadata)
        removed, _ = self.client.call('prune_adapters', store_identity=self.store_identity, keep=3)
        for item in removed['removed']:
            Path(item['path']).unlink(missing_ok=True)
            Path(item['path']).with_suffix('.json').unlink(missing_ok=True)
        return self.snapshot_id

    def safe_boundary(self, step):
        if self.state != 'ACTIVE':
            raise RuntimeError('Trainer reached a boundary without its GPU lease')
        if not self.client.poll_gpu(self.device_uuid)['yield_requested']:
            return
        self.publish_adapter(step)
        self.process._update_comfy_sample_status('Waiting for ComfyUI')
        self.suspend()
        # Reacquisition queues behind all existing requests. It advances no step,
        # scheduler, dataset iterator or RNG; cancellation remains polled.
        self.resume()
        self.process._update_comfy_sample_status('Training')

    def close(self):
        if self.state == 'ACTIVE' and self.process is not None:
            self.suspend()
        self.client.release_client()
