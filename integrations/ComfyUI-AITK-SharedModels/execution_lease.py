"""A reversible whole-prompt wrapper. Cleanup failure retains exclusive ownership."""
import asyncio
import contextvars
import functools
import inspect
import os
import threading

from aitk_shared_models.client import Client
from aitk_shared_models.protocol import PROTOCOL_VERSION, SharedModelError, default_socket, normalize_device_uuid

from .compatibility import check_execution

_CURRENT = contextvars.ContextVar('aitk_shared_prompt', default=None)
_GUARD = None


async def complete_inflight(awaitable):
    """Drain a possibly granting acquire/cleanup even if the prompt is cancelled."""
    task = asyncio.ensure_future(awaitable)
    cancellation = None
    while True:
        try:
            return await asyncio.shield(task), cancellation
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc


class PromptLease:
    def __init__(self, client, device_uuid, cleanup, cancel_check=lambda: False, prepare=None):
        self.client = client
        self.device_uuid = normalize_device_uuid(device_uuid)
        self.cleanup = cleanup
        self.cancel_check = cancel_check
        self.prepare = prepare
        self.mutex = threading.Lock()

    async def run(self, request_id, execute):
        active = _CURRENT.get()
        if active is not None:
            if active['guard'] is not self:
                raise SharedModelError('Nested prompt crossed GPU lease providers')
            return await execute()
        while not self.mutex.acquire(blocking=False):
            if self.cancel_check():
                raise InterruptedError('Comfy prompt cancelled while waiting for execution lease')
            await asyncio.sleep(0.1)
        acquired = False
        transaction = {'guard': self, 'request_id': str(request_id), 'snapshots': {}}
        token = _CURRENT.set(transaction)
        try:
            if self.prepare is not None:
                self.prepare()
            while True:
                if self.cancel_check():
                    raise InterruptedError('Comfy prompt cancelled while waiting for trainer')
                response, cancellation = await complete_inflight(
                    asyncio.to_thread(self.client.request_gpu, self.device_uuid, str(request_id)))
                if response['granted']:
                    acquired = True
                    if cancellation is not None:
                        raise cancellation
                    if self.cancel_check():
                        raise InterruptedError('Comfy prompt cancelled immediately after lease acquisition')
                    break
                if cancellation is not None:
                    raise cancellation
                await asyncio.sleep(0.1)
            return await execute()
        finally:
            cleanup_cancellation = None
            try:
                if acquired:
                    # Shield finally cleanup from task cancellation. If cleanup fails,
                    # never acknowledge quiescence or let trainer reuse the GPU.
                    result = self.cleanup()
                    if inspect.isawaitable(result):
                        _, cleanup_cancellation = await complete_inflight(result)
                    self.client.ack_quiescent(self.device_uuid)
                else:
                    self.client.call('cancel_request', device_uuid=self.device_uuid, request_id=str(request_id))
                for metadata in transaction['snapshots'].values():
                    self.client.release_adapter(metadata['snapshot_id'])
                if cleanup_cancellation is not None:
                    raise cleanup_cancellation
            finally:
                _CURRENT.reset(token)
                self.mutex.release()


def current_prompt():
    active = _CURRENT.get()
    if active is None:
        raise SharedModelError('Shared MODEL requires the enabled whole-prompt execution lease')
    return active


def cleanup_comfy():
    import gc
    import sys
    import torch
    import comfy.model_management as management
    from .shared_patcher import release_all_shared
    device = management.get_torch_device()
    torch.cuda.synchronize(device)
    management.unload_all_models()
    release_all_shared()
    management.reset_cast_buffers()
    # Pedro lookahead pools keep GPU cast-buffer references; cached CPU models stay.
    for name, module in list(sys.modules.items()):
        if name.endswith('.hunyuan_image_3.lookahead'):
            for streams in getattr(module, '_POOLS', {}).values():
                for stream in streams:
                    stream.synchronize()
            getattr(module, '_POOLS', {}).clear()
    gc.collect()
    torch.cuda.synchronize(device)
    clear_workspaces = getattr(torch._C, '_cuda_clearCublasWorkspaces', None)
    if callable(clear_workspaces):
        clear_workspaces()
    management.soft_empty_cache(force=True)
    if any(loaded.model.loaded_size() > 0 for loaded in management.current_loaded_models):
        raise SharedModelError('Comfy global model bookkeeping still has GPU-resident weights')
    # A surviving GPU graph/cast buffer is not quiescent. Fail closed until its
    # owner can be identified; never use a timeout as proof of release.
    if torch.cuda.memory_allocated(device) != 0:
        from aitk_shared_models.diagnostics import live_cuda_tensor_summary
        raise SharedModelError(f'Comfy retains {torch.cuda.memory_allocated(device)} CUDA allocator bytes after cleanup; tensors={live_cuda_tensor_summary()}')


def install(executor_class, guard, execution_module=None):
    original = check_execution(executor_class, execution_module)
    if getattr(original, '_aitk_shared_lease', False):
        if original._aitk_guard is not guard:
            raise SharedModelError('Prompt execution already has a different shared GPU lease')
        return original
    @functools.wraps(original)
    async def wrapped(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
        return await guard.run(prompt_id, lambda: original(self, prompt, prompt_id,
            {} if extra_data is None else extra_data, [] if execute_outputs is None else execute_outputs))
    wrapped._aitk_shared_lease = True
    wrapped._aitk_guard = guard
    wrapped._aitk_original = original
    executor_class.execute_async = wrapped
    return wrapped


def uninstall(executor_class):
    wrapped = executor_class.execute_async
    if getattr(wrapped, '_aitk_shared_lease', False):
        executor_class.execute_async = wrapped._aitk_original


def enable_from_environment():
    global _GUARD
    if os.environ.get('AITK_SHARED_MODELS_ENABLED') != '1':
        return
    if _GUARD is not None:
        return
    device_uuid = os.environ.get('AITK_SHARED_MODELS_DEVICE_UUID')
    if not device_uuid:
        raise SharedModelError('AITK_SHARED_MODELS_DEVICE_UUID is mandatory')
    import execution
    import comfy.model_management as management
    import torch
    actual = normalize_device_uuid(str(getattr(torch.cuda.get_device_properties(management.get_torch_device()), 'uuid', '')))
    device_uuid = normalize_device_uuid(device_uuid)
    if actual != device_uuid:
        raise SharedModelError(f'Comfy lease device UUID must match its local device ({actual})')
    client = Client(os.environ.get('AITK_SHARED_MODELS_SOCKET', default_socket()), role='comfy')
    guard = PromptLease(client, device_uuid, cleanup_comfy, management.processing_interrupted,
                        prepare=lambda: management.interrupt_current_processing(False))
    install(execution.PromptExecutor, guard, execution)
    _GUARD = guard


def capabilities():
    return {'protocol_version': PROTOCOL_VERSION, 'enabled': _GUARD is not None,
            'provider': 'aitk_shared_models', 'variant': 'instruct', 'quantization': 'int8_tensorwise',
            'whole_prompt_lease': _GUARD is not None, 'device_uuid': None if _GUARD is None else _GUARD.device_uuid,
            'socket': None if _GUARD is None else _GUARD.client.socket.getpeername(),
            'dynamic_patcher': False, 'runtime_qualified': False}
