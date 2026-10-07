"""Small CPU fixtures only. Never import Comfy's CUDA-initializing plugin stack."""
import asyncio
import copy
import gc
import importlib
import json
import os
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
import ast
import dataclasses
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages' / 'aitk_shared_models'))
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from aitk_shared_models.arena import Arena, build_checkpoint, source_identity
from aitk_shared_models.broker import Broker
from aitk_shared_models.client import Client, SharedTensorSource
from aitk_shared_models.protocol import SharedModelError, validate_manifest


def fixture(path):
    marker = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 4, 'num_experts': 2}).encode()
    values = {'bank.weight': ('I8', [2, 3, 4], bytes(range(24))),
              'bank.weight_scale': ('F32', [2, 3], struct.pack('<6f', *range(1, 7))),
              'bank.comfy_quant': ('U8', [len(marker)], marker),
              'dense.weight': ('F32', [2, 2], struct.pack('<4f', 1, 2, 3, 4))}
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in values.items():
        offset = len(payload)
        payload.extend(raw)
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, len(payload)]}
    header['__metadata__'] = {'hunyuan_variant': 'instruct'}
    raw = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


class ArenaTests(unittest.TestCase):
    def test_device_uuid_normalization(self):
        from aitk_shared_models.protocol import normalize_device_uuid
        value = '8b7a72e6-1373-b399-af57-5419747c572a'
        self.assertEqual(normalize_device_uuid('GPU-' + value.upper()), value)
        self.assertEqual(normalize_device_uuid('gpu-' + value), value)
        self.assertEqual(normalize_device_uuid(value), value)
        for malformed in ('fixture', 'MIG-' + value, '', None):
            with self.assertRaises(SharedModelError):
                normalize_device_uuid(malformed)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.source = Path(self.directory.name) / 'instruct.safetensors'
        fixture(self.source)
        self.manifest, fd = build_checkpoint(self.source, {'model_type': 'instruct'}, reserve_bytes=0)
        self.arena = Arena(self.manifest, fd)

    def tearDown(self):
        self.arena = None
        gc.collect()
        self.directory.cleanup()

    def test_alias_parameters_immutable_and_lifetime(self):
        import torch
        from aitk_shared_models.tensors import owner_of, SharedTensor
        bank = self.arena.tensor('bank.weight')
        expert = bank[1]
        parameter = torch.nn.Parameter(expert, requires_grad=False)
        self.assertIsInstance(parameter, SharedTensor)
        self.assertEqual(expert.data_ptr() - bank.data_ptr(), 12)
        self.assertIs(owner_of(parameter.data), owner_of(bank))
        self.assertEqual(parameter.tolist(), [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        for operation in (lambda: bank.add_(1), lambda: bank.clone(), lambda: bank.pin_memory(),
                          lambda: bank.float(), lambda: bank.numpy(), lambda: bank.storage(),
                          lambda: torch.add(bank, 1, out=bank)):
            with self.assertRaises(SharedModelError):
                operation()
        storage_info = bank.untyped_storage()
        with self.assertRaises(SharedModelError):
            storage_info[0] = 0
        del bank, expert
        gc.collect()
        with self.assertRaises(SharedModelError):
            self.arena.close()
        self.assertEqual(parameter[0].tolist(), [12, 13, 14, 15])
        del parameter
        gc.collect()
        with self.assertRaises(SharedModelError):
            self.arena.close()
        del storage_info
        gc.collect()
        self.arena.close()

    def test_descriptor_validation(self):
        for update in ({'offset': -1}, {'length': 1}, {'shape': [999999999], 'strides': [1]}, {'dtype': 'UNKNOWN'}):
            manifest = copy.deepcopy(self.manifest)
            manifest['tensors']['bank.weight'].update(update)
            with self.assertRaises(SharedModelError):
                validate_manifest(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest['protocol_version'] += 1
        with self.assertRaises(SharedModelError):
            validate_manifest(manifest)

    def test_readonly_fd_and_mapping(self):
        with self.assertRaises(OSError):
            os.pwrite(self.arena.fd, b'x', 0)
        with self.assertRaises(TypeError):
            self.arena.mapping[0] = 1

    def test_identity_symlink_and_source_reader(self):
        link = Path(self.directory.name) / 'link.safetensors'
        link.symlink_to(self.source)
        self.assertEqual(source_identity(self.source, {}, 'instruct', 'int8_tensorwise'), source_identity(link, {}, 'instruct', 'int8_tensorwise'))
        reader = SharedTensorSource(self.arena)
        self.assertEqual(reader.shape('bank.weight'), [2, 3, 4])
        with self.assertRaises(SharedModelError):
            reader.tensor('bank.weight', 2)
        self.assertEqual(reader.tensor('bank.weight', 1).data_ptr() - reader.tensor('bank.weight').data_ptr(), 12)

    def test_shared_module_cpu_apply_retains_master(self):
        import torch
        from aitk_shared_models.ownership import capture_masters, restore_masters, verify_masters
        model = torch.nn.Linear(2, 2, bias=False, device='meta')
        model.weight = torch.nn.Parameter(self.arena.tensor('dense.weight'), requires_grad=False)
        pointer = model.weight.data_ptr()
        capture_masters(model, strict=True)
        model.to('cpu')
        restore_masters(model)
        verify_masters(model)
        self.assertEqual(model.weight.data_ptr(), pointer)
        with self.assertRaises(SharedModelError):
            model.to(dtype=torch.float16)
        self.assertTrue(torch.equal(model(torch.ones(1, 2)), torch.tensor([[3., 7.]])))

    def test_shared_int8_importer_preserves_expert_scales_and_bias(self):
        import torch
        from aitk_shared_models.tensors import owner_of
        # Execute the real importer function with its backend class supplied as a
        # tiny fixture; avoid importing unrelated diffusion/CUDA plugin registries.
        source = ast.parse((ROOT / 'toolkit/util/comfy_quant_import.py').read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('parse_comfy_quant_blob', '_to_ostris', 'import_comfy_quantized_layers')]
        class FixtureOstris(torch.nn.Linear):
            pass
        namespace = {'torch': torch, 'json': json, 'Dict': dict, 'Tuple': tuple,
                     'OstrisLinear': FixtureOstris, 'get_ostris_quantizer': lambda _: object()}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'real-comfy-quant-import-fixture', 'exec'), namespace)
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(4, 3, bias=True, device='meta')
        weight = self.arena.tensor('bank.weight', 1)
        scales = self.arena.tensor('bank.weight_scale', 1)
        # Bias fixture uses a shared float scale-row view of the same length.
        local = {'proj.comfy_quant': self.arena.tensor('bank.comfy_quant'),
                 'proj.weight': weight, 'proj.weight_scale': scales, 'proj.bias': scales}
        remaining, count = namespace['import_comfy_quantized_layers'](model, local, strict_shared=True)
        self.assertEqual((remaining, count), ({}, 1))
        self.assertEqual(model.proj.cr8_qdata.data_ptr(), weight.data_ptr())
        self.assertEqual(model.proj.cr8_scales.data_ptr(), scales.data_ptr())
        self.assertEqual(model.proj.bias.data_ptr(), scales.data_ptr())
        self.assertIs(owner_of(model.proj.bias), owner_of(weight))

    def test_real_comfy_kitchen_cpu_wrapper_detach_and_scales(self):
        import torch
        # Only the installed CK base module: imports stdlib + torch, with no
        # comfy_kitchen package/backend registration or Comfy model-management imports.
        path = Path('/home/bart/.conda/envs/comfyui/lib/python3.12/site-packages/comfy_kitchen/tensor/base.py')
        if not path.is_file():
            self.skipTest('Installed CK base source unavailable')
        spec = importlib.util.spec_from_file_location('_aitk_fixture_ck_base', path)
        base = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = base
        spec.loader.exec_module(base)
        class Layout:
            pass
        base.register_layout_class('FixtureINT8', Layout)
        @dataclasses.dataclass(frozen=True)
        class Params(base.BaseLayoutParams):
            convrot: bool = True
            convrot_groupsize: int = 4
        scales = self.arena.tensor('bank.weight_scale')
        options = Params(scale=scales, orig_dtype=torch.float32, orig_shape=(2, 3, 4))
        comfy = types.ModuleType('comfy')
        quant = types.ModuleType('comfy.quant_ops')
        quant.QuantizedTensor = base.QuantizedTensor
        with mock.patch.dict(sys.modules, {'comfy': comfy, 'comfy.quant_ops': quant}):
            ops = integration_module('shared_ops')
            value = ops.shared_quantized_tensor(self.arena.tensor('bank.weight'), 'FixtureINT8', options)
            parameter = torch.nn.Parameter(value, requires_grad=False)
            detached = parameter.detach()
            self.assertEqual(detached._qdata.data_ptr(), value._qdata.data_ptr())
            self.assertEqual(detached._params.scale.data_ptr(), scales.data_ptr())
            self.assertTrue(detached._params.convrot)
            with self.assertRaises(SharedModelError):
                parameter.clone()
            # Comfy executes under inference mode, but whole-prompt cleanup runs
            # outside it. Reusing the cached model must preserve normal alias
            # metadata; Torch 2.11 rejects mixed wrappers at the next detach.
            from aitk_shared_models.ownership import capture_masters, restore_masters, verify_masters
            with torch.inference_mode():
                codes = self.arena.tensor('bank.weight')
                scales = self.arena.tensor('bank.weight_scale')
                options = Params(scale=scales, orig_dtype=torch.float32, orig_shape=(2, 3, 4))
                model = torch.nn.Module()
                model.weight = torch.nn.Parameter(ops.shared_quantized_tensor(codes, 'FixtureINT8', options), requires_grad=False)
                capture_masters(model)
                pointers = (codes.data_ptr(), scales.data_ptr())
                self.assertFalse(torch.is_inference(model.weight))
                self.assertFalse(torch.is_inference(codes))
            for _ in range(3):
                restore_masters(model)
                with torch.inference_mode():
                    backup = model.weight.to('cpu')
                    model.weight = torch.nn.Parameter(backup, requires_grad=False)
                    verify_masters(model)
                    self.assertFalse(torch.is_inference(model.weight))
                    self.assertEqual((model.weight._qdata.data_ptr(), model.weight._params.scale.data_ptr()), pointers)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_conditioning_cast_is_bounded(self):
        import torch
        from aitk_shared_models.tensors import bounded_cpu_cast, owner_of
        value = self.arena.tensor('dense.weight')
        converted = bounded_cpu_cast(value, torch.float64)
        self.assertEqual(converted.dtype, torch.float64)
        self.assertIsNone(owner_of(converted))
        self.assertEqual(owner_of(value).cpu_temporary_peak_bytes, 32)
        with self.assertRaises(SharedModelError):
            bounded_cpu_cast(value, torch.float64, limit_bytes=1)
        ops = integration_module('shared_ops')
        module = types.SimpleNamespace(weight=value, bias=None, weight_function=[], bias_function=[])
        weight, _ = ops.staged_weight(module, torch.ones(1, 2, dtype=torch.float64))
        self.assertTrue(torch.equal(torch.nn.functional.linear(torch.ones(1, 2, dtype=torch.float64), weight), torch.tensor([[3., 7.]], dtype=torch.float64)))


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.socket = str(Path(self.directory.name) / 'broker.sock')
        self.broker = Broker(self.socket)
        self.thread = threading.Thread(target=self.broker.serve_forever, daemon=True)
        self.thread.start()
        for _ in range(200):
            if Path(self.socket).exists():
                break
            time.sleep(.01)
        self.trainer = Client(self.socket, 'trainer')
        self.comfy = Client(self.socket, 'comfy')
        self.observer = Client(self.socket)

    def tearDown(self):
        for client in (self.trainer, self.comfy, self.observer):
            client.close()
        self.broker.close()
        self.thread.join(timeout=2)
        self.directory.cleanup()

    def test_cold_start_ready_sharing_and_importer_exit(self):
        path = Path(self.directory.name) / 'instruct.safetensors'
        fixture(path)
        identity, arena = self.trainer.load_checkpoint(path, {'model_type': 'instruct'}, reserve_bytes=0)
        attached = self.comfy.open_store(identity)
        self.assertEqual(os.fstat(arena.fd).st_ino, os.fstat(attached.fd).st_ino)
        expert = attached.tensor('bank.weight', 1)
        self.comfy.close()
        self.assertEqual(expert[0].tolist(), [12, 13, 14, 15])
        with Client(self.socket, 'comfy') as restarted:
            again = restarted.open_store(identity)
            self.assertEqual(os.fstat(arena.fd).st_ino, os.fstat(again.fd).st_ino)
        with self.assertRaises(SharedModelError):
            self.observer.open_store(identity, {'variant': 'base'})

    def test_builder_exclusion_failure_and_observer_rejection(self):
        first, _ = self.trainer.call('begin_store', identity='cold')
        second, _ = self.comfy.call('begin_store', identity='cold')
        self.assertEqual(first['state'], 'BUILD')
        self.assertEqual(second['state'], 'BUILDING')
        self.trainer.call('fail_store', identity='cold', error='fixture partial load')
        result, _ = self.comfy.call('begin_store', identity='cold')
        self.assertEqual(result['state'], 'ERROR')
        with self.assertRaises(SharedModelError):
            self.observer.call('begin_store', identity='new')

    def test_exclusive_epoch_and_stale_ack(self):
        first = self.trainer.acquire_gpu('GPU-00000000-0000-0000-0000-000000000001', 'train')
        self.assertFalse(self.comfy.request_gpu('00000000-0000-0000-0000-000000000001', 'prompt')['granted'])
        self.assertTrue(self.trainer.poll_gpu('00000000-0000-0000-0000-000000000001')['yield_requested'])
        with self.assertRaises(SharedModelError):
            self.comfy.ack_quiescent('00000000-0000-0000-0000-000000000001', first)
        self.trainer.ack_quiescent('00000000-0000-0000-0000-000000000001')
        second = self.comfy.acquire_gpu('00000000-0000-0000-0000-000000000001', 'prompt')
        self.assertGreater(second, first)
        with self.assertRaises(SharedModelError):
            self.trainer.ack_quiescent('00000000-0000-0000-0000-000000000001', first)
        self.comfy.ack_quiescent('00000000-0000-0000-0000-000000000001')

    def test_timeout_cancel_and_disconnect_never_grant(self):
        self.trainer.acquire_gpu('00000000-0000-0000-0000-000000000001', 'train')
        with self.assertRaises(TimeoutError):
            self.comfy.acquire_gpu('00000000-0000-0000-0000-000000000001', 'timeout', timeout=.01)
        with self.assertRaises(InterruptedError):
            self.comfy.acquire_gpu('00000000-0000-0000-0000-000000000001', 'cancel', cancel_check=lambda: True)
        self.assertEqual(self.observer.status()['devices']['00000000-0000-0000-0000-000000000001']['queue'], [])
        self.trainer.close()
        for _ in range(200):
            if self.observer.status()['devices']['00000000-0000-0000-0000-000000000001']['fault']:
                break
            time.sleep(.01)
        with self.assertRaises(SharedModelError):
            self.comfy.request_gpu('00000000-0000-0000-0000-000000000001', 'after-disconnect')

    def test_snapshot_latest_refs_and_hash(self):
        from aitk_shared_models.snapshot import publish_snapshot, verify_snapshot
        path = Path(self.directory.name) / 'instruct.safetensors'
        fixture(path)
        identity, arena = self.trainer.load_checkpoint(path, {'model_type': 'instruct'}, reserve_bytes=0)
        metadata = {'store_identity': identity, 'variant': 'instruct', 'rank': 2, 'alpha': 2.,
                    'preset': 'attention', 'base_content_digest': arena.manifest['content_digest']}
        generations = []
        for step in range(4):
            snapshot = publish_snapshot(Path(self.directory.name) / 'snapshots', step, metadata,
                                        lambda file: Path(file).write_bytes(b'fixture-small-adapter'))
            self.trainer.publish_adapter(snapshot)
            generations.append(snapshot)
        pinned = self.comfy.open_adapter(generations[0]['snapshot_id'], identity)
        latest = self.comfy.open_adapter('latest', identity)
        self.assertEqual(latest['step'], 3)
        verify_snapshot(pinned, arena.manifest)
        removed, _ = self.trainer.call('prune_adapters', store_identity=identity, keep=1)
        self.assertNotIn(pinned, removed['removed'])
        self.assertEqual(len(removed['removed']), 2)


def integration_module(name):
    package = '_aitk_shared_test_extension'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT / 'integrations' / 'ComfyUI-AITK-SharedModels')]
        sys.modules[package] = module
    return importlib.import_module(package + '.' + name)


class ExecutionTests(BrokerTests):
    def test_cancel_during_acquire_drains_grant(self):
        lease = integration_module('execution_lease')
        entered, release = threading.Event(), threading.Event()
        original = self.comfy.request_gpu
        def delayed(*args):
            response = original(*args)
            entered.set()
            release.wait(timeout=2)
            return response
        cleaned = []
        guard = lease.PromptLease(self.comfy, '00000000-0000-0000-0000-000000000001', lambda: cleaned.append('cleanup'))
        async def execute():
            self.fail('Cancelled prompt must never execute nodes')
        async def scenario():
            with mock.patch.object(self.comfy, 'request_gpu', delayed):
                task = asyncio.create_task(guard.run('cancel-acquire', execute))
                await asyncio.to_thread(entered.wait, 2)
                task.cancel()
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        asyncio.run(scenario())
        self.assertEqual(cleaned, ['cleanup'])
        self.assertIsNone(self.observer.status()['devices']['00000000-0000-0000-0000-000000000001']['owner'])

    def test_cancel_during_async_cleanup_still_acknowledges(self):
        lease = integration_module('execution_lease')
        entered = asyncio.Event()
        finished = []
        async def cleanup():
            entered.set()
            await asyncio.sleep(.03)
            finished.append(True)
        guard = lease.PromptLease(self.comfy, '00000000-0000-0000-0000-000000000001', cleanup)
        async def execute():
            return 'image'
        async def scenario():
            task = asyncio.create_task(guard.run('cancel-cleanup', execute))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(scenario())
        self.assertEqual(finished, [True])
        self.assertIsNone(self.observer.status()['devices']['00000000-0000-0000-0000-000000000001']['owner'])

    def test_whole_prompt_failure_finally_reentrant_and_wrapper(self):
        lease = integration_module('execution_lease')
        cleaned = []
        guard = lease.PromptLease(self.comfy, '00000000-0000-0000-0000-000000000001', lambda: cleaned.append('cleanup'))
        class Executor:
            async def execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
                self.called = True
                self.assert_owner = guard.client.poll_gpu('00000000-0000-0000-0000-000000000001')['owner']['role']
                async def nested():
                    return 4
                self.nested = await guard.run(prompt_id, nested)
                raise ValueError('decode fixture failed')
        executor = Executor()
        original = Executor.execute_async
        lease.install(Executor, guard)
        lease.install(Executor, guard)
        with self.assertRaises(ValueError):
            asyncio.run(executor.execute_async({}, 'prompt'))
        self.assertTrue(executor.called)
        self.assertEqual(executor.assert_owner, 'comfy')
        self.assertEqual(executor.nested, 4)
        self.assertEqual(cleaned, ['cleanup'])
        self.assertIsNone(self.observer.status()['devices']['00000000-0000-0000-0000-000000000001']['owner'])
        lease.uninstall(Executor)
        self.assertIs(Executor.execute_async, original)

    def test_cleanup_failure_retains_owner(self):
        lease = integration_module('execution_lease')
        def failed_cleanup():
            raise RuntimeError('outstanding CUDA fixture')
        guard = lease.PromptLease(self.comfy, '00000000-0000-0000-0000-000000000001', failed_cleanup)
        async def execute():
            return 'image'
        with self.assertRaises(RuntimeError):
            asyncio.run(guard.run('prompt', execute))
        self.assertEqual(self.observer.status()['devices']['00000000-0000-0000-0000-000000000001']['owner']['role'], 'comfy')
        self.assertFalse(self.trainer.request_gpu('00000000-0000-0000-0000-000000000001', 'train')['granted'])


class TrainerAndWorkflowTests(unittest.TestCase):
    def test_shared_patcher_gpu_copy_avoids_comfy_empty_like(self):
        import torch
        from collections import namedtuple
        tree = ast.parse((ROOT / 'integrations/ComfyUI-AITK-SharedModels/shared_patcher.py').read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SharedModelPatcher')
        node = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'patch_weight_to_device')
        calls = []
        class Weight:
            dtype = torch.float32
            def numel(self): return 4
            def to(self, **kwargs):
                calls.append(kwargs)
                return self if torch.device(kwargs['device']).type == 'cpu' else torch.ones(2,2)
        weight = Weight()
        def setter(result, **kwargs):
            self.assertFalse(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_inference(result))
            return result
        comfy = types.SimpleNamespace(
            model_patcher=types.SimpleNamespace(get_key_weight=lambda *args: (weight,setter,None)),
            model_management=types.SimpleNamespace(lora_compute_dtype=lambda device: torch.float32,
                cast_to_device=lambda *args, **kwargs: self.fail('Comfy copying path must not be called')),
            lora=types.SimpleNamespace(calculate_weight=lambda patches,value,key: value + 2),
            utils=types.SimpleNamespace(string_to_seed=lambda key: 1))
        namespace = {'torch':torch,'comfy':comfy,'SharedModelError':SharedModelError,
            'WeightBackup':namedtuple('WeightBackup',('weight','inplace_update'))}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'shared_patcher.py','exec'),namespace)
        patcher = types.SimpleNamespace(weight_inplace_update=False,patches={'proj.weight':object()},
            backup={}, model=object(),offload_device=torch.device('cpu'))
        with torch.inference_mode():
            result = namespace['patch_weight_to_device'](patcher,'proj.weight',device_to=torch.device('cuda:0'))
        self.assertTrue(torch.equal(result,torch.full((2,2),3.)))
        self.assertIs(patcher.backup['proj.weight'].weight,weight)
        self.assertEqual(calls[1], {'device':torch.device('cuda:0'),'dtype':torch.float32,'copy':True})
        self.assertFalse(torch.cuda.is_initialized())

    def test_shared_startup_waits_for_lease_without_out_of_band_unload(self):
        tree = ast.parse((ROOT / 'jobs/process/BaseSDTrainProcess.py').read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BaseSDTrainProcess')
        node = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'hook_before_model_load')
        calls = []
        namespace = {'print_acc': lambda *args: None,
            'ComfyApiClient': lambda **kwargs: types.SimpleNamespace(release_vram=lambda: calls.append('release'))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'BaseSDTrainProcess.py', 'exec'), namespace)
        config = types.SimpleNamespace(provider='aitk_shared_models', api_url='fixture')
        process = types.SimpleNamespace(_get_comfy_config_for_startup_release=lambda: config,
            model_config=types.SimpleNamespace(shared_weights={'enabled': True}),
            accelerator=types.SimpleNamespace(is_main_process=True))
        namespace['hook_before_model_load'](process)
        self.assertEqual(calls, [])
        config.provider = 'ordinary'
        namespace['hook_before_model_load'](process)
        self.assertEqual(calls, ['release'])

    def test_optimizer_plain_quantization_cache_is_parked_and_restored(self):
        import torch
        from toolkit.memory_management.shared_training import SharedTrainingCoordinator
        class DeviceFixture(torch.Tensor):
            @staticmethod
            def __new__(cls, device):
                value = torch.Tensor._make_subclass(cls, torch.zeros(256), require_grad=False)
                value.reported_device = torch.device(device)
                return value
            @property
            def device(self):
                return self.reported_device
            def to(self, device):
                return DeviceFixture(device)
        optimizer = types.SimpleNamespace(state={}, name2qmap={'dynamic': DeviceFixture('cuda:0')})
        coordinator = SharedTrainingCoordinator.__new__(SharedTrainingCoordinator)
        coordinator.process = types.SimpleNamespace(optimizer=optimizer, ema=None)
        scheduler = types.SimpleNamespace(timesteps=DeviceFixture('cuda:0'))
        coordinator.holder = types.SimpleNamespace(model=torch.nn.Module(), pipeline=None, noise_scheduler=scheduler)
        coordinator.private_modules = []
        coordinator._park_persistent_roots()
        self.assertEqual(optimizer.name2qmap['dynamic'].device.type, 'cpu')
        self.assertEqual(scheduler.timesteps.device.type, 'cpu')
        coordinator._restore_persistent_roots()
        self.assertEqual(optimizer.name2qmap['dynamic'].device, torch.device('cuda:0'))
        self.assertEqual(scheduler.timesteps.device, torch.device('cuda:0'))
        self.assertFalse(torch.cuda.is_initialized())

    def test_ui_shared_render_request_defers_without_clearing_accumulation(self):
        def method(path, class_name, name):
            tree = ast.parse((ROOT / path).read_text())
            klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            node = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)
            namespace = {'print_acc': lambda *args: None}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
            return namespace[name]
        maybe_sample = method('extensions_built_in/sd_trainer/DiffusionTrainer.py', 'DiffusionTrainer', 'maybe_sample')
        sample_helper = method('jobs/process/BaseSDTrainProcess.py', 'BaseSDTrainProcess', 'sample_with_optimizer_state_offload')
        calls = []
        process = types.SimpleNamespace(is_ui_trainer=True, should_sample=lambda: True,
            update_db_key=lambda *args: calls.append(('db', args)), step_num=7,
            sd=types.SimpleNamespace(shared_coordinator=object()),
            sample_config=types.SimpleNamespace(comfy=types.SimpleNamespace(enabled=True, provider='aitk_shared_models')),
            optimizer=types.SimpleNamespace(zero_grad=lambda: calls.append('zero_grad')),
            _shared_in_training_loop=True, _shared_sample_pending=None,
            sample=lambda *args, **kwargs: calls.append('render'))
        process.sample_with_optimizer_state_offload = lambda *args, **kwargs: sample_helper(process, *args, **kwargs)
        maybe_sample(process)
        self.assertEqual(process._shared_sample_pending, (7, False))
        self.assertEqual(calls, [('db', ('sample_now', 0))])

    def test_ui_ordinary_render_request_keeps_existing_path(self):
        tree = ast.parse((ROOT / 'extensions_built_in/sd_trainer/DiffusionTrainer.py').read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DiffusionTrainer')
        node = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'maybe_sample')
        calls = []
        namespace = {'print_acc': lambda *args: None, 'flush': lambda: calls.append('flush')}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'DiffusionTrainer.py', 'exec'), namespace)
        process = types.SimpleNamespace(is_ui_trainer=True, should_sample=lambda: True,
            update_db_key=lambda *args: calls.append('db'), step_num=7, progress_bar=None,
            sd=types.SimpleNamespace(), optimizer=types.SimpleNamespace(zero_grad=lambda: calls.append('zero_grad')),
            train_config=types.SimpleNamespace(free_u=False, unload_text_encoder=False),
            sample=lambda *args: calls.append('render'), ensure_params_requires_grad=lambda: calls.append('restore'))
        namespace['maybe_sample'](process)
        self.assertEqual(calls, ['db', 'zero_grad', 'render', 'restore', 'flush'])

    def test_tiny_training_park_resume_preserves_update_rng_optimizer(self):
        import torch
        from toolkit.memory_management.manager import MemoryManager
        from toolkit.memory_management.shared_training import SharedTrainingCoordinator
        model = torch.nn.Linear(2, 2, bias=False)
        model.requires_grad_(False)
        manager = MemoryManager(model, torch.device('cpu'))
        model._memory_manager = manager
        adapter = torch.nn.Linear(2, 2, bias=False)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
        sample = torch.ones(1, 2)
        optimizer.zero_grad(set_to_none=True)
        # Two accumulated backwards, one completed optimizer update.
        for _ in range(2):
            adapter(sample).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        rng = torch.get_rng_state()
        before = {name: value.detach().clone() for name, value in adapter.state_dict().items()}
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        calls = []
        fake_client = types.SimpleNamespace(ack_quiescent=lambda *a: calls.append('ack'),
            acquire_gpu=lambda *a, **kw: calls.append('acquire'))
        process = types.SimpleNamespace(optimizer=optimizer, ema=None,
            _iter_comfy_auxiliary_offload_modules=lambda: [adapter],
            _should_cancel_comfy_prompt_wait=lambda: False, _validation_cache={'embedding': sample})
        holder = types.SimpleNamespace(model=model, vae=None, text_encoder=[], pipeline=None, device_torch=torch.device('cpu'))
        coordinator = SharedTrainingCoordinator.__new__(SharedTrainingCoordinator)
        coordinator.state, coordinator.holder, coordinator.process = 'ACTIVE', holder, process
        coordinator.client, coordinator.device_uuid, coordinator.request_id = fake_client, 'fixture', 'train'
        # Simulate only CUDA synchronization/accounting APIs, never initialize CUDA.
        with mock.patch.object(torch.cuda, 'synchronize'), mock.patch.object(torch.cuda, 'empty_cache'), mock.patch.object(torch.cuda, 'memory_allocated', return_value=0):
            coordinator.suspend()
            self.assertEqual(manager.external_gpu_state, 'SUSPENDED')
            coordinator.resume()
        self.assertEqual(calls, ['ack', 'acquire'])
        self.assertEqual(manager.external_gpu_state, 'ACTIVE')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in adapter.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))
        for key, state in optimizer.state_dict()['state'].items():
            for field, value in state.items():
                self.assertTrue(torch.equal(value, optimizer_before['state'][key][field]))
        coordinator.state = 'ACTIVE'
        with mock.patch.object(torch.cuda, 'synchronize'), mock.patch.object(torch.cuda, 'empty_cache'), mock.patch.object(torch.cuda, 'memory_allocated', return_value=64):
            with self.assertRaisesRegex(RuntimeError, 'retains 64'):
                coordinator.suspend()
        self.assertEqual(calls, ['ack', 'acquire'])

    def test_adapter_export_key_mapping_and_workflow_text_edit(self):
        import torch
        from toolkit.lora_key_format import internal_key_to_peft_key
        from safetensors.torch import save_file, load_file
        mapping = integration_module('lora_mapping')
        prefix = internal_key_to_peft_key('diffusion_model$$model$$layers$$0$$self_attn$$qkv_proj')
        factors = {prefix + '.lora_A.weight': torch.ones(2, 4), prefix + '.lora_B.weight': torch.ones(6, 2), prefix + '.alpha': torch.tensor(2.)}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'adapter.safetensors')
            save_file(factors, path)
            saved = load_file(path)
            model = types.SimpleNamespace(named_parameters=lambda: [(prefix + '.weight', torch.empty(6, 4))])
            self.assertEqual(mapping.canonical_lora_map(model, saved, {'preset': 'attention', 'rank': 2, 'alpha': 2.}), {prefix: prefix + '.weight'})
            with self.assertRaises(SharedModelError):
                mapping.canonical_lora_map(model, saved, {'preset': 'attention', 'rank': 3, 'alpha': 2.})
        from toolkit.shared_comfy_sample import DEFAULT_WORKFLOW, build_workflow
        template = json.loads(DEFAULT_WORKFLOW.read_text())
        config = types.SimpleNamespace(vae='vae', clip_vision='vision', text_encoder='', sampler='euler', scheduler='simple')
        gen = types.SimpleNamespace(prompt='edit', width=512, height=512, seed=17, num_inference_steps=4, guidance_scale=3., network_multiplier=.75)
        coordinator = types.SimpleNamespace(store_identity='store', client=types.SimpleNamespace(socket=types.SimpleNamespace(getpeername=lambda: '/test/broker.sock')))
        edited = build_workflow(template, coordinator, 'immutable-generation', config, gen, ['one.png', 'two.png'], 'samples/test')
        self.assertEqual(edited['2']['inputs']['snapshot_id'], 'immutable-generation')
        self.assertEqual(edited['6']['inputs']['images.image_2'], ['aitk_reference_2', 0])
        self.assertEqual(edited['6']['inputs']['vit_strength'], 1.0)
        self.assertEqual(edited['6']['inputs']['latent_strength'], 1.0)
        self.assertTrue(edited['6']['inputs']['reference_vit_padding'])
        text = build_workflow(template, coordinator, 'immutable-generation', config, gen, [], 'samples/test')
        self.assertEqual(text['6']['class_type'], 'HunyuanImage3TextEncode')
        self.assertNotIn('4', text)
        self.assertNotIn('images.image_1', text['6']['inputs'])
        for image_setting in ('vit_strength', 'latent_strength', 'reference_vit_padding'):
            self.assertNotIn(image_setting, text['6']['inputs'])
        self.assertEqual(template['1']['inputs']['store_identity'], 'REPLACE_STORE_ID')

    def test_snapshot_fingerprint_unresolved_model_requests_fresh_execution(self):
        tree = ast.parse((ROOT / 'integrations/ComfyUI-AITK-SharedModels/nodes.py').read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AITKSharedLoRASnapshot')
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'fingerprint_inputs')
        fixture = ast.ClassDef(name='SnapshotFixture', bases=[], keywords=[], body=[method], decorator_list=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[])), 'nodes.py', 'exec'), namespace)
        snapshot = namespace['SnapshotFixture']
        snapshot.resolve = mock.Mock(return_value={'snapshot_id': 'generation', 'content_hash': 'digest'})
        first = snapshot.fingerprint_inputs(snapshot_id='latest')
        second = snapshot.fingerprint_inputs(snapshot_id='latest')
        self.assertNotEqual(first, second)
        snapshot.resolve.assert_not_called()
        self.assertEqual(snapshot.fingerprint_inputs(object(), 'generation', .5), ('generation', 'digest', .5))

    def test_targeted_prompt_cancellation(self):
        from toolkit.comfy_sample import ComfyApiClient
        client = ComfyApiClient()
        calls = []
        def request(method, path, payload=None):
            calls.append((method, path, payload))
            return {'queue_running': [[0, 'own-prompt']]} if method == 'GET' else {}
        client._request_json = request
        client.cancel_shared_prompt('own-prompt')
        self.assertIn(('POST', '/api/interrupt', {'prompt_id': 'own-prompt'}), calls)
        self.assertNotIn(('POST', '/api/interrupt', {}), calls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
