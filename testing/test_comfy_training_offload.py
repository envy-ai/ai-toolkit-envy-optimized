import unittest
from types import SimpleNamespace
from unittest import mock

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.comfy_sample import (
    DEFAULT_COMFY_WORKFLOW_PATH,
    DEFAULT_COMFY_QWEN_IMAGE_2_WORKFLOW_PATH,
    DEFAULT_COMFY_QWEN_IMAGE_2_BATCH_WORKFLOW_PATH,
)


class FakeTensor:
    def __init__(self, device):
        self.device = device


class FakeNetwork:
    def __init__(self, device, children=None):
        self.tensor = FakeTensor(device)
        self.children = children or []
        self.to_calls = []

    def parameters(self, recurse=True):
        return iter([self.tensor])

    def buffers(self, recurse=True):
        return iter([])

    def to(self, device):
        self.tensor.device = device
        self.to_calls.append(device)
        return self

    def get_all_modules(self):
        return self.children


class ComfyTrainingOffloadTests(unittest.TestCase):
    def test_cleanup_handles_failed_qwen_model_loading(self):
        from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model
        from toolkit.config_modules import ModelConfig

        for partially_loaded in (False, True):
            with self.subTest(partially_loaded=partially_loaded):
                model = QwenImage2Model(
                    'cpu', ModelConfig(
                        arch='qwen_image_2', name_or_path='Comfy-Org/Qwen-Image-2.1',
                        assistant_lora_path='/missing/assistant.safetensors',
                    ), dtype='float32',
                )
                model.print_and_status_update = mock.Mock()
                with self.assertRaisesRegex(FileNotFoundError, 'training adapter not found'):
                    model.load_model()
                if partially_loaded:
                    model.model = mock.Mock()

                process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
                process.sd = model
                process.optimizer = None
                process._has_pending_comfy_background_tasks = mock.Mock(return_value=False)
                with (
                    mock.patch('jobs.process.BaseSDTrainProcess.print_acc') as log,
                    mock.patch('jobs.process.BaseSDTrainProcess.flush'),
                ):
                    process._release_training_memory_before_comfy_wait()

                log.assert_not_called()
                self.assertIsNone(process.sd)
                if partially_loaded:
                    model.model.to.assert_called_once_with('cpu')

    def test_qwen_image_2_migrates_legacy_krea_workflow(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.model_config = SimpleNamespace(arch="qwen_image_2")
        comfy_config = SimpleNamespace(workflow_path=DEFAULT_COMFY_WORKFLOW_PATH)

        self.assertEqual(
            process._get_comfy_workflow_path(comfy_config),
            DEFAULT_COMFY_QWEN_IMAGE_2_WORKFLOW_PATH,
        )
        self.assertEqual(
            process._get_comfy_workflow_path(comfy_config, batch=True),
            DEFAULT_COMFY_QWEN_IMAGE_2_BATCH_WORKFLOW_PATH,
        )

    def test_startup_releases_comfy_vram_before_training_model_load(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.accelerator = SimpleNamespace(is_main_process=True)
        process.sample_config = SimpleNamespace(
            comfy=SimpleNamespace(
                enabled=True,
                api_url="http://comfy.test:8188",
                run_in_background=True,
            )
        )
        process.first_sample_config = process.sample_config

        with mock.patch(
            "jobs.process.BaseSDTrainProcess.ComfyApiClient"
        ) as client_class:
            process.hook_before_model_load()

        client_class.assert_called_once_with(
            api_url="http://comfy.test:8188",
            timeout=10,
        )
        client_class.return_value.release_vram.assert_called_once_with()
        client_class.return_value.unload_models.assert_not_called()

    def test_startup_release_hook_precedes_training_model_creation_and_load(self):
        import inspect

        source = inspect.getsource(BaseSDTrainProcess.run)

        self.assertLess(
            source.index("self.hook_before_model_load()"),
            source.index("ModelClass("),
        )
        self.assertLess(
            source.index("self.hook_before_model_load()"),
            source.index("self.sd.load_model()"),
        )

    def test_training_and_assistant_networks_are_offloaded_and_restored(self):
        training_child = FakeNetwork("cuda:0")
        training_network = FakeNetwork("cuda:0", children=[training_child])
        assistant_network = FakeNetwork("cuda:0")

        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.network = training_network
        process.sd = SimpleNamespace(
            network=training_network,
            assistant_lora=assistant_network,
            accuracy_recovery_adapter=None,
        )

        state = process._capture_comfy_auxiliary_device_state()
        process._offload_comfy_auxiliary_modules()

        self.assertEqual(training_network.tensor.device, "cpu")
        self.assertEqual(training_child.tensor.device, "cpu")
        self.assertEqual(assistant_network.tensor.device, "cpu")

        process._restore_comfy_auxiliary_modules(state)

        self.assertEqual(training_network.tensor.device, "cuda:0")
        self.assertEqual(training_child.tensor.device, "cuda:0")
        self.assertEqual(assistant_network.tensor.device, "cuda:0")

    def test_failed_comfy_render_does_not_restore_gpu_models(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.sd = mock.Mock()
        process._capture_comfy_auxiliary_device_state = mock.Mock(return_value=[])
        process._restore_comfy_auxiliary_modules = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "release failed"):
            process._run_with_models_offloaded_for_comfy(
                lambda: (_ for _ in ()).throw(RuntimeError("release failed"))
            )

        process.sd.save_device_state.assert_called_once_with()
        process.sd.restore_device_state.assert_not_called()
        process._restore_comfy_auxiliary_modules.assert_not_called()

    def test_failed_sample_keeps_optimizer_state_on_cpu(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.optimizer = object()
        process.device_torch = "cuda:0"
        process.sample = mock.Mock(side_effect=RuntimeError("sample failed"))

        with (
            mock.patch(
                "jobs.process.BaseSDTrainProcess.move_optimizer_state_to_device"
            ) as move_state,
            self.assertRaisesRegex(RuntimeError, "sample failed"),
        ):
            process.sample_with_optimizer_state_offload()

        move_state.assert_called_once_with(process.optimizer, "cpu")

    def test_release_wait_requires_pre_comfy_physical_vram_baseline(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process._get_free_cuda_memory_bytes = mock.Mock(
            return_value=21 * 1024 ** 3
        )
        process._update_comfy_sample_status = mock.Mock()
        client = mock.Mock()
        client.wait_for_vram_release.return_value = True

        self.assertTrue(process._wait_for_comfy_vram_release(
            client,
            free_bytes_before_comfy=22 * 1024 ** 3,
        ))

        client.wait_for_vram_release.assert_called_once_with(
            min_free_bytes=int(21.5 * 1024 ** 3),
            free_memory_probe=process._get_free_cuda_memory_bytes,
        )


if __name__ == "__main__":
    unittest.main()
