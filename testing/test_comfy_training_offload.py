import unittest
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


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


if __name__ == "__main__":
    unittest.main()
