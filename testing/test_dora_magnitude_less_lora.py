import os
import sys
import tempfile
import types
import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Stub:
    pass


sys.modules["toolkit.config_modules"] = types.SimpleNamespace(
    NetworkConfig=SimpleNamespace,
)
sys.modules["toolkit.lorm"] = types.SimpleNamespace(
    extract_conv=lambda *args, **kwargs: None,
    extract_linear=lambda *args, **kwargs: None,
    count_parameters=lambda *args, **kwargs: 0,
)
sys.modules["toolkit.paths"] = types.SimpleNamespace(KEYMAPS_ROOT="")
sys.modules["toolkit.saving"] = types.SimpleNamespace(
    get_lora_keymap_from_model_keymap=lambda keymap: keymap,
)
sys.modules["optimum.quanto"] = types.SimpleNamespace(
    QTensor=Stub,
    QBytesTensor=Stub,
)

from toolkit.network_mixins import ToolkitNetworkMixin


class TinyDoRANetwork(ToolkitNetworkMixin):
    def __init__(self):
        super().__init__(network_config=SimpleNamespace(type="dora"))
        self.network_type = "dora"
        self.peft_format = False
        self.use_old_lokr_format = False
        self.base_model_ref = None

    def state_dict(self):
        return OrderedDict(
            [
                ("diffusion_model.block.lora_down.weight", torch.ones(1, 2)),
                ("diffusion_model.block.lora_up.weight", torch.full((2, 1), 2.0)),
                ("diffusion_model.block.magnitude", torch.full((2,), 3.0)),
                ("diffusion_model.other.lora_magnitude_vector", torch.full((2,), 4.0)),
            ]
        )


class DoRAMagnitudeLessLoRATests(unittest.TestCase):
    def test_save_weights_writes_magnitude_less_lora_companion(self):
        network = TinyDoRANetwork()

        with tempfile.TemporaryDirectory() as temp_dir:
            dora_path = os.path.join(temp_dir, "sample_000000250.safetensors")
            lora_path = os.path.join(temp_dir, "sample_000000250-lora.safetensors")

            network.save_weights(
                dora_path,
                dtype=torch.float32,
                metadata=OrderedDict(),
                save_magnitude_less_lora=True,
            )

            dora_state = load_file(dora_path)
            lora_state = load_file(lora_path)

        self.assertIn("diffusion_model.block.magnitude", dora_state)
        self.assertIn("diffusion_model.other.lora_magnitude_vector", dora_state)
        self.assertNotIn("diffusion_model.block.magnitude", lora_state)
        self.assertNotIn("diffusion_model.other.lora_magnitude_vector", lora_state)
        self.assertTrue(
            torch.equal(
                lora_state["diffusion_model.block.lora_down.weight"],
                torch.ones(1, 2),
            )
        )
        self.assertTrue(
            torch.equal(
                lora_state["diffusion_model.block.lora_up.weight"],
                torch.full((2, 1), 2.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
