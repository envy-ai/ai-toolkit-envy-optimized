import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.optimizers.optimizer_utils import Auto8bitTensor
from toolkit.optimizers import optimizer_utils


class OptimizerStateDeviceTests(unittest.TestCase):
    def test_moves_optimizer_state_tensors_and_auto8bit_wrappers(self):
        param = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = torch.optim.AdamW([param])

        state = optimizer.state[param]
        state["step"] = 1
        state["exp_avg_sq"] = torch.ones_like(param.data)
        state["exp_avg"] = Auto8bitTensor(torch.ones_like(param.data))
        state["nested"] = {"tensor": torch.ones(1)}
        state["list_value"] = [torch.ones(1)]
        original_auto8 = state["exp_avg"]

        self.assertTrue(
            hasattr(optimizer_utils, "move_optimizer_state_to_device"),
            "optimizer state device helper is missing",
        )

        optimizer_utils.move_optimizer_state_to_device(optimizer, "meta")

        self.assertEqual(state["exp_avg_sq"].device.type, "meta")
        self.assertIs(state["exp_avg"], original_auto8)
        self.assertEqual(state["exp_avg"].quantized.device.type, "meta")
        self.assertEqual(state["nested"]["tensor"].device.type, "meta")
        self.assertEqual(state["list_value"][0].device.type, "meta")


if __name__ == "__main__":
    unittest.main()
