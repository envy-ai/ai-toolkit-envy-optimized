import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.optimizers.optimizer_utils import Auto8bitTensor
from toolkit.optimizers.prodigy_8bit import Prodigy8bit


class Prodigy8bitStateTests(unittest.TestCase):
    def test_rejects_plain_tensor_prodigy_state(self):
        param = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = Prodigy8bit([param])
        state_dict = optimizer.state_dict()
        state_dict["state"][0] = {
            "step": 1,
            "s": torch.zeros_like(param.data),
            "p0": torch.zeros_like(param.data),
            "exp_avg": torch.zeros_like(param.data),
            "exp_avg_sq": torch.zeros_like(param.data),
        }

        with self.assertRaisesRegex(ValueError, "not a Prodigy8bit state"):
            optimizer.load_state_dict(state_dict)

    def test_serializes_auto8bit_state_without_custom_objects(self):
        param = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = Prodigy8bit([param])
        optimizer.state[param]["step"] = 1
        for state_key in ("s", "p0", "exp_avg", "exp_avg_sq"):
            optimizer.state[param][state_key] = Auto8bitTensor(
                torch.zeros_like(param.data)
            )

        state_dict = optimizer.state_dict()

        self.assertEqual(state_dict["state"][0]["s"]["_type"], "Auto8bitTensor")
        self.assertIsInstance(state_dict["state"][0]["s"]["state"], dict)
        self.assertIsInstance(optimizer.state[param]["s"], Auto8bitTensor)

    def test_state_dict_does_not_break_next_step(self):
        param = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = Prodigy8bit([param])

        with patch(
            "toolkit.optimizers.prodigy_8bit.copy_stochastic",
            lambda target, source: target.copy_(source),
        ):
            param.grad = torch.ones_like(param)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer.state_dict()

            self.assertIsInstance(optimizer.state[param]["exp_avg"], Auto8bitTensor)
            param.grad = torch.ones_like(param)
            optimizer.step()
        self.assertIsInstance(optimizer.state[param]["exp_avg"], Auto8bitTensor)

    def test_loads_serialized_auto8bit_state(self):
        param = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = Prodigy8bit([param])
        state_dict = optimizer.state_dict()
        state_dict["state"][0] = {"step": 1}
        for state_key in ("s", "p0", "exp_avg", "exp_avg_sq"):
            state_dict["state"][0][state_key] = {
                "_type": "Auto8bitTensor",
                "state": Auto8bitTensor(torch.zeros_like(param.data)).state_dict(),
            }

        optimizer.load_state_dict(state_dict)

        self.assertIsInstance(optimizer.state[param]["s"], Auto8bitTensor)


if __name__ == "__main__":
    unittest.main()
