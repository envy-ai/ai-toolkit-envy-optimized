import os
import sys
import unittest
from unittest.mock import Mock

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.models.DoRA import DoRAModule


class FakeNetwork:
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def __init__(self):
        self.torch_multiplier = torch.ones(1)


class DoRAMemoryTests(unittest.TestCase):
    def test_scalar_multiplier_forward_reuses_existing_outputs(self):
        torch.manual_seed(1)
        network = FakeNetwork()
        linear = torch.nn.Linear(3, 2, bias=False)
        module = DoRAModule(
            "test",
            linear,
            lora_dim=2,
            alpha=2,
            network=network,
        )
        module.org_forward = linear.forward
        module.apply_dora = Mock(
            side_effect=AssertionError("full adapted-weight linear should be skipped")
        )
        x = torch.randn(2, 5, 3)

        org_forwarded = linear(x)
        lora_output = module._call_forward(x.to(module.lora_down.weight.dtype))
        scaled_lora_output = lora_output.to(org_forwarded.dtype)
        lora_weight = module.lora_up.weight @ module.lora_down.weight
        scaled_lora_weight = lora_weight * network.torch_multiplier.mean()
        with torch.no_grad():
            weight = module.get_orig_weight().to(
                scaled_lora_weight.device,
                dtype=scaled_lora_weight.dtype,
            )
            weight_norm = module._get_weight_norm(
                weight,
                scaled_lora_weight.detach(),
            )
        dora_scale = (module.magnitude / weight_norm).to(org_forwarded.dtype)
        expected = (org_forwarded + scaled_lora_output) * dora_scale.view(1, 1, -1)

        actual = module.forward(x)

        torch.testing.assert_close(actual, expected)
        module.apply_dora.assert_not_called()
        actual.sum().backward()
        self.assertIsNotNone(module.lora_up.weight.grad)
        self.assertIsNotNone(module.magnitude.grad)


if __name__ == "__main__":
    unittest.main()
