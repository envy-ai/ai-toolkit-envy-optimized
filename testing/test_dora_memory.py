import os
import sys
import unittest
from unittest.mock import Mock

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.models.DoRA import DoRAModule
from toolkit.models import DoRA as dora_module
from toolkit.models.lokr import AffineQuantizedTensor as LokrAffineQuantizedTensor


class FakeNetwork:
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def __init__(self):
        self.torch_multiplier = torch.ones(1)


class DoRAMemoryTests(unittest.TestCase):
    def test_affine_quantized_tensor_detection_matches_lokr(self):
        self.assertIs(dora_module.AffineQuantizedTensor, LokrAffineQuantizedTensor)

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

    def test_zero_base_rows_have_neutral_finite_dora_scale(self):
        network = FakeNetwork()
        linear = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            linear.weight[0].zero_()
            linear.weight[1].copy_(torch.tensor([1.0, -2.0, 3.0]))
            linear.bias.copy_(torch.tensor([2.0, -1.0]))

        module = DoRAModule(
            "zero-row",
            linear,
            lora_dim=2,
            alpha=2,
            network=network,
            dora_scale_chunk_size=1,
        )
        module.org_forward = linear.forward
        x = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]], requires_grad=True)

        dense_scale = module.get_dora_scale(torch.zeros_like(linear.weight))
        chunked_scale = module.get_dora_scale_from_low_rank(
            network.torch_multiplier.mean()
        )
        self.assertTrue(torch.isfinite(dense_scale).all())
        self.assertTrue(torch.isfinite(chunked_scale).all())
        torch.testing.assert_close(dense_scale, chunked_scale)
        torch.testing.assert_close(dense_scale, torch.ones_like(dense_scale))

        output = module.forward(x)
        expected = torch.nn.functional.linear(x, linear.weight, linear.bias)
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in (module.lora_up.weight, module.lora_down.weight, module.magnitude):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

        # Once the low-rank delta is nonzero, nonzero rows keep the original
        # exact magnitude/norm arithmetic in both dense and chunked paths.
        with torch.no_grad():
            module.lora_up.weight[1].fill_(0.25)
            delta = module.lora_up.weight @ module.lora_down.weight
            adapted_norm = torch.linalg.norm(linear.weight + delta, dim=1)
            previous_ratio = module.magnitude / adapted_norm
            dense_scale = module.get_dora_scale(delta)
            chunked_scale = module.get_dora_scale_from_low_rank(
                network.torch_multiplier.mean()
            )
        torch.testing.assert_close(dense_scale[1], previous_ratio[1], rtol=0, atol=0)
        torch.testing.assert_close(chunked_scale[1], previous_ratio[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
