"""CPU callable lifecycle tests; no CUDA streaming/VRAM claim."""
import unittest

import torch

from testing import test_loha as fixtures
from toolkit.lora_special import LoRAModule
from toolkit.models.DoRA import DoRAModule
from toolkit.memory_management import MemoryManager
from toolkit.util.convrot_quant import ConvRotIntNQuantizer
from toolkit.util.ostris_quant import convert_linear_to_ostris


class AdapterOffloadRestoreTests(unittest.TestCase):
    def test_float_quantized_conv_stacked_adapters_attach_detach_reattach(self):
        torch.set_num_threads(2)
        for kind in ('linear', 'quantized', 'conv'):
            for offload_first in (False, True):
                with self.subTest(kind=kind, offload_first=offload_first):
                    base = (torch.nn.Conv2d(3, 5, 3, padding=1) if kind == 'conv'
                            else torch.nn.Linear(64, 16)).requires_grad_(False)
                    if kind == 'quantized':
                        convert_linear_to_ostris(base, ConvRotIntNQuantizer(4, rot_size=16))
                    model = torch.nn.Sequential(base)
                    if offload_first:
                        MemoryManager.attach(model, torch.device('cpu'))
                    first, _, first_network = fixtures.LoHaTests()._module(base, dtype=torch.float32)
                    second, _, second_network = fixtures.LoHaTests()._module(base, dora=True, dtype=torch.float32)
                    fixtures._randomize(first)
                    fixtures._randomize(second)
                    # Networks must stay alive: adapter hooks intentionally use
                    # weak references, just as in the actual trainer.
                    self.assertTrue(first_network.is_active and second_network.is_active)
                    x = (torch.randn(2, 3, 6, 6) if kind == 'conv' else torch.randn(2, 64))
                    with torch.no_grad():
                        expected = model(x)
                        if not offload_first:
                            MemoryManager.attach(model, torch.device('cpu'))
                        torch.testing.assert_close(model(x), expected)
                        MemoryManager.detach(model)
                        self.assertFalse(hasattr(base, '_layer_memory_manager'))
                        torch.testing.assert_close(model(x), expected)
                        MemoryManager.attach(model, torch.device('cpu'))
                        torch.testing.assert_close(model(x), expected)
                        MemoryManager.detach(model)
                        torch.testing.assert_close(model(x), expected)
                    model(x).square().mean().backward()
                    self.assertGreater(first.hada_w2_b.grad.abs().sum().item(), 0)
                    self.assertGreater(second.hada_w2_b.grad.abs().sum().item(), 0)
                    self.assertIsNone(base.bias.grad)

    def test_lora_dora_and_ara_restore_preserves_outer_forward_and_gradients(self):
        for adapter_type, ara in ((LoRAModule, False), (DoRAModule, False), (LoRAModule, True)):
            for offload_first in (False, True):
                with self.subTest(adapter=adapter_type.__name__, ara=ara, offload_first=offload_first):
                    base = torch.nn.Linear(8, 6).requires_grad_(False)
                    model = torch.nn.Sequential(base)
                    network = fixtures._Network()
                    network.network_type, network.is_lorm = 'lora', False
                    if offload_first:
                        MemoryManager.attach(model, torch.device('cpu'))
                    adapter = adapter_type('test', base, network=network, lora_dim=2, alpha=1, is_ara=ara)
                    adapter.apply_to()
                    with torch.no_grad():
                        adapter.lora_up.weight.normal_(0, .1)
                        if isinstance(adapter, DoRAModule):
                            adapter.magnitude.mul_(1.1)
                    x = torch.randn(2, 8)
                    expected = model(x).detach()
                    if not offload_first:
                        MemoryManager.attach(model, torch.device('cpu'))
                    torch.testing.assert_close(model(x), expected)
                    MemoryManager.detach(model)
                    self.assertIs(base.forward.__self__, adapter)
                    torch.testing.assert_close(model(x), expected)
                    model(x).square().mean().backward()
                    self.assertGreater(adapter.lora_up.weight.grad.abs().sum().item(), 0)
                    self.assertIsNone(base.weight.grad)


if __name__ == '__main__':
    unittest.main()
