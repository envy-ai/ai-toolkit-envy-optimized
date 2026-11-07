import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.lokr import LokrModule


class Network:
    is_active = True
    is_merged_in = False
    is_lorm = False
    _multiplier = 1.0
    torch_multiplier = torch.tensor([1.0])


def randomize(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.startswith('lokr_'):
                param.normal_(0, 0.3)
        if module.use_dora:
            module.magnitude.mul_(1.2)


def comfy_adapter():
    directory = Path('/home/bart/ComfyUI/comfy/weight_adapter')
    if not (directory / 'lokr.py').is_file():
        return None
    comfy = types.ModuleType('comfy')
    management = types.ModuleType('comfy.model_management')
    management.cast_to_device = lambda tensor, device, dtype: tensor.to(device=device, dtype=dtype)
    comfy.model_management = management
    package = types.ModuleType('_lokr_test_comfy')
    package.__path__ = [str(directory)]
    with patch.dict(sys.modules, {'comfy': comfy, 'comfy.model_management': management,
                                  '_lokr_test_comfy': package}):
        for name in ('base', 'lokr'):
            spec = importlib.util.spec_from_file_location('_lokr_test_comfy.' + name, directory / (name + '.py'))
            loaded = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {spec.name: loaded}):
                spec.loader.exec_module(loaded)
            if name == 'base':
                sys.modules[spec.name] = loaded
        sys.modules.pop('_lokr_test_comfy.base', None)
    return loaded.LoKrAdapter


class LoKrDoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)

    def module(self, base=None, rank=1, **kwargs):
        base = base if base is not None else torch.nn.Linear(32, 24)
        base.requires_grad_(False)
        if not getattr(base, 'is_ostris_quantized', False):
            base.to(torch.float64)
        network = Network()
        module = LokrModule('test', base, lora_dim=rank, alpha=0.5, network=network,
                            use_dora=True, lokr_norm_chunk_size=3, **kwargs).to(torch.float64)
        module.apply_to()
        return module, base, network

    def test_config_defaults_to_plain_lokr(self):
        self.assertFalse(NetworkConfig(type='lokr').lokr_dora)
        self.assertTrue(NetworkConfig(type='lokr', lokr_dora=True).lokr_dora)

    def test_resume_signature_distinguishes_dokr(self):
        from toolkit.job_fork import resume_signature
        config = {'config': {'process': [{'type': 'ui_trainer', 'network': {'type': 'lokr', 'lokr_dora': False}}]}}
        plain = resume_signature(config)
        config['config']['process'][0]['network']['lokr_dora'] = True
        self.assertNotEqual(plain, resume_signature(config))

    def test_dense_forward_gradients_and_factor_layouts(self):
        for rank, both, factor in ((8, False, 4), (1, False, 4), (1, True, 8)):
            with self.subTest(rank=rank, both=both):
                m, base, network = self.module(rank=rank, decompose_both=both, factor=factor)
                randomize(m)
                x = torch.randn(2, 3, 32, dtype=torch.float64, requires_grad=True)
                delta = m.get_weight(base.weight)
                norm = (base.weight + delta.detach()).flatten(1).norm(dim=1)
                ratio = m.magnitude / norm
                for strength in (-1.0, 0.0, 0.5, 1.0, 2.0):
                    network._multiplier = strength
                    network.torch_multiplier = torch.tensor([strength])
                    expected = F.linear(x, base.weight + strength * ((base.weight + delta) * ratio[:, None] - base.weight), base.bias)
                    actual = base(x)
                    torch.testing.assert_close(actual, expected)
                    if strength == 0:
                        continue
                    got = torch.autograd.grad(actual.square().sum(), [x, *m.parameters()], retain_graph=True)
                    ref = torch.autograd.grad(expected.square().sum(), [x, *m.parameters()], retain_graph=True)
                    for a, b in zip(got, ref):
                        torch.testing.assert_close(a, b)
                self.assertIsNone(base.weight.grad)
                self.assertFalse(m._weight_norms(True).requires_grad)

    def test_noop_and_zero_weight_norm_are_finite(self):
        m, base, _ = self.module()
        x = torch.randn(2, 32, dtype=torch.float64)
        torch.testing.assert_close(base(x), F.linear(x, base.weight, base.bias))
        base(x).sum().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for n, p in m.named_parameters() if n.startswith('lokr_')), 0)
        with torch.no_grad():
            base.weight.zero_()
            m.magnitude.zero_()
        self.assertTrue(torch.isfinite(base(x)).all())

    def test_linear_has_no_recurring_dense_kron(self):
        m, base, _ = self.module()
        randomize(m)
        x = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
        with patch('torch.kron', side_effect=AssertionError('dense kron during linear execution')):
            m._weight_norms(True)
            base(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_conv_full_factorized_and_cp_dense_reference(self):
        for rank, cp in ((8, False), (1, False), (1, True)):
            with self.subTest(rank=rank, cp=cp):
                base = torch.nn.Conv2d(16, 24, 3, padding=1)
                m, base, network = self.module(base, rank=rank, factor=4, use_cp=cp)
                randomize(m)
                x = torch.randn(2, 16, 5, 5, dtype=torch.float64, requires_grad=True)
                delta = m.get_weight(base.weight)
                ratio = m.magnitude / (base.weight + delta.detach()).flatten(1).norm(dim=1)
                network._multiplier = -0.5
                network.torch_multiplier = torch.tensor([-0.5])
                weight = base.weight - 0.5 * ((base.weight + delta) * ratio[:, None, None, None] - base.weight)
                expected = F.conv2d(x, weight, base.bias, padding=1)
                actual = base(x)
                torch.testing.assert_close(actual, expected)
                for a, b in zip(torch.autograd.grad(actual.sum(), [x, *m.parameters()], retain_graph=True),
                                torch.autograd.grad(expected.sum(), [x, *m.parameters()])):
                    torch.testing.assert_close(a, b)

    def test_rejects_unsupported_convolutions(self):
        for base in (torch.nn.Conv2d(4, 4, 3, groups=2),
                     torch.nn.Conv2d(4, 4, 3, padding=1, padding_mode='reflect')):
            with self.assertRaisesRegex(ValueError, '(?i)(group|padding)'):
                self.module(base)

    def test_magnitude_conversion_roundtrip(self):
        m, _, _ = self.module()
        randomize(m)
        original = m.magnitude.detach().clone()
        exported = m.comfy_magnitude().detach().clone()
        with torch.no_grad():
            m.magnitude.copy_(exported.reshape(-1))
        m.load_comfy_magnitude()
        torch.testing.assert_close(m.magnitude, original)

    def test_compile_and_checkpoint(self):
        from torch.utils.checkpoint import checkpoint
        m, base, _ = self.module()
        randomize(m)
        x = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
        expected = base(x)
        ref = torch.autograd.grad(expected.square().sum(), [x, *m.parameters()])
        actual = checkpoint(torch.compile(base, backend='aot_eager', fullgraph=False), x, use_reentrant=False)
        got = torch.autograd.grad(actual.square().sum(), [x, *m.parameters()])
        torch.testing.assert_close(actual, expected)
        for a, b in zip(got, ref):
            torch.testing.assert_close(a, b)

    def test_quantized_offload_and_checkpoint_gradients(self):
        from torch.utils.checkpoint import checkpoint
        from toolkit.memory_management.manager_modules import OstrisLinearLayerMemoryManager
        from toolkit.util.convrot_quant import ConvRotInt8Quantizer, ConvRotIntNQuantizer
        from toolkit.util.ostris_quant import convert_linear_to_ostris
        for quantizer in (ConvRotInt8Quantizer(rot_size=16), ConvRotIntNQuantizer(4, rot_size=16)):
            for before in (False, True):
                base = torch.nn.Linear(64, 16).requires_grad_(False)
                convert_linear_to_ostris(base, quantizer)
                manager = types.SimpleNamespace(process_device=torch.device('cpu'))
                if before:
                    OstrisLinearLayerMemoryManager(base, manager)
                m, base, _ = self.module(base)
                m.to(torch.float32)
                randomize(m)
                if not before:
                    OstrisLinearLayerMemoryManager(base, manager)
                x = torch.randn(2, 64, requires_grad=True)
                checkpoint(base, x, use_reentrant=False).sum().backward()
                self.assertTrue(base.is_ostris_quantized)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def network_fixture(self, dora, rank=2):
        if not torch.cuda.is_available():
            with patch('torch.cuda.current_device', return_value=0), patch(
                    'torch.cuda.get_device_properties', return_value=types.SimpleNamespace(warp_size=32)), patch(
                    'triton.autotune', side_effect=lambda *args, **kwargs: lambda function: function):
                from testing import test_qwen_image_2_training_adapter as fixture
        else:
            from testing import test_qwen_image_2_training_adapter as fixture
        model = fixture.QwenImage2TrainingAdapterTests().make_model()
        model.model = fixture.QwenImage21Transformer2DModel()
        model.model.transformer_blocks[0].attn.to_q.weight.data.normal_()
        model.model.requires_grad_(False)
        config = NetworkConfig(type='lokr', linear=rank, linear_alpha=0.5, lokr_dora=dora)
        network = LoRASpecialNetwork([], model.model, lora_dim=rank, alpha=0.5,
                                    train_unet=True, train_text_encoder=False,
                                    network_type='lokr', network_config=config,
                                    transformer_only=True, is_transformer=True,
                                    target_lin_modules=model.target_lora_modules, base_model=model)
        network.apply_to([], model.model, apply_text_encoder=False, apply_unet=True)
        network.force_to(torch.device('cpu'), torch.float32)
        network._update_torch_multiplier()
        network.is_active = True
        return network, model

    def test_full_factor_import_across_nominal_ranks_preserves_unit_scale(self):
        source, source_model = self.network_fixture(True, rank=8)
        target, target_model = self.network_fixture(True, rank=9999999999)
        module = source.unet_loras[0]
        self.assertTrue(module.use_w1 and module.use_w2)
        target_module = target.unet_loras[0]
        self.assertTrue(target_module.use_w1 and target_module.use_w2)
        target_module.org_module[0].weight.data.copy_(module.org_module[0].weight.data)
        randomize(module)
        x = torch.randn(3, 2)
        expected = source_model.model(x).detach()
        exported = source.get_state_dict(dtype=torch.float32)
        target.load_weights(exported)
        self.assertEqual(target_module.scale, 1.0)
        torch.testing.assert_close(target_model.model(x), expected)
        reexported = target.get_state_dict(dtype=torch.float32)
        for key in exported:
            torch.testing.assert_close(reexported[key], exported[key])
        adapter_class = comfy_adapter()
        if adapter_class is not None:
            weights = {'test.' + k: v.detach().clone() for k, v in target_module.state_dict().items()}
            adapter = adapter_class.load('test', weights, target_module.alpha.item(), target_module.comfy_magnitude())
            base = target_module.org_module[0]
            for strength in (-1., .5, 1.):
                target._multiplier = strength
                target._update_torch_multiplier()
                patched = adapter.calculate_weight(base.weight.detach().clone(), 'test', strength, 1., None, lambda v: v)
                torch.testing.assert_close(base(x), F.linear(x, patched, base.bias), atol=3e-6, rtol=3e-5)

    def test_optimizer_checkpoint_resume_and_merge(self):
        for dora in (False, True):
            network, model = self.network_fixture(dora)
            module = network.unet_loras[0]
            randomize(module)
            optimizer = torch.optim.AdamW(network.prepare_optimizer_params(None, 0.001, 0.001))
            x = torch.randn(3, 2)
            model.model(x).square().mean().backward()
            optimizer.step()
            self.assertIsNone(module.org_module[0].weight.grad)
            if dora:
                network.merge_in()
                self.assertFalse(network.is_merged_in)
            with tempfile.TemporaryDirectory() as directory:
                filename = str(Path(directory) / 'lokr.safetensors')
                network.save_weights(filename, dtype=torch.float32)
                exported = load_file(filename)
                prefix = 'diffusion_model.transformer_blocks.0.attn.to_q'
                self.assertEqual(prefix + '.dora_scale' in exported, dora)
                expected = model.model(x).detach()
                state = {key: value.clone() for key, value in network.state_dict().items()}
                network.reset_weights()
                network.load_weights(filename)
                torch.testing.assert_close(model.model(x), expected)
                network.reset_weights()
                network.load_weights(state)
                torch.testing.assert_close(model.model(x), expected)

    def test_plain_checkpoint_to_dokr_and_disabled_rejection(self):
        network, model = self.network_fixture(False)
        module = network.unet_loras[0]
        randomize(module)
        plain = network.get_state_dict(dtype=torch.float32)
        x = torch.randn(2, 2)
        expected = model.model(x).detach()
        with patch.object(module, 'use_dora', True):
            module.magnitude = torch.nn.Parameter(module._weight_norms(False))
            network.load_weights(plain)
            torch.testing.assert_close(model.model(x), expected)
            dokr = network.get_state_dict(dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'enable network.lokr_dora'):
            network.load_weights(dokr)

    def test_one_dimensional_input_and_autocast_norms(self):
        m, base, network = self.module()
        randomize(m)
        x = torch.randn(32, dtype=torch.float64)
        delta = m.get_weight(base.weight)
        ratio = m.magnitude / (base.weight + delta.detach()).norm(dim=1)
        torch.testing.assert_close(base(x), F.linear(x, (base.weight + delta) * ratio[:, None], base.bias))
        m.float()
        base.float()
        baseline = m._weight_norms(True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = m._weight_norms(True)
        torch.testing.assert_close(actual, baseline)

    def test_plain_lokr_noop_and_forward_unchanged(self):
        base = torch.nn.Linear(32, 24).double().requires_grad_(False)
        network = Network()
        m = LokrModule('test', base, lora_dim=1, alpha=.5, network=network).double()
        m.apply_to()
        x = torch.randn(2, 32, dtype=torch.float64)
        torch.testing.assert_close(base(x), F.linear(x, base.weight, base.bias))
        randomize(m)
        for strength in (-1., .5, 1.):
            network._multiplier = strength
            network.torch_multiplier = torch.tensor([strength])
            torch.testing.assert_close(base(x), F.linear(x, base.weight + strength * m.get_weight(base.weight), base.bias))

    def test_malformed_checkpoint_rejected(self):
        network, _ = self.network_fixture(True)
        sd = network.get_state_dict(dtype=torch.float32)
        magnitude = next(k for k in sd if k.endswith('.dora_scale'))
        malformed = dict(sd)
        malformed[magnitude] = torch.ones(3, 2)
        with self.assertRaises((ValueError, RuntimeError)):
            network.load_weights(malformed)
        incomplete = dict(sd)
        del incomplete[next(k for k in sd if '.lokr_w' in k)]
        with self.assertRaisesRegex(ValueError, 'Incomplete LoKr'):
            network.load_weights(incomplete)
        unmatched = dict(sd)
        unmatched['wrong_layer.dora_scale'] = unmatched.pop(magnitude)
        with self.assertRaises(ValueError):
            network.load_weights(unmatched)

    def test_standalone_inference_rejects_dokr_checkpoint(self):
        from safetensors.torch import save_file
        from toolkit.inference_lora import InferenceLoRA
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'dokr.safetensors')
            save_file({'layer.lokr_w1': torch.ones(1, 1), 'layer.lokr_w2': torch.ones(2, 2),
                       'layer.dora_scale': torch.ones(2, 1)}, filename)
            with patch('toolkit.models.v2.resolver.resolve_lora_file', return_value=filename):
                with self.assertRaisesRegex(ValueError, 'cannot apply LoKr DoRA'):
                    InferenceLoRA(filename).load(object())

    def test_real_comfy_export_signed_strengths(self):
        adapter_class = comfy_adapter()
        if adapter_class is None:
            self.skipTest('Local ComfyUI unavailable')
        for conv in (False, True):
            base = torch.nn.Conv2d(16, 24, 3, padding=1) if conv else torch.nn.Linear(32, 24)
            m, base, network = self.module(base, factor=4)
            m.to(torch.float32)
            base.to(torch.float32)
            randomize(m)
            weights = {'test.' + k: v.detach().clone() for k, v in m.state_dict().items()}
            adapter = adapter_class.load('test', weights, m.alpha.item(), m.comfy_magnitude())
            x = torch.randn(2, 16, 5, 5) if conv else torch.randn(2, 32)
            for strength in (-1., 0., .5, 1., 2.):
                network._multiplier = strength
                network.torch_multiplier = torch.tensor([strength])
                patched = adapter.calculate_weight(base.weight.detach().clone(), 'test', strength, 1., None, lambda v: v)
                expected = F.conv2d(x, patched, base.bias, padding=1) if conv else F.linear(x, patched, base.bias)
                torch.testing.assert_close(base(x), expected, atol=3e-6, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
