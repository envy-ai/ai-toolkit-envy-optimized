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
from toolkit.models.loha import LoHaModule, _ChunkedLoHa


class _Network:
    is_active = True
    is_merged_in = False
    _multiplier = 1.0
    torch_multiplier = torch.tensor([1.0])


def _delta(module):
    return ((module.hada_w1_a @ module.hada_w1_b) *
            (module.hada_w2_a @ module.hada_w2_b)) * module.scale


def _randomize(module):
    with torch.no_grad():
        for name, value in module.named_parameters():
            if name.startswith('hada_'):
                value.normal_(0, 0.3)
        if module.use_dora:
            module.magnitude.mul_(1.15)


def _comfy_adapter():
    """Load the real local Comfy adapter without initializing CUDA/model management."""
    directory = Path('/home/bart/ComfyUI/comfy/weight_adapter')
    if not (directory / 'loha.py').is_file():
        return None
    comfy = types.ModuleType('comfy')
    management = types.ModuleType('comfy.model_management')
    management.cast_to_device = lambda tensor, device, dtype: tensor.to(device=device, dtype=dtype)
    comfy.model_management = management
    package = types.ModuleType('_loha_test_comfy')
    package.__path__ = [str(directory)]
    with patch.dict(sys.modules, {'comfy': comfy, 'comfy.model_management': management,
                                  '_loha_test_comfy': package}):
        for name in ('base', 'loha'):
            spec = importlib.util.spec_from_file_location('_loha_test_comfy.' + name, directory / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {spec.name: module}):
                spec.loader.exec_module(module)
            if name == 'base':
                sys.modules[spec.name] = module
        sys.modules.pop('_loha_test_comfy.base', None)
    return module.LoHaAdapter


class LoHaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)

    def _module(self, base=None, dora=False, dtype=torch.float64):
        base = base if base is not None else torch.nn.Linear(7, 5, bias=True)
        base.to(dtype).requires_grad_(False)
        network = _Network()
        module = LoHaModule('test', base, lora_dim=3, alpha=1.5, network=network,
                            use_dora=dora, loha_chunk_size=2).to(dtype)
        module.apply_to()
        return module, base, network

    def test_chunked_linear_forward_and_all_gradients_match_dense(self):
        values = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                  for shape in ((2, 4, 7), (5, 3), (3, 7), (5, 3), (3, 7))]
        reference = [value.detach().clone().requires_grad_() for value in values]
        output = _ChunkedLoHa.apply(*values, 2, None)
        x, a, b, c, d = reference
        expected = F.linear(x, (a @ b) * (c @ d))
        gradient = torch.randn_like(output)
        output.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(output, expected)
        for value, ref in zip(values, reference):
            torch.testing.assert_close(value.grad, ref.grad)
        self.assertTrue(torch.autograd.gradcheck(lambda *v: _ChunkedLoHa.apply(*v, 2, None), values))

    def test_convolution_forward_and_backward_match_dense(self):
        conv = ((2, 3, 3), (2, 2), (1, 1), (1, 1))
        values = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                  for shape in ((2, 2, 7, 7), (5, 3), (3, 18), (5, 3), (3, 18))]
        reference = [value.detach().clone().requires_grad_() for value in values]
        output = _ChunkedLoHa.apply(*values, 2, conv)
        x, a, b, c, d = reference
        expected = F.conv2d(x, ((a @ b) * (c @ d)).reshape(5, 2, 3, 3), stride=2, padding=1)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(output, expected)
        for value, ref in zip(values, reference):
            torch.testing.assert_close(value.grad, ref.grad)

    def test_zero_initialized_noop_has_nonzero_factor_gradient(self):
        for dora in (False, True):
            module, base, network = self._module(dora=dora)
            x = torch.randn(2, 7, dtype=torch.float64)
            torch.testing.assert_close(base(x), F.linear(x, base.weight, base.bias), atol=1e-7, rtol=1e-6)
            base(x).sum().backward()
            self.assertGreater(module.hada_w2_b.grad.abs().sum().item(), 0)
            self.assertIsNone(base.weight.grad)
            self.assertNotIn('_runtime_scale', module.state_dict())

    def test_signed_fractional_strength_and_cfg_batch_broadcast(self):
        module, base, network = self._module()
        _randomize(module)
        x = torch.randn(4, 3, 7, dtype=torch.float64)
        network._multiplier = [-1.0, 0.5]
        network.torch_multiplier = torch.tensor([-1.0, 0.5])
        expected = F.linear(x, base.weight, base.bias) + F.linear(x, _delta(module)) * torch.tensor(
            [-1, -1, 0.5, 0.5], dtype=x.dtype).reshape(4, 1, 1)
        torch.testing.assert_close(base(x), expected)
        network.is_active = False
        torch.testing.assert_close(base(x), F.linear(x, base.weight, base.bias))
        network.is_active = True
        network._multiplier = 0.0
        torch.testing.assert_close(base(x), F.linear(x, base.weight, base.bias))

    def test_doha_forward_and_gradients_use_detached_adapted_norm_bias_unchanged(self):
        module, base, network = self._module(dora=True)
        _randomize(module)
        x = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
        delta = _delta(module)
        norm = (base.weight + delta.detach()).norm(dim=1)
        ratio = module.magnitude / norm
        strength = -0.75
        network._multiplier = strength
        network.torch_multiplier = torch.tensor([strength])
        target = base.weight + strength * ((base.weight + delta) * ratio[:, None] - base.weight)
        expected = F.linear(x, target, base.bias)
        output = base(x)
        torch.testing.assert_close(output, expected)
        parameters = [x, *module.parameters()]
        got = torch.autograd.grad(output.square().sum(), parameters, retain_graph=True)
        wanted = torch.autograd.grad(expected.square().sum(), parameters)
        for actual, ref in zip(got, wanted):
            torch.testing.assert_close(actual, ref)
        torch.testing.assert_close(base(torch.zeros_like(x)), base.bias.expand_as(output))
        module.reset_weights()
        torch.testing.assert_close(base(x), F.linear(x, base.weight, base.bias))

    def test_backward_retains_only_inputs_and_factors_not_dense_weights(self):
        values = [torch.randn(*shape, requires_grad=True)
                  for shape in ((2, 4, 7), (5, 3), (3, 7), (5, 3), (3, 7))]
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x, lambda x: x):
            _ChunkedLoHa.apply(*values, 2, None).sum().backward()
        self.assertEqual([v.shape for v in saved], [v.shape for v in values])

    def test_bfloat16_autocast_produces_finite_gradients(self):
        module, base, _ = self._module(dtype=torch.bfloat16)
        _randomize(module)
        x = torch.randn(2, 7, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = base(x)
        output.float().square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_aot_compilation_and_nonreentrant_checkpointing_preserve_gradients(self):
        from torch.utils.checkpoint import checkpoint
        for dora in (False, True):
            module, base, network = self._module(dora=dora, dtype=torch.float32)
            _randomize(module)
            x = torch.randn(2, 7, requires_grad=True)
            expected = base(x)
            expected_grads = torch.autograd.grad(expected.square().sum(), [x, *module.parameters()])
            compiled = torch.compile(base, backend='aot_eager', fullgraph=False)
            actual = checkpoint(compiled, x, use_reentrant=False)
            actual_grads = torch.autograd.grad(actual.square().sum(), [x, *module.parameters()])
            torch.testing.assert_close(actual, expected)
            for got, wanted in zip(actual_grads, expected_grads):
                torch.testing.assert_close(got, wanted)

    def test_layer_offload_wrapper_works_before_or_after_adapter_attachment(self):
        from toolkit.memory_management.manager_modules import OstrisLinearLayerMemoryManager
        from toolkit.util.convrot_quant import ConvRotIntNQuantizer
        from toolkit.util.ostris_quant import convert_linear_to_ostris
        from types import SimpleNamespace
        for dora in (False, True):
            for offload_first in (False, True):
                base = torch.nn.Linear(64, 16).requires_grad_(False)
                convert_linear_to_ostris(base, ConvRotIntNQuantizer(4, rot_size=16))
                manager = SimpleNamespace(process_device=torch.device('cpu'))
                if offload_first:
                    OstrisLinearLayerMemoryManager(base, manager)
                module, base, network = self._module(base, dora=dora, dtype=torch.float32)
                _randomize(module)
                x = torch.randn(2, 64, requires_grad=True)
                expected = base(x)
                if not offload_first:
                    OstrisLinearLayerMemoryManager(base, manager)
                actual = base(x)
                torch.testing.assert_close(actual, expected)
                actual.sum().backward()
                self.assertTrue(torch.isfinite(module.hada_w2_b.grad).all())

    def test_convrot_int4_and_int8_stay_quantized_without_plain_loha_weight_reads(self):
        from toolkit.util.convrot_quant import ConvRotInt8Quantizer, ConvRotIntNQuantizer
        from toolkit.util.ostris_quant import convert_linear_to_ostris
        for quantizer in (ConvRotInt8Quantizer(rot_size=16), ConvRotIntNQuantizer(4, rot_size=16)):
            for dora in (False, True):
                base = torch.nn.Linear(64, 16).requires_grad_(False)
                self.assertTrue(convert_linear_to_ostris(base, quantizer))
                original_buffers = {name: buf.clone() for name, buf in base.named_buffers()}
                if not dora:
                    # In the plain adapter, even shape discovery must not materialize a base weight.
                    with patch.object(type(base), 'weight', property(lambda _: (_ for _ in ()).throw(
                            AssertionError('plain LoHa inspected quantized base weight')))):
                        module, base, network = self._module(base, dtype=torch.float32)
                        _randomize(module)
                        x = torch.randn(2, 64, requires_grad=True)
                        base(x).sum().backward()
                else:
                    module, base, network = self._module(base, dora=True, dtype=torch.float32)
                    _randomize(module)
                    x = torch.randn(2, 64, requires_grad=True)
                    base(x).sum().backward()
                self.assertTrue(base.is_ostris_quantized)
                self.assertNotIn('weight', dict(base.named_parameters()))
                self.assertGreater(module.hada_w2_b.grad.abs().sum().item(), 0)
                for name, buf in base.named_buffers():
                    torch.testing.assert_close(buf, original_buffers[name])

    def _qwen_network(self, dora=False):
        # The model registry imports OmniGen's Triton autotuner, which queries
        # CUDA at import time even though these Qwen fixtures run entirely on
        # CPU. Supply import-time warp metadata and bypass GPU autotuner setup
        # on CPU runners; no Triton kernels are executed by these fixtures.
        if not torch.cuda.is_available():
            with patch('torch.cuda.current_device', return_value=0), patch(
                    'torch.cuda.get_device_properties', return_value=types.SimpleNamespace(warp_size=32)), patch(
                    'triton.autotune', side_effect=lambda *args, **kwargs: lambda function: function):
                from testing import test_qwen_image_2_training_adapter as qwen_fixture
        else:
            from testing import test_qwen_image_2_training_adapter as qwen_fixture
        model = qwen_fixture.QwenImage2TrainingAdapterTests().make_model()
        model.model = qwen_fixture.QwenImage21Transformer2DModel()
        model.model.transformer_blocks[0].attn.to_q.weight.data.normal_()
        model.model.requires_grad_(False)
        config = NetworkConfig(type='loha', linear=2, linear_alpha=0.5, loha_dora=dora)
        network = LoRASpecialNetwork([], model.model, lora_dim=2, alpha=0.5,
                                    train_unet=True, train_text_encoder=False,
                                    network_type='loha', network_config=config,
                                    transformer_only=True, is_transformer=True,
                                    target_lin_modules=model.target_lora_modules, base_model=model)
        network.apply_to([], model.model, apply_text_encoder=False, apply_unet=True)
        network.force_to(torch.device('cpu'), torch.float32)
        network._update_torch_multiplier()
        network.is_active = True
        return network, model

    def test_qwen_network_training_checkpoint_export_and_resume(self):
        for dora in (False, True):
            network, model = self._qwen_network(dora)
            module = network.unet_loras[0]
            _randomize(module)
            groups = network.prepare_optimizer_params(None, 0.001, 0.001)
            optimizer = torch.optim.AdamW(groups)
            x = torch.randn(3, 2)
            model.model(x).square().mean().backward()
            optimizer.step()
            self.assertIsNone(module.org_module[0].weight.grad)
            network.enable_gradient_checkpointing()
            network.merge_in()
            self.assertFalse(network.is_merged_in)
            with tempfile.TemporaryDirectory() as directory:
                filename = str(Path(directory) / 'loha.safetensors')
                network.save_weights(filename, dtype=torch.float32)
                exported = load_file(filename)
                prefix = 'diffusion_model.transformer_blocks.0.attn.to_q'
                self.assertIn(prefix + '.hada_w1_a', exported)
                self.assertEqual(exported[prefix + '.alpha'].item(), 0.5)
                self.assertEqual(prefix + '.dora_scale' in exported, dora)
                expected = model.model(x).detach()
                raw_state = {key: value.clone() for key, value in network.state_dict().items()}
                network.reset_weights()
                module._set_runtime_scale(1)
                self.assertIsNone(network.load_weights(filename))
                self.assertEqual(module.scale, 0.25)
                torch.testing.assert_close(model.model(x), expected)
                network.reset_weights()
                self.assertIsNone(network.load_weights(raw_state))
                torch.testing.assert_close(model.model(x), expected)

    def test_plain_checkpoint_can_initialize_doha_but_doha_cannot_silently_drop_magnitude(self):
        network, model = self._qwen_network(False)
        module = network.unet_loras[0]
        _randomize(module)
        plain = network.get_state_dict(dtype=torch.float32)
        with patch.object(module, 'use_dora', True):
            module.magnitude = torch.nn.Parameter(module._weight_norms(False))
            network.load_weights(plain)
            x = torch.randn(2, 2)
            torch.testing.assert_close(model.model(x), F.linear(x, module.org_module[0].weight + _delta(module)))
            doha = network.get_state_dict(dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'enable network.loha_dora'):
            network.load_weights(doha)
        with self.assertRaisesRegex(ValueError, 'No matching LoHa'):
            network.load_weights({'transformer.wrong.lora_A.weight': torch.zeros(2, 2)})

    def test_export_matches_real_comfy_adapter_at_signed_strengths_linear_and_conv(self):
        adapter_class = _comfy_adapter()
        if adapter_class is None:
            self.skipTest('Local ComfyUI not available for export compatibility test')
        for dora in (False, True):
            for conv in (False, True):
                base = torch.nn.Conv2d(2, 5, 3, padding=1) if conv else torch.nn.Linear(7, 5)
                module, base, network = self._module(base, dora=dora, dtype=torch.float32)
                _randomize(module)
                weights = {'test.' + key: value.detach().clone() for key, value in module.state_dict().items()}
                dora_scale = module.comfy_magnitude() if dora else None
                adapter = adapter_class.load('test', weights, module.alpha.item(), dora_scale)
                x = torch.randn(2, 2, 8, 8) if conv else torch.randn(2, 3, 7)
                for strength in (-1.0, 0.0, 0.5, 1.0, 2.0):
                    network._multiplier = strength
                    network.torch_multiplier = torch.tensor([strength])
                    patched = adapter.calculate_weight(base.weight.detach().clone(), 'test', strength,
                                                       1.0, None, lambda value: value)
                    expected = F.conv2d(x, patched, base.bias, padding=1) if conv else F.linear(x, patched, base.bias)
                    torch.testing.assert_close(base(x), expected, atol=3e-6, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
