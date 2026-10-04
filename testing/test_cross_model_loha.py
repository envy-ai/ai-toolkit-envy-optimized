"""Native target discovery, file roundtrips and installed Comfy loader parity.

Real tiny transformer/conditioner classes; no downloads or CUDA initialization.
Comfy's generic native-prefix mapping and load_lora dispatch run unchanged.
Only unrelated GPU/model-family imports are stubbed.
"""
import importlib.util
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from extensions_built_in.diffusion_models.anima.anima import AnimaModel, AnimaTrainableModel
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleStreamDiT, SingleMMDiTConfig
from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
from extensions_built_in.diffusion_models.ideogram4.src.transformer import Ideogram4Config, Ideogram4Transformer2DModel
from toolkit.models.v2.diffusion_models.cosmos import CosmosTransformer3DModel
from toolkit.models.v2.text_encoders.anima import AnimaTextConditioner
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from extensions_built_in.diffusion_models.anima.anima import AnimaPromptEmbeds
from toolkit.memory_management import MemoryManager
from testing.test_ideogram4_dora import make_tiny_ideogram4_transformer
from testing.test_loha import _comfy_adapter, _randomize


def installed_comfy_loader():
    adapter = _comfy_adapter()
    path = Path('/home/bart/ComfyUI/comfy/lora.py')
    if adapter is None or not path.is_file():
        return None
    comfy = types.ModuleType('comfy')
    modules = {'comfy': comfy}
    for name in ('memory_management', 'utils', 'model_management', 'model_base', 'weight_adapter'):
        module = types.ModuleType('comfy.' + name)
        setattr(comfy, name, module)
        modules[module.__name__] = module
    comfy.weight_adapter.adapters = [adapter]
    # These mappings describe legacy Diffusers UNets, not the native-prefix
    # Krea/Ideogram/Anima keys verified here. No selected generic map is mocked.
    comfy.utils.unet_to_diffusers = lambda config: {}
    for name in set(re.findall(r'comfy\.model_base\.(\w+)', path.read_text())):
        setattr(comfy.model_base, name, type(name, (), {}))
    spec = importlib.util.spec_from_file_location('_cross_model_comfy_lora', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class CrossModelLoHaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(16)
        torch.set_num_threads(2)

    def fixture(self, arch, dora, conditioner=False, quantizer=None, offload=False):
        if arch == 'krea2':
            model = Krea2Model.__new__(Krea2Model)
            model.model = SingleStreamDiT(SingleMMDiTConfig(features=32, tdim=16, txtdim=8,
                heads=4, multiplier=2, layers=1, patch=2, channels=4, kvheads=2,
                txtlayers=2, txtheads=2, txtkvheads=2))
            targets = ['SingleStreamDiT']
        elif arch == 'ideogram4':
            model = Ideogram4Model.__new__(Ideogram4Model)
            model.model = make_tiny_ideogram4_transformer() if quantizer is None and not offload else Ideogram4Transformer2DModel(
                Ideogram4Config(emb_dim=32, num_layers=1, num_heads=4, intermediate_size=64,
                    adanln_dim=16, in_channels=4, llm_features_dim=32, mrope_section=(2, 1, 1)))
            targets = ['Ideogram4Transformer2DModel']
        else:
            model = AnimaModel.__new__(AnimaModel)
            model.train_text_conditioner = conditioner
            transformer = CosmosTransformer3DModel(in_channels=4, out_channels=4,
                num_attention_heads=2, attention_head_dim=16, num_layers=1,
                mlp_ratio=2, text_embed_dim=8, adaln_lora_dim=4,
                max_size=(2, 8, 8), extra_pos_embed_type=None)
            text = AnimaTextConditioner(source_dim=8, target_dim=8, model_dim=8,
                num_layers=1, num_attention_heads=2, target_vocab_size=16, min_sequence_length=2)
            model.model = AnimaTrainableModel(transformer, text)
            targets = ['CosmosTransformer3DModel'] + (['AnimaTextConditioner'] if conditioner else [])
        model.use_old_lokr_format = False
        model.model.requires_grad_(False)
        if quantizer is not None:
            from toolkit.util.ostris_quant import convert_linear_to_ostris
            for base in list(model.model.modules()):
                if isinstance(base, torch.nn.Linear) and quantizer.can_quantize(base):
                    self.assertTrue(convert_linear_to_ostris(base, quantizer))
        if offload:
            # Match actual startup: frozen component streaming is attached
            # before the adapter hooks and does not own the trainable factors.
            MemoryManager.attach(model.model, torch.device('cpu'), offload_percent=1.)
        config = NetworkConfig(type='loha', linear=2, linear_alpha=.5, loha_dora=dora,
                               transformer_only=True)
        network = LoRASpecialNetwork([], model.model, lora_dim=2, alpha=.5,
            train_unet=True, train_text_encoder=False, target_lin_modules=targets,
            network_type='loha', network_config=config, transformer_only=True,
            is_transformer=True, base_model=model)
        network.apply_to([], model.model, False, True)
        network.force_to(torch.device('cpu'), torch.float32)
        network._update_torch_multiplier()
        network.is_active = True
        return model, network

    def test_native_checkpointed_predictions_with_real_memory_manager_and_pretrained_reload(self):
        from toolkit.util.convrot_quant import ConvRotInt8Quantizer, ConvRotIntNQuantizer

        for arch, conditioner in [('krea2', False), ('anima', False), ('anima', True), ('ideogram4', False)]:
            for dora in (False, True):
                for bits in (None, 4, 8):
                    quantizer = (None if bits is None else ConvRotInt8Quantizer(rot_size=16)
                                 if bits == 8 else ConvRotIntNQuantizer(4, rot_size=16))
                    with self.subTest(arch=arch, dora=dora, bits=bits, conditioner=conditioner):
                        model, network = self.fixture(arch, dora, conditioner, quantizer, offload=True)
                        model.device_torch = torch.device('cpu')
                        model.torch_dtype = torch.float32
                        model.noise_scheduler = model.get_train_scheduler()
                        model.model_config = SimpleNamespace(model_kwargs={})
                        model.is_edit = model.kv_cache = model.is_multistage = False
                        model.use_old_lokr_format = False
                        transformer = model.model.transformer if arch == 'anima' else model.model
                        transformer.enable_gradient_checkpointing()
                        model.model.train()
                        if arch == 'krea2':
                            prompt = AdvancedPromptEmbeds(text_embeds=[torch.randn(2, 16), torch.randn(3, 16)])
                        elif arch == 'anima':
                            prompt = AnimaPromptEmbeds(torch.randn(2, 3, 8), torch.arange(3).expand(2, -1),
                                torch.tensor([[1, 1, 0], [1, 1, 1]]), torch.ones(2, 3, dtype=torch.long))
                        else:
                            dim = transformer.config.llm_features_dim
                            prompt = AdvancedPromptEmbeds(text_embeds=[torch.randn(2, dim), torch.randn(3, dim)])
                        latent = torch.randn(2, 4, 4, 6, requires_grad=True)
                        times = torch.tensor([250., 500.])
                        def predict():
                            return model.predict_noise(latents=latent, timestep=times,
                                conditional_embeddings=prompt, unconditional_embeddings=None,
                                guidance_scale=1., guidance_embedding_scale=1., batch=None)
                        # No-op inference must compare the SAME numerical path.
                        # CPU ConvRot deliberately uses fake-quantized activations
                        # in STE training, but WnA16 in its no-hardware inference.
                        with torch.no_grad():
                            network.is_active = False
                            baseline = predict()
                            network.is_active = True
                            torch.testing.assert_close(predict(), baseline)
                        for module in network.unet_loras:
                            _randomize(module)
                        managed = [base for base in model.model.modules() if hasattr(base, '_layer_memory_manager')]
                        self.assertTrue(managed)
                        frozen_parameters = {id(param): param.detach().clone() for param in model.model.parameters()}
                        frozen_buffers = {name: tensor.clone() for name, tensor in model.model.named_buffers()}
                        params = [param for param in network.parameters() if param.requires_grad]
                        prediction = predict()
                        prediction.square().mean().backward()
                        actual = [None if param.grad is None else param.grad.clone() for param in [latent, *params]]
                        self.assertGreater(sum(grad.abs().sum().item() for grad in actual[1:] if grad is not None), 0)
                        # BaseModel.predict_noise constructs/scales noisy inputs
                        # under no_grad in ordinary training; adapter factors,
                        # not the input image latents, are optimized here.
                        self.assertIsNone(actual[0])
                        self.assertTrue(all(param.grad is None for param in model.model.parameters()))
                        self.assertTrue(all(param.device.type == 'cpu' for param in model.model.parameters()))
                        self.assertTrue(all(param.device.type == 'cpu' for param in params))
                        # Test numerical parity through the exact manager/wrapper,
                        # not merely isolated native projection layers.
                        network.zero_grad(set_to_none=True)
                        latent.grad = None
                        transformer.disable_gradient_checkpointing()
                        expected = predict()
                        torch.testing.assert_close(prediction, expected, atol=3e-5, rtol=3e-4)
                        expected.square().mean().backward()
                        for param, grad in zip([latent, *params], actual):
                            if grad is None:
                                self.assertIsNone(param.grad)
                            else:
                                torch.testing.assert_close(param.grad, grad, atol=3e-5, rtol=3e-4)
                        for param in model.model.parameters():
                            torch.testing.assert_close(param, frozen_parameters[id(param)])
                            self.assertIsNone(param.grad)
                        for name, tensor in model.model.named_buffers():
                            torch.testing.assert_close(tensor, frozen_buffers[name])
                        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
                            path = str(Path(directory) / 'pretrained.safetensors')
                            network.save_weights(path, dtype=torch.float32)
                            before = []
                            for strength in (-1., 0., .5, 1.):
                                network.multiplier = strength
                                before.append(predict())
                            network.reset_weights()
                            self.assertIsNone(network.load_weights(path))
                            for strength, value in zip((-1., 0., .5, 1.), before):
                                network.multiplier = strength
                                torch.testing.assert_close(predict(), value, atol=3e-5, rtol=3e-4)
                        self.assertTrue(all(hasattr(base, '_layer_memory_manager') for base in managed))
                        # Detaching a component after training must restore the
                        # base callable without losing its already-applied LoHa.
                        with torch.no_grad():
                            value = predict()
                            MemoryManager.detach(model.model)
                            self.assertFalse(hasattr(model.model, '_memory_manager'))
                            torch.testing.assert_close(predict(), value, atol=3e-5, rtol=3e-4)

    def test_real_architecture_targets_noop_gradients_and_frozen_conditioner(self):
        for arch in ('krea2', 'anima', 'ideogram4'):
            for dora in (False, True):
                with self.subTest(arch=arch, dora=dora):
                    model, network = self.fixture(arch, dora)
                    self.assertTrue(network.unet_loras)
                    for module in network.unet_loras:
                        base = module.org_module[0]
                        self.assertIsInstance(base, torch.nn.Linear)
                        x = torch.randn(2, base.in_features)
                        torch.testing.assert_close(base(x), F.linear(x, base.weight, base.bias))
                        base(x).square().mean().backward()
                        self.assertIsNone(base.weight.grad)
                        self.assertIsNotNone(module.hada_w2_b.grad)
                    self.assertTrue(any(m.hada_w2_b.grad.abs().sum() > 0 for m in network.unet_loras))
                    if arch == 'anima':
                        self.assertTrue(all('transformer_blocks' in m.lora_name for m in network.unet_loras))
                        self.assertTrue(all(p.grad is None for p in model.trainable_model.text_conditioner.parameters()))

    def test_actual_targets_convrot_int4_int8_checkpointed_factor_gradients(self):
        from torch.utils.checkpoint import checkpoint
        from toolkit.util.convrot_quant import ConvRotInt8Quantizer, ConvRotIntNQuantizer
        for arch in ('krea2', 'anima', 'ideogram4'):
            for dora in (False, True):
                for quantizer in (ConvRotInt8Quantizer(rot_size=16), ConvRotIntNQuantizer(4, rot_size=16)):
                    with self.subTest(arch=arch, dora=dora, quantizer=type(quantizer).__name__):
                        model, network = self.fixture(arch, dora, quantizer=quantizer)
                        modules = [module for module in network.unet_loras
                                   if getattr(module.org_module[0], 'is_ostris_quantized', False)]
                        self.assertTrue(modules, 'No native transformer targets were quantized')
                        module = modules[0]
                        _randomize(module)
                        base = module.org_module[0]
                        buffers = {key: value.clone() for key, value in base.named_buffers()}
                        x = torch.randn(2, base.in_features, requires_grad=True)
                        checkpoint(base, x, use_reentrant=False).square().mean().backward()
                        self.assertTrue(base.is_ostris_quantized)
                        self.assertNotIn('weight', dict(base.named_parameters()))
                        for name, value in base.named_buffers():
                            torch.testing.assert_close(value, buffers[name])
                        for factor in ('hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b'):
                            self.assertIsNotNone(getattr(module, factor).grad)
                            self.assertTrue(torch.isfinite(getattr(module, factor).grad).all())
                        self.assertGreater(module.hada_w2_b.grad.abs().sum().item(), 0)
                        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_native_exports_load_through_comfy_and_resume_at_signed_strengths(self):
        loader = installed_comfy_loader()
        if loader is None:
            self.skipTest('Installed Comfy loader unavailable')
        for arch, conditioner in [('krea2', False), ('anima', False), ('anima', True), ('ideogram4', False)]:
            for dora in (False, True):
                with self.subTest(arch=arch, dora=dora, conditioner=conditioner):
                    model, network = self.fixture(arch, dora, conditioner)
                    for module in network.unet_loras:
                        _randomize(module)
                    # Use the actual model-specific converter on BASE layer
                    # keys, as opposed to deriving a map from exported factors.
                    base_state = {'transformer.' + key: tensor.detach().clone()
                                  for key, tensor in model.model.state_dict().items()}
                    comfy_state = model.convert_lora_weights_before_save(base_state)
                    fake_comfy_model = types.SimpleNamespace(state_dict=lambda: comfy_state,
                        model_config=types.SimpleNamespace(unet_config={}))
                    key_map = loader.model_lora_keys_unet(fake_comfy_model, {})
                    with tempfile.TemporaryDirectory() as directory:
                        path = str(Path(directory) / 'adapter.safetensors')
                        network.save_weights(path, dtype=torch.float32)
                        exported = load_file(path)
                        with patch.object(loader.logging, 'warning') as warning:
                            patches = loader.load_lora(exported, key_map)
                        warning.assert_not_called()  # Every factor/alpha/magnitude was consumed.
                        self.assertEqual(len(patches), len(network.unet_loras))
                        samples = []
                        for module in network.unet_loras:
                            internal = module.lora_name.replace('$$', '.')
                            public = next(iter(model.convert_lora_weights_before_save({internal: None})))
                            self.assertTrue(public.startswith('diffusion_model.'))
                            self.assertIn(public + '.hada_w1_a', exported)
                            self.assertEqual(exported[public + '.alpha'].item(), .5)
                            self.assertEqual(public + '.dora_scale' in exported, dora)
                            base = module.org_module[0]
                            x = torch.randn(2, base.in_features)
                            for strength in (-1., 0., .5, 1.):
                                network.multiplier = strength
                                patched = patches[public + '.weight'].calculate_weight(
                                    base.weight.detach().clone(), public, strength, 1., None, lambda value: value)
                                torch.testing.assert_close(base(x), F.linear(x, patched, base.bias), atol=3e-6, rtol=3e-5)
                            samples.append((base, x, base(x).detach().clone()))
                        network.reset_weights()
                        self.assertIsNone(network.load_weights(path))
                        for base, x, expected in samples:
                            torch.testing.assert_close(base(x), expected, atol=3e-6, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
