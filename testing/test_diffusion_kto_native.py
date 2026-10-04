"""KTO through real tiny native transformers, wrappers and ordinary LoRAs.

CPU-only, no checkpoint downloads. Complements scalar stochastic replay fixtures;
this tests native packing/conditioning/time/velocity and adapter gradients.
"""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file

from testing import test_diffusion_kto as kto_fixtures
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleStreamDiT, SingleMMDiTConfig
from extensions_built_in.diffusion_models.anima.anima import AnimaModel, AnimaTrainableModel, AnimaPromptEmbeds
from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
from extensions_built_in.diffusion_models.ideogram4.src.transformer import Ideogram4Config, Ideogram4Transformer2DModel
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel
from toolkit.models.v2.diffusion_models.cosmos import CosmosTransformer3DModel
from toolkit.models.v2.text_encoders.anima import AnimaTextConditioner
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import NetworkConfig
from toolkit.flow_kto import flow_kto_terms, kto_reference_point
from toolkit.lora_special import LoRASpecialNetwork


class NativeDiffusionKTOTests(unittest.TestCase):
    def native_model(self, arch, network_type='lora'):
        if arch == 'krea2':
            model = Krea2Model.__new__(Krea2Model)
            model.model = SingleStreamDiT(SingleMMDiTConfig(features=32, tdim=16, txtdim=8,
                heads=4, multiplier=2, layers=1, patch=2, channels=4, kvheads=2,
                txtlayers=2, txtheads=2, txtkvheads=2))
            model.is_edit = model.kv_cache = False
            targets, prompt_dim = ['SingleStreamDiT'], 16
        elif arch == 'anima':
            model = AnimaModel.__new__(AnimaModel)
            model.train_text_conditioner = False
            transformer = CosmosTransformer3DModel(in_channels=4, out_channels=4,
                num_attention_heads=2, attention_head_dim=16, num_layers=1,
                mlp_ratio=2, text_embed_dim=8, adaln_lora_dim=4,
                max_size=(2, 8, 8), extra_pos_embed_type=None)
            conditioner = AnimaTextConditioner(source_dim=8, target_dim=8, model_dim=8,
                num_layers=1, num_attention_heads=2, target_vocab_size=16, min_sequence_length=2)
            model.model = AnimaTrainableModel(transformer, conditioner)
            targets, prompt_dim = ['CosmosTransformer3DModel'], 8
        elif arch == 'ideogram4':
            model = Ideogram4Model.__new__(Ideogram4Model)
            model.model = Ideogram4Transformer2DModel(Ideogram4Config(emb_dim=32,
                num_layers=1, num_heads=4, intermediate_size=64, adanln_dim=16,
                in_channels=4, llm_features_dim=8, mrope_section=(2, 1, 1)))
            targets, prompt_dim = ['Ideogram4Transformer2DModel'], 8
        else:
            model = QwenImage2Model.__new__(QwenImage2Model)
            model.model = QwenImage21Transformer2DModel(in_channels=4, out_channels=4,
                num_layers=1, num_attention_heads=2, attention_head_dim=16,
                context_in_dim=8, axes_dims_rope=(8, 4, 4))
            targets, prompt_dim = ['QwenImage21Transformer2DModel'], 8
        model.device_torch = torch.device('cpu')
        model.torch_dtype = torch.float32
        model.noise_scheduler = model.get_train_scheduler()
        model.is_multistage = False
        model.use_old_lokr_format = False
        model.model.requires_grad_(False)
        model.model.train()
        transformer = model.model.transformer if arch == 'anima' else model.model
        transformer.enable_gradient_checkpointing()
        network = LoRASpecialNetwork([], model.model, lora_dim=2, alpha=1,
            train_unet=True, train_text_encoder=False, target_lin_modules=targets,
            network_type=network_type, network_config=NetworkConfig(type=network_type, linear=2, linear_alpha=1),
            transformer_only=True, is_transformer=True, base_model=model)
        network.apply_to([], model.model, False, True)
        network.force_to(torch.device('cpu'), torch.float32)
        network._update_torch_multiplier()
        return model, network, prompt_dim

    def test_native_noop_scores_and_exact_optimizer_gradients_all_four(self):
        torch.set_num_threads(2)
        for arch in ('qwen_image_2', 'krea2', 'anima', 'ideogram4'):
            with self.subTest(arch=arch):
                torch.manual_seed(71)
                native, network, prompt_dim = self.native_model(arch)
                trainer, _, _ = kto_fixtures.DiffusionKTOTests().fixture(arch)
                trainer.network = network
                # REAL BaseModel prediction and architecture wrappers perform
                # prompt validation, scheduler scaling, native packing/time/sign.
                trainer.sd = native
                trainer.params = [parameter for parameter in network.parameters() if parameter.requires_grad]
                trainer.optimizer = torch.optim.SGD(trainer.params, lr=.05)
                batches = []
                for index, count in enumerate((1, 2)):
                    text = torch.randn(count, 2 + index, prompt_dim)
                    if arch == 'anima':
                        prompt = AnimaPromptEmbeds(text, torch.ones(count, 3, dtype=torch.long),
                            torch.ones(count, 2 + index, dtype=torch.long), torch.ones(count, 3, dtype=torch.long))
                    else:
                        prompt = AdvancedPromptEmbeds(text_embeds=list(text))
                        if arch == 'qwen_image_2':
                            prompt.attention_mask = [torch.ones(2 + index, dtype=torch.bool) for _ in range(count)]
                            prompt.image_slot_mask = [torch.zeros(2 + index, dtype=torch.bool) for _ in range(count)]
                    batches.append(SimpleNamespace(latents=torch.randn(count, 4, 4, 4 + index * 2),
                        prompt_embeds=prompt, file_items=[SimpleNamespace(dataset_config=SimpleNamespace(
                            kto_label='liked' if index == 0 else 'disliked')) for _ in range(count)],
                        loss_multiplier_list=[1.] * count))
                captured = []
                original_prepare = trainer._prepare_replay
                def prepare(values):
                    result = original_prepare(values)
                    captured.extend(trainer._kto_replay_queue)
                    return result
                trainer._prepare_replay = prepare
                for step in range(2):
                    captured.clear()
                    before = [parameter.detach().clone() for parameter in trainer.params]
                    trainer.hook_train_loop(batches)
                    after = [parameter.detach().clone() for parameter in trainer.params]
                    if step == 0:
                        for record in captured:
                            torch.testing.assert_close(record['policy_error'], record['reference_error'])
                    else:
                        self.assertTrue(any(not torch.equal(record['policy_error'], record['reference_error']) for record in captured))
                    self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, after)))
                    self.assertTrue(all(parameter.grad is None for parameter in native.model.parameters()))
                    # Restore adapter weights only, then evaluate a full-graph
                    # oracle on scored inputs; no time/noise resampling.
                    with torch.no_grad():
                        for parameter, value in zip(trainer.params, before):
                            parameter.copy_(value)
                    trainer.optimizer.zero_grad(set_to_none=True)
                    network.is_active, network.multiplier = True, 1.
                    policy = torch.cat([trainer._record_error(record) for record in captured])
                    reference = torch.cat([record['reference_error'] for record in captured])
                    labels = torch.cat([record['liked'] for record in captured])
                    point = kto_reference_point(reference - policy)
                    losses, _, _ = flow_kto_terms(policy, reference, labels, trainer.kto_settings, reference_point=point)
                    losses.mean().backward()
                    for parameter, value, update in zip(trainer.params, before, after):
                        grad = torch.zeros_like(value) if parameter.grad is None else parameter.grad
                        torch.testing.assert_close(update, value - .05 * grad, atol=2e-6, rtol=2e-5)
                    self.assertTrue(all(parameter.grad is None for parameter in native.model.parameters()))
                    with torch.no_grad():
                        for parameter, value in zip(trainer.params, after):
                            parameter.copy_(value)
                # Exercise actual architecture key converters and ordinary
                # safetensors export/reload at signed and fractional strengths.
                with tempfile.TemporaryDirectory() as directory, torch.no_grad():
                    path = str(Path(directory) / 'kto.safetensors')
                    network.save_weights(path, dtype=torch.float32)
                    exported = load_file(path)
                    self.assertTrue(any(value.ndim == 2 for value in exported.values()))
                    expected = []
                    for strength in (-1., 0., .5, 1.):
                        network.multiplier = strength
                        expected.append(trainer._record_error(captured[0]))
                    network.reset_weights()
                    self.assertIsNone(network.load_weights(path))
                    for strength, value in zip((-1., 0., .5, 1.), expected):
                        network.multiplier = strength
                        torch.testing.assert_close(trainer._record_error(captured[0]), value, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
