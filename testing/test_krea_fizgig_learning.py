"""CPU learning probes through real Krea, adapter and Fizgig implementations."""
from types import SimpleNamespace
import unittest

import torch

from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleMMDiTConfig, SingleStreamDiT
from extensions_built_in.sd_trainer.FizgigSliderTrainer import FizgigSliderTrainer
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.flow_training import FlowTrainingProfile
from toolkit.lora_special import LoRASpecialNetwork


class KreaFizgigLearningTests(unittest.TestCase):
    def fixture(self, mode, network_type, checkpointing=True, quantized=False):
        torch.manual_seed(123)
        sd = Krea2Model('cpu', ModelConfig(arch='krea2', name_or_path='unused'), dtype='fp32')
        sd.model = SingleStreamDiT(SingleMMDiTConfig(
            features=32, tdim=16, txtdim=8, heads=4, multiplier=2,
            layers=2, patch=2, channels=4, kvheads=2, txtlayers=2,
            txtheads=2, txtkvheads=2,
        ))
        sd.model.requires_grad_(False)
        sd.model.gradient_checkpointing = checkpointing
        if quantized:
            from toolkit.util.quantize import quantize_model
            sd.model_config.quantize = True
            sd.model_config.qtype = 'qfloat8'
            quantize_model(sd, sd.model)
        # BaseSDTrainProcess freezes the loaded/quantized backbone before
        # attaching trainable adapters. Quantization can replace Parameters.
        sd.model.requires_grad_(False)
        sd.noise_scheduler = sd.get_train_scheduler()
        network = LoRASpecialNetwork(
            text_encoder=None, unet=sd.model, lora_dim=2, alpha=1,
            train_unet=True, train_text_encoder=False, is_transformer=True,
            transformer_only=True, target_lin_modules=['SingleStreamDiT'],
            base_model=sd,
            network_type=network_type, network_config=NetworkConfig(type=network_type, linear=2, linear_alpha=1),
        )
        network.apply_to(None, sd.model, apply_text_encoder=False, apply_unet=True)
        network.requires_grad_(True)
        network._update_torch_multiplier()
        network.signed_dora_slider = network_type == 'dora'
        sd.network = network
        trainer = object.__new__(FizgigSliderTrainer)
        trainer.network, trainer.sd = network, sd
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=0, max_denoising_steps=999)
        trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': 'krea2'})
        trainer.slider_mode = mode
        trainer.slider_diff_weight, trainer.slider_cfg, trainer.slider_guidance = 1, 1, 3
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        embeds = [AdvancedPromptEmbeds(text_embeds=[torch.randn(3, 16)]) for _ in range(3)]
        clean = torch.randn(1, 4, 4, 4)
        if mode == 'prompt_pairs':
            trainer.slider_bank = [(clean, 0)]
            trainer.slider_embeds = [tuple(embeds)]
            trainer.slider_cfg_negative_embeds = None
            batch = None
        else:
            trainer._pair_key = lambda item: 'pair'
            trainer.negative_latents = {'pair': clean[0] + .5}
            batch = SimpleNamespace(latents=clean, prompt_embeds=embeds[0], file_items=[object()])
        return trainer, batch, clean, embeds[0]

    def test_both_objectives_learn_with_lora_dora_and_float8_base(self):
        for mode in ('prompt_pairs', 'image_pairs'):
            for kind in ('lora', 'dora'):
                for quantized in (False, True):
                    with self.subTest(mode=mode, network=kind, quantized=quantized):
                        trainer, batch, clean, embed = self.fixture(mode, kind, quantized=quantized)
                        network = trainer.network
                        optimizer = torch.optim.AdamW(network.parameters(), lr=.003)
                        losses = []
                        for _ in range(12):
                            optimizer.zero_grad()
                            # Fixed noise/time isolates learning from sampling variance.
                            torch.manual_seed(456)
                            loss = trainer.train_single_accumulation(batch)
                            losses.append(loss.item())
                            grads = [p.grad for p in network.parameters() if p.grad is not None]
                            self.assertTrue(grads)
                            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                            optimizer.step()
                        self.assertLess(losses[-1], losses[0])
                        self.assertTrue(all(p.grad is None for p in trainer.sd.model.parameters()))
                        network.is_active = True
                        with torch.no_grad():
                            network.multiplier = -1
                            minus = trainer._predict(clean, torch.tensor([500.]), embed)
                            network.multiplier = 1
                            plus = trainer._predict(clean, torch.tensor([500.]), embed)
                        self.assertGreater((plus - minus).abs().max().item(), 1e-5)

    def test_checkpoint_recomputation_preserves_signed_gradients(self):
        for mode in ('prompt_pairs', 'image_pairs'):
            for kind in ('lora', 'dora'):
                results = []
                for checkpointing in (False, True):
                    trainer, batch, _, _ = self.fixture(mode, kind, checkpointing)
                    torch.manual_seed(456)
                    loss = trainer.train_single_accumulation(batch)
                    results.append((loss, {n: p.grad for n, p in trainer.network.named_parameters() if p.grad is not None}))
                with self.subTest(mode=mode, network=kind):
                    torch.testing.assert_close(results[0][0], results[1][0])
                    self.assertEqual(results[0][1].keys(), results[1][1].keys())
                    for name in results[0][1]:
                        torch.testing.assert_close(results[0][1][name], results[1][1][name])
