"""CPU diagnostics for static sequence/RoPE/attention and signed-DoRA gradients.

This does not qualify XLA kernels, full-model memory, or TPU execution.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from workers.kaggle_tpu.export_paired_cache import qwen_module
QwenImage21Transformer2DModel = qwen_module("src.transformer").QwenImage21Transformer2DModel
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from workers.kaggle_tpu.static_qwen import StaticQwen


def run():
    torch.manual_seed(1701)
    for checkpoint_blocks in (False, True):
        for padded in (False, True):
            original = QwenImage21Transformer2DModel(in_channels=4, out_channels=4, num_layers=2,
                                                    attention_head_dim=8, num_attention_heads=2,
                                                    context_in_dim=16, axes_dims_rope=(2, 2, 4))
            adapted = copy.deepcopy(original)
            networks = []
            for model in (original, adapted):
                model.requires_grad_(False)
                config = NetworkConfig(type="dora", linear=2, linear_alpha=2)
                network = LoRASpecialNetwork(text_encoder=[], unet=model, lora_dim=2, alpha=2,
                                            network_type="dora", network_config=config,
                                            train_unet=True, train_text_encoder=False,
                                            is_transformer=True,
                                            target_lin_modules=["QwenImage21Transformer2DModel"])
                network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
                network.signed_dora_slider, network.is_active = True, True
                network._update_torch_multiplier()
                networks.append(network)
            with torch.no_grad():
                for module in networks[0].get_all_modules():
                    module.lora_up.weight.normal_(std=0.01)
                    module.magnitude.mul_(1.01)
            networks[1].load_state_dict(networks[0].state_dict(), strict=True)
            latents = torch.randn(1, 4, 2, 2)
            prompt = torch.randn(1, 3, 16)
            mask = torch.tensor([[True, True, not padded]])
            static = StaticQwen(adapted, mask, 2, 2, checkpoint_blocks=checkpoint_blocks)
            for strength in (1.0, -1.0):
                for network in networks:
                    network.zero_grad(set_to_none=True)
                    network.multiplier = strength
                expected = original(hidden_states=latents.flatten(2).transpose(1, 2),
                                    encoder_hidden_states=prompt, timestep=torch.tensor([0.4]),
                                    img_shapes=[[(1, 2, 2)]], img_mask=torch.tensor([[False, False, False, True]]),
                                    encoder_hidden_states_mask=mask, return_dict=False)[0][:, -4:]
                expected = expected.transpose(1, 2).reshape_as(latents)
                actual = static(latents, torch.tensor([0.4]), prompt)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                expected.square().mean().backward()
                actual.square().mean().backward()
                for (name, first), (other_name, second) in zip(networks[0].named_parameters(), networks[1].named_parameters()):
                    assert name == other_name and first.grad is not None and second.grad is not None
                    torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-4)
    print("PASS: static Qwen matches original outputs and signed DoRA gradients for both strengths, padded/unpadded prompts, checkpointed/uncheckpointed blocks")


if __name__ == "__main__":
    run()
