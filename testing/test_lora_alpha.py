import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"proj": torch.nn.Linear(4, 4, bias=False)})]
        )


class OstrisLinear(torch.nn.Module):
    """Quantized-linear stand-in whose weight must not be materialized."""

    is_ostris_quantized = True

    def __init__(self):
        super().__init__()
        self.in_features = 4
        self.out_features = 4
        self.bias = None

    @property
    def weight(self):
        raise AssertionError("ordinary LoRA discovery dequantized the base weight")


class FakeQuantizedTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"proj": OstrisLinear()})]
        )


class FakeBaseModel:
    arch = "fake_transformer"
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return ["transformer_blocks"]

    @staticmethod
    def convert_lora_weights_before_save(state_dict):
        return {
            key.replace("transformer.", "diffusion_model."): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def convert_lora_weights_before_load(state_dict):
        return {
            key.replace("diffusion_model.", "transformer."): value
            for key, value in state_dict.items()
        }


class LoraAlphaTests(unittest.TestCase):
    def test_quantized_lora_discovery_does_not_materialize_base_weight(self):
        network = LoRASpecialNetwork(
            text_encoder=[],
            unet=FakeQuantizedTransformer(),
            lora_dim=4,
            alpha=4,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["FakeQuantizedTransformer"],
            network_config=NetworkConfig(linear=4, linear_alpha=4),
            is_transformer=True,
            is_assistant_adapter=True,
            base_model=FakeBaseModel(),
        )

        self.assertEqual(len(network.unet_loras), 1)

    def test_peft_assistant_adapter_preserves_explicit_alpha(self):
        network = LoRASpecialNetwork(
            text_encoder=[],
            unet=FakeTransformer(),
            lora_dim=64,
            alpha=8,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["FakeTransformer"],
            network_config=NetworkConfig(linear=64, linear_alpha=8),
            is_transformer=True,
            is_assistant_adapter=True,
            base_model=FakeBaseModel(),
        )

        self.assertEqual(len(network.unet_loras), 1)
        self.assertEqual(network.unet_loras[0].alpha.item(), 8)
        self.assertEqual(network.unet_loras[0].scale, 0.125)

    def test_peft_training_network_preserves_explicit_alpha(self):
        base_model = FakeBaseModel()
        network = LoRASpecialNetwork(
            text_encoder=[],
            unet=FakeTransformer(),
            lora_dim=64,
            alpha=8,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["FakeTransformer"],
            network_config=NetworkConfig(linear=64, linear_alpha=8),
            is_transformer=True,
            is_assistant_adapter=False,
            base_model=base_model,
        )
        network.apply_to(None, None, False, True)

        self.assertEqual(len(network.unet_loras), 1)
        self.assertEqual(network.unet_loras[0].alpha.item(), 8)
        self.assertEqual(network.unet_loras[0].scale, 0.125)

        state = network.get_state_dict(dtype=torch.float32)
        alpha_key = "diffusion_model.transformer_blocks.0.proj.alpha"
        self.assertIn(alpha_key, state)
        self.assertEqual(state[alpha_key].item(), 8)

    def test_peft_alpha_round_trips_and_refreshes_runtime_scale(self):
        source_base_model = FakeBaseModel()
        source = LoRASpecialNetwork(
            text_encoder=[],
            unet=FakeTransformer(),
            lora_dim=64,
            alpha=8,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["FakeTransformer"],
            network_config=NetworkConfig(linear=64, linear_alpha=8),
            is_transformer=True,
            base_model=source_base_model,
        )
        source.apply_to(None, None, False, True)
        state = source.get_state_dict(dtype=torch.float32)
        target_base_model = FakeBaseModel()
        target = LoRASpecialNetwork(
            text_encoder=[],
            unet=FakeTransformer(),
            lora_dim=64,
            alpha=64,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["FakeTransformer"],
            network_config=NetworkConfig(linear=64, linear_alpha=64),
            is_transformer=True,
            base_model=target_base_model,
        )
        target.apply_to(None, None, False, True)

        target.load_weights(state)

        self.assertEqual(target.unet_loras[0].alpha.item(), 8)
        self.assertEqual(target.unet_loras[0].scale, 0.125)

    def test_old_peft_checkpoint_without_alpha_defaults_to_rank(self):
        source_base_model = FakeBaseModel()
        source = LoRASpecialNetwork(
            text_encoder=[],
            unet=FakeTransformer(),
            lora_dim=64,
            alpha=8,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["FakeTransformer"],
            network_config=NetworkConfig(linear=64, linear_alpha=8),
            is_transformer=True,
            base_model=source_base_model,
        )
        source.apply_to(None, None, False, True)
        state = {
            key: value
            for key, value in source.get_state_dict(dtype=torch.float32).items()
            if not key.endswith(".alpha")
        }
        target_base_model = FakeBaseModel()
        target = LoRASpecialNetwork(
            text_encoder=[],
            unet=FakeTransformer(),
            lora_dim=64,
            alpha=8,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["FakeTransformer"],
            network_config=NetworkConfig(linear=64, linear_alpha=8),
            is_transformer=True,
            base_model=target_base_model,
        )
        target.apply_to(None, None, False, True)

        target.load_weights(state)

        self.assertEqual(target.unet_loras[0].alpha.item(), 64)
        self.assertEqual(target.unet_loras[0].scale, 1.0)


if __name__ == "__main__":
    unittest.main()
