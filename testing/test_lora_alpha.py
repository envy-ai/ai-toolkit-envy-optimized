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


class FakeBaseModel:
    arch = "fake_transformer"
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return ["transformer_blocks"]


class LoraAlphaTests(unittest.TestCase):
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

    def test_peft_training_network_keeps_existing_no_alpha_behavior(self):
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
            base_model=FakeBaseModel(),
        )

        self.assertEqual(len(network.unet_loras), 1)
        self.assertEqual(network.unet_loras[0].alpha.item(), 64)
        self.assertEqual(network.unet_loras[0].scale, 1.0)


if __name__ == "__main__":
    unittest.main()
