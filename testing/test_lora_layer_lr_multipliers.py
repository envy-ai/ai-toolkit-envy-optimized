import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.DoRA import DoRAModule


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Linear(4, 4, bias=False)
        self.ff = torch.nn.Linear(4, 4, bias=False)


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])


class FakeBaseModel:
    arch = "fake_transformer"
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return ["blocks"]


def make_network(**kwargs):
    return LoRASpecialNetwork(
        text_encoder=[],
        unet=FakeTransformer(),
        lora_dim=2,
        alpha=2,
        train_text_encoder=False,
        train_unet=True,
        target_lin_modules=["FakeTransformer"],
        network_config=NetworkConfig(
            type=kwargs.get("network_type", "lora"),
            linear=2,
            linear_alpha=2,
        ),
        is_transformer=True,
        base_model=FakeBaseModel(),
        **kwargs,
    )


class LayerLrMultiplierTests(unittest.TestCase):
    def test_layer_lr_multipliers_split_unet_optimizer_groups_by_matched_name(self):
        network = make_network(
            layer_lr_multipliers=[
                {"match": "transformer.blocks.0.", "multiplier": 0.1},
                {"match": "transformer.blocks.1.ff", "multiplier": 0.25},
            ]
        )

        params = network.prepare_optimizer_params(
            text_encoder_lr=1.0,
            unet_lr=1.0,
            default_lr=1.0,
        )

        lr_to_num_tensors = {
            group["lr"]: sum(1 for _ in group["params"]) for group in params
        }
        self.assertEqual(lr_to_num_tensors[0.1], 4)
        self.assertEqual(lr_to_num_tensors[0.25], 2)
        self.assertEqual(lr_to_num_tensors[1.0], 2)

    def test_layer_lr_multipliers_work_for_dora_modules(self):
        network = make_network(
            network_type="dora",
            layer_lr_multipliers={
                "transformer.blocks.1.": 0.05,
            },
        )

        self.assertTrue(all(isinstance(lora, DoRAModule) for lora in network.unet_loras))

        params = network.prepare_optimizer_params(
            text_encoder_lr=1.0,
            unet_lr=1.0,
            default_lr=1.0,
        )

        lr_to_num_tensors = {
            group["lr"]: sum(1 for _ in group["params"]) for group in params
        }
        self.assertEqual(lr_to_num_tensors[0.05], 6)
        self.assertEqual(lr_to_num_tensors[1.0], 6)


if __name__ == "__main__":
    unittest.main()
