import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
    Ideogram4Config,
    Ideogram4Transformer2DModel,
)
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.DoRA import DoRAModule


class FakeIdeogram4BaseModel:
    arch = "ideogram4"
    is_transformer = True
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        return Ideogram4Model.convert_lora_weights_before_save(self, state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        return Ideogram4Model.convert_lora_weights_before_load(self, state_dict)


def make_tiny_ideogram4_transformer():
    return Ideogram4Transformer2DModel(
        Ideogram4Config(
            emb_dim=12,
            num_layers=1,
            num_heads=3,
            intermediate_size=16,
            adanln_dim=4,
            in_channels=4,
            llm_features_dim=8,
            mrope_section=(1, 1, 0),
        )
    )


class Ideogram4DoRATests(unittest.TestCase):
    def test_ideogram4_dora_targets_transformer_layers(self):
        base_model = FakeIdeogram4BaseModel()
        network = LoRASpecialNetwork(
            text_encoder=[],
            unet=make_tiny_ideogram4_transformer(),
            lora_dim=2,
            alpha=2,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["Ideogram4Transformer2DModel"],
            network_config=NetworkConfig(
                type="dora",
                linear=2,
                linear_alpha=2,
                transformer_only=True,
            ),
            network_type="dora",
            transformer_only=True,
            is_transformer=True,
            base_model=base_model,
        )

        self.assertGreater(len(network.unet_loras), 0)
        self.assertTrue(all(isinstance(lora, DoRAModule) for lora in network.unet_loras))
        self.assertTrue(all("layers" in lora.lora_name for lora in network.unet_loras))

    def test_ideogram4_dora_state_dict_saves_and_loads_magnitude_keys(self):
        transformer = make_tiny_ideogram4_transformer()
        base_model = FakeIdeogram4BaseModel()
        network = LoRASpecialNetwork(
            text_encoder=[],
            unet=transformer,
            lora_dim=2,
            alpha=2,
            train_text_encoder=False,
            train_unet=True,
            target_lin_modules=["Ideogram4Transformer2DModel"],
            network_config=NetworkConfig(type="dora", linear=2, linear_alpha=2),
            network_type="dora",
            is_transformer=True,
            base_model=base_model,
        )
        network.apply_to(None, transformer, False, True)

        save_dict = network.get_state_dict(dtype=torch.float32)

        self.assertIn(
            "diffusion_model.layers.0.attention.qkv.magnitude",
            save_dict,
        )
        extra = network.load_weights(save_dict)
        self.assertIsNone(extra)

    def test_ideogram4_model_returns_transformer_for_training(self):
        model = Ideogram4Model.__new__(Ideogram4Model)
        transformer = make_tiny_ideogram4_transformer()
        model.model = transformer

        self.assertIs(model.get_model_to_train(), transformer)


if __name__ == "__main__":
    unittest.main()
