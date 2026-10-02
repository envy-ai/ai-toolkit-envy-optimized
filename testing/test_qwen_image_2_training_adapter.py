import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from toolkit.assistant_lora import load_assistant_lora_from_path
from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model


class QwenImage21Transformer2DModel(torch.nn.Module):
    """Small transformer-shaped fixture using the public Qwen LoRA key layout."""

    def __init__(self):
        super().__init__()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(2, 2, bias=False)
        block.attn.to_q.weight.data.zero_()
        self.transformer_blocks = torch.nn.ModuleList([block])

    def forward(self, value):
        return self.transformer_blocks[0].attn.to_q(value)


class QwenImage2TrainingAdapterTests(unittest.TestCase):
    def make_model(self, **config):
        model = object.__new__(QwenImage2Model)
        model.model_config = ModelConfig(
            arch='qwen_image_2', name_or_path='Comfy-Org/Qwen-Image-2.1', **config
        )
        model.torch_dtype = torch.float32
        model.device_torch = torch.device('cpu')
        model.is_transformer = True
        model.arch = 'qwen_image_2'
        model.target_lora_modules = ['QwenImage21Transformer2DModel']
        model.use_old_lokr_format = False
        model.component_load_kwargs = Mock(return_value={})
        model.print_and_status_update = Mock()
        return model

    def test_fizgig_style_adapter_loads_frozen_beside_trainable_lora(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assistant.safetensors'
            prefix = 'transformer.transformer_blocks.0.attn.to_q'
            save_file({
                prefix + '.lora_A.weight': torch.ones(1, 2),
                prefix + '.lora_B.weight': torch.ones(2, 1),
                prefix + '.alpha': torch.tensor(1.0),
            }, str(path))
            model = self.make_model()
            model.model = QwenImage21Transformer2DModel()
            model.text_encoder = []
            # Qwen's minimal pipeline does not expose transformer/text_encoder.
            model.pipeline = SimpleNamespace()
            assistant = load_assistant_lora_from_path(str(path), model, strict=True)
            inputs = torch.ones(1, 2)
            self.assertTrue(torch.equal(model.model(inputs), torch.full((1, 2), 2.0)))
            self.assertTrue(all(not param.requires_grad for param in assistant.parameters()))
            assistant.is_active = False
            self.assertTrue(torch.equal(model.model(inputs), torch.zeros(1, 2)))
            assistant.is_active = True

            network_config = NetworkConfig(linear=1, linear_alpha=1, transformer_only=True)
            trainable = LoRASpecialNetwork(
                text_encoder=[], unet=model.model, lora_dim=1, alpha=1,
                multiplier=1.0, train_unet=True, train_text_encoder=False,
                network_config=network_config, network_type='lora',
                transformer_only=True, is_transformer=True,
                target_lin_modules=model.target_lora_modules, base_model=model,
            )
            trainable.apply_to([], model.model, apply_text_encoder=False, apply_unet=True)
            trainable._update_torch_multiplier()
            trainable.is_active = True
            model.model(inputs).sum().backward()
            self.assertTrue(any(param.grad is not None for param in trainable.parameters()))
            self.assertTrue(all(param.grad is None for param in assistant.parameters()))
            self.assertEqual(len(trainable.unet_loras), 1)

    def test_load_model_uses_external_encoder_and_loads_assistant(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder_path = Path(directory) / 'qwen3vl_8b.safetensors'
            assistant_path = Path(directory) / 'assistant.safetensors'
            encoder_path.touch()
            assistant_path.touch()
            model = self.make_model(
                text_encoder_path=str(encoder_path), assistant_lora_path=str(assistant_path)
            )
            module = 'extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2'
            with patch(module + '.QwenImage21Transformer2DModel.load') as transformer_load, \
                 patch(module + '.QwenImage21TextEncoder.load_processor') as processor_load, \
                 patch(module + '.QwenImage21TextEncoder.load_model') as encoder_load, \
                 patch(module + '.AutoencoderKLQwenImage21.load'), \
                 patch(module + '.load_assistant_lora_from_path') as assistant_load, \
                 patch.object(QwenImage2Model, 'get_train_scheduler'), \
                 patch(module + '.flush'):
                model.load_model()

            encoder_load.assert_called_once_with(
                str(encoder_path), dtype=torch.float32, subfolder='text_encoder',
                config_path='Qwen/Qwen-Image-2.1',
            )
            processor_load.assert_called_once_with('Qwen/Qwen-Image-2.1')
            assistant_load.assert_called_once_with(str(assistant_path), model, strict=True)
            self.assertIs(model.assistant_lora, assistant_load.return_value)
            self.assertIs(model.model, transformer_load.return_value)

    def test_missing_paths_fail_before_loading_transformer(self):
        for config in (
            {'text_encoder_path': '/missing/qwen3vl_8b.safetensors'},
            {'assistant_lora_path': '/missing/assistant.safetensors'},
        ):
            with self.subTest(config=config):
                model = self.make_model(**config)
                with patch('extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2.QwenImage21Transformer2DModel.load') as load:
                    with self.assertRaises(FileNotFoundError):
                        model.load_model()
                load.assert_not_called()

    def test_split_mlp_fusion_preserves_both_updates_and_their_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.safetensors'
            prefix = 'transformer.transformer_blocks.0.img_mlp'
            gate_a, gate_b = torch.tensor([[1., 2.]]), torch.tensor([[3.], [4.]])
            proj_a, proj_b = torch.eye(2), torch.tensor([[5., 6.], [7., 8.]])
            weights = {
                prefix + '.gate_layer.lora_A.weight': gate_a,
                prefix + '.gate_layer.lora_B.weight': gate_b,
                prefix + '.gate_layer.alpha': torch.tensor(0.5),
                prefix + '.proj.lora_A.weight': proj_a,
                prefix + '.proj.lora_B.weight': proj_b,
                prefix + '.proj.alpha': torch.tensor(4.),
                'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.ones(1, 2),
                'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.ones(2, 1),
            }
            save_file(weights, str(path))
            model = self.make_model()
            model.model = QwenImage21Transformer2DModel()
            mlp = torch.nn.Module()
            mlp.gate_up = torch.nn.Linear(2, 4, bias=False)
            mlp.gate_up.weight.data.zero_()
            model.model.transformer_blocks[0].img_mlp = mlp
            model.text_encoder = []
            model.pipeline = SimpleNamespace()

            assistant = load_assistant_lora_from_path(str(path), model, strict=True)
            inputs = torch.ones(1, 2)
            expected = inputs @ torch.cat([gate_b @ gate_a * 0.5, proj_b @ proj_a * 2.]).T

            self.assertTrue(torch.equal(mlp.gate_up(inputs), expected))
            self.assertEqual(sorted(lora.lora_dim for lora in assistant.unet_loras), [1, 3])

    def test_invalid_adapter_cannot_silently_load_partially(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unmatched.safetensors'
            prefix = 'transformer.transformer_blocks.0.attn.unknown'
            save_file({
                prefix + '.lora_A.weight': torch.ones(1, 2),
                prefix + '.lora_B.weight': torch.ones(2, 1),
                'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.ones(1, 2),
                'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.ones(2, 1),
            }, str(path))
            model = self.make_model()
            model.model = QwenImage21Transformer2DModel()
            model.text_encoder = []
            model.pipeline = SimpleNamespace()
            with self.assertRaisesRegex(ValueError, 'unmatched LoRA weights'):
                load_assistant_lora_from_path(str(path), model, strict=True)

    def test_external_encoder_change_invalidates_embedding_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.safetensors'
            path.write_bytes(b'first')
            model = self.make_model(text_encoder_path=str(path))
            first_version = model.text_embedding_space_version
            path.write_bytes(b'updated encoder')
            self.assertNotEqual(first_version, model.text_embedding_space_version)
            self.assertEqual(self.make_model().text_embedding_space_version, 'qwen_image_2_bucketed_refs_v2')

    def test_native_sampling_disables_then_restores_training_adapter(self):
        model = self.make_model(assistant_lora_path='/tmp/assistant.safetensors')
        model.model = torch.nn.Linear(2, 2)
        model.network = None
        model.adapter = None
        model.assistant_lora = SimpleNamespace(is_active=True)
        model.inference_lora_network = None
        model.invert_assistant_lora = False
        model.save_device_state = Mock()
        model.restore_device_state = Mock()
        model._install_sample_step_hooks = Mock(return_value=Mock())
        model.set_device_state_preset = Mock(
            side_effect=lambda preset: self.assertFalse(model.assistant_lora.is_active)
        )

        model.generate_images([], pipeline=Mock())

        self.assertTrue(model.assistant_lora.is_active)


if __name__ == '__main__':
    unittest.main()
