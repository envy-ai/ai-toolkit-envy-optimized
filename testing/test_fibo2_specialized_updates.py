"""Run specialized flow trainer steps through a tiny real FIBO2 transformer on CPU."""

import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from extensions_built_in.concept_slider.ConceptSliderTrainer import ConceptSliderTrainer
from extensions_built_in.diffusion_models.fibo2.fibo2 import Fibo2Model
from extensions_built_in.diffusion_models.fibo2.src.transformer import Fibo2Params, Fibo2Transformer
from extensions_built_in.sd_trainer.DiffusionKTOTrainer import DiffusionKTOTrainer
from extensions_built_in.sd_trainer.FizgigSliderTrainer import FizgigSliderTrainer
from extensions_built_in.sd_trainer.QwenFlowDPOTrainer import QwenFlowDPOTrainer
from extensions_built_in.sd_trainer.QwenGuidanceDistillationTrainer import QwenGuidanceDistillationTrainer
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import NetworkConfig
from toolkit.flow_kto import FlowKTOSettings
from toolkit.flow_training import FlowTrainingProfile
from toolkit.fizgig_image_objective import _image_loss
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.sliderspace_network import SliderSpaceNetwork
from extensions_built_in.sd_trainer.SliderSpaceTrainer import SliderSpaceTrainer


def tiny_model(with_adapter=True):
    params = Fibo2Params(in_channels=4, dim=16, n_heads=2, n_layers=5,
        n_refiner_layers=1, injection_layers=(0, 1, 2, 3, 4), cap_feat_dim=12,
        perceiver_layers=1, min_gist_tokens=1, max_gist_tokens=2, gist_step=1,
        gist_min_text_len=2, gist_max_text_len=8, ffn_dim=24, axes_dims=(4, 2, 2))
    model = object.__new__(Fibo2Model)
    model.device_torch = torch.device('cpu')
    model.torch_dtype = torch.float32
    model.transformer = Fibo2Transformer(params)
    model.transformer.requires_grad_(False)
    model.model = model.transformer
    model.target_lora_modules = ['Fibo2Transformer']
    model.use_old_lokr_format = False
    model.arch = 'fibo2'
    model.is_edit = True
    model.vae = None
    if with_adapter:
        network = LoRASpecialNetwork(text_encoder=[], unet=model.transformer, multiplier=1.0,
            lora_dim=2, alpha=2, train_unet=True, train_text_encoder=False,
            network_config=NetworkConfig(linear=2, linear_alpha=2, transformer_only=True),
            network_type='lora', transformer_only=True, is_transformer=True,
            target_lin_modules=['Fibo2Transformer'], base_model=model,
            only_if_contains=['layers.2.attention.qkv'])
        network.apply_to([], model.transformer, apply_text_encoder=False, apply_unet=True)
        network._update_torch_multiplier()
        with torch.no_grad():
            network.unet_loras[0].lora_down.weight.fill_(0.2)
            network.unet_loras[0].lora_up.weight.fill_(0.4)
        network.is_active = True
        model.network = network
    else:
        model.network = None
    model.assistant_lora = None

    def predict_noise(**kwargs):
        return model.predict_embeddings(kwargs['latents'], kwargs['timestep'],
                                        kwargs['conditional_embeddings'])
    model.predict_noise = predict_noise
    return model


def embeds(seed):
    generator = torch.Generator().manual_seed(seed)
    return AdvancedPromptEmbeds(text_embeds=[torch.randn(6, 4, 12, generator=generator)])


def optimizer_step(model, loss):
    parameters = tuple(parameter for parameter in model.network.parameters() if parameter.requires_grad)
    before = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer = torch.optim.SGD(parameters, lr=1e-2)
    assert torch.isfinite(loss)
    # These trainer methods own the complete accumulation step and call their
    # accelerator's backward internally; some return detached reporting losses.
    # Apply the gradients produced by that real trainer path instead of calling
    # backward on its reporting value a second time.
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)
    optimizer.step()
    assert any(not torch.equal(old, parameter.detach()) for old, parameter in zip(before, parameters))


def flow_batch(model, *, label=None, seed=2):
    batch = SimpleNamespace(latents=torch.randn(1, 4, 2, 2), prompt_embeds=embeds(seed),
                            loss_multiplier_list=[1.0], control_tensor=None, control_tensor_list=None)
    if label:
        item = SimpleNamespace(dataset_config=SimpleNamespace(kto_label=label))
        batch.file_items = [item]
    return batch


class Fibo2SpecializedUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)

    def test_fizgig_image_and_prompt_slider_paths_update_real_adapter(self):
        for mode in ('image_pairs', 'prompt_pairs'):
            with self.subTest(mode=mode):
                model = tiny_model()
                trainer = SimpleNamespace(network=model.network, slider_mode=mode,
                    slider_diff_weight=0.0, slider_guidance=1.0, slider_cfg=1.0,
                    device_torch=torch.device('cpu'), sd=model,
                    train_config=SimpleNamespace(dtype='fp32'),
                    accelerator=SimpleNamespace(backward=lambda loss: loss.backward()),
                    negative_latents={}, slider_cfg_negative_embeds=None)
                trainer._predict = lambda noisy, time, prompt, negative=None: model.predict_embeddings(noisy, time, prompt)
                trainer._image_loss = lambda clean, other, prompt: _image_loss(trainer, clean, other, prompt)
                trainer._noised_state = lambda clean: (clean * .5, torch.tensor([500.]), torch.randn_like(clean)-clean)
                if mode == 'image_pairs':
                    item = SimpleNamespace(path='positive/a.png', unconditional_path='negative/a.png',
                        scale_to_width=32, scale_to_height=32, crop_x=0, crop_y=0,
                        crop_width=32, crop_height=32, flip_x=False, flip_y=False)
                    trainer._pair_key = QwenFlowDPOTrainer._pair_key
                    trainer.negative_latents[trainer._pair_key(item)] = torch.randn(4, 2, 2)
                    batch = SimpleNamespace(latents=torch.randn(1, 4, 2, 2), prompt_embeds=embeds(3), file_items=[item])
                else:
                    trainer.slider_bank = [(torch.randn(1, 4, 2, 2), 0)]
                    trainer.slider_embeds = [(embeds(4), embeds(5), embeds(6))]
                    batch = None
                loss = FizgigSliderTrainer.train_single_accumulation(trainer, batch)
                optimizer_step(model, loss)
                self.assertTrue(model.network.is_active)

    def test_flow_dpo_preferred_rejected_reference_replay_updates_adapter(self):
        model = tiny_model()
        item = SimpleNamespace(path='preferred/a.png', unconditional_path='rejected/a.png',
            scale_to_width=32, scale_to_height=32, crop_x=0, crop_y=0,
            crop_width=32, crop_height=32, flip_x=False, flip_y=False)
        key = QwenFlowDPOTrainer._pair_key(item)
        trainer = object.__new__(QwenFlowDPOTrainer)
        trainer.sd, trainer.network = model, model.network
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=250, max_denoising_steps=750)
        trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': 'fibo2'})
        trainer.rejected_latents = {key: torch.randn(4, 2, 2)}
        trainer.dpo_beta, trainer.dpo_sft_weight = 1.0, 0.1
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        trainer.additional_logs = {}
        result = QwenFlowDPOTrainer.train_single_accumulation(trainer,
            SimpleNamespace(latents=torch.randn(1, 4, 2, 2), prompt_embeds=embeds(7),
                            file_items=[item], loss_multiplier_list=[1.0]))
        optimizer_step(model, result)
        self.assertIn('dpo/margin', trainer.additional_logs)
        self.assertTrue(model.network.is_active)

    def test_guidance_distillation_teacher_reference_and_student_step(self):
        model = tiny_model()
        trainer = object.__new__(QwenGuidanceDistillationTrainer)
        trainer.sd, trainer.network = model, model.network
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=250,
            max_denoising_steps=750, loss_type='mse')
        trainer.teacher_cfg_scale = 3.0
        trainer.distillation_objective = 'full_guidance'
        trainer._teacher_embeds = lambda _batch: (embeds(8), None)
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        trainer.additional_logs = {}
        result = QwenGuidanceDistillationTrainer.train_single_accumulation(
            trainer, flow_batch(model, seed=9))
        optimizer_step(model, result)
        self.assertIn('distill/teacher_correction_rms', trainer.additional_logs)
        self.assertTrue(model.network.is_active)

    def test_diffusion_kto_scores_and_replays_real_adapter(self):
        model = tiny_model()
        trainer = object.__new__(DiffusionKTOTrainer)
        trainer.sd, trainer.network = model, model.network
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=250, max_denoising_steps=750)
        trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': 'fibo2'})
        trainer.kto_settings = FlowKTOSettings.parse({'beta': 1.2})
        trainer._kto_replay_queue = deque()
        trainer._kto_preparing = False
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        trainer.additional_logs = {}
        batch = flow_batch(model, label='liked', seed=10)
        trainer._prepare_replay([batch])
        result = DiffusionKTOTrainer.train_single_accumulation(trainer, batch)
        optimizer_step(model, result)
        self.assertFalse(trainer._kto_replay_queue)
        self.assertIn('kto/reference_point', trainer.additional_logs)

    def test_concept_slider_sequential_signed_checkpoint_backwards_update_adapter(self):
        model = tiny_model()
        model.transformer.enable_gradient_checkpointing()
        trainer = object.__new__(ConceptSliderTrainer)
        trainer.sd, trainer.network = model, model.network
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32')
        trainer.slider = SimpleNamespace(guidance_strength=1.5, anchor_strength=1.0)
        trainer.positive_prompt_embeds = embeds(11)
        trainer.target_class_embeds = embeds(12)
        trainer.negative_prompt_embeds = embeds(13)
        trainer.anchor_class_embeds = None
        noisy = torch.randn(1, 4, 2, 2)
        loss = ConceptSliderTrainer._fibo_guided_loss(trainer, noisy, torch.tensor([500.]), flow_batch(model))
        optimizer_step(model, loss)
        self.assertEqual(model.network.multiplier, 1.0)
        self.assertTrue(model.network.is_active)

    def test_sliderspace_decodes_semantic_direction_and_updates_selected_adapter(self):
        model = tiny_model(with_adapter=False)
        model.network = SliderSpaceNetwork(text_encoder=[], unet=model.transformer, multiplier=1.0,
            lora_dim=2, alpha=2, train_unet=True, train_text_encoder=False,
            network_config=NetworkConfig(linear=2, linear_alpha=2, transformer_only=True),
            network_type='lora', transformer_only=True, is_transformer=True,
            target_lin_modules=['Fibo2Transformer'], base_model=model,
            only_if_contains=['layers.2.attention.qkv'], num_directions=2)
        model.network.apply_to([], model.transformer, apply_text_encoder=False, apply_unet=True)
        model.network._update_torch_multiplier()
        model.network.is_active = False
        model.decode_latents = lambda latent: latent

        trainer = object.__new__(SliderSpaceTrainer)
        trainer.sd, trainer.network = model, model.network
        trainer.device_torch = torch.device('cpu')
        trainer.train_config = SimpleNamespace(dtype='fp32', min_denoising_steps=250, max_denoising_steps=750)
        trainer.sliderspace = SimpleNamespace(num_directions=2, seed=17, cfg_scale=1.0, loss_weight=1.0)
        trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': 'fibo2'})
        trainer.step_num = 0
        trainer.discovery_entries = [{'caption': 'fixture concept'}]
        trainer.bank_embeds = [embeds(21)]
        trainer.negative_embeds = None
        trainer.direction_vectors = torch.randn(1, 16)
        trainer.semantic_encoder = lambda pixels: pixels.flatten(1)
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        with TemporaryDirectory(dir='tmp') as directory:
            latent_path = Path(directory) / 'latent.safetensors'
            save_file({'latent': torch.randn(1, 4, 2, 2)}, str(latent_path))
            trainer._bank_paths = lambda _index: (Path(directory) / 'source.png', latent_path)
            result = SliderSpaceTrainer.train_single_accumulation(trainer, batch=None)

        active_parameters = tuple(parameter for parameter in model.network.directions[0].parameters()
                                  if parameter.requires_grad)
        self.assertTrue(torch.isfinite(result))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in active_parameters))
        self.assertTrue(model.network.is_active is False)
        self.assertEqual(model.network.active_direction, 0)
        self.assertFalse(model.network.directions[1].is_active)


if __name__ == '__main__':
    unittest.main()
