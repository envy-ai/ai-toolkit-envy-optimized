"""Batched FIBO2 conditioning and signed adapter-strength CPU regressions."""

import unittest
import copy
import tempfile
from pathlib import Path

import torch

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from extensions_built_in.diffusion_models.fibo2.fibo2 import Fibo2Model
from extensions_built_in.diffusion_models.fibo2.src.transformer import Fibo2Params, Fibo2Transformer


def tiny_params():
    return Fibo2Params(
        in_channels=4,
        dim=16,
        n_heads=2,
        n_layers=5,
        n_refiner_layers=1,
        injection_layers=(0, 1, 2, 3, 4),
        cap_feat_dim=12,
        perceiver_layers=1,
        min_gist_tokens=1,
        max_gist_tokens=2,
        gist_step=1,
        gist_min_text_len=2,
        gist_max_text_len=8,
        ffn_dim=24,
        axes_dims=(4, 2, 2),
    )


def model_with_adapter(network_type, *, doha=False):
    wrapper = object.__new__(Fibo2Model)
    wrapper.device_torch = torch.device("cpu")
    wrapper.torch_dtype = torch.float32
    wrapper.transformer = Fibo2Transformer(tiny_params())
    wrapper.transformer.requires_grad_(False)
    wrapper.model = wrapper.transformer
    wrapper.target_lora_modules = ["Fibo2Transformer"]
    wrapper.use_old_lokr_format = False
    wrapper.arch = "fibo2"
    network = LoRASpecialNetwork(
        text_encoder=[],
        unet=wrapper.transformer,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        train_unet=True,
        train_text_encoder=False,
        network_config=NetworkConfig(linear=2, linear_alpha=2, transformer_only=True, loha_dora=doha),
        network_type=network_type,
        transformer_only=True,
        is_transformer=True,
        target_lin_modules=["Fibo2Transformer"],
        base_model=wrapper,
        only_if_contains=["layers.2.attention.qkv"],
    )
    network.apply_to([], wrapper.transformer, apply_text_encoder=False, apply_unet=True)
    network._update_torch_multiplier()
    if len(network.unet_loras) != 1:
        raise AssertionError(f"expected one targeted adapter, got {len(network.unet_loras)}")
    with torch.no_grad():
        for adapter in network.unet_loras:
            if hasattr(adapter, "lora_down"):
                adapter.lora_down.weight.fill_(0.25)
                adapter.lora_up.weight.fill_(0.5)
            else:
                for parameter in adapter.parameters():
                    parameter.normal_(mean=0.0, std=0.1)
    network.is_active = True
    wrapper.network = network
    wrapper.assistant_lora = None
    return wrapper


class Fibo2BatchedAdapterTests(unittest.TestCase):
    def test_masked_batched_transformer_matches_unpadded_single_examples(self):
        torch.manual_seed(37)
        model = Fibo2Transformer(tiny_params())
        latents = torch.randn(2, 4, 2, 2)
        timestep = torch.tensor([0.35, 0.72])
        padded_context = torch.randn(2, 6, 5, 12)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]],
        )

        batched = model(latents, timestep, padded_context, mask=mask)
        singles = []
        for index, length in enumerate((3, 5)):
            singles.append(
                model(
                    latents[index : index + 1],
                    timestep[index : index + 1],
                    padded_context[index : index + 1, :, :length],
                    mask=torch.ones(1, length, dtype=torch.bool),
                )
            )

        torch.testing.assert_close(batched, torch.cat(singles), rtol=1e-5, atol=1e-6)

    def test_signed_lora_and_dora_batch_strengths_match_individual_forwards(self):
        for network_type in ("lora", "dora"):
            with self.subTest(network_type=network_type):
                torch.manual_seed(41)
                model = model_with_adapter(network_type)
                network = model.network
                strengths = [0.75, -0.5]
                network.multiplier = strengths
                vector_before = network.torch_multiplier.clone()
                model.transformer.enable_gradient_checkpointing()

                contexts = [torch.randn(6, 3, 12), torch.randn(6, 6, 12)]
                embeds = AdvancedPromptEmbeds(text_embeds=contexts)
                latents = torch.randn(2, 4, 2, 2)
                timesteps = torch.tensor([400.0, 600.0])
                batched = model.predict_embeddings(latents, timesteps, embeds)
                self.assertEqual(network.multiplier, strengths)
                torch.testing.assert_close(network.torch_multiplier, vector_before)
                network.is_active = False
                with torch.no_grad():
                    base = model.predict_embeddings(latents, timesteps, embeds)
                network.is_active = True
                adapter_parameters = tuple(parameter for parameter in network.parameters()
                                            if parameter.requires_grad)
                batched_gradients = torch.autograd.grad(
                    batched.square().mean(), adapter_parameters
                )

                singles = []
                single_gradients = [torch.zeros_like(parameter) for parameter in adapter_parameters]
                for index, strength in enumerate(strengths):
                    network.multiplier = strength
                    single_embeds = AdvancedPromptEmbeds(text_embeds=[contexts[index]])
                    single = model.predict_embeddings(
                        latents[index : index + 1],
                        timesteps[index : index + 1],
                        single_embeds,
                    )
                    singles.append(single.detach())
                    per_example_gradients = torch.autograd.grad(
                        single.square().mean() / len(strengths), adapter_parameters
                    )
                    for total, gradient in zip(single_gradients, per_example_gradients):
                        total.add_(gradient)

                torch.testing.assert_close(batched, torch.cat(singles), rtol=1e-5, atol=1e-6)
                self.assertGreater((batched - base).abs().max().item(), 1e-5)
                for batch_gradient, single_gradient in zip(batched_gradients, single_gradients):
                    self.assertTrue(torch.isfinite(batch_gradient).all())
                    self.assertTrue(torch.isfinite(single_gradient).all())
                    torch.testing.assert_close(batch_gradient, single_gradient, rtol=3e-5, atol=3e-6)
                self.assertTrue(any(gradient.abs().sum().item() > 0 for gradient in batched_gradients))

    def test_flow_velocity_target_drives_a_real_tiny_lora_optimizer_step(self):
        torch.manual_seed(53)
        model = model_with_adapter("lora")
        network = model.network
        network.multiplier = 1.0
        network._update_torch_multiplier()
        context = torch.randn(6, 4, 12)
        embeds = AdvancedPromptEmbeds(text_embeds=[context])
        clean = torch.randn(1, 4, 2, 2)
        noise = torch.randn_like(clean)
        timestep = torch.tensor([500.0])
        noisy = clean * 0.5 + noise * 0.5
        target = model.get_loss_target(noise=noise, batch=type("Batch", (), {"latents": clean})())
        torch.testing.assert_close(target, noise - clean)

        optimizer = torch.optim.SGD(network.parameters(), lr=1e-3)
        parameters = tuple(parameter for parameter in network.parameters() if parameter.requires_grad)
        before = tuple(parameter.detach().clone() for parameter in parameters)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.predict_embeddings(noisy, timestep, embeds)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in parameters))
        self.assertTrue(all(parameter.grad is None for parameter in model.transformer.parameters()))
        optimizer.step()

        self.assertTrue(any(not torch.equal(old, parameter.detach())
                            for old, parameter in zip(before, parameters)))

    def test_ordinary_lora_dora_loha_doha_and_lokr_train_export_and_reload(self):
        for network_type, doha in (("lora", False), ("dora", False), ("loha", False),
                                   ("loha", True), ("lokr", False)):
            with self.subTest(network_type=network_type, doha=doha):
                torch.manual_seed(59)
                model = model_with_adapter(network_type, doha=doha)
                network = model.network
                has_dora = any(getattr(module, 'use_dora', False) or module.__class__.__name__ == 'DoRAModule'
                               for module in network.unet_loras)
                self.assertEqual(has_dora, network_type == "dora" or (network_type == "loha" and doha))
                base_state = copy.deepcopy(model.transformer.state_dict())
                inputs = torch.randn(1, 4, 2, 2)
                embeds = AdvancedPromptEmbeds(text_embeds=[torch.randn(6, 4, 12)])
                time = torch.tensor([500.0])
                before_output = model.predict_embeddings(inputs, time, embeds).detach()
                params = tuple(p for p in network.parameters() if p.requires_grad)
                optimizer = torch.optim.SGD(params, lr=1e-2)
                optimizer.zero_grad(set_to_none=True)
                output = model.predict_embeddings(inputs, time, embeds)
                loss = output.square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in params))
                optimizer.step()
                after_output = model.predict_embeddings(inputs, time, embeds).detach()
                self.assertGreater((after_output - before_output).abs().max().item(), 0.0)

                with tempfile.TemporaryDirectory() as directory:
                    path = str(Path(directory) / f"{network_type}-{doha}.safetensors")
                    network.save_weights(path, dtype=torch.float32)
                    restored = model_with_adapter(network_type, doha=doha)
                    restored.transformer.load_state_dict(base_state)
                    restored.network.load_weights(path)
                    actual = restored.predict_embeddings(inputs, time, embeds).detach()
                    torch.testing.assert_close(actual, after_output, rtol=1e-5, atol=1e-6)

    def test_full_finetune_optimizer_step_and_transformer_export_reload(self):
        torch.manual_seed(61)
        model = Fibo2Transformer(tiny_params())
        wrapper = object.__new__(Fibo2Model)
        wrapper.device_torch = torch.device("cpu")
        wrapper.torch_dtype = torch.float32
        wrapper.transformer = model
        wrapper.model = model
        wrapper.network = None
        wrapper.assistant_lora = None
        params = tuple(model.parameters())
        before = tuple(parameter.detach().clone() for parameter in params)
        latents = torch.randn(1, 4, 2, 2)
        context = torch.randn(1, 6, 4, 12)
        target = torch.randn_like(latents)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(latents, torch.tensor([0.5]), context)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        updated_prediction = model(latents, torch.tensor([0.5]), context).detach()
        self.assertTrue(any(not torch.equal(old, parameter.detach()) for old, parameter in zip(before, params)))
        self.assertTrue(all(parameter.grad is not None for parameter in params))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transformer.safetensors"
            from safetensors.torch import load_file, save_file
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(path))
            reloaded = Fibo2Transformer(tiny_params())
            incompatible = reloaded.load_state_dict(load_file(str(path)), strict=True)
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            torch.testing.assert_close(reloaded(latents, torch.tensor([0.5]), context), updated_prediction,
                                       rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
