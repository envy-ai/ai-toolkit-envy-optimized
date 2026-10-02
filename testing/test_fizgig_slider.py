from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.FizgigSliderTrainer import (
    FizgigSliderTrainer,
    pair_difference_weights,
    parse_cfg_negative_prompts,
    parse_prompt_triplets,
    validate_image_slider_pairs,
)
from toolkit.models.DoRA import DoRAModule


class FizgigSliderTests(unittest.TestCase):
    def test_pair_difference_weight_targets_changed_latents(self):
        positive = torch.zeros(1, 2, 2, 2)
        negative = positive.clone()
        negative[:, :, 0, 0] = 1
        weights = pair_difference_weights(positive, negative, 1.0)
        self.assertEqual(weights.shape, (1, 4))
        self.assertGreater(weights[0, 0], weights[0, 1])
        self.assertAlmostEqual(weights.mean().item(), 1.0)
        self.assertTrue(torch.all(pair_difference_weights(positive, positive, 1.0) == 1))
        self.assertTrue(torch.all(pair_difference_weights(positive, negative, 0.0) == 1))

    def test_qwen_flow_target_matches_noising_equation(self):
        fake = SimpleNamespace(
            device_torch=torch.device("cpu"),
            train_config=SimpleNamespace(dtype="fp32", min_denoising_steps=0, max_denoising_steps=999),
        )
        clean = torch.ones(2, 4, 4, 4)
        noisy, timesteps, target = FizgigSliderTrainer._noised_state(fake, clean)
        t = (timesteps / 1000).view(-1, 1, 1, 1)
        reconstructed_noise = target + clean
        self.assertTrue(torch.allclose(
            noisy, (1 - t) * clean + t * reconstructed_noise, atol=1e-6, rtol=1e-5,
        ))
        self.assertTrue(torch.all(timesteps >= 0))
        self.assertTrue(torch.all(timesteps <= 999))

    def test_image_pairs_validate_all_files_and_route_negative_as_target(self):
        with tempfile.TemporaryDirectory() as directory:
            positive, negative = Path(directory) / "positive", Path(directory) / "negative"
            positive.mkdir()
            negative.mkdir()
            Image.new("RGB", (64, 64)).save(positive / "a.png")
            Image.new("RGB", (64, 64)).save(negative / "a.webp")
            dataset = {
                "folder_path": str(positive),
                "control_path_1": str(negative),
                "cache_latents_to_disk": True,
            }
            validate_image_slider_pairs([dataset])

            config = {
                "type": "fizgig_image_slider",
                "model": {"arch": "qwen_image_2"},
                "network": {"type": "lora"},
                "train": {"noise_scheduler": "flowmatch"},
                "datasets": [dataset],
            }
            with patch.object(DiffusionTrainer, "__init__", return_value=None):
                trainer = FizgigSliderTrainer(0, None, config)
            self.assertEqual(trainer.slider_mode, "image_pairs")
            self.assertEqual(dataset["unconditional_path"], str(negative))
            self.assertNotIn("control_path_1", dataset)

            Image.new("RGB", (32, 64)).save(negative / "a.webp")
            with self.assertRaisesRegex(ValueError, "different dimensions"):
                validate_image_slider_pairs([{
                    "folder_path": str(positive), "control_path_1": str(negative),
                    "cache_latents_to_disk": True,
                }])

    def test_prompt_slider_uses_no_dataset_and_backprops_both_strengths(self):
        class Network:
            multiplier = 1.0
            is_active = False

        class Embed:
            def __init__(self, value):
                self.value = value

            def to(self, *args, **kwargs):
                return self

        network = Network()
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        strengths_at_backward = []
        cfg_negatives_at_prediction = []

        class Accelerator:
            def backward(self, loss):
                strengths_at_backward.append(network.multiplier)
                loss.backward()

        fake = SimpleNamespace(
            network=network,
            slider_mode="prompt_pairs",
            slider_bank=[(torch.zeros(1, 1, 2, 2), 1)],
            slider_embeds=[
                (Embed(0), Embed(100), Embed(99)),
                (Embed(0), Embed(1), Embed(-1)),
            ],
            slider_cfg_negative_embeds=[
                (Embed(90), Embed(91), Embed(92)),
                (Embed(0), Embed(-2), Embed(2)),
            ],
            slider_cfg=3.0,
            slider_guidance=1.0,
            device_torch=torch.device("cpu"),
            sd=SimpleNamespace(torch_dtype=torch.float32),
            accelerator=Accelerator(),
        )
        fake._noised_state = lambda clean: (clean, torch.tensor([500.0]), None)
        def predict(noisy, timestep, embed, unconditional):
            cfg_negatives_at_prediction.append(unconditional.value)
            cfg_prediction = unconditional.value + fake.slider_cfg * (embed.value - unconditional.value)
            return torch.ones_like(noisy) * (parameter * network.multiplier + cfg_prediction)
        fake._predict = predict
        loss = FizgigSliderTrainer.train_single_accumulation(fake, None)
        self.assertEqual(strengths_at_backward, [1.0, -1.0])
        # The two opposite CFG negatives make the teacher delta 14, rather
        # than 6 with one shared unconditional embedding.
        self.assertAlmostEqual(loss.item(), 196.0)
        self.assertAlmostEqual(parameter.grad.item(), -28.0)
        self.assertEqual(network.multiplier, 1.0)
        self.assertFalse(network.is_active)
        self.assertEqual(cfg_negatives_at_prediction, [0, -2, 2, 0, 0])

    def test_predict_forwards_cfg_to_noise_model_only_when_negative_is_present(self):
        calls = []
        def predict_noise(**kwargs):
            calls.append(kwargs)
            return torch.ones(1, 1, 2, 2)
        fake = SimpleNamespace(sd=SimpleNamespace(predict_noise=predict_noise), slider_cfg=4.5)
        noisy = torch.zeros(1, 1, 2, 2)
        timestep = torch.tensor([500.0])
        FizgigSliderTrainer._predict(fake, noisy, timestep, "positive", "negative")
        FizgigSliderTrainer._predict(fake, noisy, timestep, "positive")
        self.assertEqual(calls[0]["unconditional_embeddings"], "negative")
        self.assertEqual(calls[0]["guidance_scale"], 4.5)
        self.assertIsNone(calls[1]["unconditional_embeddings"])
        self.assertEqual(calls[1]["guidance_scale"], 1.0)

    def test_prompt_slider_constructor_discards_dataset(self):
        config = {
            "type": "fizgig_prompt_slider",
            "model": {"arch": "qwen_image_2"},
            "network": {"type": "lora"},
            "train": {"noise_scheduler": "flowmatch"},
            "datasets": [{"folder_path": "ignored"}],
            "fizgig_slider": {
                "neutral_prompt": "neutral", "positive_prompt": "bright",
                "negative_prompt": "dark",
            },
        }
        with patch.object(DiffusionTrainer, "__init__", return_value=None):
            trainer = FizgigSliderTrainer(0, None, config)
        self.assertEqual(config["datasets"], [])
        self.assertEqual(trainer.slider_prompt_triplets, [("neutral", "bright", "dark")])
        self.assertTrue(config["train"]["unload_text_encoder"])
        self.assertEqual(trainer.slider_cfg, 1.0)
        self.assertIsNone(trainer.slider_cfg_negative_prompts)
        config["fizgig_slider"]["cfg_scale"] = 2.0
        config["fizgig_slider"]["cfg_negative_prompt"] = "no artifacts"
        with patch.object(DiffusionTrainer, "__init__", return_value=None):
            trainer = FizgigSliderTrainer(0, None, config)
        self.assertEqual(trainer.slider_cfg_negative_prompts, [
            ("no artifacts", "no artifacts", "no artifacts")
        ])

    def test_prompt_triplet_list_and_bank_size_validation(self):
        triplets = [
            {"neutral_prompt": "portrait", "positive_prompt": "clear portrait",
             "negative_prompt": "hazy portrait"},
            {"neutral_prompt": "landscape", "positive_prompt": "clear landscape",
             "negative_prompt": "hazy landscape"},
        ]
        self.assertEqual(parse_prompt_triplets({"prompt_triplets": triplets}), [
            ("portrait", "clear portrait", "hazy portrait"),
            ("landscape", "clear landscape", "hazy landscape"),
        ])
        with self.assertRaisesRegex(ValueError, "triplet 2 requires"):
            parse_prompt_triplets({"prompt_triplets": [triplets[0], {**triplets[1], "negative_prompt": ""}]})
        config = {
            "type": "fizgig_prompt_slider", "model": {"arch": "qwen_image_2"},
            "network": {"type": "lora"}, "train": {"noise_scheduler": "flowmatch"},
            "fizgig_slider": {"prompt_triplets": triplets, "bank_size": 1},
        }
        with self.assertRaisesRegex(ValueError, "at least the number of prompt triplets"):
            FizgigSliderTrainer(0, None, config)
        config["fizgig_slider"]["bank_size"] = 2
        with patch.object(DiffusionTrainer, "__init__", return_value=None):
            trainer = FizgigSliderTrainer(0, None, config)
        self.assertEqual(len(trainer.slider_prompt_triplets), 2)
        config["network"]["type"] = "dora"
        with patch.object(DiffusionTrainer, "__init__", return_value=None):
            trainer = FizgigSliderTrainer(0, None, config)
        self.assertEqual(len(trainer.slider_prompt_triplets), 2)

    def test_signed_dora_matches_comfy_weight_delta_at_both_poles(self):
        class Network:
            is_lorm = False
            is_active = True
            is_merged_in = False
            signed_dora_slider = True
            _multiplier = 1.0
            torch_multiplier = torch.tensor([1.0])

        network = Network()
        linear = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[0.3, -0.4, 0.5], [-0.2, 0.6, 0.1]]))
            linear.bias.copy_(torch.tensor([0.7, -0.3]))
        module = DoRAModule("test", linear, network=network, lora_dim=2, alpha=1)
        with torch.no_grad():
            module.lora_down.weight.copy_(torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.2, -0.2]]))
            module.lora_up.weight.copy_(torch.tensor([[0.3, -0.2], [-0.1, 0.4]]))
            module.magnitude.mul_(torch.tensor([1.2, 0.8]))
        module.apply_to()
        x = torch.tensor([
            [[0.4, -0.8, 0.6], [-0.5, 0.2, 0.9]],
            [[0.1, 0.3, -0.2], [0.8, -0.6, 0.4]],
        ])
        base_weight = module.get_orig_weight()
        lora_delta = (module.lora_up.weight @ module.lora_down.weight) * 0.5
        comfy_plus = (base_weight + lora_delta) * (
            module.magnitude / module._comfy_base_norm
        ).unsqueeze(1)
        for strength in (-1.0, 0.0, 1.0):
            network._multiplier = strength
            network.torch_multiplier = torch.tensor([strength])
            expected_weight = base_weight + strength * (comfy_plus - base_weight)
            self.assertTrue(torch.allclose(
                linear(x), F.linear(x, expected_weight, linear.bias), atol=1e-6,
            ))
        network._multiplier = -1.0
        network.torch_multiplier = torch.tensor([-1.0])
        linear(x).sum().backward()
        self.assertIsNotNone(module.magnitude.grad)
        self.assertIsNotNone(module.lora_up.weight.grad)

    def test_slider_enables_signed_dora_before_training(self):
        trainer = object.__new__(FizgigSliderTrainer)
        trainer.network_config = SimpleNamespace(type="dora")
        trainer.network = SimpleNamespace(signed_dora_slider=False)
        trainer.slider_mode = "image_pairs"
        with patch.object(trainer, "_cache_negative_latents") as cache, patch.object(
            DiffusionTrainer, "hook_before_train_loop", return_value=None,
        ):
            trainer.hook_before_train_loop()
        self.assertTrue(trainer.network.signed_dora_slider)
        cache.assert_called_once()

    def test_prompt_slider_encodes_each_positive_and_cfg_negative_prompt(self):
        class Embed:
            def detach(self):
                return self

            def to(self, *args, **kwargs):
                return self

        encoded = []
        def encode_prompt(prompts):
            encoded.append(prompts[0])
            return Embed()

        trainer = object.__new__(FizgigSliderTrainer)
        trainer.network_config = SimpleNamespace(type="lora")
        trainer.slider_mode = "prompt_pairs"
        trainer.slider_prompt_triplets = [("subject", "full body shot\n\nsubject", "close-up portrait\n\nsubject")]
        trainer.slider_cfg_negative_prompts = [
            ("", "close-up portrait\n\nsubject", "full body shot\n\nsubject")
        ]
        trainer.slider_cfg_negative_embeds = None
        trainer.sd = SimpleNamespace(
            set_device_state_preset=lambda preset: None,
            encode_prompt=encode_prompt,
            restore_device_state=lambda: None,
        )
        with patch.object(DiffusionTrainer, "hook_before_train_loop", return_value=None), patch.object(
            trainer, "_build_prompt_bank", return_value=None,
        ):
            trainer.hook_before_train_loop()
        self.assertEqual(encoded, [
            "subject", "full body shot\n\nsubject", "close-up portrait\n\nsubject",
            "", "close-up portrait\n\nsubject", "full body shot\n\nsubject",
        ])
        self.assertEqual(len(trainer.slider_embeds[0]), 3)
        self.assertEqual(len(trainer.slider_cfg_negative_embeds[0]), 3)

    def test_simplified_prefixes_expand_with_one_blank_line_and_mix_with_specific(self):
        slider = {
            "positive_prefix": " clear and crisp \n detailed ",
            "negative_prefix": " hazy ",
            "prompt_entries": [
                {"kind": "simple", "prompt": " a portrait\nwith a blue coat "},
                {"kind": "specific", "neutral_prompt": "landscape",
                 "positive_prompt": "bright landscape", "negative_prompt": "dark landscape"},
            ],
        }
        self.assertEqual(parse_prompt_triplets(slider), [
            ("a portrait\nwith a blue coat",
             "clear and crisp \n detailed\n\na portrait\nwith a blue coat",
             "hazy\n\na portrait\nwith a blue coat"),
            ("landscape", "bright landscape", "dark landscape"),
        ])
        slider["negative_prefix"] = " "
        with self.assertRaisesRegex(ValueError, "require \+1 and -1 prefixes"):
            parse_prompt_triplets(slider)

    def test_cfg_negative_prompts_match_each_entry(self):
        slider = {
            "positive_prefix": "clear", "negative_prefix": "hazy",
            "cfg_negative_prefix": " no grain ",
            "cfg_negative_prefix_positive": " no haze ",
            "cfg_negative_prefix_negative": " no clarity ",
            "prompt_entries": [
                {"kind": "simple", "prompt": " a portrait "},
                {"kind": "specific", "neutral_prompt": "landscape",
                 "positive_prompt": "bright landscape", "negative_prompt": "dark landscape",
                 "cfg_negative_prompt": " no artifacts ",
                 "cfg_negative_prompt_positive": " no darkness ",
                 "cfg_negative_prompt_negative": " no brightness "},
            ],
        }
        triplets = parse_prompt_triplets(slider)
        self.assertEqual(parse_cfg_negative_prompts(slider, triplets), [
            ("no grain\n\na portrait", "no haze\n\na portrait", "no clarity\n\na portrait"),
            ("no artifacts", "no darkness", "no brightness"),
        ])
        slider["cfg_negative_prefix"] = ""
        self.assertEqual(parse_cfg_negative_prompts(slider, triplets)[0][0], "")
        legacy = {"neutral_prompt": "portrait", "positive_prompt": "bright",
                  "negative_prompt": "dark", "cfg_negative_prompt": "no noise"}
        self.assertEqual(parse_cfg_negative_prompts(legacy, parse_prompt_triplets(legacy)), [
            ("no noise", "no noise", "no noise")
        ])

    def test_practice_images_are_distributed_and_tagged_by_triplet(self):
        class Embed:
            def __init__(self, value):
                self.value = value

            def to(self, *args, **kwargs):
                return self

        class Network:
            multiplier = 1.0
            is_active = True

        rendered = []

        def pipeline(neutral, **kwargs):
            rendered.append((neutral.value, kwargs["unconditional_embeds"].value,
                             kwargs["guidance_scale"]))
            return [Image.new("RGB", (64, 64))]

        sd = SimpleNamespace(
            assistant_lora=None, torch_dtype=torch.float32, vae_torch_dtype=torch.float32,
            save_device_state=lambda: None, restore_device_state=lambda: None,
            text_encoder_to=lambda device: None, unet=SimpleNamespace(to=lambda device: None),
            vae=SimpleNamespace(to=lambda device: None), vae_device_torch=torch.device("cpu"),
            get_generation_pipeline=lambda: pipeline,
            encode_images=lambda pixels: torch.zeros(1, 2, 2, 2),
        )
        fake = SimpleNamespace(
            sd=sd, network=Network(), device_torch=torch.device("cpu"),
            slider_prompt_triplets=[("first", "a", "b"), ("second", "c", "d")],
            slider_embeds=[(Embed("first"), None, None), (Embed("second"), None, None)],
            slider_cfg_negative_embeds=[
                (Embed("negative first"), Embed("+1 first"), Embed("-1 first")),
                (Embed("negative second"), Embed("+1 second"), Embed("-1 second")),
            ],
            slider_cfg=3.0,
            slider_bank=[], slider_bank_size=4, slider_bank_resolution=64, slider_bank_steps=1,
            sample_config=SimpleNamespace(seed=42), print=lambda *args: None,
        )
        FizgigSliderTrainer._build_prompt_bank(fake)
        self.assertEqual(rendered, [
            ("first", "negative first", 3.0), ("second", "negative second", 3.0),
            ("first", "negative first", 3.0), ("second", "negative second", 3.0),
        ])
        self.assertEqual([index for _, index in fake.slider_bank], [0, 1, 0, 1])

    def test_negative_latents_are_cached_before_dataloader_workers_start(self):
        class Item:
            path = "positive/a.png"
            unconditional_path = "negative/a.png"
            scale_to_width = scale_to_height = crop_width = crop_height = 32
            crop_x = crop_y = 0
            flip_x = flip_y = False
            has_unconditional = True

            def load_unconditional_image(self):
                self.unconditional_tensor = torch.zeros(3, 32, 32)

            def cleanup_unconditional(self):
                self.unconditional_tensor = None

        item = Item()
        device_states = []
        sd = SimpleNamespace(
            vae=object(), vae_device_torch=torch.device("cpu"), vae_torch_dtype=torch.float32,
            set_device_state_preset=lambda name: device_states.append(name),
            restore_device_state=lambda: device_states.append("restored"),
            encode_images=lambda pixels: torch.ones(1, 2, 2, 2),
        )
        fake = SimpleNamespace(sd=sd, negative_latents={}, data_loader=object(), print=lambda *args: None)
        fake._pair_key = FizgigSliderTrainer._pair_key
        with patch(
            "extensions_built_in.sd_trainer.FizgigSliderTrainer.get_dataloader_datasets",
            return_value=[SimpleNamespace(file_list=[item])],
        ):
            FizgigSliderTrainer._cache_negative_latents(fake)
        self.assertEqual(device_states, ["cache_latents", "restored"])
        self.assertIn(FizgigSliderTrainer._pair_key(item), fake.negative_latents)
        self.assertFalse(item.has_unconditional)
        self.assertIsNone(item.unconditional_tensor)

    def test_image_slider_backprops_each_pole_before_flipping_strength(self):
        class Network:
            multiplier = 1.0
            is_active = False

        class Embed:
            def to(self, *args, **kwargs):
                return self

            def detach(self):
                return self

        network = Network()
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        strengths_at_backward = []

        class Accelerator:
            def backward(self, loss):
                strengths_at_backward.append(network.multiplier)
                loss.backward()

        item = SimpleNamespace(
            path="positive/a.png", unconditional_path="negative/a.png",
            scale_to_width=32, scale_to_height=32, crop_x=0, crop_y=0,
            crop_width=32, crop_height=32, flip_x=False, flip_y=False,
        )
        key = FizgigSliderTrainer._pair_key(item)
        batch = SimpleNamespace(
            latents=torch.ones(1, 1, 2, 2), file_items=[item], prompt_embeds=Embed(),
        )
        fake = SimpleNamespace(
            network=network,
            slider_mode="image_pairs",
            negative_latents={key: torch.zeros(1, 2, 2)},
            device_torch=torch.device("cpu"),
            train_config=SimpleNamespace(dtype="fp32"),
            accelerator=Accelerator(),
        )
        fake._pair_key = FizgigSliderTrainer._pair_key
        fake._image_loss = lambda clean, other, embeds: (
            parameter * network.multiplier - (1.0 if clean.mean() > 0 else -1.0)
        ).square()
        loss = FizgigSliderTrainer.train_single_accumulation(fake, batch)
        self.assertEqual(strengths_at_backward, [1.0, -1.0])
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(parameter.grad.item(), -2.0)


if __name__ == "__main__":
    unittest.main()
