"""CPU contracts for MOSS-Image specialized training modes and native paths.

Tiny random modules keep this coverage independent of model downloads and CUDA.
"""
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import torch
import yaml

from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from extensions_built_in.diffusion_models.moss_image.src.pipeline import (
    DEFAULT_UNCOND_SKIP_LAYERS,
    MossImagePipeline,
    moss_sigmas,
)
from extensions_built_in.diffusion_models.moss_image.src.transformer import (
    MossImageParams,
    MossImageTransformer,
)
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.flow_cache_identity import specialized_text_cache_identity
from toolkit.flow_cache_identity import resolved_flow_components
from toolkit.flow_cache_identity import prepare_specialized_conditioning
from toolkit.flow_training import (
    FlowTrainingProfile,
    flow_reference_prediction_kwargs,
    guided_flow_prediction,
    training_reference_context,
)
from toolkit.training_capabilities import (
    FLOW_TRAINING_MODELS,
    SPECIALIZED_MODES,
    canonical_flow_arch,
    normalize_moss_training_config,
    validate_edit_references,
    validate_specialized_model,
)


class MossTrainingModeTests(unittest.TestCase):
    def config(self, mode="flow_dpo", variant="moss_image:base", **kwargs):
        return {
            "type": mode,
            "model": {"arch": variant, "model_kwargs": kwargs},
            "network": {"type": "lora"},
        }

    def test_every_specialized_mode_accepts_moss_t2i_base(self):
        self.assertEqual(canonical_flow_arch("moss_image:edit"), "moss_image")
        self.assertIn("moss_image", FLOW_TRAINING_MODELS)
        for mode in SPECIALIZED_MODES:
            with self.subTest(mode=mode):
                config = self.config(mode=mode)
                capability = validate_specialized_model(config)
                self.assertEqual(capability, FLOW_TRAINING_MODELS["moss_image"])
                self.assertEqual(config["model"]["arch"], "moss_image")
                self.assertFalse(config["model"]["model_kwargs"].get("edit", False))
        self.assertEqual(len(SPECIALIZED_MODES), 6)

    def test_variant_defaults_preserve_explicit_sources_and_memory_options(self):
        cases = {
            "moss_image": ("OpenMOSS-Team/MOSS-Image-SFT", False),
            "moss_image:base": ("OpenMOSS-Team/MOSS-Image-Base", False),
            "moss_image:edit": ("OpenMOSS-Team/MOSS-Image-SFT", True),
            "moss_image:base_edit": ("OpenMOSS-Team/MOSS-Image-Base", True),
        }
        for variant, (source, edit) in cases.items():
            config = self.config(variant=variant)
            normalize_moss_training_config(config)
            model = config["model"]
            self.assertEqual(model["arch"], "moss_image")
            self.assertEqual(model["name_or_path"], source)
            self.assertEqual(model["model_kwargs"].get("edit", False), edit)
            self.assertEqual(model["model_kwargs"].get("uncond_skip_layers"), [22])
            if edit:
                self.assertTrue(model["model_kwargs"]["kv_cache"])
                self.assertTrue(model["model_kwargs"]["match_target_res"])

        explicit = self.config(variant="moss_image:edit")
        explicit["model"].update(name_or_path="/models/local-base", quantize=True, qtype="int8")
        explicit["model"]["model_kwargs"].update(edit=False, kv_cache=False, custom_memory_flag=True)
        normalize_moss_training_config(explicit)
        self.assertEqual(explicit["model"]["name_or_path"], "/models/local-base")
        self.assertEqual(explicit["model"]["model_kwargs"], {
            "edit": False, "kv_cache": False, "custom_memory_flag": True,
            "match_target_res": True, "uncond_skip_layers": [22],
        })
        self.assertTrue(explicit["model"]["quantize"])
        self.assertEqual(explicit["model"]["qtype"], "int8")

    def test_edit_is_limited_to_dpo_and_distillation_with_fixed_references(self):
        for mode in ("flow_dpo", "guidance_distillation"):
            with self.subTest(mode=mode):
                config = self.config(mode=mode, variant="moss_image:edit")
                config["datasets"] = [{"control_path_1": "/rejected", "control_path_2": "/edit-source"}]
                validate_specialized_model(config)
                self.assertTrue(validate_edit_references(
                    config, config["datasets"], paired=(mode == "flow_dpo")
                ))

                missing = self.config(mode=mode, variant="moss_image:edit")
                missing["datasets"] = [{"folder_path": "/data/a"}, {"control_path": "/source"}]
                validate_specialized_model(missing)
                with self.assertRaisesRegex(ValueError, "every dataset"):
                    validate_edit_references(
                        missing, missing["datasets"], paired=(mode == "flow_dpo")
                    )

        for mode in ("fizgig_image_slider", "fizgig_prompt_slider", "sliderspace", "diffusion_kto"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "text-to-image"):
                validate_specialized_model(self.config(mode=mode, variant="moss_image:edit"))

        t2i = self.config(mode="flow_dpo", variant="moss_image:base")
        t2i["datasets"] = [{"control_path_1": "/rejected", "control_path_2": "/source"}]
        validate_specialized_model(t2i)
        with self.assertRaisesRegex(ValueError, "edit=true"):
            validate_edit_references(t2i, t2i["datasets"], paired=True)

    def test_native_only_preview_and_invalid_skip_layers_fail_preload(self):
        comfy = self.config()
        comfy["sample"] = {"comfy": {"enabled": True}}
        with self.assertRaisesRegex(ValueError, "native previews"):
            validate_specialized_model(comfy)
        for values in ([42], [-1], [1.5], "22"):
            config = self.config(uncond_skip_layers=values)
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "zero-based"):
                validate_specialized_model(config)

    def test_six_shipped_moss_examples_match_specialized_mode_contracts(self):
        root = Path(__file__).resolve().parents[1]
        names = (
            "train_flow_dpo_moss_image.yaml",
            "train_guidance_distillation_moss_image.yaml",
            "train_fizgig_image_slider_moss_image.yaml",
            "train_fizgig_prompt_slider_moss_image.yaml",
            "train_sliderspace_moss_image.yaml",
            "train_diffusion_kto_moss_image.yaml",
        )
        for name in names:
            with self.subTest(example=name):
                loaded = yaml.safe_load((root / "config/examples" / name).read_text())
                process = loaded["config"]["process"][0]
                capability = validate_specialized_model(process)
                self.assertEqual(capability, FLOW_TRAINING_MODELS["moss_image"])
                self.assertEqual(process["model"]["arch"], "moss_image")
                network_kwargs = process.get("network", {}).get("network_kwargs", {})
                self.assertNotIn("layers.22.", network_kwargs.get("ignore_if_contains", []))
                self.assertEqual(process["model"]["model_kwargs"].get("uncond_skip_layers"), [22])

    def test_fixed_shift_five_and_reference_branch_skips_block_22(self):
        profile = FlowTrainingProfile.from_model_config({"arch": "moss_image:base"})
        self.assertEqual(profile.arch, "moss_image")
        self.assertEqual(profile.fixed_mu, math.log(5))
        self.assertEqual(profile.shift(32, 64), math.log(5))
        model = SimpleNamespace(arch="moss_image", uncond_skip_layers=(22,))
        self.assertEqual(flow_reference_prediction_kwargs(model), {"skip_layers": (22,)})
        calls = []

        def predict_noise(**kwargs):
            calls.append(kwargs)
            return torch.ones_like(kwargs["latents"]) * kwargs.get("skip_layers", (0,))[0] if kwargs.get("skip_layers") else torch.ones_like(kwargs["latents"])

        model.predict_noise = predict_noise
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.ones(2, 4)])
        latent = torch.zeros(1, 4, 2, 2)
        result = guided_flow_prediction(model, latent, torch.tensor([500]), embeds, embeds, 3)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("skip_layers", calls[0])
        self.assertEqual(calls[1]["skip_layers"], (22,))
        torch.testing.assert_close(result, torch.full_like(latent, -41.0))


class MossNativePathTests(unittest.TestCase):
    @staticmethod
    def tiny_transformer(num_layers=2):
        return MossImageTransformer(MossImageParams(
            dim=32, ffn_dim=64, num_heads=1, num_layers=num_layers,
            text_dim=4, freq_dim=16, latent_channels=4,
        )).float()

    @staticmethod
    def prompt(value):
        return AdvancedPromptEmbeds(text_embeds=[torch.full((2, 4), value)])

    def test_native_sampler_uses_shift_five_skip_22_and_separate_edit_caches(self):
        class SpyTransformer:
            params = SimpleNamespace(latent_channels=4)

            def __init__(self):
                self.calls = []

            def __call__(self, latents, timestep, embeds, mask, **kwargs):
                cache = kwargs.get("ref_kv_cache")
                self.calls.append((timestep.detach().clone(), kwargs.copy()))
                if cache is not None and cache.get("kv") is None:
                    cache.update(kv={0: torch.ones(1)}, mask=torch.ones(1, 1, dtype=torch.bool))
                return torch.ones_like(latents)

        transformer = SpyTransformer()
        model = SimpleNamespace(
            device_torch=torch.device("cpu"), torch_dtype=torch.float32,
            transformer=transformer, sample_step_hook=None,
            decode_latents=lambda latent, **kwargs: torch.zeros(1, 3, 16, 16),
        )
        refs = [[torch.ones(4, 1, 1)]]
        image = MossImagePipeline(model)(
            self.prompt(1), self.prompt(0), height=16, width=16,
            num_inference_steps=3, guidance_scale=3, latents=torch.zeros(1, 4, 1, 1),
            ref_latents=refs, isolate_refs=True,
        )[0]
        sigmas = moss_sigmas(3, shift=5)
        self.assertEqual(image.size, (16, 16))
        self.assertEqual(len(transformer.calls), 2 * (len(sigmas) - 1))
        self.assertIs(transformer.calls[0][1]["ref_kv_cache"], transformer.calls[2][1]["ref_kv_cache"])
        self.assertIsNot(transformer.calls[0][1]["ref_kv_cache"], transformer.calls[1][1]["ref_kv_cache"])
        self.assertEqual(transformer.calls[0][1]["isolate_refs"], True)
        self.assertNotIn("skip_layers", transformer.calls[0][1])
        self.assertEqual(transformer.calls[1][1]["skip_layers"], DEFAULT_UNCOND_SKIP_LAYERS)
        torch.testing.assert_close(transformer.calls[0][0], (sigmas[0] * 1000).reshape(1))

    def test_real_tiny_transformer_checkpoint_gradients_and_reference_kv_reuse(self):
        transformer = self.tiny_transformer()
        transformer.enable_gradient_checkpointing()
        noisy = torch.randn(1, 4, 1, 2, requires_grad=True)
        text = torch.randn(1, 2, 4)
        refs = [[torch.randn(4, 1, 1)]]
        output = transformer(noisy, torch.tensor([500.]), text,
            torch.ones(1, 2, dtype=torch.bool), ref_latents=refs, isolate_refs=True,
            skip_layers=(1,))
        output.square().mean().backward()
        self.assertIsNotNone(noisy.grad)
        self.assertTrue(torch.isfinite(noisy.grad).all())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in transformer.parameters()))

        cache = {}
        with torch.no_grad():
            first = transformer(noisy.detach(), torch.tensor([500.]), text,
                torch.ones(1, 2, dtype=torch.bool), ref_latents=refs,
                isolate_refs=True, ref_kv_cache=cache, skip_layers=(1,))
            self.assertEqual(set(cache["kv"]), {0})
            cached_kv = cache["kv"]
            second = transformer(noisy.detach(), torch.tensor([500.]), text,
                torch.ones(1, 2, dtype=torch.bool), ref_latents=refs,
                isolate_refs=True, ref_kv_cache=cache, skip_layers=(1,))
        torch.testing.assert_close(first, second)
        self.assertIs(cache["kv"], cached_kv)

    def test_edit_reference_vae_inputs_are_detached_and_shared_for_the_batch_scope(self):
        model = object.__new__(MossImageModel)
        source = torch.ones(1, 3, 16, 16, requires_grad=True)
        encoded = torch.ones(4, 1, 1, requires_grad=True)
        model._encode_ref_latents = Mock(return_value=[encoded])
        batch = SimpleNamespace(control_tensor_list=[[source]], control_tensor=None)
        with model.training_reference_context(batch):
            first = model._batch_ref_latents(batch, 1, 256)
            second = model._batch_ref_latents(batch, 1, 256)
            self.assertIs(first, second)
            self.assertFalse(first[0][0].requires_grad)
            with self.assertRaisesRegex(ValueError, "original training batch"):
                other = SimpleNamespace(control_tensor_list=[[source]], control_tensor=None)
                model._batch_ref_latents(other, 1, 256)
        model._encode_ref_latents.assert_called_once_with([source], target_pixels=256)
        self.assertFalse(hasattr(model, "_training_reference_scope"))

    def test_frozen_vae_decode_keeps_semantic_gradient(self):
        class VAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def dtype(self):
                return torch.float32

            def decode(self, value):
                return value * self.scale

        model = object.__new__(MossImageModel)
        model.vae = VAE()
        model.vae_device_torch = torch.device("cpu")
        model.vae_torch_dtype = torch.float32
        latents = torch.randn(1, 4, 2, 2, requires_grad=True)
        model.decode_latents(latents).square().mean().backward()
        self.assertIsNotNone(latents.grad)
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertIsNone(model.vae.scale.grad)

    def test_moss_cache_identity_tracks_encoder_and_image_processor(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = Path(directory) / "encoder.safetensors"
            encoder.write_bytes(b"first encoder")
            model = SimpleNamespace(
                arch="moss_image",
                max_text_length=256,
                model_config=SimpleNamespace(
                    name_or_path="OpenMOSS-Team/MOSS-Image-SFT",
                    text_encoder_path=None,
                    te_name_or_path=str(encoder),
                    model_kwargs={"edit": True, "sample_shift": 5, "uncond_skip_layers": [22]},
                    quantize_te=False, qtype_te=None,
                ),
                text_encoder=SimpleNamespace(config=SimpleNamespace(_name_or_path=str(encoder), _commit_hash="abc")),
                tokenizer=SimpleNamespace(name_or_path="moss-tokenizer", padding_side="right"),
                image_processor=SimpleNamespace(name_or_path="processor-v1"),
            )
            baseline = specialized_text_cache_identity(model)
            model.model_config.model_kwargs.update(sample_shift=7, uncond_skip_layers=[])
            self.assertEqual(specialized_text_cache_identity(model), baseline)
            model.image_processor.name_or_path = "processor-v2"
            self.assertNotEqual(specialized_text_cache_identity(model), baseline)
            model.image_processor.name_or_path = "processor-v1"
            model.tokenizer.chat_template = "changed chat template"
            self.assertNotEqual(specialized_text_cache_identity(model), baseline)
            model.tokenizer.chat_template = None
            encoder.write_bytes(b"changed encoder")
            self.assertNotEqual(specialized_text_cache_identity(model), baseline)

    def test_default_vae_file_provenance_changes_resolved_component_identity(self):
        from extensions_built_in.diffusion_models.moss_image import moss_image as moss_module

        class FakeVae:
            def eval(self):
                return self

            def requires_grad_(self, enabled):
                self.grad_enabled = enabled
                return self

        with tempfile.TemporaryDirectory() as directory:
            selected_vae = Path(directory) / "comfy-flux2-vae.safetensors"
            selected_vae.write_bytes(b"first VAE weights")
            model = object.__new__(MossImageModel)
            model.arch = "moss_image"
            model.latent_space_version = "moss_image"
            model.model_config = SimpleNamespace(
                name_or_path="OpenMOSS-Team/MOSS-Image-SFT", vae_path=None,
                latent_space_version=None,
                model_kwargs={}, quantize=False, qtype=None,
                quantize_te=False, qtype_te=None,
            )
            model.vae_torch_dtype = torch.float32
            model.torch_dtype = torch.float32
            model.transformer = SimpleNamespace()
            model.print_and_status_update = Mock()

            with unittest.mock.patch.object(moss_module, "comfy_vae_file", return_value=str(selected_vae)), \
                 unittest.mock.patch.object(moss_module, "load_file", return_value={"vae.weight": torch.ones(1)}) as load_file, \
                 unittest.mock.patch.object(moss_module.AutoEncoder, "load_from_state_dict", return_value=FakeVae()):
                model.vae = model._load_vae()
                first_identity = resolved_flow_components(model)
                load_file.assert_called_once_with(str(selected_vae))
                self.assertEqual(model.vae.aitk_load_source, {"path": str(selected_vae.resolve())})
                self.assertEqual(model.vae.aitk_load_files, [str(selected_vae.resolve())])
                self.assertFalse(model.vae.grad_enabled)
                prepare_specialized_conditioning(model)
                first_cache_key = model.get_latent_space_version()
                prepare_specialized_conditioning(model)
                self.assertEqual(model.get_latent_space_version(), first_cache_key)
                # Target latent identity stays independent of text/adapter and
                # guidance settings because they do not affect VAE encoding.
                model.model_config.model_kwargs.update(rank=32, cfg_scale=7)
                self.assertEqual(model.get_latent_space_version(), first_cache_key)

                selected_vae.write_bytes(b"different VAE weights, changed stat")
                second_identity = resolved_flow_components(model)
                prepare_specialized_conditioning(model)
                self.assertNotEqual(model.get_latent_space_version(), first_cache_key)

            self.assertNotEqual(first_identity["vae"], second_identity["vae"])


class MossUiContractTests(unittest.TestCase):
    def test_ordinary_moss_explicit_encoder_invalidates_text_cache(self):
        model = object.__new__(MossImageModel)
        model.arch = "moss_image"
        model.model_config = SimpleNamespace(
            te_name_or_path=None, text_encoder_path=None, model_kwargs={})
        self.assertEqual(model.get_text_embedding_space_version(), "moss_image")

        with tempfile.TemporaryDirectory() as directory:
            encoder = Path(directory) / "encoder.safetensors"
            encoder.write_bytes(b"first encoder")
            model.model_config.text_encoder_path = str(encoder)
            first_key = model.get_text_embedding_space_version()
            self.assertNotEqual(first_key, "moss_image")
            self.assertEqual(first_key, model.get_text_embedding_space_version())
            encoder.write_bytes(b"replaced encoder weights")
            self.assertNotEqual(first_key, model.get_text_embedding_space_version())

        model._specialized_text_cache_identity = "specialized"
        self.assertEqual(model.get_text_embedding_space_version(),
                         "moss_image_flow_components_v1_specialized")

    def test_specialized_moss_rechecks_native_preview_renderer_for_live_samples(self):
        comfy_sample = SimpleNamespace(comfy=SimpleNamespace(enabled=True))
        specialized = object.__new__(MossImageModel)
        specialized.arch = "moss_image"
        prepare_specialized_conditioning(specialized)
        with self.assertRaisesRegex(ValueError, "native previews"):
            specialized.validate_sample_config(comfy_sample)

        ordinary = object.__new__(MossImageModel)
        ordinary.arch = "moss_image"
        custom_sample = SimpleNamespace(comfy=SimpleNamespace(
            enabled=True, workflow_path="custom/moss_sample.json"))
        ordinary.validate_sample_config(custom_sample)
        with self.assertRaisesRegex(ValueError, "Krea 2 ComfyUI workflows are incompatible"):
            ordinary.validate_sample_config(SimpleNamespace(comfy=SimpleNamespace(
                enabled=True, workflow_path="config/comfy_templates/krea2_lora_sample.json.njk")))

    def test_ui_capability_table_and_variant_options_include_moss_modes(self):
        root = Path(__file__).resolve().parents[1]
        capabilities = (root / "ui/src/app/jobs/new/trainingCapabilities.ts").read_text()
        model_options = (root / "extensions_built_in/diffusion_models/ui.tsx").read_text()
        self.assertIn("moss_image: { label: 'MOSS-Image', bucketDivisibility: 16, editReferences: true }", capabilities)
        for variant in ("moss_image", "moss_image:base", "moss_image:edit", "moss_image:base_edit"):
            self.assertIn(f'name: "{variant}"', model_options)
        self.assertIn("MOSS-Image specialized modes require native previews", capabilities)
        self.assertIn("MOSS-Image edit training requires fixed edit-source images for every dataset", capabilities)


if __name__ == "__main__":
    unittest.main()
