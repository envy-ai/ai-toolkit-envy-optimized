import importlib.util
import pathlib
import sys
import types
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Krea2DiffLoraExtractionTests(unittest.TestCase):
    def test_default_checkpoint_filename_uses_repo_suffix(self):
        from scripts.extract_krea2_diff_lora import default_checkpoint_filename

        self.assertEqual(
            default_checkpoint_filename("krea/Krea-2-Turbo"),
            "turbo.safetensors",
        )
        self.assertEqual(
            default_checkpoint_filename("krea/Krea-2-Raw"),
            "raw.safetensors",
        )

    def test_extracts_block_linear_diffs_to_peft_keys(self):
        from scripts.extract_krea2_diff_lora import (
            extract_krea2_lora_diff_from_state_dicts,
        )

        block_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 100
        base_state_dict = {
            "blocks.0.attn.wq.weight": torch.zeros(6, 4),
            "txtmlp.1.weight": torch.zeros(6, 4),
        }
        tuned_state_dict = {
            "blocks.0.attn.wq.weight": block_weight,
            "txtmlp.1.weight": torch.ones(6, 4),
        }

        lora_state_dict, stats = extract_krea2_lora_diff_from_state_dicts(
            base_state_dict,
            tuned_state_dict,
            dim=2,
            alpha=2,
            device="cpu",
            save_dtype=torch.float32,
            module_name_filter="",
            blocks_only=True,
            eps=0.0,
        )

        self.assertEqual(stats["extracted"], 1)
        self.assertIn(
            "transformer.blocks.0.attn.wq.lora_A.weight",
            lora_state_dict,
        )
        self.assertIn(
            "transformer.blocks.0.attn.wq.lora_B.weight",
            lora_state_dict,
        )
        self.assertEqual(
            lora_state_dict["transformer.blocks.0.attn.wq.lora_A.weight"].shape,
            (2, 4),
        )
        self.assertEqual(
            lora_state_dict["transformer.blocks.0.attn.wq.lora_B.weight"].shape,
            (6, 2),
        )
        self.assertFalse(any("txtmlp" in key for key in lora_state_dict))


def _install_krea_import_stubs(load_calls):
    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class Stub:
        pass

    class EntryNotFoundError(Exception):
        pass

    module(
        "huggingface_hub",
        hf_hub_download=lambda *args, **kwargs: "",
    )
    module("huggingface_hub.errors", EntryNotFoundError=EntryNotFoundError)
    module("diffusers", AutoencoderKLQwenImage=Stub)
    module("diffusers.utils", torch_utils=types.SimpleNamespace())
    module(
        "diffusers.utils.torch_utils",
        randn_tensor=lambda *args, **kwargs: torch.randn(*args, **kwargs),
    )
    module(
        "transformers",
        AutoTokenizer=Stub,
        Qwen2TokenizerFast=Stub,
        Qwen3VLForConditionalGeneration=Stub,
    )
    module(
        "optimum.quanto",
        freeze=lambda *args, **kwargs: None,
        QTensor=Stub,
    )
    module(
        "toolkit.config_modules",
        GenerateImageConfig=Stub,
        ModelConfig=Stub,
    )
    module("toolkit.models.base_model", BaseModel=object)
    module("toolkit.basic", flush=lambda *args, **kwargs: None)
    module("toolkit.advanced_prompt_embeds", AdvancedPromptEmbeds=Stub)
    module(
        "toolkit.samplers.custom_flowmatch_sampler",
        CustomFlowMatchEulerDiscreteScheduler=Stub,
    )
    module("toolkit.accelerator", unwrap_model=lambda model: model)
    module("toolkit.metadata", get_meta_for_safetensors=lambda meta, name: meta)
    module(
        "toolkit.util.quantize",
        quantize=lambda *args, **kwargs: None,
        get_qtype=lambda *args, **kwargs: None,
        quantize_model=lambda *args, **kwargs: None,
    )
    module(
        "toolkit.memory_management",
        MemoryManager=types.SimpleNamespace(attach=lambda *args, **kwargs: None),
    )

    assistant = types.SimpleNamespace(
        is_active=True,
        force_to=lambda *args, **kwargs: None,
    )

    def load_assistant_lora_from_path(path, model):
        load_calls.append((path, model))
        return assistant

    module(
        "toolkit.assistant_lora",
        load_assistant_lora_from_path=load_assistant_lora_from_path,
    )
    return assistant


def _load_krea_module(load_calls):
    assistant = _install_krea_import_stubs(load_calls)
    package_name = "extensions_built_in.diffusion_models.krea2"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT / "extensions_built_in/diffusion_models/krea2")]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.krea2",
        REPO_ROOT / "extensions_built_in/diffusion_models/krea2/krea2.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, assistant


class Krea2InferenceLoraTests(unittest.TestCase):
    def test_registers_inference_lora_without_activating_it(self):
        load_calls = []
        krea_module, assistant = _load_krea_module(load_calls)
        model = krea_module.Krea2Model.__new__(krea_module.Krea2Model)
        model.model_config = types.SimpleNamespace(
            inference_lora_path="/tmp/krea2_raw_to_turbo_r256.safetensors"
        )

        model._load_inference_lora()

        self.assertIs(model.assistant_lora, assistant)
        self.assertEqual(
            load_calls,
            [("/tmp/krea2_raw_to_turbo_r256.safetensors", model)],
        )
        self.assertFalse(model.assistant_lora.is_active)


if __name__ == "__main__":
    unittest.main()
