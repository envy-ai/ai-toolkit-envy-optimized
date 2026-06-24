import os
import sys
import types
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def install_base_model_import_stubs():
    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class Stub:
        pass

    module(
        "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl",
        rescale_noise_cfg=lambda *args, **kwargs: None,
    )
    module(
        "diffusers",
        StableDiffusionPipeline=Stub,
        StableDiffusionXLPipeline=Stub,
        T2IAdapter=Stub,
        DDPMScheduler=Stub,
        LCMScheduler=Stub,
        Transformer2DModel=Stub,
        AutoencoderTiny=Stub,
        ControlNetModel=Stub,
        AutoencoderKL=Stub,
        UNet2DConditionModel=Stub,
        PixArtAlphaPipeline=Stub,
        logging=types.SimpleNamespace(
            set_verbosity=lambda *args, **kwargs: None,
            ERROR=40,
        ),
    )
    module(
        "transformers",
        CLIPTextModel=Stub,
        CLIPTokenizer=Stub,
        CLIPTextModelWithProjection=Stub,
    )
    transforms_module = module(
        "torchvision.transforms",
        Resize=Stub,
        transforms=types.SimpleNamespace(),
        functional=types.SimpleNamespace(),
    )
    module("torchvision", transforms=transforms_module)

    module("toolkit.clip_vision_adapter", ClipVisionAdapter=Stub)
    module("toolkit.custom_adapter", CustomAdapter=Stub)
    module("toolkit.dequantize", patch_dequantization_on_save=lambda *args, **kwargs: None)
    module("toolkit.ip_adapter", IPAdapter=Stub)
    module("toolkit.models.decorator", Decorator=Stub)
    module("toolkit.paths", KEYMAPS_ROOT="", get_path=lambda *args, **kwargs: "")
    module(
        "toolkit.prompt_utils",
        inject_trigger_into_prompt=lambda prompt, *args, **kwargs: prompt,
        PromptEmbeds=Stub,
        concat_prompt_embeds=lambda embeds: embeds,
    )
    module("toolkit.reference_adapter", ReferenceAdapter=Stub)
    module(
        "toolkit.sd_device_states_presets",
        empty_preset={
            "vae": {},
            "unet": {},
            "refiner_unet": {},
            "text_encoder": {},
            "adapter": {},
        },
    )
    module(
        "toolkit.train_tools",
        get_torch_dtype=lambda dtype: torch.float32,
        apply_noise_offset=lambda *args, **kwargs: None,
    )
    module("toolkit.pipelines", CustomStableDiffusionXLPipeline=Stub)
    module(
        "toolkit.accelerator",
        get_accelerator=lambda: None,
        unwrap_model=lambda model: model,
    )
    module("toolkit.print", print_acc=lambda *args, **kwargs: None)
    module("toolkit.basic", flush=lambda *args, **kwargs: None)


install_base_model_import_stubs()

from toolkit.models.base_model import BaseModel


class RecordingBaseModel(BaseModel):
    def save_device_state(self):
        self.saved_device_state = True

    def set_device_state(self, state):
        self.recorded_state = state


def make_recording_model():
    model = RecordingBaseModel.__new__(RecordingBaseModel)
    model.device_torch = torch.device("cuda")
    model.te_device_torch = torch.device("cuda")
    model.vae_device_torch = torch.device("cuda")
    model.text_encoder = object()
    model.adapter = None
    model.refiner_unet = None
    return model


class BaseModelGenerationDeviceStateTests(unittest.TestCase):
    def test_generate_preset_activates_text_encoder_by_default(self):
        model = make_recording_model()

        model.set_device_state_preset("generate")

        self.assertEqual(model.recorded_state["vae"]["device"], torch.device("cuda"))
        self.assertEqual(model.recorded_state["unet"]["device"], torch.device("cuda"))
        self.assertEqual(
            model.recorded_state["text_encoder"]["device"],
            torch.device("cuda"),
        )

    def test_generate_preset_can_keep_large_modules_on_cpu(self):
        model = make_recording_model()
        model.generation_cpu_offload_modules = {"vae", "unet", "text_encoder"}

        model.set_device_state_preset("generate")

        self.assertEqual(model.recorded_state["vae"]["device"], "cpu")
        self.assertEqual(model.recorded_state["unet"]["device"], "cpu")
        self.assertEqual(model.recorded_state["text_encoder"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
