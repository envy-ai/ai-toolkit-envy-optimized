# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


# This is for generic training (LoRA, Dreambooth, FineTuning)
class SDTrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "sd_trainer"

    # name is the name of the extension for printing
    name = "SD Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .SDTrainer import SDTrainer

        return SDTrainer


# This is for generic training (LoRA, Dreambooth, FineTuning)
class UITrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "ui_trainer"

    # name is the name of the extension for printing
    name = "UI Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .UITrainer import UITrainer

        return UITrainer


# This is a universal trainer that can be from ui or api
class DiffusionTrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "diffusion_trainer"

    # name is the name of the extension for printing
    name = "Diffusion Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .DiffusionTrainer import DiffusionTrainer

        return DiffusionTrainer


class FizgigImageSliderExtension(Extension):
    uid = "fizgig_image_slider"
    name = "Fizgig Image Slider"

    @classmethod
    def get_process(cls):
        from .FizgigSliderTrainer import FizgigSliderTrainer
        return FizgigSliderTrainer


class FizgigPromptSliderExtension(Extension):
    uid = "fizgig_prompt_slider"
    name = "Fizgig Prompt Slider"

    @classmethod
    def get_process(cls):
        from .FizgigSliderTrainer import FizgigSliderTrainer
        return FizgigSliderTrainer


class QwenFlowDPOExtension(Extension):
    uid = "qwen_flow_dpo"
    name = "Qwen Image 2.1 Flow-DPO (LoRA)"

    @classmethod
    def get_process(cls):
        from .QwenFlowDPOTrainer import QwenFlowDPOTrainer
        return QwenFlowDPOTrainer


class QwenGuidanceDistillationExtension(Extension):
    uid = "qwen_guidance_distillation"
    name = "Qwen Image 2.1 Guidance Distillation (LoRA)"

    @classmethod
    def get_process(cls):
        from .QwenGuidanceDistillationTrainer import QwenGuidanceDistillationTrainer
        return QwenGuidanceDistillationTrainer


class FlowDPOExtension(QwenFlowDPOExtension):
    uid = 'flow_dpo'
    name = 'Flow-DPO LoRA'


class GuidanceDistillationExtension(QwenGuidanceDistillationExtension):
    uid = 'guidance_distillation'
    name = 'Guidance Distillation LoRA'


class DiffusionKTOExtension(Extension):
    uid = 'diffusion_kto'
    name = 'Diffusion-KTO LoRA (experimental flow adaptation)'

    @classmethod
    def get_process(cls):
        from .DiffusionKTOTrainer import DiffusionKTOTrainer
        return DiffusionKTOTrainer


# Semantic discovery directions, initially ordinary Qwen Image 2.1 LoRAs.
class SliderSpaceExtension(Extension):
    uid = 'sliderspace'
    name = 'SliderSpace (LoRAs)'

    @classmethod
    def get_process(cls):
        from .SliderSpaceTrainer import SliderSpaceTrainer
        return SliderSpaceTrainer


# for backwards compatability
class TextualInversionTrainer(SDTrainerExtension):
    uid = "textual_inversion_trainer"


AI_TOOLKIT_EXTENSIONS = [
    # you can put a list of extensions here
    SDTrainerExtension,
    TextualInversionTrainer,
    UITrainerExtension,
    DiffusionTrainerExtension,
    FizgigImageSliderExtension,
    FizgigPromptSliderExtension,
    QwenFlowDPOExtension,
    QwenGuidanceDistillationExtension,
    FlowDPOExtension,
    GuidanceDistillationExtension,
    DiffusionKTOExtension,
    SliderSpaceExtension,
]
