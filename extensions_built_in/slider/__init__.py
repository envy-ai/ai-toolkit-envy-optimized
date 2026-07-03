from toolkit.extension import Extension


class SliderTrainerExtension(Extension):
    uid = "slider"
    name = "Slider LoRA Trainer"

    @classmethod
    def get_process(cls):
        from .SliderTrainer import SliderTrainer

        return SliderTrainer


AI_TOOLKIT_EXTENSIONS = [
    SliderTrainerExtension,
]
