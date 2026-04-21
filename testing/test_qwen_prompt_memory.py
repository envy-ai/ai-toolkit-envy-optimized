import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.qwen_image.qwen_image import (
    QwenImageModel,
    _PipelineLoraController,
    _quanto_skip_missing_base_weight_load,
)
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.dataloader_mixins import TextEmbeddingCachingMixin
from toolkit.models.base_model import BaseModel


class RecordingBaseModel(BaseModel):
    def save_device_state(self):
        self.saved_device_state = True

    def set_device_state(self, state):
        self.recorded_state = state


class FakeTokenBatch:
    def __init__(self):
        self.input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
        self.attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class FakeTokenizer:
    model_max_length = 1024

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        self.last_text = text
        self.last_max_length = max_length
        return FakeTokenBatch()


class FakeInnerTextModel:
    def __init__(self):
        self.called = False

    def __call__(self, **kwargs):
        self.called = True
        hidden = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        return type("FakeOutput", (), {"hidden_states": (hidden,)})


class FakeQwenTextEncoder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model = FakeInnerTextModel()

    def to(self, device, *args, **kwargs):
        self.device = torch.device(device)
        return self

    def __call__(self, *args, **kwargs):
        raise AssertionError("outer causal LM forward should not be called")


class FakeQwenPipeline:
    prompt_template_encode = "{}"
    prompt_template_encode_start_idx = 1
    tokenizer_max_length = 1024

    def __init__(self):
        self.text_encoder = FakeQwenTextEncoder()
        self.tokenizer = FakeTokenizer()

    def _extract_masked_hidden(self, hidden_states, attention_mask):
        return [
            hidden_states[idx][attention_mask[idx].bool()]
            for idx in range(hidden_states.shape[0])
        ]

    def encode_prompt(self, *args, **kwargs):
        raise AssertionError("pipeline encode_prompt should not be used")


class FakePromptEmbeds:
    def save(self, path):
        with open(path, "w") as f:
            f.write("cached")


class FakeTextEmbeddingFileItem:
    encode_control_in_text_embeddings = False
    caption = "anime digital painting"

    def __init__(self, path):
        self.path = path
        self.is_text_embedding_cached = False
        self.latent_load_device = None

    def get_text_embedding_path(self, recalculate=False):
        return self.path


class FakeTextEmbeddingSD:
    device = "cuda"

    def __init__(self):
        self.device_state_restored = False
        self.device_state_preset = None

    def set_device_state_preset(self, preset):
        self.device_state_preset = preset

    def restore_device_state(self):
        self.device_state_restored = True

    def encode_prompt(self, caption):
        return FakePromptEmbeds()


class FakeAccelerator:
    def __init__(self):
        self.prepared = []
        self.even_batches = None

    def prepare(self, obj):
        self.prepared.append(obj.name)
        return obj


class FakeModule:
    def __init__(self, name):
        self.name = name


class FakePipelineTransformer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.moves = []

    def to(self, device=None, dtype=None, *args, **kwargs):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self.moves.append((self.device, self.dtype))
        return self


class FakeLoraPipeline:
    def __init__(self):
        self.transformer = FakePipelineTransformer()
        self.loaded_on_device = None
        self.low_cpu_mem_usage = None
        self.disabled = False
        self.enabled = False
        self.adapter_name = None

    def load_lora_weights(
        self, lora_path, adapter_name=None, low_cpu_mem_usage=None
    ):
        self.loaded_on_device = self.transformer.device
        self.adapter_name = adapter_name
        self.low_cpu_mem_usage = low_cpu_mem_usage

    def disable_lora(self):
        self.disabled = True

    def enable_lora(self):
        self.enabled = True


class FakeAdapterModule:
    def __init__(self):
        self.moves = []

    def to(self, device=None, dtype=None, *args, **kwargs):
        self.moves.append((torch.device(device), dtype))
        return self


class FakePeftLayer:
    def __init__(self):
        self.lora_A = {"inference": FakeAdapterModule()}
        self.lora_B = {"inference": FakeAdapterModule()}


class FakePeftTransformer(FakePipelineTransformer):
    def __init__(self):
        super().__init__()
        self.peft_layer = FakePeftLayer()

    def modules(self):
        return [self, self.peft_layer]


class FakePeftLoraPipeline(FakeLoraPipeline):
    def __init__(self):
        super().__init__()
        self.transformer = FakePeftTransformer()


class QwenPromptMemoryTests(unittest.TestCase):
    def test_base_model_cache_text_encoder_preset_activates_only_text_encoder(self):
        model = RecordingBaseModel.__new__(RecordingBaseModel)
        model.device_torch = torch.device("cuda")
        model.te_device_torch = torch.device("cuda")
        model.vae_device_torch = torch.device("cuda")
        model.text_encoder = [object()]
        model.adapter = None
        model.refiner_unet = None

        model.set_device_state_preset("cache_text_encoder")

        self.assertEqual(model.recorded_state["vae"]["device"], "cpu")
        self.assertEqual(model.recorded_state["unet"]["device"], "cpu")
        self.assertEqual(
            model.recorded_state["text_encoder"][0]["device"], torch.device("cuda")
        )

    def test_quantized_module_cache_path_is_temporarily_disabled(self):
        model = BaseModel.__new__(BaseModel)
        model.model_config = SimpleNamespace(
            cache_quantized_models=True,
            quantized_model_cache_dir="models/.quantized_training_cache",
        )
        model.arch = "qwen_image"
        model.dtype = "bf16"

        cache_path = model.get_quantized_module_cache_path(
            component_name="transformer",
            qtype="uint6",
            source_ref={"repo": "Qwen/Qwen-Image-2512"},
        )

        self.assertIsNone(cache_path)

    def test_qwen_prompt_embeddings_skip_outer_causal_lm_logits(self):
        model = QwenImageModel.__new__(QwenImageModel)
        model.device_torch = torch.device("cpu")
        model.torch_dtype = torch.float32
        model.pipeline = FakeQwenPipeline()

        prompt_embeds = QwenImageModel.get_prompt_embeds(model, "anime digital painting")

        self.assertTrue(model.pipeline.text_encoder.model.called)
        self.assertEqual(prompt_embeds.text_embeds.shape, (1, 2, 2))
        self.assertEqual(prompt_embeds.attention_mask.tolist(), [[1, 1]])

    def test_text_embedding_cache_restores_device_state_after_encoding(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = TextEmbeddingCachingMixin.__new__(TextEmbeddingCachingMixin)
            dataset.dataset_path = tmp_dir
            dataset.sd = FakeTextEmbeddingSD()
            dataset.file_list = [
                FakeTextEmbeddingFileItem(os.path.join(tmp_dir, "prompt.safetensors"))
            ]

            TextEmbeddingCachingMixin.cache_text_embeddings(dataset)

        self.assertEqual(dataset.sd.device_state_preset, "cache_text_encoder")
        self.assertTrue(dataset.sd.device_state_restored)

    def test_prepare_accelerator_does_not_prepare_vae_when_latents_are_cached(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.accelerator = FakeAccelerator()
        process.is_latents_cached = True
        process.sd = type(
            "FakeSD",
            (),
            {
                "vae": FakeModule("vae"),
                "unet": FakeModule("unet"),
                "text_encoder": None,
                "refiner_unet": None,
                "network": None,
            },
        )()
        process.train_config = type(
            "FakeTrainConfig",
            (),
            {"train_text_encoder": False, "train_refiner": False},
        )()
        process.adapter = None
        process.adapter_config = None
        process.optimizer = FakeModule("optimizer")
        process.lr_scheduler = None
        process.modules_being_trained = []

        BaseSDTrainProcess.prepare_accelerator(process)

        self.assertNotIn("vae", process.accelerator.prepared)
        self.assertIn("unet", process.accelerator.prepared)

    def test_qwen_inference_lora_registers_diffusers_loader_controller(self):
        model = QwenImageModel.__new__(QwenImageModel)
        model.model_config = SimpleNamespace(
            inference_lora_path="lightning.safetensors"
        )
        model.pipeline = FakeLoraPipeline()
        model.device_torch = torch.device("cuda")
        model.torch_dtype = torch.bfloat16
        model.assistant_lora = None

        model._load_inference_lora()

        self.assertIsInstance(model.assistant_lora, _PipelineLoraController)
        self.assertFalse(model.assistant_lora.is_active)

    def test_qwen_inference_lora_moves_transformer_before_diffusers_load(self):
        pipeline = FakeLoraPipeline()
        controller = _PipelineLoraController(
            pipeline,
            adapter_name="inference",
            lora_path="lightning.safetensors",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

        controller.is_active = True

        self.assertEqual(pipeline.loaded_on_device, torch.device("cuda"))
        self.assertEqual(pipeline.transformer.dtype, torch.bfloat16)
        self.assertEqual(pipeline.adapter_name, "inference")
        self.assertFalse(pipeline.low_cpu_mem_usage)
        self.assertTrue(pipeline.enabled)

    def test_qwen_inference_lora_offloads_only_adapter_when_inactive(self):
        pipeline = FakePeftLoraPipeline()
        controller = _PipelineLoraController(
            pipeline,
            adapter_name="inference",
            lora_path="lightning.safetensors",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

        controller.is_active = True
        transformer_moves_after_load = list(pipeline.transformer.moves)
        controller.is_active = False
        controller.force_to("cpu", torch.bfloat16)

        self.assertEqual(pipeline.transformer.moves, transformer_moves_after_load)
        self.assertEqual(
            pipeline.transformer.peft_layer.lora_A["inference"].moves[-1],
            (torch.device("cpu"), torch.bfloat16),
        )
        self.assertEqual(
            pipeline.transformer.peft_layer.lora_B["inference"].moves[-1],
            (torch.device("cpu"), torch.bfloat16),
        )

    def test_qwen_inference_lora_stays_offloaded_after_transformer_to_cuda(self):
        pipeline = FakePeftLoraPipeline()
        controller = _PipelineLoraController(
            pipeline,
            adapter_name="inference",
            lora_path="lightning.safetensors",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

        controller.is_active = True
        controller.is_active = False
        controller.force_to("cpu", torch.bfloat16)
        pipeline.transformer.peft_layer.lora_A["inference"].moves.clear()

        pipeline.transformer.to(device=torch.device("cuda"), dtype=torch.bfloat16)

        self.assertEqual(
            pipeline.transformer.peft_layer.lora_A["inference"].moves[-1],
            (torch.device("cpu"), torch.bfloat16),
        )

    def test_qwen_inference_lora_reactivation_moves_adapter_to_target(self):
        pipeline = FakePeftLoraPipeline()
        controller = _PipelineLoraController(
            pipeline,
            adapter_name="inference",
            lora_path="lightning.safetensors",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

        controller.is_active = True
        controller.is_active = False
        controller.force_to("cpu", torch.bfloat16)
        pipeline.transformer.peft_layer.lora_A["inference"].moves.clear()

        controller.is_active = True

        self.assertEqual(
            pipeline.transformer.peft_layer.lora_A["inference"].moves[-1],
            (torch.device("cuda"), torch.bfloat16),
        )

    def test_quanto_loader_patch_allows_adapter_only_state_dicts(self):
        from optimum.quanto import freeze, qfloat8, quantize

        module = torch.nn.Linear(4, 4, bias=False)
        quantize(module, weights=qfloat8)
        freeze(module)

        with self.assertRaises(KeyError):
            module.load_state_dict({}, strict=False)

        with _quanto_skip_missing_base_weight_load():
            module.load_state_dict({}, strict=False)

    def test_qwen_bare_transformer_lora_keys_are_prefixed_before_loading(self):
        model = QwenImageModel.__new__(QwenImageModel)
        converted = model.convert_lora_weights_before_load(
            {
                "transformer_blocks.0.attn.to_q.lora_down.weight": torch.ones(1),
                "transformer.transformer_blocks.0.attn.to_k.lora_down.weight": torch.ones(1),
            }
        )

        self.assertIn(
            "transformer.transformer_blocks.0.attn.to_q.lora_down.weight",
            converted,
        )
        self.assertIn(
            "transformer.transformer_blocks.0.attn.to_k.lora_down.weight",
            converted,
        )
        self.assertNotIn(
            "transformer.transformer.transformer_blocks.0.attn.to_k.lora_down.weight",
            converted,
        )


if __name__ == "__main__":
    unittest.main()
