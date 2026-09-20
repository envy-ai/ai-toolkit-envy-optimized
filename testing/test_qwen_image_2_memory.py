import os
import importlib.util
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PIPELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "extensions_built_in/diffusion_models/qwen_image_2/src/pipeline.py",
)
SPEC = importlib.util.spec_from_file_location("qwen_image_2_pipeline_memory_test", PIPELINE_PATH)
PIPELINE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE_MODULE)
QwenImage21PromptEncoder = PIPELINE_MODULE.QwenImage21PromptEncoder


class _HookHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _FakeNorm:
    def __init__(self):
        self.handle = None

    def register_forward_hook(self, hook):
        self.hook = hook
        self.handle = _HookHandle()
        return self.handle


class _FakeInnerTextModel:
    def __init__(self):
        self.language_model = SimpleNamespace(norm=_FakeNorm())
        self.called = False
        self.forward_kwargs = None

    def __call__(self, **kwargs):
        self.called = True
        self.forward_kwargs = kwargs
        sequence = kwargs["input_ids"].shape[1]
        hidden = torch.arange(sequence * 3, dtype=torch.float32).reshape(
            1, sequence, 3
        )
        return SimpleNamespace(hidden_states=(hidden,))


class _FakeTextEncoder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = _FakeInnerTextModel()
        self.outer_called = False

    def __call__(self, **kwargs):
        self.outer_called = True
        raise AssertionError("the causal-LM wrapper must not run for prompt encoding")


class _FakeBatch:
    def __init__(self):
        self.input_ids = torch.tensor([[1, 2, 10, 11]], dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class _FakeTokenizer:
    def encode(self, text):
        return [99]


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize, return_dict):
        return [[1, 2]]

    def __call__(self, **kwargs):
        return _FakeBatch()


class QwenImage21PromptMemoryTests(unittest.TestCase):
    def test_prompt_encoding_skips_lm_head_and_kv_cache(self):
        text_encoder = _FakeTextEncoder()
        encoder = QwenImage21PromptEncoder(text_encoder, _FakeProcessor())

        embeds, masks, slots = encoder.encode(["paint a lighthouse"])

        self.assertFalse(text_encoder.outer_called)
        self.assertTrue(text_encoder.model.called)
        self.assertFalse(text_encoder.model.forward_kwargs["use_cache"])
        self.assertEqual(embeds[0].shape, (2, 3))
        self.assertEqual(masks[0].tolist(), [1, 1])
        self.assertEqual(slots[0].tolist(), [False, False])
        self.assertTrue(text_encoder.model.language_model.norm.handle.removed)


if __name__ == "__main__":
    unittest.main()
