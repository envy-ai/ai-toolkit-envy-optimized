"""FIBO2 prompt bundles retain sample-batch semantics through generic CFG."""

import tempfile
import unittest
from pathlib import Path

import torch

from extensions_built_in.diffusion_models.fibo2.fibo2 import Fibo2Model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds


class TinyTextEncoder:
    device = torch.device('cpu')

    def to(self, device):
        self.device = torch.device(device)
        return self


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context_scale = torch.nn.Parameter(torch.tensor(1.25))
        self.last_contexts = []

    @property
    def device(self):
        return self.context_scale.device

    @property
    def dtype(self):
        return self.context_scale.dtype

    def forward(self, latents, sigma, context, ref_latents=None):
        self.last_contexts.append(context.detach().clone())
        context_value = context.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        return latents * 0.1 + context_value * self.context_scale


class TinyNoiseScheduler:
    @staticmethod
    def scale_model_input(latents, _timestep):
        return latents


def make_model():
    model = object.__new__(Fibo2Model)
    model.device_torch = torch.device('cpu')
    model.torch_dtype = torch.float32
    model.text_encoder = TinyTextEncoder()
    model.unet = TinyTransformer()
    model.noise_scheduler = TinyNoiseScheduler()

    def encode_text(text, _images, _text_images, *, is_unconditional=False):
        tokens = max(1, len(text.split()) + 1)
        value = float(sum(text.encode('utf-8')) % 97 + (100 if is_unconditional else 0))
        return torch.full((6, tokens, 4), value, dtype=torch.float32)

    model._encode_text = encode_text
    model.network = None
    return model


def cached_single_prompt(embeds, index, path):
    single = AdvancedPromptEmbeds(
        text_embeds=[embeds.text_embeds[index].clone()],
        base_text_embeds=[embeds.base_text_embeds[index].clone()],
    )
    single.save(str(path))
    return AdvancedPromptEmbeds.load(str(path))


class Fibo2PredictNoiseBatchTests(unittest.TestCase):
    def test_batch_one_cfg_survives_negative_cache_concat_and_split(self):
        model = make_model()
        positive = model.get_prompt_embeds('a bright landscape')
        negative = model.get_prompt_embeds('', is_unconditional=True)

        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            negative = cached_single_prompt(
                negative, 0, Path(directory) / 'negative.safetensors'
            )

        joined = concat_prompt_embeds([negative, positive])
        self.assertIsInstance(joined, AdvancedPromptEmbeds)
        self.assertEqual(len(joined.text_embeds), 2)
        split = AdvancedPromptEmbeds.split_prompt_embeds(joined, num_parts=2)
        torch.testing.assert_close(split[0].text_embeds[0], negative.text_embeds[0])
        torch.testing.assert_close(split[1].text_embeds[0], positive.text_embeds[0])

        latents = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        guided = BaseModel.predict_noise(
            model,
            latents,
            conditional_embeddings=positive,
            unconditional_embeddings=negative,
            timestep=torch.tensor([500.0]),
            guidance_scale=3.5,
        )
        negative_value = negative.text_embeds[0].mean()
        positive_value = positive.text_embeds[0].mean()
        expected = latents * 0.1 + (
            negative_value + 3.5 * (positive_value - negative_value)
        ) * model.transformer.context_scale
        torch.testing.assert_close(guided, expected)
        self.assertEqual(len(model.transformer.last_contexts), 2)

    def test_batch_two_cfg_handles_unequal_bundle_lengths_and_gradient(self):
        model = make_model()
        prompts = ['short', 'a substantially longer second prompt']
        negatives = ['', 'a longer negative prompt']
        positive = model.get_prompt_embeds(prompts)
        negative = model.get_prompt_embeds(negatives, is_unconditional=True)
        self.assertNotEqual(
            positive.text_embeds[0].shape[1], positive.text_embeds[1].shape[1]
        )
        self.assertNotEqual(
            negative.text_embeds[0].shape[1], negative.text_embeds[1].shape[1]
        )

        joined = concat_prompt_embeds([negative, positive])
        self.assertEqual(len(joined.text_embeds), 4)
        split = AdvancedPromptEmbeds.split_prompt_embeds(joined, num_parts=4)
        for part, expected in zip(
            split,
            [*negative.text_embeds, *positive.text_embeds],
        ):
            torch.testing.assert_close(part.text_embeds[0], expected)

        latents = torch.ones((2, 1, 2, 2), dtype=torch.float32)
        guided = BaseModel.predict_noise(
            model,
            latents,
            conditional_embeddings=positive,
            unconditional_embeddings=negative,
            timestep=torch.tensor([250.0, 750.0]),
            guidance_scale=2.0,
        )
        negative_values = torch.stack([item.mean() for item in negative.text_embeds])
        positive_values = torch.stack([item.mean() for item in positive.text_embeds])
        expected_values = negative_values + 2.0 * (positive_values - negative_values)
        expected = latents * 0.1 + expected_values.view(2, 1, 1, 1) * model.transformer.context_scale
        torch.testing.assert_close(guided, expected)

        # Conditional-only prediction is the ordinary training path. The
        # generic wrapper prepares/scales inputs under no_grad but leaves the
        # model prediction differentiable.
        train_output = BaseModel.predict_noise(
            model,
            latents,
            conditional_embeddings=positive,
            timestep=torch.tensor([250.0, 750.0]),
        )
        train_output.sum().backward()
        self.assertIsNotNone(model.transformer.context_scale.grad)
        self.assertTrue(torch.isfinite(model.transformer.context_scale.grad))

    def test_generic_tensor_and_list_prompt_embeddings_keep_batch_behavior(self):
        model = make_model()
        model.get_noise_prediction = lambda **kwargs: torch.zeros_like(
            kwargs['latent_model_input']
        )
        latents = torch.ones((1, 1, 2, 2))

        tensor_embeds = PromptEmbeds(torch.zeros((1, 3, 4)))
        result = BaseModel.predict_noise(
            model, latents, text_embeddings=tensor_embeds, timestep=torch.tensor([1.0])
        )
        torch.testing.assert_close(result, torch.zeros_like(latents))

        list_embeds = PromptEmbeds(torch.zeros((1, 3, 4)))
        list_embeds.text_embeds = [torch.zeros((3, 4))]
        result = BaseModel.predict_noise(
            model, latents, text_embeddings=list_embeds, timestep=torch.tensor([1.0])
        )
        torch.testing.assert_close(result, torch.zeros_like(latents))


if __name__ == '__main__':
    unittest.main()
