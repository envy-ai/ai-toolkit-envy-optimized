"""CPU boundary regression for FIBO2 preview offload, glyph routing, CFG and Euler updates."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from extensions_built_in.diffusion_models.fibo2.fibo2 import Fibo2Model
from extensions_built_in.diffusion_models.fibo2.src.pipeline import Fibo2Pipeline
from extensions_built_in.diffusion_models.fibo2.src.transformer import Fibo2Params, Fibo2Transformer
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


class RecordingVAE:
    def __init__(self, model):
        self.model = model
        self.device = 'cuda:0'

    def to(self, device):
        self.device = str(device)
        self.model.lifecycle.append(('vae_to', self.device))
        return self


class TinyPreviewModel:
    def __init__(self):
        params = Fibo2Params(in_channels=4, dim=16, n_heads=2, n_layers=5,
            n_refiner_layers=1, injection_layers=(0, 1, 2, 3, 4), cap_feat_dim=12,
            perceiver_layers=1, min_gist_tokens=1, max_gist_tokens=2, gist_step=1,
            gist_min_text_len=2, gist_max_text_len=8, ffn_dim=24, axes_dims=(4, 2, 2))
        self.device_torch = torch.device('cpu')
        self.torch_dtype = torch.float32
        self.transformer = Fibo2Transformer(params).eval()
        self.model = self.transformer
        self.network = self.assistant_lora = None
        self.text_encoder = object()
        self.lifecycle = []
        self.vae = RecordingVAE(self)
        self.calls = []
        self.decoded = None

    def text_encoder_to(self, device):
        self.lifecycle.append(('text_encoder_to', str(device)))

    def predict_embeddings(self, state, timestep, embeds, refs=None):
        self.assert_parked()
        t = float(timestep.reshape(-1)[0])
        selected = 'base_text_embeds' if t > 900 and 'base_text_embeds' in embeds else 'text_embeds'
        self.calls.append({'timestep': t, 'selected': selected, 'refs': refs, 'state': state.detach().clone()})
        prediction = Fibo2Model.predict_embeddings(self, state, timestep, embeds, refs)
        self.calls[-1]['prediction'] = prediction.detach().clone()
        return prediction

    def assert_parked(self):
        assert self.lifecycle[-2:] == [('text_encoder_to', 'cpu'), ('vae_to', 'cpu')], self.lifecycle

    def decode_latents(self, state):
        self.decoded = state.detach().clone()
        return F.interpolate(state[:, :3], size=(32, 32), mode='nearest')


class Fibo2PipelineTests(unittest.TestCase):
    def test_invalid_canvas_and_step_counts_fail_before_parking_or_denoising(self):
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.randn(6, 4, 12)])
        invalid = (
            {'width': 520, 'height': 32, 'num_inference_steps': 3},
            {'width': 32, 'height': 17, 'num_inference_steps': 3},
            {'width': 32, 'height': 32, 'num_inference_steps': 0},
        )
        for options in invalid:
            with self.subTest(**options):
                model = TinyPreviewModel()
                with self.assertRaises(ValueError):
                    Fibo2Pipeline(model)(embeds, latents=torch.zeros(1, 4, 2, 2), **options)
                self.assertEqual(model.lifecycle, [])
                self.assertEqual(model.calls, [])

    def test_parks_components_routes_glyph_threshold_shares_cfg_refs_and_runs_euler(self):
        torch.manual_seed(918)
        model = TinyPreviewModel()
        positive = AdvancedPromptEmbeds(
            text_embeds=[torch.randn(6, 4, 12)],
            base_text_embeds=[torch.randn(6, 4, 12)],
        )
        negative = AdvancedPromptEmbeds(
            text_embeds=[torch.randn(6, 4, 12)],
            base_text_embeds=[torch.randn(6, 4, 12)],
        )
        refs = [[torch.randn(4, 2, 2)]]
        initial = torch.randn(1, 4, 2, 2)
        image = Fibo2Pipeline(model)(positive, negative, height=32, width=32,
            num_inference_steps=3, guidance_scale=2.5,
            generator=torch.Generator().manual_seed(99), latents=initial, ref_latents=refs)[0]

        self.assertEqual(model.lifecycle,
            [('text_encoder_to', 'cpu'), ('vae_to', 'cpu')])
        self.assertEqual([call['selected'] for call in model.calls],
            ['base_text_embeds', 'base_text_embeds', 'text_embeds', 'text_embeds', 'text_embeds', 'text_embeds'])
        self.assertTrue(all(call['refs'] is refs for call in model.calls))
        self.assertTrue(torch.isfinite(model.decoded).all())
        self.assertEqual(image.size, (32, 32))

        # Independent flow_mu schedule and explicit Euler accumulation from
        # the recorded conditional/CFG predictions.
        base = torch.linspace(1.0, 1 / 3, 3)
        mu = 0.5 + (2 * 2 - 256) / (4096 - 256) * (1.15 - 0.5)
        e_mu = torch.exp(torch.tensor(mu))
        schedule = e_mu * base / (1 + (e_mu - 1) * base)
        schedule = torch.cat((schedule, torch.zeros(1)))
        expected = initial.clone()
        for index in range(3):
            conditional, unconditional = model.calls[index * 2:index * 2 + 2]
            self.assertTrue(torch.equal(conditional['state'], expected))
            guided = unconditional['prediction'] + 2.5 * (conditional['prediction'] - unconditional['prediction'])
            expected = expected + (schedule[index + 1] - schedule[index]) * guided
        self.assertTrue(torch.allclose(model.decoded, expected, atol=2e-6, rtol=2e-6))
        expected_timesteps = [round(float(sigma) * 1000) for sigma in schedule[:-1] for _ in range(2)]
        self.assertEqual([round(call['timestep']) for call in model.calls], expected_timesteps)


if __name__ == '__main__':
    unittest.main()
