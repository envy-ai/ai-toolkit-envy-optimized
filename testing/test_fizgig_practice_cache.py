"""Resume the real Fizgig practice builders from their persisted latents."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from PIL import Image

from extensions_built_in.sd_trainer.FizgigSliderTrainer import FizgigSliderTrainer
from toolkit.prompt_utils import PromptEmbeds


def embedding(value):
    return PromptEmbeds(torch.full((1, 2, 3), float(value)))


class PracticeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / 'base.safetensors'
        self.base.write_bytes(b'original model identity')
        self.renders = 0
        self.encodes = 0

    def trainer(self):
        config = SimpleNamespace(name_or_path=str(self.base), model_kwargs={},
            quantize=False, qtype='qfloat8', quantize_te=False, qtype_te='qfloat8')
        transformer = torch.nn.Linear(2, 2)
        vae = torch.nn.Linear(2, 2)
        def generate(*args, **kwargs):
            self.renders += 1
            return Image.new('RGB', (64, 64), 'red')
        def encode(*args, **kwargs):
            self.encodes += 1
            return torch.full((1, 4, 2, 2), float(self.encodes))
        sd = SimpleNamespace(arch='krea2', model_config=config,
            transformer=transformer, unet=transformer, vae=vae, text_encoder=None,
            tokenizer=None, processor=None, vl_processor=None,
            torch_dtype=torch.float32, vae_torch_dtype=torch.float32, te_torch_dtype=torch.float32,
            max_text_length=256, save_device_state=lambda: None,
            restore_device_state=lambda: None, text_encoder_to=lambda device: None,
            vae_device_torch=torch.device('cpu'), get_generation_pipeline=lambda: object(),
            generate_single_image=generate, encode_images=encode)
        trainer = object.__new__(FizgigSliderTrainer)
        trainer.sd = sd
        trainer.model_config = config
        trainer.save_root = str(self.root)
        trainer.device_torch = torch.device('cpu')
        trainer.network = SimpleNamespace(is_active=True, multiplier=1.)
        trainer.sample_config = SimpleNamespace(seed=42)
        trainer.slider_cfg = 1.
        trainer.slider_bank_resolution = 64
        trainer.slider_bank_steps = 2
        trainer.slider_bank_size = 3
        trainer.slider_prompt_triplets = [('cat', 'large cat', 'small cat')]
        trainer.slider_embeds = [(embedding(0), embedding(1), embedding(-1))]
        trainer.slider_cfg_negative_embeds = None
        trainer.slider_cfg_negative_prompts = None
        trainer.slider_bank = []
        trainer.slider_anchor_prompts = [('dog', '')]
        trainer.slider_anchor_embeds = [(embedding(5), None)]
        trainer.slider_anchor_bank = []
        trainer.print = lambda *args: None
        trainer.record_training_examples = True
        trainer.logger = SimpleNamespace()
        return trainer

    def test_resume_reuses_exact_latents_without_render_or_vae(self):
        first = self.trainer()
        first._build_prompt_bank()
        self.assertEqual((self.renders, self.encodes), (3, 3))
        initial = [latent.clone() for latent, _ in first.slider_bank]
        resumed = self.trainer()
        resumed._build_prompt_bank()
        self.assertEqual((self.renders, self.encodes), (3, 3))
        self.assertEqual([index for _, index in resumed.slider_bank], [0, 0, 0])
        for index, ((latent, _), expected) in enumerate(zip(resumed.slider_bank, initial)):
            torch.testing.assert_close(latent, expected)
            source = resumed._loss_report_bank_sources[id(latent)]
            self.assertEqual(Path(source.path).name, f'practice_{index:06d}.png')

    def test_missing_or_corrupt_entries_rebuild_individually(self):
        self.trainer()._build_prompt_bank()
        folder = self.root / 'fizgig_practice_cache'
        (folder / 'prompt_000000.safetensors').unlink()
        (folder / 'prompt_000001.safetensors').write_bytes(b'broken cache')
        resumed = self.trainer()
        resumed._build_prompt_bank()
        self.assertEqual((self.renders, self.encodes), (5, 5))
        self.assertEqual([float(latent.mean()) for latent, _ in resumed.slider_bank], [4., 5., 3.])

    def test_changed_settings_or_base_model_invalidate_cache(self):
        self.trainer()._build_prompt_bank()
        changed = self.trainer()
        changed.slider_bank_steps = 3
        changed._build_prompt_bank()
        self.assertEqual(self.renders, 6)
        self.base.write_bytes(b'updated model identity')
        changed = self.trainer()
        changed.slider_bank_steps = 3
        changed._build_prompt_bank()
        self.assertEqual(self.renders, 9)

    def test_anchor_practice_bank_resumes_too(self):
        first = self.trainer()
        first._build_anchor_prompt_bank()
        self.assertEqual((self.renders, self.encodes), (1, 1))
        resumed = self.trainer()
        resumed._build_anchor_prompt_bank()
        self.assertEqual((self.renders, self.encodes), (1, 1))
        latent, positive, negative = resumed.slider_anchor_bank[0]
        self.assertEqual(float(latent.mean()), 1.)
        self.assertIs(positive, resumed.slider_anchor_embeds[0][0])
        self.assertIsNone(negative)
        self.assertEqual(Path(resumed._loss_report_bank_sources[id(latent)].path).name,
                         'anchor_000000.png')
