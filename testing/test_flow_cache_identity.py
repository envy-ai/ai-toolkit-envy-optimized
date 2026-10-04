import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from toolkit.flow_cache_identity import specialized_text_cache_identity, prepare_specialized_conditioning


class FlowCacheIdentityTests(unittest.TestCase):
    def model(self, arch, path):
        return SimpleNamespace(arch=arch, max_text_length=256, max_sequence_length=512,
            model_config=SimpleNamespace(name_or_path=path, text_encoder_path=path, model_kwargs={},
                quantize_te=True, qtype_te='convrotint4'),
            text_encoder=SimpleNamespace(config=SimpleNamespace(_name_or_path=path, _commit_hash='123')),
            tokenizer=SimpleNamespace(name_or_path=path, padding_side='right', model_max_length=256),
            trainable_model=SimpleNamespace(text_conditioner=SimpleNamespace(config=SimpleNamespace(_name_or_path=path))))

    def test_components_length_conditioner_revision_change_but_cfg_and_preview_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.safetensors'
            path.write_bytes(b'weights')
            for arch in ('krea2', 'anima', 'ideogram4'):
                model = self.model(arch, str(path))
                baseline = specialized_text_cache_identity(model)
                model.model_config.model_kwargs.update(ideogram_cfg_reference='negative_prompt',
                    schedule_mu=2, ideogram_schedule_std=2, preview_auto=True)
                self.assertEqual(specialized_text_cache_identity(model), baseline)
                model.max_text_length = 512
                self.assertNotEqual(specialized_text_cache_identity(model), baseline)
                model.max_text_length = 256
                model.text_encoder.config._commit_hash = '456'
                self.assertNotEqual(specialized_text_cache_identity(model), baseline)
                model.text_encoder.config._commit_hash = '123'
                path.write_bytes(path.read_bytes() + b'changed')
                self.assertNotEqual(specialized_text_cache_identity(model), baseline)
            self.assertIsNone(specialized_text_cache_identity(self.model('qwen_image_2', str(path))))
            self.assertIsNone(specialized_text_cache_identity(self.model('flux', str(path))))

    def test_tokenizer_and_edit_settings_change_identity(self):
        model = self.model('krea2', 'remote/model@revision')
        baseline = specialized_text_cache_identity(model)
        model.model_config.model_kwargs['edit'] = True
        self.assertNotEqual(specialized_text_cache_identity(model), baseline)
        model.model_config.model_kwargs.clear()
        model.tokenizer.padding_side = 'left'
        self.assertNotEqual(specialized_text_cache_identity(model), baseline)

    def test_qwen_kto_opt_in_tracks_encoder_changes_without_touching_legacy_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.safetensors'
            path.write_bytes(b'encoder')
            model = self.model('qwen_image_2', str(path))
            baseline = specialized_text_cache_identity(model, include_qwen=True)
            self.assertIsNotNone(baseline)
            self.assertIsNone(specialized_text_cache_identity(model))
            model.model_config.model_kwargs.update(beta=2, score_window_size=8, schedule_mu=3)
            self.assertEqual(specialized_text_cache_identity(model, include_qwen=True), baseline)
            path.write_bytes(b'new encoder weights')
            self.assertNotEqual(specialized_text_cache_identity(model, include_qwen=True), baseline)
            self.assertIsNone(specialized_text_cache_identity(model))

    def test_krea_reference_vlm_budget_and_processor_change_identity(self):
        model = self.model('krea2', 'remote/model@revision')
        baseline = specialized_text_cache_identity(model)
        model.model_config.model_kwargs['vlm_max_pixels'] = 256 * 256
        self.assertNotEqual(specialized_text_cache_identity(model), baseline)
        model.model_config.model_kwargs.clear()
        model.vl_processor = SimpleNamespace(name_or_path='processor-v2')
        self.assertNotEqual(specialized_text_cache_identity(model), baseline)

    def test_specialized_krea_composited_presentation_changes_identity(self):
        model = self.model('krea2', 'remote/model@revision')
        model.is_edit = True
        baseline = specialized_text_cache_identity(model)
        prepare_specialized_conditioning(model)
        self.assertNotEqual(specialized_text_cache_identity(model), baseline)
        self.assertTrue(model.cache_processed_control_text_embeddings)


if __name__ == '__main__':
    unittest.main()
