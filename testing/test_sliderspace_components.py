"""Resolved loader provenance and complete SliderSpace cache/resume fixtures."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import save_file, load_file

from testing import test_sliderspace as fixtures
from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.flow_cache_identity import (resolved_flow_components, resolved_semantic_components,
                                        component_identity_digest, specialized_text_cache_identity)
from toolkit.flow_training import FlowTrainingProfile
from toolkit.sliderspace import SliderSpaceConfig, discovery_entries, scan_discovery_images
from extensions_built_in.diffusion_models.krea2.krea2 import _load_mmdit_state_dict
from extensions_built_in.diffusion_models.ideogram4.ideogram4 import _load_component_state_dict


class LoadedLinear(torch.nn.Linear, OstrisModelMixin):
    pass


class SliderSpaceComponentTests(unittest.TestCase):
    def test_mixin_records_resolved_comfy_file_and_pretrained_source_without_extra_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / 'encoder.safetensors'
            save_file({'weight': torch.ones(2, 2)}, str(weight))
            with patch.object(LoadedLinear, 'resolve_comfy_weights', return_value=str(weight)), \
                 patch.object(LoadedLinear, '_load_single_file', return_value=LoadedLinear(2, 2)) as load:
                model = LoadedLinear.load_model('author/encoder')
            self.assertEqual(model.aitk_load_source, {'path': str(weight.resolve()),
                'config_path': 'author/encoder', 'subfolder': None})
            self.assertEqual(load.call_count, 1)
            with patch.object(LoadedLinear, 'aitk_from_pretrained', return_value=LoadedLinear(2, 2)) as load:
                model = LoadedLinear.load_model('author/encoder', use_comfy_weights=False, revision='pinned')
            self.assertEqual(model.aitk_load_source, {'path': 'author/encoder', 'subfolder': None, 'revision': 'pinned'})
            self.assertEqual(load.call_count, 1)

    def test_actual_native_state_loaders_record_local_single_and_remote_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / 'krea.safetensors'
            save_file({'a': torch.ones(2)}, str(weight))
            files = []
            self.assertEqual(set(_load_mmdit_state_dict(str(weight), None, source_files=files)), {'a'})
            self.assertEqual(files, [str(weight)])
            files = []
            with patch('extensions_built_in.diffusion_models.krea2.krea2.huggingface_hub.hf_hub_download', return_value=str(weight)):
                _load_mmdit_state_dict('author/Krea-2-Raw', None, source_files=files)
            self.assertEqual(files, [str(weight)])
            folder = root / 'transformer'
            folder.mkdir()
            index = folder / 'diffusion_pytorch_model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}}))
            for name, key in [('one.safetensors', 'a'), ('two.safetensors', 'b')]:
                save_file({key: torch.ones(2)}, str(folder / name))
            files = []
            state = _load_component_state_dict(str(root), 'transformer', 'diffusion_pytorch_model', source_files=files)
            self.assertEqual(set(state), {'a', 'b'})
            self.assertEqual(files, [str(index), str(folder / 'one.safetensors'), str(folder / 'two.safetensors')])
            files = []
            with patch('extensions_built_in.diffusion_models.ideogram4.ideogram4.huggingface_hub.hf_hub_download',
                       side_effect=lambda **kwargs: str(folder / Path(kwargs['filename']).name)) as download:
                _load_component_state_dict('author/ideogram', 'transformer', 'diffusion_pytorch_model', source_files=files)
            self.assertEqual(download.call_count, 3)
            self.assertEqual(files, [str(index), str(folder / 'one.safetensors'), str(folder / 'two.safetensors')])

    def native_sources(self, sd, arch, directory):
        root = Path(directory)
        for name in ('transformer', 'vae', 'encoder', 'conditioner', 'feature'):
            path = root / f'{name}.safetensors'
            if not path.exists():
                save_file({'weight': torch.ones(2, 2)}, str(path))
        def component(name):
            model = LoadedLinear(2, 2)
            model.aitk_load_source = {'path': str(root / f'{name}.safetensors')}
            model.config = SimpleNamespace(_name_or_path='author/' + name, _commit_hash='v1')
            return model
        sd.arch = arch
        sd.transformer = component('transformer')
        sd.vae = component('vae')
        sd.text_encoder = component('encoder')
        sd.trainable_model = SimpleNamespace(transformer=sd.transformer, text_conditioner=component('conditioner'))
        sd.model_config = SimpleNamespace(name_or_path='author/base', text_encoder_path=None, vae_path=None,
            model_kwargs={}, quantize=True, qtype='convrotint4', quantize_te=True, qtype_te='qfloat8')
        sd.tokenizer = SimpleNamespace(name_or_path='author/tokenizer', padding_side='right', model_max_length=256)
        sd.te_torch_dtype = torch.float32
        return component

    def new_trainer(self, directory, arch):
        trainer, _ = fixtures.SliderSpaceTrainerTests().make_trainer(directory)
        self.native_sources(trainer.sd, arch, directory)
        trainer.flow_profile = FlowTrainingProfile.from_model_config({'arch': arch})
        trainer.discovery_components = None
        trainer.resume_components = None
        trainer.hook_after_model_load()
        feature = LoadedLinear(2, 2)
        feature.config = SimpleNamespace(_name_or_path='author/feature', _commit_hash='v1', image_size=224)
        feature.aitk_load_source = {'path': str(Path(directory) / 'feature.safetensors')}
        processor = SimpleNamespace(name_or_path='author/feature', image_mean=[.5] * 3, image_std=[.25] * 3)
        trainer.semantic_encoder = SimpleNamespace(cache_identity=resolved_semantic_components(feature, processor, 'author/feature'))
        trainer._resolve_feature_identity()
        return trainer

    def test_resolved_identities_detect_implicit_component_files_and_conditioner_not_preview(self):
        for arch in ('krea2', 'anima', 'ideogram4'):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                trainer = self.new_trainer(directory, arch)
                baseline = resolved_flow_components(trainer.sd)
                te = specialized_text_cache_identity(trainer.sd)
                trainer.sd.model_config.model_kwargs.update(preview_auto=True, schedule_mu=2)
                self.assertEqual(resolved_flow_components(trainer.sd), baseline)
                for name in ('transformer', 'vae', 'encoder', 'conditioner'):
                    path = Path(directory) / f'{name}.safetensors'
                    old = path.read_bytes()
                    path.write_bytes(old + b'changed')
                    changed = resolved_flow_components(trainer.sd)
                    if name != 'conditioner' or arch == 'anima':
                        self.assertNotEqual(changed, baseline, name)
                    if name == 'encoder' or (name == 'conditioner' and arch == 'anima'):
                        self.assertNotEqual(specialized_text_cache_identity(trainer.sd), te, name)
                    path.write_bytes(old)
                    # Restoring bytes does not restore mtime: recapture the
                    # baseline for the next independent file-identity change.
                    baseline = resolved_flow_components(trainer.sd)
                    te = specialized_text_cache_identity(trainer.sd)

    def test_complete_bank_save_reload_rejects_component_feature_and_file_changes(self):
        for arch in ('krea2', 'anima', 'ideogram4'):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                trainer = self.new_trainer(directory, arch)
                trainer.network.select_direction(1)
                trainer.network.is_active = True
                trainer.network.multiplier = 1.
                trainer.sd.unet(torch.ones(2, 4)).square().mean().backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer.save(4)
                path = trainer.get_latest_save_path()
                expected = {key: value.clone() for key, value in trainer.network.state_dict().items()}
                resumed = self.new_trainer(directory, arch)
                resumed.load_weights(path)
                resumed._resolve_feature_identity()
                for key, value in resumed.network.state_dict().items():
                    torch.testing.assert_close(value, expected[key])
                resumed.resume_bank_path = path
                resumed.optimizer.load_state_dict(torch.load(resumed.get_optimizer_state_path(), weights_only=True))
                self.assertTrue(resumed.optimizer.state)
                for old, new in zip(trainer.optimizer.state.values(), resumed.optimizer.state.values()):
                    for key in old:
                        torch.testing.assert_close(old[key], new[key])
                self.assertEqual(resumed.step_num, 4)
                resumed.discovery_components['flow']['vae']['commit'] = 'different'
                with self.assertRaisesRegex(ValueError, 'resolved components changed'):
                    resumed.load_weights(path)
                resumed.discovery_components['flow'] = copy.deepcopy(trainer.discovery_components['flow'])
                resumed.semantic_encoder.cache_identity['model']['commit'] = 'different'
                with self.assertRaisesRegex(ValueError, 'feature encoder changed'):
                    resumed._resolve_feature_identity()
                optimizer_path = Path(trainer.get_optimizer_state_path())
                with optimizer_path.open('ab') as handle:
                    handle.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'checkpoint files changed'):
                    trainer.get_latest_save_path()

    def test_missing_latest_published_bank_never_silently_starts_a_new_job(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.new_trainer(directory, 'krea2')
            trainer.save(2)
            latest = Path(trainer.get_latest_save_path())
            latest.unlink()
            with self.assertRaisesRegex(ValueError, 'published checkpoint is incomplete'):
                trainer.get_latest_save_path()

    def test_missing_optimizer_or_direction_export_rejects_latest_generation(self):
        for missing in ('optimizer', 'export'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                trainer = self.new_trainer(directory, 'anima')
                trainer.save_config.max_step_saves_to_keep = 2
                trainer.save(1)
                trainer.save(2)
                latest = Path(trainer.get_latest_save_path())
                marker = json.loads((trainer._state_root() / 'checkpoint_000000002.json').read_text())
                target = (latest.with_suffix('.optimizer.pt') if missing == 'optimizer'
                          else Path(trainer.save_root) / marker['exports'][0])
                target.unlink()
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    trainer.get_latest_save_path()
                # An older valid generation exists, but silently reverting would
                # discard completed updates and is not a valid resume strategy.
                self.assertTrue((trainer._state_root() / 'bank_000000001.safetensors').is_file())

    def test_save_without_optimizer_cannot_publish_partial_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.new_trainer(directory, 'ideogram4')
            trainer.optimizer = None
            before = sorted(Path(trainer.save_root).rglob('*.safetensors'))
            with self.assertRaisesRegex(ValueError, 'require optimizer state'):
                trainer.save(2)
            self.assertEqual(sorted(Path(trainer.save_root).rglob('*.safetensors')), before)
            self.assertFalse((trainer._state_root() / 'checkpoint_000000002.json').exists())

    def test_discovery_bank_reuse_records_components_and_changes_root_without_touching_sources(self):
        for arch, alignment in [('krea2', 16), ('anima', 32), ('ideogram4', 16)]:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'source'
                source.mkdir()
                for index, color in enumerate(((190, 20, 30), (20, 190, 30), (30, 20, 190))):
                    Image.new('RGB', (160, 96), color).save(source / f'{index}.png')
                trainer = self.new_trainer(directory, arch)
                trainer.bucket_divisibility = alignment
                trainer.sliderspace = SliderSpaceConfig.parse({'num_directions': 2, 'discovery_mode': 'provided',
                    'discovery_datasets': [{'folder_path': str(source)}], 'discovery_buckets': True,
                    'concept_prompts': ['cat'], 'resolution': 128})
                trainer.discovery_entries = discovery_entries(trainer.sliderspace, scan_discovery_images(trainer.sliderspace, alignment))
                identity = trainer.semantic_encoder.cache_identity
                class Encoder:
                    cache_identity = identity
                    def __call__(self, pixels): return pixels.mean((2, 3))
                trainer.semantic_encoder = Encoder()
                trainer.sd.get_generation_pipeline = lambda: self.fail('Provided-only must not create a pipeline')
                calls = []
                trainer.sd.encode_images = lambda pixels: calls.append(pixels.shape) or torch.cat((pixels, pixels[:, :1]), dim=1)
                trainer._build_discovery_bank()
                root = trainer._discovery_root()
                manifest = json.loads((root / 'manifest.json').read_text())
                self.assertEqual(manifest['components'], trainer.discovery_components)
                self.assertEqual(root.name, component_identity_digest(trainer.discovery_components))
                trainer._build_discovery_bank()
                self.assertEqual(len(calls), 3)
                manifest['components']['flow']['vae']['commit'] = 'tampered'
                (root / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'discovery cache changed'):
                    trainer._build_discovery_bank()
                trainer.discovery_components['feature']['model']['commit'] = 'next encoder'
                self.assertNotEqual(trainer._discovery_root(), root)
                self.assertTrue((root / 'features.safetensors').exists())


if __name__ == '__main__':
    unittest.main()
