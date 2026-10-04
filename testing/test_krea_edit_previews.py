"""Native preprocessing/preview graphs; no Comfy/GPU models are loaded."""
import ast
import copy
import io
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from comfy_nodes.ai_toolkit_krea_preview.reference import composite_reference, prepare_reference
from toolkit.comfy_sample import ComfyApiClient, get_workflow_for_sample, validate_model_workflow
from toolkit.control_image import KREA_EDIT_CONTROL_PRESENTATION, load_control_rgb
from testing.test_krea_edit_objectives import ReferenceItem
from testing import test_cross_model_comfy_templates as templates
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def node_logic():
    # Execute the actual node class unchanged. Only Comfy's node/return-container
    # glue is substituted; the real torch preprocessing and encode logic run.
    source = Path('comfy_nodes/ai_toolkit_krea_preview/nodes.py')
    node = next(item for item in ast.parse(source.read_text()).body if isinstance(item, ast.ClassDef))
    helpers = Path('/home/bart/ComfyUI/node_helpers.py')
    template_source = Path('/home/bart/ComfyUI/comfy/text_encoders/krea2.py')
    if not helpers.exists() or not template_source.exists():
        raise unittest.SkipTest('Installed Comfy conditioning helper/Krea template unavailable')
    helper = next(item for item in ast.parse(helpers.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == 'conditioning_set_values')
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(helpers), 'exec'), namespace)
    template = next(ast.literal_eval(item.value) for item in ast.parse(template_source.read_text()).body
        if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'KREA2_TEMPLATE' for target in item.targets))
    namespace.update(json=json, torch=torch, prepare_reference=prepare_reference, composite_reference=composite_reference,
        _krea_template=lambda: template, node_helpers=SimpleNamespace(conditioning_set_values=namespace['conditioning_set_values']),
        io=SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values: values))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['AIToolkitKreaEditConditioning']


class KreaEditPreviewTests(unittest.TestCase):
    def native(self, dtype, kwargs):
        model = object.__new__(Krea2Model)
        model.device_torch, model.torch_dtype = torch.device('cpu'), dtype
        model.model_config = SimpleNamespace(model_kwargs=kwargs)
        model.vae_scale_factor, model.patch_size = 8, 2
        return model

    def test_vlm_and_normalized_vae_inputs_match_actual_native_methods(self):
        image = torch.rand(1, 117, 83, 3, generator=torch.Generator().manual_seed(31))
        for precision, dtype in [('float32', torch.float32), ('bfloat16', torch.bfloat16), ('float16', torch.float16)]:
            for match in (False, True):
                with self.subTest(precision=precision, match=match):
                    kwargs = {'vlm_max_pixels': 56 ** 2, 'control_image_max_pixels': 64 ** 2, 'match_target_res': match}
                    model = self.native(dtype, kwargs)
                    captured = []
                    model.encode_images = lambda images, **kw: captured.append(images.clone()) or torch.zeros(1, 16, 2, 2)
                    native_vlm = model._prep_vlm_images([image.movedim(-1, 1)[0].to(dtype)])[0]
                    model._encode_ref_latents([image.movedim(-1, 1)[0]], target_pixels=32 * 64)
                    vlm, pixels = prepare_reference(image, dtype=precision, width=32, height=64,
                        vlm_max_pixels=56 ** 2, control_image_max_pixels=64 ** 2, match_target_res=match)
                    torch.testing.assert_close(vlm.movedim(-1, 1)[0], native_vlm, atol=0, rtol=0)
                    torch.testing.assert_close((pixels.movedim(-1, 1) * 2 - 1).to(dtype), captured[0], atol=0, rtol=0)
                    self.assertEqual(pixels.shape[1] % 16, 0)
                    self.assertEqual(pixels.shape[2] % 16, 0)

    def test_alpha_matches_pil_and_native_loader(self):
        rgba = np.random.default_rng(19).integers(0, 256, size=(32, 48, 4), dtype=np.uint8)
        original = Image.fromarray(rgba, 'RGBA')
        background = [19, 71, 139]
        expected = Image.new('RGB', original.size, tuple(background))
        expected.paste(original, mask=original.getchannel('A'))
        image = torch.from_numpy(rgba[..., :3].copy()).float().unsqueeze(0) / 255
        mask = 1 - torch.from_numpy(rgba[..., 3].copy()).float().unsqueeze(0) / 255
        actual = composite_reference(image, mask, background)
        self.assertTrue(np.array_equal((actual[0].numpy() * 255).round().astype(np.uint8), np.array(expected)))
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'reference.png')
            original.save(path)
            model = self.native(torch.float32, {})
            model._specialized_control_presentation = KREA_EDIT_CONTROL_PRESENTATION
            model._specialized_preview_background = tuple(background)
            self.assertTrue(np.array_equal(np.array(model.load_sample_control_image(path)), np.array(expected)))
            del model._specialized_control_presentation
            self.assertTrue(np.array_equal(np.array(model.load_sample_control_image(path)), np.array(original.convert('RGB'))))

    def test_node_encodes_refs_once_and_shares_them_skipping_cfg_one_negatives(self):
        Node = node_logic()
        for cfg in (1, 3):
            with self.subTest(cfg=cfg):
                images, encodes = [], []
                def tokenize(text, **kwargs):
                    images.append((text, kwargs))
                    return text
                clip = SimpleNamespace(tokenize=tokenize,
                    encode_from_tokens_scheduled=lambda tokens: [[torch.zeros(1, 2, 4), {'attention_mask': torch.ones(1, 2)}]])
                vae = SimpleNamespace(encode=lambda pixels: encodes.append(pixels) or torch.rand(1, 16, 2, 2))
                image = torch.rand(1, 32, 48, 3)
                positive, negative = Node.execute(clip, vae, 'edit', '', cfg, 64, 32, 56 ** 2, 64 ** 2,
                    False, 'bfloat16', '[0, 0, 0]', image1=image, image2=image)
                self.assertEqual(len(encodes), 2)
                self.assertEqual(len(images), 1 if cfg <= 1 else 2)
                self.assertIs(positive[0][1]['reference_latents'], negative[0][1]['reference_latents'])
                self.assertIn('Picture 2:', images[0][0])
                self.assertIn('attention_mask', positive[0][1])
                if cfg > 1:
                    self.assertEqual(images[1][0], images[0][0][:-len('edit')])
                    self.assertIs(images[0][1]['images'], images[1][1]['images'])
                with self.assertRaisesRegex(ValueError, 'consecutive slots'):
                    Node.execute(clip, vae, 'edit', '', cfg, 64, 32, 56 ** 2, 64 ** 2,
                        False, 'bfloat16', '[0, 0, 0]', image2=image)

    def test_edit_graph_links_and_prompts(self):
        for cfg in (1, 3):
            request = replace(templates.CrossModelComfyTemplateTests().request('krea2', cfg), specialized_flow=True,
                model_kwargs={'edit': True, 'vlm_max_pixels': 65536, 'control_image_max_pixels': 262144,
                    'match_target_res': True, 'kv_cache': True},
                control_image='reference.png', control_image_2='second.png', control_background=[19, 71, 139])
            graph = get_workflow_for_sample('config/comfy_templates/krea2_lora_sample.json.njk', request)
            encoder = graph['1']
            self.assertEqual(encoder['class_type'], 'AIToolkitKreaEditConditioning')
            self.assertEqual(encoder['inputs']['vlm_max_pixels'], 65536)
            self.assertEqual(encoder['inputs']['background'], '[19, 71, 139]')
            self.assertEqual(encoder['inputs']['image2'], ['91', 0])
            self.assertEqual(encoder['inputs']['mask2'], ['91', 1])
            self.assertEqual(graph['85']['inputs']['kv_cache'], True)
            self.assertEqual(graph['81']['inputs']['model'], ['85', 0])
            self.assertNotIn('7', graph)
            broken = copy.deepcopy(graph)
            broken['1']['inputs']['vlm_max_pixels'] = 42
            with self.assertRaisesRegex(ValueError, 'preprocessing must match'):
                validate_model_workflow(broken, request)
            if cfg > 1:
                self.assertEqual(graph['81']['inputs']['negative'], ['1', 1])
                broken = copy.deepcopy(graph)
                broken['81']['inputs']['negative'] = ['1', 0]
                with self.assertRaisesRegex(ValueError, 'share the same'):
                    validate_model_workflow(broken, request)

    def test_preview_background_settings_and_batch_fallback(self):
        process = object.__new__(BaseSDTrainProcess)
        process.flow_profile = object()
        process.model_config = SimpleNamespace(arch='krea2', model_kwargs={'edit': True})
        process.train_config = SimpleNamespace(dtype='bf16')
        process.sd = SimpleNamespace(torch_dtype=torch.bfloat16)
        process.dataset_configs = [SimpleNamespace(control_transparent_color=[19, 71, 139])]
        self.assertEqual(process._get_krea_preview_control_settings(),
            {'model_dtype': 'bfloat16', 'control_background': [19, 71, 139]})
        self.assertEqual(process.sd._specialized_preview_background, (19, 71, 139))
        process._render_comfy_samples = lambda configs, sample, step=None: (configs, sample, step)
        self.assertEqual(process._render_comfy_sample_batch(['first', 'second'], 'sample', step=7),
            (['first', 'second'], 'sample', 7))
        process.dataset_configs.append(SimpleNamespace(control_transparent_color=[0, 0, 0]))
        with self.assertRaisesRegex(ValueError, 'different alpha backgrounds'):
            process._get_krea_preview_control_settings()
        process.model_config.model_kwargs['preview_control_transparent_color'] = [255, 255, 255]
        self.assertEqual(process._get_krea_preview_control_settings()['control_background'], [255, 255, 255])

    def test_specialized_alias_paths_are_identical_for_vlm_and_vae(self):
        model = self.native(torch.float32, {})
        config = SimpleNamespace(ctrl_img='one.png', ctrl_img_1='one.png', ctrl_img_2='two.png', ctrl_img_3=None)
        self.assertEqual(model.get_sample_control_image_paths(config), ['one.png', 'one.png', 'two.png'])
        model._specialized_control_presentation = KREA_EDIT_CONTROL_PRESENTATION
        self.assertEqual(model.get_sample_control_image_paths(config), ['one.png', 'two.png'])
        config.ctrl_img_1 = 'explicit.png'
        self.assertEqual(model.get_sample_control_image_paths(config), ['explicit.png', 'two.png'])

    def test_source_formats_match_actual_raw_training_and_native_preview_loaders(self):
        palette = Image.new('P', (16, 32), 1)
        palette.putpalette([0, 0, 0, 210, 71, 39] + [0] * (768 - 6))
        palette.info['transparency'] = 1
        grayscale = Image.fromarray(np.arange(512, dtype=np.uint16).reshape(32, 16) * 100)
        la = Image.new('LA', (16, 32), (120, 127))
        oriented = Image.new('RGB', (16, 32), (17, 61, 113))
        exif = oriented.getexif()
        exif[274] = 6
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.native(torch.float32, {})
            model._specialized_control_presentation = KREA_EDIT_CONTROL_PRESENTATION
            model._specialized_preview_background = (19, 71, 139)
            model.cache_processed_control_text_embeddings = False
            for kind, source in [('palette', palette), ('16bit', grayscale), ('la', la), ('exif', oriented)]:
                with self.subTest(kind=kind):
                    path = root / f'{kind}.png'
                    source.save(path, **({'exif': exif} if kind == 'exif' else {}))
                    item = ReferenceItem(root, model)
                    item.control_path = str(path)
                    item.load_control_image()
                    expected = load_control_rgb(str(path), (19, 71, 139))
                    np.testing.assert_array_equal(np.array(model.load_sample_control_image(str(path))), np.array(expected))
                    torch.testing.assert_close(item.control_tensor,
                        torch.from_numpy(np.array(expected)).movedim(-1, 0).float() / 255, atol=0, rtol=0)
                    if kind == 'palette':
                        np.testing.assert_array_equal(np.array(expected)[0, 0], [19, 71, 139])
                        # Ordinary/legacy raw control presentation is unchanged.
                        item.text_embedding_control_presentation = None
                        item.load_control_image()
                        torch.testing.assert_close(item.control_tensor[:, 0, 0], torch.tensor([210, 71, 39]) / 255)
                    if kind == '16bit':
                        np.testing.assert_array_equal(np.array(expected)[-1, -1], [255, 255, 255])
                    if kind == 'exif':
                        self.assertEqual(expected.size, (32, 16))
            path = root / 'animation.gif'
            Image.new('RGB', (16, 32), 'red').save(path, save_all=True,
                append_images=[Image.new('RGB', (16, 32), 'blue')], duration=50, loop=0)
            np.testing.assert_array_equal(np.array(load_control_rgb(str(path)))[0, 0], [255, 0, 0])

    def test_prepared_upload_is_canonical_rgb_and_content_specific_without_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.png'
            Image.new('RGBA', (16, 32), (200, 40, 70, 127)).save(path)
            original = path.read_bytes()
            response = MagicMock()
            response.__enter__.return_value.read.return_value = b'{"name":"ref.png","subfolder":"ai-toolkit","type":"input"}'
            requests = []
            def upload(request, **kwargs):
                requests.append(request)
                return response
            with patch('toolkit.comfy_sample.urllib.request.urlopen', side_effect=upload):
                for background in [(0, 0, 0), (255, 255, 255), (0, 0, 0)]:
                    prepared = load_control_rgb(str(path), background)
                    ComfyApiClient().upload_image(str(path), prepared_image=prepared)
                    body = requests[-1].data
                    payload = body.split(b'Content-Type: image/png\r\n\r\n', 1)[1].split(b'\r\n--', 1)[0]
                    decoded = Image.open(io.BytesIO(payload))
                    self.assertEqual(decoded.mode, 'RGB')
                    np.testing.assert_array_equal(np.array(decoded), np.array(prepared))
            self.assertNotEqual(requests[0].data, requests[1].data)
            filenames = [request.data.split(b'filename="', 1)[1].split(b'"', 1)[0] for request in requests]
            self.assertNotEqual(filenames[0], filenames[1])
            self.assertEqual(filenames[0], filenames[2])
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
