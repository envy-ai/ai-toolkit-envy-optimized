"""CPU checks with installed V3 Schema/ComfyNode/NodeOutput, not replacement glue.

Each probe is a separate ai-toolkit Python process. Only unrelated V3 video
implementations are isolated (this environment's older PyAV cannot import them).
No model manager, weights, CUDA, server, deployment or dependency changes. Actual
live extension loading/encoder execution remains a separate acceptance check.
"""
import ast
import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


COMFY_ROOT = Path(os.environ.get('COMFYUI_ROOT', '/home/bart/ComfyUI'))
REPO_ROOT = Path(__file__).resolve().parents[1]


def probe(mode):
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(COMFY_ROOT))
    import torch
    from comfy.cli_args import args
    args.cpu, args.disable_dynamic_vram = True, True
    video = types.ModuleType('comfy_api.latest._input_impl')
    video.VideoFromFile = video.VideoFromComponents = video.VideoFromList = object
    sys.modules[video.__name__] = video
    from comfy_api.latest import ComfyExtension, io
    from comfy_nodes.ai_toolkit_krea_preview import comfy_entrypoint
    from comfy_nodes.ai_toolkit_krea_preview import nodes as implementation
    case = unittest.TestCase()
    extension = asyncio.run(comfy_entrypoint())
    case.assertIsInstance(extension, ComfyExtension)
    classes = asyncio.run(extension.get_node_list())
    case.assertEqual(classes, [implementation.AIToolkitKreaEditConditioning])
    node = classes[0]
    case.assertTrue(issubclass(node, io.ComfyNode))
    schema = node.GET_SCHEMA()  # real class + finalized schema validation
    inputs = node.INPUT_TYPES()
    case.assertEqual(schema.node_id, 'AIToolkitKreaEditConditioning')
    case.assertEqual(node.RETURN_TYPES, ['CONDITIONING', 'CONDITIONING'])
    case.assertEqual(node.RETURN_NAMES, ['positive', 'negative'])
    case.assertEqual(set(inputs['optional']), {f'{kind}{i}' for kind in ('image', 'mask') for i in (1, 2, 3)})
    case.assertEqual(inputs['required']['dtype'][1]['options'], ['float32', 'bfloat16', 'float16'])
    case.assertNotIn('comfy.model_management', sys.modules)
    case.assertNotIn('comfy.text_encoders.krea2', sys.modules)

    if mode == 'graph':
        from dataclasses import replace
        from testing.test_cross_model_comfy_templates import CrossModelComfyTemplateTests
        from toolkit.comfy_sample import get_workflow_for_sample
        from toolkit.comfy_schema import validate_workflow_schema
        for cfg in (1, 4):
            request = replace(CrossModelComfyTemplateTests().request('krea2', cfg),
                specialized_flow=True, model_kwargs={'edit': True},
                control_image='first.png', control_image_2='second.png', control_image_3='third.png')
            graph = get_workflow_for_sample('config/comfy_templates/krea2_lora_sample.json.njk', request)
            # The exact rendered encoder node, with non-model source fixtures.
            # Other template nodes already have separate live-registry coverage.
            selected = {'1': copy.deepcopy(graph['1'])}
            registry = {schema.node_id: {'input': inputs, 'output': node.RETURN_TYPES}}
            source_types = {'clip': ['MODEL', 'CLIP'], 'vae': ['VAE'],
                            **{f'{kind}{i}': ['IMAGE', 'MASK'] for kind in ('image', 'mask') for i in (1, 2, 3)}}
            for key, value in selected['1']['inputs'].items():
                if key in source_types:
                    source_id, _ = value
                    name = 'Source_' + source_id
                    selected[source_id] = {'class_type': name, 'inputs': {}}
                    registry[name] = {'input': {}, 'output': source_types[key]}
            validate_workflow_schema(selected, registry)
            original = copy.deepcopy(selected)
            selected['1']['inputs']['dtype'] = 'float64'
            with case.assertRaisesRegex(ValueError, 'not an available choice'):
                validate_workflow_schema(selected, registry)
            selected = original
            selected['1']['inputs']['image1'] = selected['1']['inputs']['vae']
            with case.assertRaisesRegex(ValueError, 'linked output is VAE'):
                validate_workflow_schema(selected, registry)

    if mode in ('execute', 'invalid'):
        source = ast.parse((COMFY_ROOT / 'comfy/text_encoders/krea2.py').read_text())
        template = next(ast.literal_eval(item.value) for item in source.body if isinstance(item, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'KREA2_TEMPLATE' for target in item.targets))
        # Resolve the real installed constant without loading encoder/model code.
        with patch.object(implementation, '_krea_template', return_value=template):
            image = torch.rand(1, 32, 48, 3, requires_grad=True)
            calls, latents = [], []
            def tokenize(text, **kwargs):
                calls.append((text, kwargs))
                case.assertFalse(torch.is_grad_enabled())
                return text
            def vae_encode(pixels):
                case.assertFalse(torch.is_grad_enabled())
                value = pixels.mean().expand(1, 16, 1, 2, 3).clone()
                latents.append(value)
                return value
            attention_mask = torch.ones(1, 2, dtype=torch.bool)
            clip = types.SimpleNamespace(tokenize=tokenize, encode_from_tokens_scheduled=lambda tokens:
                [[torch.ones(1, 2, 4), {'attention_mask': attention_mask}]])
            vae = types.SimpleNamespace(encode=vae_encode)
            keywords = dict(clip=clip, vae=vae, prompt='edit the image', negative_prompt='haze',
                cfg=4, width=64, height=32, vlm_max_pixels=56 ** 2, control_image_max_pixels=64 ** 2,
                match_target_res=False, dtype='bfloat16', background='[19, 71, 139]')
            if mode == 'execute':
                for cfg in (0, 1, 4):
                    calls.clear()
                    latents.clear()
                    result = node.execute(**{**keywords, 'cfg': cfg}, image1=image, image2=image, image3=image)
                    case.assertIsInstance(result, io.NodeOutput)
                    positive, negative = result.result
                    case.assertEqual(len(latents), 3)
                    case.assertEqual(len(calls), 1 if cfg <= 1 else 2)
                    case.assertIs(positive[0][1]['reference_latents'], negative[0][1]['reference_latents'])
                    case.assertIs(positive[0][1]['attention_mask'], attention_mask)
                    case.assertIn('Picture 3:', calls[0][0])
                    case.assertEqual(calls[0][1]['llama_template'], template)
                    case.assertTrue(all(not latent.requires_grad for latent in latents))
                    if cfg <= 1:
                        case.assertIs(negative, positive)
                    else:
                        case.assertTrue(calls[1][0].endswith('haze'))
                        case.assertIs(calls[0][1]['images'], calls[1][1]['images'])
            else:
                with case.assertRaisesRegex(ValueError, 'consecutive slots'):
                    node.execute(**keywords, image2=image)
                with case.assertRaisesRegex(ValueError, 'RGB list'):
                    node.execute(**{**keywords, 'background': '[999, 0, 0]'}, image1=image)
                with case.assertRaisesRegex(ValueError, 'one RGB image'):
                    node.execute(**keywords, image1=image.expand(2, -1, -1, -1))
                with case.assertRaisesRegex(ValueError, 'alpha mask'):
                    node.execute(**keywords, image1=image, mask1=torch.ones(1, 2, 2))
                case.assertFalse(latents)
                case.assertFalse(calls)
    case.assertFalse(torch.cuda.is_initialized())
    case.assertNotIn('comfy.model_management', sys.modules)
    print(json.dumps({'case': mode, 'status': 'passed', 'actual_v3_api': True,
                      'unrelated_video_impl_isolated': True, 'models_loaded': False, 'cuda_initialized': False}))


@unittest.skipUnless((COMFY_ROOT / 'comfy_api/latest/_io.py').is_file(), 'Installed Comfy V3 API unavailable')
class KreaPreviewV3Tests(unittest.TestCase):
    def check_probe(self, mode):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe', mode],
            cwd=REPO_ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report['case'], mode)
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['cuda_initialized'])

    def test_extension_registration_and_real_schema_without_model_imports(self):
        self.check_probe('schema')

    def test_rendered_encoder_graph_matches_real_v3_input_output_specs(self):
        self.check_probe('graph')

    def test_real_nodeoutput_cfg_skip_shared_reference_latents_and_metadata(self):
        self.check_probe('execute')

    def test_real_node_rejects_invalid_controls_before_encoding(self):
        self.check_probe('invalid')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--probe']:
        probe(sys.argv[2])
    else:
        unittest.main()
