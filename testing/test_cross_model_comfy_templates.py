"""Render new templates without contacting Comfy or allocating model weights."""
import unittest
from dataclasses import replace, fields
import ast
from pathlib import Path
import re
from types import SimpleNamespace

import torch

from toolkit.comfy_sample import (ComfySampleRequest, ComfyBatchSampleRequest, get_workflow_for_sample,
    get_workflow_for_samples, validate_model_workflow, build_template_context)


class CrossModelComfyTemplateTests(unittest.TestCase):
    def request(self, arch, cfg=4, policy='image_only'):
        return ComfySampleRequest(prompt='a cat', width=768, height=1024, steps=20, cfg=cfg, seed=123,
            model='base.safetensors', vae='vae.safetensors', text_encoder='te.safetensors', sampler='euler',
            scheduler='normal', inference_lora='', inference_lora_strength=1, output_format='png', output_quality='high',
            training_lora_path='/output/direction.safetensors', training_lora_filename='direction.safetensors',
            filename_prefix='aitk/preview', training_lora_strength=-.5, negative_prompt='haze', model_arch=arch,
            model_kwargs={'ideogram_cfg_reference': policy, 'ideogram_schedule_mu': .25, 'ideogram_schedule_std': 1.5})

    def test_anima_loader_latent_scheduler_and_signed_strength(self):
        request = self.request('anima')
        workflow = get_workflow_for_sample('config/comfy_templates/anima_lora_sample.json.njk', request)
        self.assertEqual(workflow['2']['inputs']['type'], 'qwen_image') # Qwen3-0.6B is auto-detected as Anima TE.
        self.assertEqual(workflow['8']['class_type'], 'EmptySD3LatentImage')
        self.assertEqual(workflow['9']['inputs']['shift'], 3)
        self.assertEqual(workflow['4']['inputs']['lora_strength'], -.5)
        self.assertEqual(workflow['7']['inputs']['text'], 'haze')
        self.assertEqual(workflow['10']['inputs']['cfg'], 4)

    def test_krea_native_sigmas_match_pipeline_and_installed_parser(self):
        from extensions_built_in.diffusion_models.krea2.src.pipeline import timesteps
        path = Path('/home/bart/ComfyUI/comfy_extras/nodes_custom_sampler.py')
        if not path.exists():
            self.skipTest('Installed Comfy ManualSigmas source unavailable')
        # Execute the installed parser body unchanged, without importing GPU
        # management or any sampler models. Only its UI return container is stubbed.
        node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == 'ManualSigmas')
        execute = next(method for method in node.body if isinstance(method, ast.FunctionDef) and method.name == 'execute')
        execute.decorator_list = []
        namespace = {'re': re, 'torch': torch, 'io': SimpleNamespace(NodeOutput=lambda value: value)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[execute], type_ignores=[])), str(path), 'exec'), namespace)
        for width, height, kwargs in [(768, 1024, {}), (1024, 1024, {'schedule_mu': 1.15}),
            (512, 768, {'schedule_mu': -30.}),
            (768, 1024, {'schedule_min_res': 128, 'schedule_max_res': 1536, 'schedule_y1': .2, 'schedule_y2': 1.5})]:
            with self.subTest(width=width, height=height, kwargs=kwargs):
                request = replace(self.request('krea2'), specialized_flow=True, width=width, height=height, model_kwargs=kwargs)
                context = build_template_context(request)
                actual = namespace['execute'](None, context['krea_native_sigmas'])
                expected = timesteps((width // 16) * (height // 16), request.steps,
                    (kwargs.get('schedule_min_res', 256) // 16) ** 2,
                    (kwargs.get('schedule_max_res', 1280) // 16) ** 2,
                    kwargs.get('schedule_y1', .5), kwargs.get('schedule_y2', 1.15), mu=kwargs.get('schedule_mu'))
                torch.testing.assert_close(actual, torch.tensor(expected), atol=0, rtol=0)
                self.assertEqual(actual[-1], 0)
                self.assertEqual(len(actual), request.steps + 1)
                self.assertNotIn('e-', context['krea_native_sigmas'])

    def test_specialized_krea_single_and_batch_blank_cfg_one_signed_preview(self):
        for cfg in (1., 3.):
            request = replace(self.request('krea2', cfg), specialized_flow=True, negative_prompt='', model_kwargs={})
            values = {key: value for key, value in vars(request).items() if key in {field.name for field in fields(ComfyBatchSampleRequest)}}
            batch = ComfyBatchSampleRequest(**values, prompts=['cat', 'dog'], seeds=[123, 456])
            for path, value in [('config/comfy_templates/krea2_lora_sample.json.njk', request),
                ('config/comfy_templates/krea2_lora_sample_batch_easy_use.json.njk', batch)]:
                with self.subTest(path=path, cfg=cfg):
                    workflow = (get_workflow_for_samples(path, value) if value is batch else get_workflow_for_sample(path, value))
                    self.assertEqual(workflow['2']['class_type'], 'EmptySD3LatentImage')
                    self.assertEqual(workflow['17']['class_type'], 'SamplerCustomAdvanced')
                    self.assertEqual(workflow['36']['inputs']['lora_strength'], -.5)
                    self.assertEqual(workflow['81']['class_type'], 'BasicGuider' if cfg <= 1 else 'CFGGuider')
                    positive_key = 'conditioning' if cfg <= 1 else 'positive'
                    self.assertEqual(workflow['81']['inputs'][positive_key], ['71' if value is batch else '1', 0])
                    if cfg > 1:
                        self.assertEqual(workflow['7']['class_type'], 'CLIPTextEncode')
                        self.assertEqual(workflow['7']['inputs']['text'], '')
                    else:
                        self.assertNotIn('7', workflow)
                    if value is batch:
                        self.assertEqual(workflow['80']['inputs']['noise_seed'], ['73', 0])
            ordinary = get_workflow_for_sample('config/comfy_templates/krea2_lora_sample.json.njk',
                replace(request, specialized_flow=False))
            self.assertEqual(ordinary['17']['class_type'], 'KSampler')
            self.assertEqual(ordinary['7']['class_type'], 'ConditioningZeroOut')

    def test_specialized_krea_rejects_changed_schedule_and_zeroed_blank(self):
        request = replace(self.request('krea2'), specialized_flow=True, negative_prompt='', model_kwargs={})
        path = 'config/comfy_templates/krea2_lora_sample.json.njk'
        workflow = get_workflow_for_sample(path, request)
        workflow['83']['inputs']['sigmas'] = '1, 0'
        with self.assertRaisesRegex(ValueError, 'schedule must match'):
            validate_model_workflow(workflow, request)
        workflow = get_workflow_for_sample(path, request)
        workflow['7'] = {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['1', 0]}}
        with self.assertRaisesRegex(ValueError, 'including an empty prompt'):
            validate_model_workflow(workflow, request)
        for kwargs in ({'width': 765}, {'steps': 0}, {'model_kwargs': {'schedule_max_res': 128}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_template_context(replace(request, **kwargs))

    def test_specialized_krea_batch_supports_large_seeds_and_requires_seed_count(self):
        request = replace(self.request('krea2'), specialized_flow=True, model_kwargs={})
        values = {key: value for key, value in vars(request).items()
            if key in {field.name for field in fields(ComfyBatchSampleRequest)}}
        batch = ComfyBatchSampleRequest(**values, prompts=['cat', 'dog', 'bird'],
            seeds=[2 ** 32 - 1, 2 ** 32, 2 ** 64 - 1])
        path = 'config/comfy_templates/krea2_lora_sample_batch_easy_use.json.njk'
        workflow = get_workflow_for_samples(path, batch)
        for index, seed in enumerate(batch.seeds):
            node = workflow[str(1100 + index)]
            self.assertEqual(node['class_type'], 'Seed')
            self.assertEqual(node['inputs'], {'seed': seed})
        self.assertEqual(workflow['1300']['inputs']['any_1'], ['1100', 3])
        self.assertEqual(workflow['1300']['inputs']['any_2'], ['1101', 3])
        self.assertEqual(workflow['1301']['inputs']['any_1'], ['1300', 0])
        self.assertEqual(workflow['1301']['inputs']['any_2'], ['1102', 3])
        single = get_workflow_for_sample('config/comfy_templates/krea2_lora_sample.json.njk',
            replace(request, seed=2 ** 64 - 1))
        self.assertEqual(single['80']['inputs']['noise_seed'], 2 ** 64 - 1)
        self.assertEqual(single['26']['inputs']['seed'], 2 ** 64 - 1)
        ordinary = get_workflow_for_samples(path, replace(batch, specialized_flow=False))
        self.assertEqual(ordinary['1100']['class_type'], 'easy int')
        self.assertEqual(ordinary['1300']['inputs']['any_1'], ['1100', 0])
        with self.assertRaisesRegex(ValueError, 'seed count'):
            build_template_context(replace(batch, seeds=[42]))
        for seed in (-1, 2 ** 64, 1.5, True):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, '64-bit'):
                build_template_context(replace(request, seed=seed))

    def test_ideogram_reference_branches_schedule_and_cfg_one(self):
        for policy, cfg, reference in [('image_only', 4, None), ('negative_prompt', 4, ['7', 0]),
                                       ('negative_prompt', 1, None)]:
            with self.subTest(policy=policy, cfg=cfg):
                request = self.request('ideogram4', cfg, policy)
                workflow = get_workflow_for_sample('config/comfy_templates/ideogram4_lora_sample.json.njk', request)
                self.assertEqual(workflow['9']['inputs'].get('negative'), reference)
                self.assertEqual('7' in workflow, reference is not None)
                self.assertEqual(workflow['12']['inputs']['mu'], .25)
                self.assertEqual(workflow['12']['inputs']['std'], 1.5)
                self.assertEqual(workflow['8']['class_type'], 'EmptyFlux2LatentImage')
                self.assertEqual(workflow['4']['inputs']['lora_strength'], -.5)

    def test_ideogram_rejects_blank_substitution_and_wrong_or_ignored_negatives(self):
        request = self.request('ideogram4')
        workflow = get_workflow_for_sample('config/comfy_templates/ideogram4_lora_sample.json.njk', request)
        workflow['9']['inputs']['negative'] = ['fake', 0]
        workflow['fake'] = {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['6', 0]}}
        with self.assertRaisesRegex(ValueError, 'unconnected negative'):
            validate_model_workflow(workflow, request)
        with self.assertRaisesRegex(ValueError, 'encode the configured'):
            validate_model_workflow(workflow, replace(request, model_kwargs={'ideogram_cfg_reference': 'negative_prompt'}))
        workflow['2']['inputs']['type'] = 'krea2'
        with self.assertRaisesRegex(ValueError, 'Ideogram text encoder'):
            validate_model_workflow(workflow, request)

    def test_structured_prompt_digest_matches_native_without_modifying_saved_text(self):
        request = self.request('ideogram4')
        from toolkit.ideogram_caption import digest_caption_string
        prompt = '{"compositional_deconstruction":{"medium":"photography","description":"cat"}}'
        request.prompt = prompt
        self.assertEqual(build_template_context(request)['prompt'], digest_caption_string(prompt))
        self.assertEqual(request.prompt, prompt)
        self.assertEqual(build_template_context(replace(request, prompt='clear\n\n' + prompt))['prompt'], 'clear\n\n' + prompt)


if __name__ == '__main__':
    unittest.main()
