"""Exercise real batch dispatch and rendered templates without a Comfy server."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.comfy_sample import (DEFAULT_COMFY_BATCH_WORKFLOW_PATH,
    DEFAULT_COMFY_QWEN_IMAGE_EDIT_BATCH_WORKFLOW_PATH)
from toolkit.config_modules import GenerateImageConfig


class ComfyBatchSettingsTests(unittest.TestCase):
    def render(self, samples, *, workflow_path=DEFAULT_COMFY_BATCH_WORKFLOW_PATH, arch='krea2'):
        with tempfile.TemporaryDirectory() as directory:
            trainer = object.__new__(BaseSDTrainProcess)
            trainer.save_root = directory
            trainer.model_config = SimpleNamespace(arch=arch, model_kwargs={})
            trainer.flow_profile = object()
            for name in (
                '_cleanup_legacy_comfy_sample_loras', '_log_comfy_sample_generation_start',
                '_log_comfy_prompt_submitted', '_create_comfy_image_progress_bar',
                '_close_comfy_image_progress_bar', '_update_comfy_sample_status',
                '_log_comfy_sample_generation_time', 'sample_step_hook',
                '_should_cancel_comfy_prompt_wait',
            ):
                setattr(trainer, name, Mock())
            trainer._save_current_network_for_comfy = Mock(return_value='/slider.safetensors')
            trainer._get_comfy_lora_display_filename = Mock(return_value='slider.safetensors')
            trainer._get_comfy_training_lora_path = Mock(return_value='/slider.safetensors')
            trainer._prepare_comfy_inference_lora_path = Mock(return_value='')
            trainer._run_with_models_offloaded_for_comfy = lambda render: render(False, False)
            comfy = SimpleNamespace(
                workflow_path=workflow_path, api_url='http://unused',
                timeout=1, run_in_background=False, negative_prompt='',
                model='krea.safetensors', vae='vae.safetensors', audio_vae='',
                text_encoder='te.safetensors', sampler='euler', scheduler='simple',
                inference_lora='', inference_lora_strength=1,
                output_format='png', output_quality='high',
            )
            for i, config in enumerate(samples):
                config.output_folder = directory
                config.output_filename_no_ext = f'sample_{i}'
            graphs = []
            client = Mock()
            client.upload_image.side_effect = lambda path: path
            def post(graph):
                graphs.append(graph)
                return len(graphs) - 1
            client.post_prompt.side_effect = post
            client.wait_for_images.side_effect = lambda index, **kwargs: [
                f'{index}:{i}' for i in range(graphs[index].get('41', {}).get('inputs', {}).get('total', 1))
            ]
            with patch('jobs.process.BaseSDTrainProcess.ComfyApiClient', return_value=client), patch(
                'jobs.process.BaseSDTrainProcess.flush'
            ):
                trainer._render_comfy_sample_batch(samples, SimpleNamespace(comfy=comfy), step=100)
            downloads = [(call.args[0], Path(call.args[1]).name) for call in client.download_image.call_args_list]
            return graphs, downloads

    def sample(self, strength=1, **kwargs):
        return GenerateImageConfig(prompt='a leather couch', seed=42,
            width=1024, height=1024, num_inference_steps=12, guidance_scale=1,
            network_multiplier=strength, output_path='/unused/sample.png', **kwargs)

    def test_slider_endpoints_render_at_their_own_strength(self):
        graphs, downloads = self.render([self.sample(s) for s in (-1, 0, 1)])
        self.assertEqual([g['36']['inputs']['lora_strength'] for g in graphs], [-1, 0, 1])
        self.assertEqual([name for _, name in downloads], ['sample_0.png', 'sample_1.png', 'sample_2.png'])
        self.assertTrue(all('41' not in g for g in graphs))

    def test_compatible_samples_still_batch_and_keep_original_output_indices(self):
        graphs, downloads = self.render([self.sample(s) for s in (-1, 1, -1)])
        self.assertEqual([g['36']['inputs']['lora_strength'] for g in graphs], [-1, 1])
        self.assertEqual(graphs[0]['41']['inputs']['total'], 2)
        self.assertEqual(downloads, [('0:0', 'sample_0.png'), ('0:1', 'sample_2.png'), ('1:0', 'sample_1.png')])

    def test_sampling_overrides_do_not_inherit_first_samples_settings(self):
        samples = [self.sample() for _ in range(4)]
        samples[1].width = 768
        samples[2].num_inference_steps = 20
        samples[3].guidance_scale = 3
        graphs, _ = self.render(samples)
        self.assertEqual(len(graphs), 4)
        self.assertEqual(graphs[1]['2']['inputs']['width'], 768)
        self.assertEqual(len(graphs[2]['83']['inputs']['sigmas'].split(',')), 21)
        self.assertEqual(graphs[3]['81']['inputs']['cfg'], 3)

    def test_qwen_edit_batch_splits_different_strengths_too(self):
        graphs, downloads = self.render([
            self.sample(-1, ctrl_img_1='/source-a.png'),
            self.sample(1, ctrl_img_1='/source-b.png'),
        ], workflow_path=DEFAULT_COMFY_QWEN_IMAGE_EDIT_BATCH_WORKFLOW_PATH,
            arch='qwen_image_edit')
        self.assertEqual([g['36']['inputs']['lora_strength'] for g in graphs], [-1, 1])
        self.assertEqual([name for _, name in downloads], ['sample_0.png', 'sample_1.png'])
