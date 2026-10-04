"""CPU-only safety checks for the manually invoked GPU smoke harness."""
import unittest
import importlib
import os
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from PIL import Image

from testing.smoke_flow_comfy_gpu import evaluate_identity_renders, release_smoke_vram
from toolkit.comfy_sample import COMFY_CACHE_MONITOR_RELEASE_PATH


class SmokeCleanupSafetyTests(unittest.TestCase):
    def test_import_does_not_hide_cuda_from_other_tests(self):
        import testing.smoke_flow_comfy_gpu as harness
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '7'}):
            importlib.reload(harness)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '7')

    def test_idle_queue_uses_only_cache_preserving_endpoint(self):
        client, report = Mock(), {}
        client._request_json.side_effect = [dict(queue_running=[], queue_pending=[]), dict(released=True)]
        release_smoke_vram(client, report)
        self.assertEqual(client._request_json.call_args_list[1].args,
                         ('POST', COMFY_CACHE_MONITOR_RELEASE_PATH))
        self.assertTrue(report['vram_cleanup']['released'])
        client.release_vram.assert_not_called()
        client.unload_models.assert_not_called()

    def test_another_queued_render_is_not_offloaded(self):
        for running, pending in [(['another-render'], []), ([], ['another-render'])]:
            client, report = Mock(), {}
            client._request_json.return_value = dict(queue_running=running, queue_pending=pending)
            release_smoke_vram(client, report)
            self.assertEqual(client._request_json.call_count, 1)
            self.assertIn('skipped', report['vram_cleanup'])

    def test_unavailable_endpoint_never_falls_back_to_cache_eviction(self):
        client, report = Mock(), {}
        client._request_json.side_effect = [dict(queue_running=[], queue_pending=[]), TimeoutError('offline')]
        release_smoke_vram(client, report)
        self.assertEqual(client._request_json.call_count, 2)
        self.assertFalse(report['vram_cleanup']['cache_eviction_attempted'])
        client.release_vram.assert_not_called()
        client.unload_models.assert_not_called()


class SmokeIdentityTests(unittest.TestCase):
    def test_zero_adapter_must_preserve_base_at_all_strengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'base.png'
            Image.new('RGB', (4, 3), (10, 20, 30)).save(path)
            result = evaluate_identity_renders([
                dict(strength=strength, image=str(path)) for strength in (-1., 0., .5, 1.)])
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['comparisons']), 3)
        self.assertTrue(all(item['max_absolute_channel_difference'] == 0
                            for item in result['comparisons']))

    def test_successful_render_does_not_hide_quantized_identity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            base, drift = Path(directory) / 'base.png', Path(directory) / 'drift.png'
            Image.new('RGB', (4, 3), (10, 20, 30)).save(base)
            Image.new('RGB', (4, 3), (14, 24, 34)).save(drift)
            result = evaluate_identity_renders([
                dict(strength=0., image=str(base)), dict(strength=1., image=str(drift))])
        self.assertFalse(result['passed'])
        self.assertEqual(result['comparisons'][0]['mean_absolute_channel_difference'], 4)
        self.assertEqual(result['comparisons'][0]['max_absolute_channel_difference'], 4)

    def test_incomplete_or_incompatible_comparisons_are_not_passes(self):
        for renders in ([], [dict(strength=1., image='unused')],
                        [dict(strength=0., image='unused')],
                        [dict(strength=0., image='unused')] * 2):
            with self.assertRaises(ValueError):
                evaluate_identity_renders(renders)
        with tempfile.TemporaryDirectory() as directory:
            small, large = Path(directory) / 'small.png', Path(directory) / 'large.png'
            Image.new('RGB', (2, 2)).save(small)
            Image.new('RGB', (3, 3)).save(large)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                evaluate_identity_renders([
                    dict(strength=0., image=str(small)), dict(strength=1., image=str(large))])


if __name__ == '__main__':
    unittest.main()
