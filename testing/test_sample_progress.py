"""CPU checks of visible per-image progress, including modular pipelines."""
import io
import unittest
from types import SimpleNamespace

from diffusers import DiffusionPipeline
from toolkit.sample_progress import SampleProgressMixin, configure_sample_progress


class SampleProgressTests(unittest.TestCase):
    def assert_two_step_bars(self, pipeline):
        stream = io.StringIO()
        pipeline.set_progress_bar_config(file=stream, disable=True, mininterval=0)
        for image_index in range(2):
            configure_sample_progress(pipeline, image_index, 2)
            completed = list(pipeline.progress_bar(range(20)))
            self.assertEqual(completed, list(range(20)))
        # Carriage-return redraws stay on each image's line; leave=True ends
        # each finished image with a newline, including when output is captured.
        lines = [line for line in stream.getvalue().split("\n") if line]
        self.assertEqual(len(lines), 2)
        for index, line in enumerate(lines):
            self.assertIn(f"Sample {index + 1}/2", line)
            self.assertIn("20/20", line)
            self.assertIn("100%", line)

    def test_custom_pipeline_two_twenty_step_images(self):
        self.assert_two_step_bars(SampleProgressMixin())

    def test_diffusers_pipeline_two_twenty_step_images(self):
        pipeline = DiffusionPipeline()
        pipeline.register_to_config()
        self.assert_two_step_bars(pipeline)

    def test_nested_modular_denoising_loop(self):
        loop = SampleProgressMixin()
        stream = io.StringIO()
        loop.set_progress_bar_config(file=stream, disable=True)
        pipeline = SimpleNamespace(_blocks=SimpleNamespace(sub_blocks={
            "wrapper": SimpleNamespace(sub_blocks={"denoising": loop})}))
        configure_sample_progress(pipeline, 1, 3)
        list(loop.progress_bar(range(20)))
        self.assertIn("Sample 2/3", stream.getvalue())
        self.assertIn("20/20", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
