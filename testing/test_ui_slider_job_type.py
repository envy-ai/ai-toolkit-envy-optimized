import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class UISliderJobTypeTests(unittest.TestCase):
    def test_base_model_can_use_absolute_comfy_checkpoint_path(self):
        simple_job_source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        docs_source = (REPO_ROOT / "ui/src/docs.tsx").read_text()

        self.assertIn("Base Model Source", simple_job_source)
        self.assertIn("AI Toolkit / Hugging Face", simple_job_source)
        self.assertIn("ComfyUI Checkpoint Path", simple_job_source)
        self.assertIn("comfyCheckpointPathSelected", simple_job_source)
        self.assertIn("config.process[0].model.name_or_path", simple_job_source)
        self.assertIn("Paste the full absolute path", simple_job_source)

        self.assertIn("ComfyUI checkpoint", docs_source)
        self.assertIn("absolute path", docs_source)

    def test_slider_job_type_hides_datasets_and_uses_repeatable_targets(self):
        options_source = (REPO_ROOT / "ui/src/app/jobs/new/options.ts").read_text()
        simple_job_source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        types_source = (REPO_ROOT / "ui/src/types.ts").read_text()
        job_config_source = (REPO_ROOT / "ui/src/app/jobs/new/jobConfig.ts").read_text()

        self.assertIn("value: 'slider'", options_source)
        self.assertIn("label: 'Slider LoRA'", options_source)
        self.assertIn("'datasets'", options_source)
        self.assertIn("config.config.process[0].train.unload_text_encoder = true", options_source)
        self.assertIn("config.config.process[0].train.cache_text_embeddings = false", options_source)
        self.assertIn("config.config.process[0].train.max_denoising_steps", options_source)

        self.assertIn("targets?: SliderTargetConfig[];", types_source)
        self.assertIn("positive: string;", types_source)
        self.assertIn("negative: string;", types_source)

        self.assertIn("targets: [", job_config_source)
        self.assertIn("positive: 'person who is happy'", job_config_source)
        self.assertIn("negative: 'person who is sad'", job_config_source)
        self.assertIn("batch_full_slide: false", job_config_source)

        self.assertIn("!disableSections.includes('datasets')", simple_job_source)
        self.assertIn("Slider Targets", simple_job_source)
        self.assertIn("config.process[0].slider.targets", simple_job_source)
        self.assertIn("Add Prompt Set", simple_job_source)
        self.assertIn("Full Slide Batch", simple_job_source)
        self.assertIn("config.process[0].slider.batch_full_slide", simple_job_source)

        dataset_slot_index = simple_job_source.index("{!disableSections.includes('datasets')")
        slider_targets_index = simple_job_source.index("{sliderTargetsEditor}")
        self.assertGreater(
            slider_targets_index,
            dataset_slot_index,
            "Slider prompt targets should render in the full-width dataset section area",
        )

    def test_prodigy_optimizer_shows_d_adaptation_knobs(self):
        simple_job_source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        types_source = (REPO_ROOT / "ui/src/types.ts").read_text()

        self.assertIn("d0?: number;", types_source)
        self.assertIn("d_coef?: number;", types_source)

        self.assertIn("startsWith('prodigy')", simple_job_source)
        self.assertIn("Initial D Estimate", simple_job_source)
        self.assertIn("D Coefficient", simple_job_source)
        self.assertIn("config.process[0].train.optimizer_params.d0", simple_job_source)
        self.assertIn("config.process[0].train.optimizer_params.d_coef", simple_job_source)


if __name__ == "__main__":
    unittest.main()
