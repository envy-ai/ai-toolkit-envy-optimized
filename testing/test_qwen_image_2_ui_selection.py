import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class QwenImage2UiSelectionTests(unittest.TestCase):
    def test_architecture_change_receives_runtime_model_registry(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        call_start = source.index("handleModelArchChange(")
        call = source[call_start : source.index(");", call_start) + 2]

        self.assertIn("modelArchs,", call)
        self.assertIn("jobConfig.config.process[0].model.arch,", call)
        self.assertIn("value,", call)
        self.assertIn("jobConfig,", call)
        self.assertIn("setJobConfig,", call)

    def test_qwen_image_2_selection_sets_a_valid_base_model(self):
        source = (REPO_ROOT / "extensions_built_in/diffusion_models/ui.tsx").read_text()
        arch_start = source.index('name: "qwen_image_2"')
        arch_end = source.index('\n  {', arch_start)
        arch = source[arch_start:arch_end]

        self.assertIn('label: "Qwen-Image-2.1"', arch)
        self.assertIn('"Comfy-Org/Qwen-Image-2.1"', arch)
        self.assertIn('"config.process[0].model.name_or_path"', arch)
        self.assertIn('"config.process[0].sample.comfy.workflow_path"', arch)
        self.assertIn('"config/comfy_templates/qwen_image_2_lora_sample.json.njk"', arch)
        self.assertIn('"model.assistant_lora_path"', arch)
        self.assertIn('"model.text_encoder_path"', arch)

    def test_qwen_image_2_workflow_is_selectable(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()

        self.assertIn("config/comfy_templates/qwen_image_2_lora_sample.json.njk", source)
        self.assertIn("Qwen Image 2.1 LoRA image", source)

    def test_invalid_empty_model_path_is_not_saved(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/page.tsx").read_text()

        self.assertIn("const nameOrPath = jobConfig.config.process[0].model.name_or_path", source)
        self.assertIn("nameOrPath.trim() === ''", source)

    def test_job_construction_failure_does_not_call_unbound_job(self):
        source = (REPO_ROOT / "run.py").read_text()

        self.assertIn("job = None", source)
        self.assertGreaterEqual(source.count("if job is not None:"), 2)


if __name__ == "__main__":
    unittest.main()
