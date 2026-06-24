import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class SampleInferenceLoraFieldTests(unittest.TestCase):
    def test_sample_card_exposes_inference_lora_model_path(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        sample_start = source.index('<Card title="Sample">')
        sample_prompts_start = source.index("Sample Prompts", sample_start)
        sample_section = source[sample_start:sample_prompts_start]

        self.assertIn('label="Inference LoRA Path"', sample_section)
        self.assertIn(
            "jobConfig.config.process[0].model.inference_lora_path",
            sample_section,
        )
        self.assertIn(
            "config.process[0].model.inference_lora_path",
            sample_section,
        )

    def test_model_config_type_and_docs_include_inference_lora_path(self):
        types_source = (REPO_ROOT / "ui/src/types.ts").read_text()
        docs_source = (REPO_ROOT / "ui/src/docs.tsx").read_text()

        self.assertIn("inference_lora_path?: string;", types_source)
        self.assertIn("'config.process[0].model.inference_lora_path'", docs_source)


if __name__ == "__main__":
    unittest.main()
