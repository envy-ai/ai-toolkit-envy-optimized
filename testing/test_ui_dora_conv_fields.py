import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class DoRAConvFieldTests(unittest.TestCase):
    def test_dora_selection_materializes_conv_fields_for_yaml(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()

        self.assertIn("useEffect", source)
        self.assertIn("networkType != 'dora'", source)
        self.assertIn("'config.process[0].network.conv'", source)
        self.assertIn("'config.process[0].network.conv_alpha'", source)
        self.assertIn("setJobConfig(16, 'config.process[0].network.conv')", source)
        self.assertIn("setJobConfig(convValue, 'config.process[0].network.conv_alpha')", source)

    def test_dora_selection_can_enable_magnitude_less_lora_output(self):
        simple_job_source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        types_source = (REPO_ROOT / "ui/src/types.ts").read_text()
        default_config_source = (REPO_ROOT / "ui/src/app/jobs/new/jobConfig.ts").read_text()
        backend_config_source = (REPO_ROOT / "toolkit/config_modules.py").read_text()
        train_process_source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()

        self.assertIn("save_magnitude_less_lora?: boolean", types_source)
        self.assertIn("save_magnitude_less_lora: false", default_config_source)
        self.assertIn("self.save_magnitude_less_lora", backend_config_source)
        self.assertIn('label="Save magnitude-less LoRAs"', simple_job_source)
        self.assertIn("networkType == 'dora'", simple_job_source)
        self.assertIn(
            "'config.process[0].network.save_magnitude_less_lora'",
            simple_job_source,
        )
        self.assertIn(
            "save_magnitude_less_lora=self.network.network_config.save_magnitude_less_lora",
            train_process_source,
        )

    def test_magnitude_less_lora_sidecars_do_not_count_against_save_retention(self):
        train_process_source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()

        self.assertIn("not f.endswith('-lora.safetensors')", train_process_source)
        self.assertIn("sidecar_file = os.path.splitext(item)[0] + '-lora.safetensors'", train_process_source)
        self.assertIn("if os.path.exists(sidecar_file):", train_process_source)


if __name__ == "__main__":
    unittest.main()
