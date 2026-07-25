import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class LiveSampleEditingTests(unittest.TestCase):
    def test_running_training_jobs_open_sample_only_editor(self):
        jobs_source = (REPO_ROOT / "ui/src/utils/jobs.ts").read_text()
        action_bar_source = (REPO_ROOT / "ui/src/components/JobActionBar.tsx").read_text()

        self.assertIn("canEditSamples", jobs_source)
        self.assertIn("job.status === 'running'", jobs_source)
        self.assertIn("sampleOnly=1", action_bar_source)
        self.assertIn("canEditSamples", action_bar_source)

    def test_training_form_posts_sample_only_edits_and_disables_full_editor(self):
        page_source = (REPO_ROOT / "ui/src/app/jobs/new/page.tsx").read_text()
        simple_job_source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()

        self.assertIn("const sampleOnlyMode = searchParams.get('sampleOnly') === '1'", page_source)
        self.assertIn("sample_only: sampleOnlyMode", page_source)
        self.assertIn("!sampleOnlyMode && showAdvancedView", page_source)
        self.assertIn("sampleOnlyMode={sampleOnlyMode}", page_source)
        self.assertIn("sampleOnlyMode?: boolean", simple_job_source)
        self.assertIn("sampleOnlyLockedClass", simple_job_source)
        self.assertIn("<div className={sampleOnlyLockedClass}>", simple_job_source)
        self.assertIn('<Card title="Sample">', simple_job_source)

    def test_jobs_api_whitelists_sample_only_updates_and_writes_running_snapshot(self):
        route_source = (REPO_ROOT / "ui/src/app/api/jobs/route.ts").read_text()

        self.assertIn("mergeSampleOnlyJobConfig", route_source)
        self.assertIn("existingConfig.config.process[0].sample", route_source)
        self.assertIn("incomingConfig.config.process[0].sample", route_source)
        self.assertIn("inference_lora_path", route_source)
        self.assertIn("writeRunningJobConfigSnapshot", route_source)
        self.assertIn("fs.renameSync", route_source)
        self.assertIn(".job_config.json", route_source)
        self.assertIn("status !== 'running'", route_source)

    def test_trainer_reloads_sample_config_from_disk_before_sample_decision(self):
        toolkit_config_source = (REPO_ROOT / "toolkit/config.py").read_text()
        base_job_source = (REPO_ROOT / "jobs/BaseJob.py").read_text()
        train_process_source = (REPO_ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()

        self.assertIn("__config_path", toolkit_config_source)
        self.assertIn("self.config_path = config.get('__config_path'", base_job_source)
        self.assertIn("def _refresh_live_sample_config", train_process_source)
        self.assertIn("self._refresh_live_sample_config()", train_process_source)
        self.assertLess(
            train_process_source.index("self._refresh_live_sample_config()"),
            train_process_source.index("is_sample_step = ("),
        )
        self.assertIn("self.sample_config = SampleConfig", train_process_source)


if __name__ == "__main__":
    unittest.main()
