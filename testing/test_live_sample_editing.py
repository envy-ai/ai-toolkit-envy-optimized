import pathlib
import shutil
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class LiveSampleEditingTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which('node') and (REPO_ROOT / 'ui/node_modules/typescript').is_dir(),
        'Node and the UI TypeScript dependency are needed for the API merge check',
    )
    def test_api_live_toggle_preserves_other_save_settings(self):
        script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('./ui/node_modules/typescript');
const source = ts.createSourceFile('route.ts', fs.readFileSync('ui/src/app/api/jobs/route.ts', 'utf8'), ts.ScriptTarget.Latest, true);
const functions = source.statements.filter(statement =>
    ts.isVariableStatement(statement) && statement.declarationList.declarations.some(declaration =>
        ['cloneJson', 'mergeSampleOnlyJobConfig'].includes(declaration.name.getText(source))
    )
).map(statement => statement.getText(source)).join('\n');
const compiled = ts.transpileModule(functions, {compilerOptions: {target: ts.ScriptTarget.ES2020}}).outputText;
const merge = vm.runInNewContext(compiled + '\nmergeSampleOnlyJobConfig');
const wrap = process => ({config: {process: [process]}});
for (const enabled of [false, true]) {
    const existing = wrap({save: {sample_on_record_low: !enabled, save_every: 100, max_record_low_saves_to_keep: 5}, sample: {sample_every: 400}});
    const original = JSON.stringify(existing);
    const incoming = wrap({save: {sample_on_record_low: enabled, save_every: 1, max_record_low_saves_to_keep: 1}, sample: {sample_every: 800}});
    const result = JSON.parse(JSON.stringify(merge(existing, incoming))).config.process[0];
    assert.equal(result.save.sample_on_record_low, enabled);
    assert.equal(result.sample.sample_every, 800);
    assert.equal(result.save.save_every, 100);
    assert.equal(result.save.max_record_low_saves_to_keep, 5);
    assert.equal(JSON.stringify(existing), original);
}
const existing = wrap({save: {sample_on_record_low: false}, sample: {}});
const result = merge(existing, wrap({save: {save_every: 1}, sample: {}}));
assert.equal(result.config.process[0].save.sample_on_record_low, false);
'''
        result = subprocess.run(
            ['node', '-e', script], cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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
            train_process_source.index("is_sample_step = sample_reason is not None"),
        )
        self.assertIn("self.sample_config = SampleConfig", train_process_source)


if __name__ == "__main__":
    unittest.main()
