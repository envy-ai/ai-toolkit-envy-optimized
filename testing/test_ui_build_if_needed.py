import json
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class UIBuildIfNeededTests(unittest.TestCase):
    def test_package_scripts_use_guarded_build_before_start(self):
        package_json = json.loads((REPO_ROOT / "ui/package.json").read_text())
        scripts = package_json["scripts"]

        self.assertEqual(scripts["build_if_needed"], "node scripts/build-if-needed.mjs")
        self.assertEqual(scripts["start_if_needed"], "npm run build_if_needed && npm run start")
        self.assertIn("npm run build_if_needed", scripts["build_and_start"])
        self.assertNotIn("npm run build && npm run start", scripts["build_and_start"])

    def test_guarded_build_checks_required_outputs_and_source_inputs(self):
        script_source = (REPO_ROOT / "ui/scripts/build-if-needed.mjs").read_text()

        self.assertIn("'.next', 'BUILD_ID'", script_source)
        self.assertIn("'dist', 'cron', 'worker.js'", script_source)
        self.assertIn("'src'", script_source)
        self.assertIn("'cron'", script_source)
        self.assertIn("'prisma', 'schema.prisma'", script_source)
        self.assertIn("newestSourceMtime", script_source)
        self.assertIn("oldestOutputMtime", script_source)
        self.assertIn("spawnSync('npm', ['run', 'build']", script_source)
        self.assertIn("--dry-run", script_source)


if __name__ == "__main__":
    unittest.main()
