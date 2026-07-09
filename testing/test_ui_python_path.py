import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class UIPythonPathTests(unittest.TestCase):
    def test_resolver_prefers_conda_env_python_before_system_python(self):
        source = (REPO_ROOT / "ui/cron/pythonPath.ts").read_text()

        self.assertIn("process.env.CONDA_PREFIX", source)
        self.assertIn("'.conda', 'envs', 'ai-toolkit'", source)
        self.assertLess(source.index("process.env.CONDA_PREFIX"), source.index("return isWindows ? 'python.exe' : 'python3';"))


if __name__ == "__main__":
    unittest.main()
