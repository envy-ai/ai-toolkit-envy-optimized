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


if __name__ == "__main__":
    unittest.main()
