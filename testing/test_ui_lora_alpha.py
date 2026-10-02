import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class LoraAlphaFormTests(unittest.TestCase):
    def test_linear_alpha_is_visible_and_documented_for_lora(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        docs = (REPO_ROOT / "ui/src/docs.tsx").read_text()

        self.assertIn('label="Linear Alpha"', source)
        self.assertIn('docKey="config.process[0].network.linear_alpha"', source)
        self.assertNotIn(
            "{networkType == 'dora' && (\n                  <NumberInput\n                    label=\"Linear Alpha\"",
            source,
        )
        self.assertIn("'config.process[0].network.linear_alpha'", docs)
        self.assertIn("alpha / rank", docs)

    def test_changing_rank_preserves_a_custom_alpha(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()

        self.assertIn("const currentRank = jobConfig.config.process[0].network?.linear", source)
        self.assertIn("const currentAlpha = jobConfig.config.process[0].network?.linear_alpha", source)
        self.assertIn("currentAlpha === currentRank", source)


if __name__ == "__main__":
    unittest.main()
