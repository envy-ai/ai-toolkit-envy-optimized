from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class QwenFlowDPOFormTests(unittest.TestCase):
    def test_job_type_and_slider_style_pairing_are_on_form(self):
        options = (ROOT / "ui/src/app/jobs/new/options.tsx").read_text()
        page = (ROOT / "ui/src/app/jobs/new/page.tsx").read_text()
        form = (ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        self.assertIn("value: 'qwen_flow_dpo'", options)
        self.assertIn("option.value !== 'qwen_flow_dpo'", page)
        self.assertIn("Control Dataset 1 (Rejected Images)", form)
        self.assertIn("Control Dataset 2 (Edit Source 1)", form)
        self.assertIn("Target Dataset contains preferred outputs", form)
        self.assertIn("label=\"DPO Beta\"", form)
        self.assertIn("label=\"Preferred Image Loss Weight\"", form)
        self.assertIn("Cache Preferred Latents (required)", form)


if __name__ == "__main__":
    unittest.main()
