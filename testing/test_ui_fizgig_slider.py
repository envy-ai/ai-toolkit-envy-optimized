import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class FizgigSliderFormTests(unittest.TestCase):
    def test_both_job_types_are_qwen_only_in_selector(self):
        options = (REPO_ROOT / "ui/src/app/jobs/new/options.tsx").read_text()
        page = (REPO_ROOT / "ui/src/app/jobs/new/page.tsx").read_text()
        self.assertIn("value: 'fizgig_image_slider'", options)
        self.assertIn("value: 'fizgig_prompt_slider'", options)
        self.assertIn("model.arch === 'qwen_image_2'", page)
        self.assertIn("option.value.startsWith('fizgig_')", page)

    def test_image_negative_uses_control_dataset_one_and_prompt_fields_are_full_width(self):
        form = (REPO_ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text()
        self.assertIn("Control Dataset 1 (−1 Images)", form)
        for title in ("Neutral Prompt (0)", "Positive Prompt (+1)", "Negative Prompt (−1)"):
            self.assertIn(f'<Card title="{title}">', form)
        self.assertIn("rows={5}", form)
        for label in ("Negative Prefix (−1)", "Positive Prefix (+1)", "Base Prompt (0)",
                      "Add Simplified Prompt", "Add Specific Triplet"):
            self.assertIn(label, form)
        self.assertIn("updatePromptEntries", form)
        self.assertIn("DoRA (signed slider)", form)
        options = (REPO_ROOT / "ui/src/app/jobs/new/options.tsx").read_text()
        self.assertIn("prompt_entries: [{ kind: 'simple'", options)
        self.assertIn("cfg_scale: 1.0", options)
        self.assertIn('label="CFG (Practice + Training)"', form)
        for pole in ("Neutral (0)", "Positive (+1)", "Negative (−1)"):
            self.assertIn(f'label="CFG Negative Prefix for {pole}"', form)
            self.assertIn(f'<Card title="CFG Negative Prompt for {pole}">', form)
        self.assertIn('disabled={!sliderCfgEnabled || !hasSimplifiedPrompts}', form)
        self.assertIn('disabled={!sliderCfgEnabled}', form)


if __name__ == "__main__":
    unittest.main()
