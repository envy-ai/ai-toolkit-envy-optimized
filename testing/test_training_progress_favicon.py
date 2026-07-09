import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TrainingProgressFaviconTests(unittest.TestCase):
    def test_root_layout_mounts_training_progress_favicon(self):
        layout_source = (REPO_ROOT / "ui/src/app/layout.tsx").read_text()

        self.assertIn("TrainingProgressFavicon", layout_source)
        self.assertIn("<TrainingProgressFavicon />", layout_source)

    def test_favicon_component_polls_train_jobs_and_restores_default_when_idle(self):
        component_source = (REPO_ROOT / "ui/src/components/TrainingProgressFavicon.tsx").read_text()

        self.assertIn("useJobsList({ onlyActive: true, reloadInterval: 5000, job_type: 'train' })", component_source)
        self.assertIn("getRunningTrainingProgress(jobs)", component_source)
        self.assertIn("getTrainingQueueFaviconState(jobs)", component_source)
        self.assertIn("drawTrainingProgressFavicon", component_source)
        self.assertIn("restoreDefaultFavicon", component_source)

    def test_idle_favicon_restores_transparent_icon_background(self):
        util_source = (REPO_ROOT / "ui/src/utils/faviconProgress.ts").read_text()
        component_source = (REPO_ROOT / "ui/src/components/TrainingProgressFavicon.tsx").read_text()

        self.assertIn("const DEFAULT_FAVICON_HREF = '/icon.png';", util_source)
        self.assertIn("normalFavicon.href = DEFAULT_FAVICON_HREF;", util_source)
        self.assertIn("defaultFaviconHref.current = restoreDefaultFavicon();", component_source)

    def test_progress_utility_filters_running_train_jobs_and_draws_yellow_bar_under_icon(self):
        util_source = (REPO_ROOT / "ui/src/utils/faviconProgress.ts").read_text()

        self.assertIn("job.job_type === 'train'", util_source)
        self.assertIn("job.status === 'running'", util_source)
        self.assertIn("return null", util_source)
        self.assertIn("Math.min(1, Math.max(0", util_source)
        self.assertIn("context.fillStyle = '#facc15'", util_source)
        self.assertIn("const y = size - barHeight", util_source)
        self.assertIn("drawProgressUnderlay();", util_source)
        self.assertLess(
            util_source.index("drawProgressUnderlay();"),
            util_source.index("context.drawImage(image, 0, 0, size, size);"),
        )

    def test_favicon_draws_queue_count_and_running_state_only_when_queue_has_items(self):
        util_source = (REPO_ROOT / "ui/src/utils/faviconProgress.ts").read_text()
        component_source = (REPO_ROOT / "ui/src/components/TrainingProgressFavicon.tsx").read_text()

        self.assertIn("job.status === 'queued'", util_source)
        self.assertIn("queueCount", util_source)
        self.assertIn("queueCount <= 0", util_source)
        self.assertIn("context.strokeText", util_source)
        self.assertIn("context.fillText", util_source)
        self.assertIn("PLAY_PAUSE_COLOR = '#67e8f9'", util_source)
        self.assertIn("QUEUE_COUNT_COLOR = '#bef264'", util_source)
        self.assertIn("context.textAlign = 'right';", util_source)
        self.assertIn("context.textAlign = 'left';", util_source)
        self.assertIn("'\\u25b6'", util_source)
        self.assertIn("'\\u23f8'", util_source)
        self.assertIn("queueState === null && progress === null", component_source)


if __name__ == "__main__":
    unittest.main()
