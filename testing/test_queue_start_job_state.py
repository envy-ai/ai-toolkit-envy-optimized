import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class QueueStartJobStateTests(unittest.TestCase):
    def test_queue_worker_start_clears_stale_return_to_queue_flag(self):
        start_job_source = (REPO_ROOT / "ui/cron/actions/startJob.ts").read_text()

        self.assertIn("status: 'running'", start_job_source)
        self.assertIn("stop: false", start_job_source)
        self.assertIn("return_to_queue: false", start_job_source)


if __name__ == "__main__":
    unittest.main()
