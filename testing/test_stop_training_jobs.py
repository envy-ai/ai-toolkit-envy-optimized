import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stop_training_jobs import stop_training_jobs


JOB_SCHEMA = """
CREATE TABLE Job (
    id TEXT NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    gpu_ids TEXT NOT NULL,
    job_config TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'stopped',
    stop BOOLEAN NOT NULL DEFAULT false,
    return_to_queue BOOLEAN NOT NULL DEFAULT false,
    step INTEGER NOT NULL DEFAULT 0,
    info TEXT NOT NULL DEFAULT '',
    speed_string TEXT NOT NULL DEFAULT '',
    queue_position INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    job_type TEXT NOT NULL DEFAULT 'train',
    job_ref TEXT
)
"""


def insert_job(conn, job_id, status, job_type, stop=0, return_to_queue=0, pid=None):
    conn.execute(
        """
        INSERT INTO Job (
            id, name, gpu_ids, job_config, status, stop, return_to_queue, pid, job_type
        )
        VALUES (?, ?, '0', '{}', ?, ?, ?, ?, ?)
        """,
        (job_id, job_id, status, stop, return_to_queue, pid, job_type),
    )


class StopTrainingJobsTests(unittest.TestCase):
    def test_marks_only_training_jobs_as_stopped(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "aitk_db.db")
            with sqlite3.connect(db_path) as conn:
                conn.execute(JOB_SCHEMA)
                insert_job(
                    conn,
                    "train-running",
                    "running",
                    "train",
                    return_to_queue=1,
                    pid=123,
                )
                insert_job(conn, "train-queued", "queued", "train", pid=456)
                insert_job(conn, "caption-running", "running", "caption", pid=789)

            changed = stop_training_jobs(db_path)

            with sqlite3.connect(db_path) as conn:
                rows = {
                    row[0]: row
                    for row in conn.execute(
                        """
                        SELECT id, status, stop, return_to_queue, info, pid
                        FROM Job
                        ORDER BY id
                        """
                    )
                }

        self.assertEqual(changed, 2)
        self.assertEqual(rows["train-running"], ("train-running", "stopped", 1, 0, "Job stopped", None))
        self.assertEqual(rows["train-queued"], ("train-queued", "stopped", 1, 0, "Job stopped", None))
        self.assertEqual(rows["caption-running"], ("caption-running", "running", 0, 0, "", 789))


if __name__ == "__main__":
    unittest.main()
