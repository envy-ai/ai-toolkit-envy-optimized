"""Prepare an isolated SQLite database and tiny synthetic dataset for the regex UI E2E.

Run inside the ai-toolkit Conda environment. The script refuses to replace an
existing database and never reads the toolkit's active database or user files.
"""
import argparse
import json
import sqlite3
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("tmp/caption-cleanup/e2e.db"))
    parser.add_argument("--root", type=Path, default=Path("tmp/caption-cleanup"))
    args = parser.parse_args()
    database = args.database.resolve()
    root = args.root.resolve()
    if database.exists():
        raise SystemExit(f"Refusing to replace existing isolated DB: {database}")

    datasets = root / "datasets"
    training = root / "output"
    for name, color in (
        ("regex_fixture", (42, 111, 196)),
        ("regex_fixture_second", (156, 78, 142)),
        ("regex_fixture_third", (70, 150, 80)),
    ):
        folder = datasets / name
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color).save(folder / "tiny_fixture.png")
    training.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)

    api_preferences = {
        "type": "APICaptioner",
        "caption": {
            "api_url": "http://127.0.0.1:9/v1",
            "model_name_or_path": "regex-e2e-fixture-model",
            "system_prompt": "Return a concise caption.",
            "caption_prompt": "Describe the synthetic image.",
            "max_new_tokens": 4096,
            "api_timeout": 3,
            "api_max_tokens_field": "max_tokens",
            "image_megapixels": 1,
            "caption_extension": "txt",
            "extensions": ["png"],
            "recaption": False,
        },
        "gpu_ids": "cpu",
    }
    rows = {
        "DATASETS_FOLDER": str(datasets),
        "TRAINING_FOLDER": str(training),
        "CAPTION_DEFAULTS": json.dumps(api_preferences, separators=(",", ":")),
    }

    with sqlite3.connect(database) as connection:
        connection.executescript(
            '''
            CREATE TABLE "Settings" (
              "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
              "key" TEXT NOT NULL UNIQUE,
              "value" TEXT NOT NULL
            );
            CREATE TABLE "Queue" (
              "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
              "gpu_ids" TEXT NOT NULL UNIQUE,
              "is_running" BOOLEAN NOT NULL DEFAULT false
            );
            CREATE INDEX "Queue_gpu_ids_idx" ON "Queue"("gpu_ids");
            CREATE TABLE "Job" (
              "id" TEXT NOT NULL PRIMARY KEY,
              "name" TEXT NOT NULL UNIQUE,
              "gpu_ids" TEXT NOT NULL,
              "job_config" TEXT NOT NULL,
              "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              "updated_at" DATETIME NOT NULL,
              "status" TEXT NOT NULL DEFAULT 'stopped',
              "stop" BOOLEAN NOT NULL DEFAULT false,
              "return_to_queue" BOOLEAN NOT NULL DEFAULT false,
              "step" INTEGER NOT NULL DEFAULT 0,
              "total_steps" INTEGER,
              "info" TEXT NOT NULL DEFAULT '',
              "speed_string" TEXT NOT NULL DEFAULT '',
              "queue_position" INTEGER NOT NULL DEFAULT 0,
              "pid" INTEGER,
              "job_type" TEXT NOT NULL DEFAULT 'train',
              "job_ref" TEXT,
              "save_now" BOOLEAN NOT NULL DEFAULT false,
              "sample_now" BOOLEAN NOT NULL DEFAULT false
            );
            CREATE INDEX "Job_status_idx" ON "Job"("status");
            CREATE INDEX "Job_gpu_ids_idx" ON "Job"("gpu_ids");
            CREATE INDEX "Job_job_type_idx" ON "Job"("job_type");
            CREATE INDEX "Job_job_ref_idx" ON "Job"("job_ref");
            '''
        )
        connection.executemany('INSERT INTO "Settings" ("key", "value") VALUES (?, ?)', rows.items())
        connection.execute('INSERT INTO "Queue" ("gpu_ids", "is_running") VALUES (?, false)', ("cpu",))

    print(json.dumps({
        "database": str(database),
        "dataset_names": ["regex_fixture", "regex_fixture_second", "regex_fixture_third"],
        "image_count": 3,
        "initial_captioner": "APICaptioner",
        "provider_url": "http://127.0.0.1:9/v1 (never called by browser test)",
    }))


if __name__ == "__main__":
    main()
