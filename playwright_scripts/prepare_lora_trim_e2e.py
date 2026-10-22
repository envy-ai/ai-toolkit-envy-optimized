"""Create isolated synthetic fixtures for the LoRA trimming browser smoke.

Run with ``conda run -n ai-toolkit python playwright_scripts/prepare_lora_trim_e2e.py``.
It only recreates ``tmp/lora-trim/e2e.db`` and synthetic files under that
ignored directory; point the Next server at the DB with
``AI_TOOLKIT_DATABASE_URL=file:<absolute-path>/tmp/lora-trim/e2e.db``. The
Playwright runner accepts ``PLAYWRIGHT_MODULE`` if Playwright is not installed
in the repository's Node module resolution path.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "tmp" / "lora-trim"
DATABASE = ARTIFACTS / "e2e.db"
OUTPUT = ARTIFACTS / "e2e-output"
JOB_ID = "trim-e2e"
JOB_NAME = "trim_fixture"


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    job_folder = OUTPUT / JOB_NAME
    job_folder.mkdir(parents=True, exist_ok=True)
    for old_fixture in job_folder.glob("*.safetensors"):
        old_fixture.unlink()
    if DATABASE.exists():
        DATABASE.unlink()

    connection = sqlite3.connect(DATABASE)
    try:
        connection.executescript(
            """
            CREATE TABLE Settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL
            );
            CREATE TABLE Job (
                id TEXT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL UNIQUE,
                gpu_ids TEXT NOT NULL,
                job_config TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'stopped',
                stop BOOLEAN NOT NULL DEFAULT false,
                return_to_queue BOOLEAN NOT NULL DEFAULT false,
                step INTEGER NOT NULL DEFAULT 0,
                total_steps INTEGER,
                info TEXT NOT NULL DEFAULT '',
                speed_string TEXT NOT NULL DEFAULT '',
                queue_position INTEGER NOT NULL DEFAULT 0,
                pid INTEGER,
                job_type TEXT NOT NULL DEFAULT 'train',
                job_ref TEXT,
                save_now BOOLEAN NOT NULL DEFAULT false,
                sample_now BOOLEAN NOT NULL DEFAULT false
            );
            CREATE INDEX Job_status_idx ON Job(status);
            CREATE INDEX Job_gpu_ids_idx ON Job(gpu_ids);
            CREATE INDEX Job_job_type_idx ON Job(job_type);
            CREATE INDEX Job_job_ref_idx ON Job(job_ref);
            """
        )
        connection.execute(
            "INSERT INTO Settings(key, value) VALUES(?, ?)",
            ("TRAINING_FOLDER", str(OUTPUT)),
        )
        config = {
            "config": {
                "process": [
                    {
                        "type": "diffusion_trainer",
                        "train": {"steps": 8},
                        "model": {"name_or_path": "synthetic/local-base", "arch": "synthetic"},
                        "network": {"type": "lora", "linear": 8, "linear_alpha": 8},
                        "sample": {"samples": []},
                        "datasets": [],
                    }
                ]
            }
        }
        connection.execute(
            """INSERT INTO Job(
                id, name, gpu_ids, job_config, created_at, updated_at, status,
                step, total_steps, info, speed_string, job_type
            ) VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'stopped', 0, 8, ?, '', 'train')""",
            (JOB_ID, JOB_NAME, "0", json.dumps(config), "Synthetic browser fixture"),
        )
        connection.commit()
    finally:
        connection.close()

    singular = torch.tensor([16.0, 8.0, 0.5, 0.05, 0.01, 0.005, 0.001, 0.0005]).sqrt()
    down = torch.zeros((8, 64), dtype=torch.float32)
    up = torch.zeros((64, 8), dtype=torch.float32)
    for index, value in enumerate(singular):
        down[index, index] = value
        up[index, index] = value
    safe = {
        "diffusion_model.block.proj.lora_down.weight": down,
        "diffusion_model.block.proj.lora_up.weight": up,
    }
    save_file(safe, job_folder / "synthetic_lora.safetensors", metadata={"ss_base_model": "synthetic/local-base"})
    # Valid DoRA source plus a matching entirely synthetic floating base for
    # the explicit bake flow. These files never touch a user's checkpoint.
    dora = dict(safe)
    dora["diffusion_model.block.proj.alpha"] = torch.tensor(8.0)
    dora["diffusion_model.block.proj.dora_scale"] = torch.ones(64)
    save_file(dora, job_folder / "synthetic_dora.safetensors",
              metadata={"network_module": "networks.dora", "use_dora": "true"})
    torch.manual_seed(431)
    save_file({"diffusion_model.block.proj.weight": torch.randn(64, 64)},
              job_folder / "synthetic_base.safetensors")

    # Unsupported DoHa keeps exercising the visible-validation-error path;
    # plain DoRA is now intentionally supported by the reducer.
    unsupported = {
        "diffusion_model.block.proj.hada_w1_a": torch.ones((4, 2)),
        "diffusion_model.block.proj.hada_w1_b": torch.ones((2, 8)),
        "diffusion_model.block.proj.hada_w2_a": torch.ones((4, 2)),
        "diffusion_model.block.proj.hada_w2_b": torch.ones((2, 8)),
        "diffusion_model.block.proj.hada_t1": torch.ones((2, 2)),
    }
    save_file(unsupported, job_folder / "unsupported_doha.safetensors")

    loha = {
        "diffusion_model.block.proj.hada_w1_a": torch.randn((8, 2)),
        "diffusion_model.block.proj.hada_w1_b": torch.randn((2, 64)),
        "diffusion_model.block.proj.hada_w2_a": torch.randn((8, 2)),
        "diffusion_model.block.proj.hada_w2_b": torch.randn((2, 64)),
        "diffusion_model.block.proj.alpha": torch.tensor(2.0),
    }
    save_file(loha, job_folder / "synthetic_loha.safetensors")
    print(json.dumps({"database": str(DATABASE), "job_id": JOB_ID, "source": str(job_folder / "synthetic_lora.safetensors")}))


if __name__ == "__main__":
    main()
