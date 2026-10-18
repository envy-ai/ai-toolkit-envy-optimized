# Private Kaggle checkpoint relay

`checkpoint_relay.py` transports completed save boundaries through a **dedicated private Kaggle dataset**, independently of notebook output publication. It does not submit notebooks, train, deserialize optimizer files, or promise fresh-process resume correctness. The training runner owns those operations and must validate its complete resume contract.

Each Kaggle dataset version contains exactly `checkpoint.bin` and `relay.json`. `checkpoint.bin` holds ZIP bytes under an opaque extension because Kaggle expands uploaded `.zip` files. Its contents include an inference export under `export/`, optional opaque resume files under `resume/`, and a checksummed `manifest.json`. The 40,000-update runner publishes an export every 100 updates and includes full FP32 resume state every 1,000 updates and at each stop or final save. The outer record commits the archive and inner manifest SHA-256 plus job, attempt, and completed optimizer step. The publisher retains every old dataset version and acknowledges a save only after processing reaches `ready` and a version-pinned read verifies the uploaded bytes. An upload error is never retried automatically as a mutation: a network failure can occur after Kaggle has committed a version. Inspect the dedicated dataset and watcher state before deciding whether to repeat publication.

Use one publisher per dataset. Parallel publishers are unsupported and detected when the observed version increment differs from one. Dataset privacy is checked before each upload/download; publication refuses an existing public dataset. Changing privacy externally remains outside the relay's control. Keep the dataset private, grant credentials only to this notebook, and do not use a dataset shared with unrelated jobs.

## Worker integration

At a completed optimizer update, synchronize ranks, finish writing the export and any scheduled resume state, then call:

```python
from workers.kaggle_tpu.checkpoint_relay import (
    KaggleDataset, build_checkpoint, publish_checkpoint,
)

checkpoint = build_checkpoint(
    "/kaggle/working/checkpoints", job_id="qwen2.1-anime-34_copy-diffguidance",
    attempt_id=attempt_id, step=completed_updates,
    files={
        "export/adapter.safetensors": export_path,
        # Include resume/training.pt only at full resume boundaries.
        "resume/training.pt": training_state_path,
    },
    metadata={"identity": training_identity, "world_size": world_size},
)
receipt = publish_checkpoint(
    KaggleDataset("OWNER/DEDICATED-PRIVATE-DATASET"), checkpoint,
    create=first_dataset_publication,
)
```

Set `first_dataset_publication=True` only for the first checkpoint of a new dataset, and `False` thereafter, including new TPU sessions. `create=True` refuses to replace an existing dataset. Kaggle may return HTTP 403 for an absent private dataset; the relay treats that as missing only when the slug owner matches the authenticated username and the exact reference is absent from every page of the authenticated `mine` listing. It keeps 403 as an error otherwise. Never create an empty bootstrap version: every relay version must be a valid checkpoint. Local checkpoint names are immutable; use a new attempt ID when restarting an interrupted attempt.

When present, `training.pt` must include adapter parameters, optimizer and scheduler state, completed updates/sample counts, per-rank XLA RNG state, and the data cursor. Its immutable identities must include base/helper/cache hashes, adapter representation and quantization contract, topology, effective batch and objective settings. Restore must reject incompatible identities. An export-only save supports inference and local archival, but cannot resume AdamW training at that exact step; after an interrupted session, resume from the most recent full checkpoint. The relay checks bytes and identity labels; the runner must check these semantic requirements before resuming. The existing one-step smoke checkpoint is diagnostic evidence and does not by itself establish this full resume contract.

Retrieve notebook credentials from Kaggle Secrets directly into the subprocess environment, before constructing `KaggleDataset`. For a current Kaggle access token:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
```

Alternatively configure Secrets named `KAGGLE_USERNAME` and `KAGGLE_KEY` for the official legacy authentication path. Enable the selected Secrets for this private notebook. Never print values, serialize the environment, put credentials in the code/cache bundle, or pass credentials on the command line. Supply only explicit checkpoint files and credential-free identity metadata. Do not dump raw Kaggle exceptions or signed URLs into notebook logs. The standalone watcher suppresses raw exception text for that reason.

One save acknowledgment defaults to a 900-second transfer/processing timeout. This timeout does **not** bound TPU queue time, session duration, training progress, or the local watcher. Leave sufficient time before the separately managed nine-hour TPU runtime boundary for a final save and acknowledged publication. Abrupt termination may lose updates after the last acknowledged save.

## Persistent local watcher

Use a new output folder, separate from the current local training job. The watcher refuses a nonempty folder on first initialization, and refuses either folder being an ancestor of the other:

```bash
conda run -n ai-toolkit python -m workers.kaggle_tpu.checkpoint_relay \
  --dataset OWNER/DEDICATED-PRIVATE-DATASET \
  --job-id qwen2.1-anime-34_copy-diffguidance \
  --current-job /absolute/path/to/current/local/job \
  --destination /absolute/path/to/new/tpu/output
```

Authenticate the local Kaggle client using its existing private credential configuration. The process polls indefinitely, including while the dataset is absent before its first save. It consumes every unseen version in order, downloads each exact version, rejects unsafe ZIP members and hash mismatches, and installs complete checkpoints by atomic rename under `checkpoints/ATTEMPT--step-NNNNNNNNN/`. It never unpickles downloaded state. It stops nonzero on a corrupt version, ownership mismatch or API error, leaving previous verified files and the resume pointer intact. Run under a persistent service with restart-on-failure; an unavailable API must remain visible as a failed/restarting service, not successful progress.

Example user systemd unit (replace all paths and identifiers before installing):

```ini
[Unit]
Description=Private Kaggle TPU checkpoint pull
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=/home/bart/ai-toolkit-qwen
ExecStart=/absolute/path/to/conda run --no-capture-output -n ai-toolkit python -m workers.kaggle_tpu.checkpoint_relay --dataset OWNER/DEDICATED-PRIVATE-DATASET --job-id qwen2.1-anime-34_copy-diffguidance --current-job /absolute/current/job --destination /absolute/new/output
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
```

`relay-owner.json` fences the local folder to its job and dataset. A nonblocking file lock allows only one standalone watcher. `watcher-state.json` records the last completely installed dataset version. `latest-export.json` points to the highest verified export; `latest.json` points to the highest verified **full resume** checkpoint, with dataset version, attempt ID, archive/manifest hashes, installed checkpoint path and opaque resume-file paths. The latter may be absent until the first 1,000-update or stop save. Reprocessing after interruption is idempotent only when the same identity has exactly the same manifest. Older saves cannot move pointers backwards, and different checkpoints claiming an equal step cause an explicit error.

An orchestrator can use `initialize_destination`, `watcher_lock` and `poll_once(..., on_checkpoint=callback)` instead of the CLI. The callback receives a verified install record; the durable `latest.json` is authoritative after an orchestrator restart. Notebook status polling, nine-hour session boundaries, launching a subsequent notebook version and stopping after 40,000 completed updates belong to that orchestrator. It must wait for a terminal notebook state and select the latest verified resume checkpoint before restarting, and must not apply an overall TPU queue deadline. The callback alone is not a durable event queue.

The concrete local controller is `session_orchestrator.py`; see [its operator guide](session_orchestrator.md). Run either its integrated watcher or this standalone watcher for a destination, never both at once.

## Sources and limits

The official [dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) document private creation, dataset version publication and version retention. The [current CLI implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py) implements `dataset_create_new(public=False)`, `dataset_create_version(delete_old_versions=False)`, `dataset_status` and `dataset_download_files("owner/slug/version")`; its generated SDK exposes dataset privacy and current version. This module targets that API shape. [Kaggle's dataset specifications](https://www.kaggle.com/docs/datasets) state a 200 GB limit per dataset and across private datasets. The 192 DoRA target shapes in the local base-model safetensors metadata imply about 162.3 MB per BF16 export and 973.6 MB per full FP32 adapter/AdamW state. Four hundred exports plus 40 regular full states total about 103.9 GB before archive metadata; each additional stop boundary with full state adds about 1.14 GB if it is an extra save, or 0.974 GB if it replaces a scheduled export-only save. The remote base/helper asset dataset totals 7.341 GB; the local paired cache occupies about 0.268 GB. The local source profile also names a 10.594 GB text encoder, but the paired cache replaces that model during TPU training and it is absent from the remote asset dataset. These are conservative raw-byte estimates, not measured TPU checkpoint files. An older [Kaggle staff announcement](https://www.kaggle.com/discussions/product-feedback/50755) said the private quota counted only the latest version of each dataset; current specifications do not clarify historical-version accounting. No live private dataset upload or notebook mutation is performed by offline tests. Kaggle dataset quotas, processing latency and remote service availability remain external limits.

A tiny live private transport probe on 2026-10-16 used `nerfviking/aitk-relay-opaque-probe-20261016-v1`: version 2 retained the 627-byte `checkpoint.bin` intact, and a version-pinned download passed archive, manifest and recovered export hash checks. Version 1 also passed a read-only pinned download after its first publication returned an ambiguous local error; the publisher did not retry that mutation. This verifies opaque archive transport for small payloads, not full checkpoint size, sustained publication rate or TPU resume.
