# Local Kaggle TPU session orchestration

`session_orchestrator.py` is the local, restartable controller for the private
`qwen2.1-anime-34_copy-diffguidance` 40,000-update, eight-rank training job. It
uses `checkpoint_relay.py` to pull every private dataset version into a new
local folder, verifies each archive and manifest, and selects the highest
completed optimizer step. The controller then submits a private Kaggle notebook
version, waits through the batch TPU queue without an overall timeout, and
submits another version only after the preceding notebook is terminal and a
newer verified checkpoint is installed. It stops at update 40,000.

The default launch submits the full 40,000-update training session directly.
It does not require a separate smoke notebook. The actual TPU run remains
responsible for reporting model fit, compilation and training failures. The
runner validates the private source bundle, exact model/cache profile,
installed TPU runtime, eight-rank topology and resume identities before
training. A failed session with no new verified checkpoint stops the controller.

The training notebook needs the full source bundle from `build_full_bundle.py`,
the paired cache and base/helper assets. The metadata input references must
match the actual private inputs. Leave `is_private`, `enable_tpu`, Internet and
`machine_shape: TpuV5E8` enabled. Set `docker_image` explicitly to `""` in the
training metadata and leave the notebook free of `kaggle.dockerImageVersionId`.
The empty value clears the CPU draft's inherited image. The controller requests
`acc="TpuV5E8"` and verifies the resulting server metadata before adoption.
Kaggle may resolve the empty image to
`gcr.io/kaggle-private-byod/python-tpuvm@sha256:<64 hex digits>`; that TPU
image is accepted. CPU images and other nonempty image references are rejected.

An optional smoke gate remains available with `--smoke-kernel`,
`--smoke-report-dataset`, `--cache-manifest` and `--asset-manifest` together.
That mode adopts and verifies an already-submitted one-rank smoke and publishes
a private pinned report before training; omit these options for direct launch.

Kaggle attaches Secrets to an existing notebook, so first create the **private
CPU draft** at the eventual training slug. The ready-to-push folder is
`workers/kaggle_tpu/train_draft/`; its metadata names
`nerfviking/qwen21-fizgig-train`, title `Qwen21 Fizgig Train`, Internet enabled,
GPU and TPU disabled. Its only code cell prints a message. To submit that draft
separately, use the installed official CLI without a timeout or accelerator
override:

```bash
conda run -n ai-toolkit kaggle kernels push -p workers/kaggle_tpu/train_draft
```

Wait until that version is `COMPLETE`, then enable the account Secret in that
private notebook's Settings. Kaggle may create another draft version when
the Secret is enabled. The controller accepts any terminal version with the
exact private CPU draft code and title as its initial baseline. It waits if
the draft is absent or still running, and rejects a failed, public, GPU, TPU,
renamed or modified draft. The later controller push replaces it with the
actual private TPU training notebook; the draft itself consumes no TPU session.

Enable the Kaggle Secret `KAGGLE_API_TOKEN` on the private training notebook. Its value is
read by `kaggle_secrets` inside the notebook and enters only the training
subprocess environment. Authenticate the *local* official Kaggle client with
its existing private credential configuration. Never put a token in a command,
metadata, notebook, source bundle, relay dataset or local state file. The relay
dataset must be dedicated to this one training job and remain private. Its
first version is created by the first completed save, not by an empty bootstrap.

The controller does **not** submit training until a local Secret readiness
marker has been created. While it is absent, relay polling and any explicitly configured optional smoke
gate continue indefinitely. After you have enabled
the Secret on the private training notebook, create the marker named by
`--secret-ready-file` with exactly this credential-free JSON (substitute your
job and kernel identifiers):

```json
{
  "schema": 1,
  "job_id": "qwen2.1-anime-34_copy-diffguidance",
  "kernel": "nerfviking/qwen21-fizgig-train",
  "secret_name": "KAGGLE_API_TOKEN",
  "confirmed": true
}
```

The controller reads the marker as a regular file, records the confirmation in
`orchestrator-state.json` before its first notebook push, and then can survive
a restart without requiring the marker again. No Secret value is stored there.
The marker is a user confirmation; the local controller cannot inspect which
Kaggle Secrets are enabled in the remote notebook settings.

The destination must be a new/empty folder separate from the current local job
folder. Use this command after the private training-source dataset is ready;
the controller can run before the Secret marker exists:

```bash
conda run --no-capture-output -n ai-toolkit python -m workers.kaggle_tpu.session_orchestrator \
  --dataset nerfviking/qwen21-fizgig-checkpoint-relay \
  --kernel nerfviking/qwen21-fizgig-train \
  --job-id qwen2.1-anime-34_copy-diffguidance \
  --secret-ready-file /home/bart/ai-toolkit-qwen/output/qwen2.1-anime-34_copy-diffguidance-kaggle-tpu/secret-ready.json \
  --current-job /home/bart/ai-toolkit-qwen/output/qwen2.1-anime-34_copy-diffguidance \
  --destination /home/bart/ai-toolkit-qwen/output/qwen2.1-anime-34_copy-diffguidance-kaggle-tpu
```

The command performs remote notebook submissions. Running syntax checks and
offline tests does not. No overall job timeout or batch-queue timeout is
passed to Kaggle. The training notebook records `allocation_started_unix` in
its first code cell and passes that to the runner; its separate nine-hour TPU
session budget therefore includes setup, loading, training and final save.
`--checkpoint-wait-seconds` defaults to 1800 and applies only after a notebook
is terminal and a newer verified checkpoint has not yet arrived. This bound
prevents a failed session from replaying indefinitely from an unchanged step.

`relay-owner.json` and `orchestrator-state.json` bind the destination to the
job, dataset and training kernel. Optional smoke mode additionally retains
`smoke-gate-state.json` and `smoke/version-N/` proof. `submissions/` holds the exact generated notebook for
each attempt, `checkpoints/` holds immutable verified saves, and `latest.json`
is the durable resume pointer. The relay watcher lock prevents concurrent
controllers or a separate watcher from writing the same folder. Restart the
same command after a local interruption; the controller reconciles its
recorded notebook version and source before polling or submitting again. An
ambiguous push stays in `pushing` state and is never retried as a mutation.
Inspect the remote version and staged notebook before deciding how to recover
if reconciliation cannot prove that the push committed. Likewise, an external
notebook edit, a nonterminal existing session, a corrupt relay version or a
terminal session without a newer verified checkpoint stops the controller
nonzero rather than launching a duplicate or silently discarding progress.

The notebook downloads the pinned private dataset version directly with the
official Kaggle client. It verifies the archive and manifest via
`install_download`, then compares the step and manifest hash with the local
pointer embedded in that attempt's notebook. Every new session uses a new
attempt ID and an empty Kaggle output directory. The runner rejects changed
base/helper/cache/source/runtime identity, optimizer contract, topology and
batch schedule before accepting a resume. The local folder can be used for
inference export from `checkpoints/<attempt>/export/adapter.safetensors` and
later local training only after checking the chosen workflow's adapter format.

The controller uses the installed official Kaggle Python API:
`kernels_push(folder, timeout=None, acc="TpuV5E8")`, `kernels_status`, `kernels_output`
and the versioned kernel source returned by `get_kernel`. It does not depend on a
fixed Kaggle queue duration. API errors stop nonzero with credential-bearing
details suppressed. For unattended operation use a user service with
`Restart=on-failure`, retaining the same destination and arguments. A failed
service remains an actionable error; a restart never blindly repeats an
ambiguous push. Kaggle quotas, batch TPU availability, processing latency and
actual full-model TPU fit remain external acceptance conditions.

`--poll-seconds` controls the local controller interval, including queue waits;
`--poll-seconds 28800` checks every eight hours without setting a queue deadline.
The notebook's own five-second rank/deadline monitor remains active during TPU
training. The first staged code cell records allocation time before dependency
setup; session configuration injection preserves that timestamp.
