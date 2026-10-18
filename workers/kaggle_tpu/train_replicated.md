`train_replicated.py` runs the full private eight-rank Qwen/Fizgig training
job directly, without requiring a separate smoke report. It validates the source,
cache, base/helper identities and qualified Kaggle runtime, launches through
`torch_xla.launch`, and requires exactly eight logical devices. The first real
session establishes model fit and training behavior; failures remain visible
and stop the complete launcher process group.

The fixed contract is 40,000 completed optimizer updates, per-rank batch one,
one accumulation step and global batch eight. Every rank owns the complete
frozen ConvRot transformer and helper plus signed DoRA. The objective uses the
same shared Fizgig implementation as local training. AdamW retains learning
rate, weight decay and epsilon `1e-6`, with FP32 adapter parameters/moments.
Each rank averages the +1/-1 endpoint losses, one explicit collective averages
adapter gradients across eight ranks, clipping happens after that reduction,
and ordinary `optimizer.step()` performs no additional reduction.

CPU model loading and resume staging run one replica at a time. Every adapter
tensor is hashed to verify matching initialization, and all ranks verify the
schedule fingerprint. Canonical cache exposures retain configured repeats;
deterministic epoch shuffles allocate distinct presentations to each rank.
Incomplete global batches are dropped and counted. The supplied 355 exposures
produce 44 global batches and three dropped exposures per epoch. The first
version deliberately requires one exact image/text shape and validity layout,
preserving existing positions without artificial padding or reference inputs.

```bash
PJRT_DEVICE=TPU python workers/kaggle_tpu/train_replicated.py \
  --cache /kaggle/working/paired-cache \
  --asset-manifest /kaggle/input/model-assets/asset-manifest.json \
  --allocation-started-unix "$ALLOCATION_STARTED_UNIX" \
  --job-id qwen21-fizgig --attempt-id session-001 \
  --relay-dataset nerfviking/PRIVATE-CHECKPOINT-RELAY --relay-create \
  --output /kaggle/working/session-001
```

The optional `--smoke-report` flag additionally verifies a passing one-rank
full-model diagnostic before launch. Credentials come from the official Kaggle client environment and never
enter configuration or checkpoint metadata. The supervisor cleans inherited
TPU worker-address variables and launches a separate process group. An error
rank, watchdog expiry or hard session deadline terminates the entire group.

Every 100 updates, at requested stop boundaries, and at completion, rank zero
serializes a BF16 inference adapter. It also writes full FP32 resume state
every 1,000 updates and at each stop or final save. The resume state
contains optimizer, constant scheduler, completed updates/samples, exact epoch
and global cursor, every rank's XLA RNG, model/cache/source/runtime identities,
and the effective batch contract. CPU RNG is used only during serialized
initialization; training randomness uses device RNG and deterministic local
schedule generators. The cursor identifies the next unconsumed global batch.

The private relay publishes and acknowledges a pinned dataset version before
the save counts as durable. Publication failure preserves local files
and stops all ranks. Remote versions remain available; the worker keeps its
two newest acknowledged full resume directories and current export locally.
The local watcher installs every export and tracks the latest full state for
resuming. An interruption between full boundaries resumes from the preceding
full checkpoint, while all later export-only LoRAs remain locally available
for inference. The default session
budget is nine hours from the actual TPU allocation start, with a final-save request 30 minutes before the end and
a 15-minute publication timeout. Compilation/update time still consumes that
reserve; an update that cannot reach a save boundary before the hard deadline
can lose work since the last acknowledged version.

Active training requires either `--allocation-started-unix` from the actual
allocation or an absolute `--session-deadline-unix`. Record that time before
notebook dependency installation; launcher startup cannot establish a new
budget. The runner rejects a missing timestamp or a remaining budget already
within the final-save reserve. Host memory is sampled across the launcher
process group; summed RSS may count shared pages more than once, and the
diagnostic labels that limitation.

For another session, attach/download a verified relay checkpoint directory,
pass `--resume /path/to/verified/checkpoint`, use a new `--attempt-id` and empty
output directory, and omit `--relay-create`. Resume rejects changed topology,
batch, objective/cache, base/helper, runtime versions or training source.
An already verified terminal checkpoint at update 40,000 is accepted without
launching replicas or publishing another version; its identity and sample/rank
state remain validated against the selected profile.
Creating `/kaggle/working/session-001/stop.json` requests a checkpoint at the
next completed update boundary. Merely closing the notebook is not a save.

CPU checks cover the repeat-aware schedule, exact cursor continuation, optional failed
smoke-gate rejection and archive traversal guard:

```bash
conda run -n ai-toolkit python workers/kaggle_tpu/check_replicated_schedule.py \
  tmp/kaggle_qwen21_pairs
```

Run local diagnostic commands inside the required 16-GiB/no-swap systemd
scope. Actual all-device update agreement, sustained fit, throughput, publication
and fresh-session resume remain TPU acceptance work; dense attention and the
peak-HBM measurement limitations remain in effect.
