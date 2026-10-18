# Forking a training job

The training job gear menu's **Fork this job** opens the existing new-job form
with the source configuration and a unique suggested name. Review or edit the
settings, then create the fork. It is created **stopped** and displays the saved
checkpoint step. Its total step limit stays unchanged; increase it if the
checkpoint already reached that limit. Starting the job remains a separate
action. The source must be stopped, completed, or in error, with no live process
in its database/PID records. Pause a running source and wait for its process to
exit before forking.

Fork follows the toolkit's existing checkpoint resume semantics. It restores
weights, optimizer state, metadata step/epoch, and required trainer caches;
it does not promise identical random-number streams or dataset ordering.

Creation copies the latest complete resume generation, selected at creation
time. Ordinary jobs use the trainer's newest checkpoint by filesystem ctime,
including final unnumbered checkpoints. The training step comes from checkpoint
metadata; numbered filenames must agree. The weights are renamed for the new
job's normal resume discovery. `optimizer.pt` and enabled learnable SNR state
are copied independently. Full-model diffusers folders retain their contents
and `aitk_meta.yaml`. Samples, logs, and historical checkpoints are omitted.
An existing `network.pretrained_lora_path` can remain in the configuration:
the fork's saved checkpoint takes precedence, even if that path no longer exists.
Forking requires a saved checkpoint; pretrained initialization alone is the
separate Clone workflow.

SliderSpace forks copy the latest published complete bank generation, its
paired optimizer, its direction exports, and the active discovery cache that
defines its PCA directions and training inputs. Export filenames and manifest
references are updated for the new job. Other discovery caches and generations
are omitted. Fizgig jobs also copy their current practice latent cache when
present, so unchanged practice inputs can be reused without rendering them
again. Changing practice prompts or rendering settings uses the trainer's
normal cache invalidation.

Checkpoint files and resume caches are real copies, never hardlinks or symlinks.
Creation checks source file identities before and after copying and checks job
status/PIDs again before creating the database row. Failures remove the new
folder and leave the source intact. No source job is stopped or restarted by
this action.

Loss settings, learning rate, training length, datasets, sampling settings, and
ordinary trainer guidance/performance settings can be edited. Model identity,
network parameter layout, optimizer type, and trained model components must
remain compatible. SliderSpace also preserves its discovery identity, as its
trainer requires. `train.start_step` is removed from the preview; resume uses
the copied metadata. A `.fork_resume.json` guard makes fork starts and later
resumes fail visibly if weights, optimizer, metadata step, or parameter layout
cannot be restored, instead of continuing with a fresh optimizer or step zero.

Combined adapter/embedding/decorator training, refiner training, merged-network
saves, unsupported checkpoint formats, missing metadata, missing/corrupt
optimizer files, symlinks, and inconsistent SliderSpace manifests are rejected
with a persistent form error and error details. Existing ordinary checkpoints
do not publish an explicit optimizer pairing manifest: their singleton optimizer
is validated as a loadable optimizer state and restored using the source job's
normal resume behavior. If its timestamp predates the weights, the form reports
that pairing cannot be proved from the legacy files; timestamps alone do not
block a valid resume. This cannot prove pairing for manually restored files.
Per-parameter optimizer counters may legitimately differ from the trainer step
after prior resumes and are restored without an equality requirement.

Implementation: `ui/src/server/jobFork.ts` coordinates source checks and calls
`toolkit/job_fork.py`; `GET /api/jobs?forkId=ID` prepares the form and
`POST /api/jobs` with `fork_from_id` creates the independent job. The same Python
helper validates guarded forks before model loading in `BaseSDTrainProcess`.
