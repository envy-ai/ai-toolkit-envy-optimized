# Render samples from a saved job

Training jobs that are stopped, completed, or in error expose **Render Samples**
in the gear menu. Running jobs retain **Sample Next Step**. Use the mouse or
focus **Job actions** with Tab, open the menu with Enter/Space, and select the
action with the arrow keys and Enter. Escape closes the menu or details dialog.

The action renders the job's currently saved sample configuration from its
latest checkpoint. It does not restart training, create an optimizer, load
datasets, request another training step, or change the job's status, progress,
PID, configuration, save/sample flags, or stop flags. It writes ordinary outputs
to the existing `samples` folder, so the Samples view displays them normally.
The details dialog identifies the checkpoint and its saved training step.

The supported path is an ordinary diffusion/UI/SD trainer with an unmerged
LoRA checkpoint and the ordinary ComfyUI workflow renderer. Per-sample prompt
whitespace, seeds, dimensions, steps, guidance, scales and reference images use
the same sample builder and workflow renderer as training. Configured batching
and Comfy path replacements still apply. Comfy's background setting is ignored
for this subprocess so completion and errors cover the entire request.

Native internal sampling, shared models, specialized training previews, trained
adapters, embeddings and decorators are rejected with an explanation. Those
paths depend on trainer-owned model/network attachment, architecture-specific
adapter mapping, or preview state that this inference-only runner does not
reconstruct. The generic model loader's `lora_path` merge is not equivalent to
the configured trainable network, particularly for DoRA/LoHA and per-sample
network scales.

Only `<job name>.safetensors` and `<job name>_<nine-digit step>.safetensors`
in the job's own output folder are considered. The most recently modified save
is selected; its header and tensor boundaries are validated before rendering.
Unrelated, helper and record-index files are excluded. A damaged latest save
produces an error instead of using an older checkpoint or base-model weights.

ComfyUI configures its own GPU independently of the job's GPU list. Until that
mapping is available, running and stopping toolkit jobs block saved sample
rendering. Completed jobs may retain obsolete PIDs, which do not block a render.
Queued jobs may remain queued. Rendering also requires an empty Comfy queue.
A durable gate defers new toolkit queue launches until the render finishes. The
renderer does not stop jobs, clear queues, cancel someone else's prompts, or
restart services. Work
submitted by another Comfy client after the empty-queue check remains outside
the toolkit's launch gate.

Rendering runs asynchronously. Details show progress and retain full error
messages and tracebacks. Closing the dialog leaves the process running; the
image/spinner button reopens it. Status is read on a detail page or when opening
a row's gear menu, then polled while running. A failed status request keeps
actions disabled and retries, rather than treating the render as finished.

`POST /api/jobs/<id>/render_samples` validates and starts the request (202).
`GET` on the same endpoint returns `idle`, `running`, `completed`, or `error`
with message, checkpoint, saved step, renderer PID, and error/stack when present.
Requests and status live in `<job folder>/.sample_render`; subprocess output is
in its `log.txt`. Launch gates live under `data/gpu-launch-locks` and reclaim
dead process owners. A hard-killed renderer is reported as an error on the next
status read and its gate can be reclaimed by the next launch.
