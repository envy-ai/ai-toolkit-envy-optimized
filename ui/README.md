This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

The eye icon beside a training job on the dashboard, queue, or jobs list opens
its **Samples** tab directly (`/jobs/<id>?tab=samples`). Click it or focus it with
Tab and press Enter. Job-name links still open the overview.

Each safetensors checkpoint in the training console's **Checkpoints** panel has
a compact **Reduce adapter** icon (Google Material Compress), beside download
and delete. Hover for its title, or use Tab then Enter/Space to open the shared
**Reduce adapter** dialog. **Rank policy** offers retained energy (99.99% by
default), relative reconstruction error (1%), fixed rank (32), singular-value
ratio (1000), cumulative singular values (99.99%), and Frobenius norm retention
(99.99%). Percentage inputs are converted to fractions for the backend.
Energy/error choose the smallest rank meeting the measured bound after storage
rounding. Other policies select from the singular-value spectrum, with any
rank limits applied, and report the resulting retention/error. Frobenius norm
retention squares the requested norm fraction to select by energy.

Click **Analyze** for a preview that writes nothing and shows estimated
file size, per-layer factor ranks, formats, measurement scopes, retained energy,
and reconstruction error. **Advanced rank limits and memory** expands with a
click or Enter/Space. By default, DoRA magnitude tensors remain unchanged;
its visible warning and table identify measurements as direction-only, before
normalization. They do
not bound final normalized weight error or visual changes without the base.
Analysis automatically selects CUDA when available, otherwise CPU. The existing
layer-count summary identifies the actual device/GPU; its hover title shows the
Python interpreter. CUDA errors or insufficient VRAM stop the operation with
details rather than silently retrying on CPU. CLI users can select `--device cpu`
or `--device cuda:0` explicitly. Checkpoint decoding and final serialization stay
on CPU; per-layer decomposition and reconstruction measurements use the selected
device. `run.sh` exports `AI_TOOLKIT_PYTHON` after activating Conda `ai-toolkit`;
the shared script/job resolver honors that exact executable, then active Conda
before repository virtual environments. An invalid explicit override is an error.
The separate **Bake DoRA to ordinary LoRA before reducing** switch is off by
default. Enabling it requires a **Matching base safetensors checkpoint** path
to a single local checkpoint, and an explicit **DoRA convention**:
**ComfyUI** or **AI Toolkit / native**, according to the loader normally used.
No convention is selected automatically. The reducer prefills the editable
base path from the training job when its configured model
is an absolute `.safetensors` ComfyUI checkpoint path. Repository IDs and model
directories leave the field blank; checkbox toggles preserve manual edits.
Baking computes the final effective DoRA update with those base weights, then
reduces it to ordinary LoRA. It preserves the source and base files; the preview identifies baked effective
update measurements separately from native DoRA direction-only measurements.
Floating weights, explicitly marked Comfy ConvRot8 / per-row int8, E4M3 FP8,
NVFP4 (including AWQ), W4A8, and ConvRot W4A4 are supported. Tensor markers and
Comfy header quantization metadata are recognized; legacy scaled FP8 is also
supported. **Quantized base precision**
selects FP32, BF16 or FP16 reconstruction before baking, initialized from the
job's training dtype when supported (otherwise FP32). Select the precision used
during training; checkpoint codes do not record it. This selector uses the shared
searchable select and supports keyboard selection. Manual path and precision
edits survive checkbox toggles; reopening initializes from the current job again.
Changing precision invalidates analysis. Floating checkpoint weights retain their
stored precision. Mismatched, malformed or unmarked quantized tensors fail with
details. Baking uses reconstructed logical weights; activation quantization and
loader arithmetic can differ. AWQ outputs depend on the chosen consumer: ComfyUI
uses stored-weight deltas, while native toolkit uses raw-input adapter deltas.
The preview warns against interchanging them.
MOSS split Q/K/V factors with one fused magnitude are reduced jointly. DoHa
(LoHa with DoRA magnitude), including its supported Conv2d layouts, can be baked
with this same switch; choose ComfyUI for exported `.dora_scale` and native for
an internal `.magnitude`. Native DoRA's scaled base bias is preserved as `.diff_b`
when needed; the preview identifies this requirement for a bias-capable consumer.
LoHa/LoKr require the separate **Convert LoHa / LoKr to ordinary LoRA** switch,
off by default. Conversion produces an ordinary LoRA and supports only the
documented 2D layouts. Enabling either conversion or DoRA baking exposes a
shared **Conversion / baking working-memory limit (MiB)** in Advanced options
(512 MiB by default); an over-budget estimate fails before expanded allocation
or dense baking/decomposition work.
Changing analysis settings clears the preview. Enter in a text/number field
analyzes; **Create trimmed copy** requires its own explicit button activation
after a current analysis. Use a new `.safetensors` filename in the source
folder; existing files cannot be overwritten. Escape or the backdrop closes an
idle dialog and returns focus to the Reduce icon. While analysis or creation runs, closing
and duplicate submissions are disabled. Errors persist with expandable stderr
and stack details, and a failed save keeps the preview for retry. Numerical
reconstruction measurements do not predict visual quality. See
[supported formats, scaling, and trimming limits](../docs/lora_trimming.md).

Shared form controls in `src/components/formInputs.tsx` supply the existing
text, number, and searchable-select appearance. `TextInput`, `NumberInput`,
`SelectInput`, and `TextAreaInput` accept an optional `id` that associates their
visible label with the native input (`inputId` for react-select). API caption cleanup uses
the shared textareas' IDs for its From/To labels. The trimming workflow uses
these controls and the shared `Modal`; numeric validation uses
`clampOnBlur={false}` to reject invalid values without silently changing them.

For ordinary LoKr training, select **LoKr** under **Target Type** and enable
**DoRA Weight Decomposition (DoKr)** beneath the LoKr factor selector. The
shared checkbox supports clicking its label or focusing it with Tab and pressing
Space/Enter. It saves `network.lokr_dora`, which defaults to false. **LoKr Full
Rank** preserves the new-job default of true; turn it off to use the editable
Linear and Conv Rank/Alpha fields. An imported config without the full-rank flag
uses the backend default of false. The shared Conv Rank/Alpha controls are
available for LoRA, DoRA, LoHa, and LoKr across model architectures and training
forms; they affect the network only when its targets include Conv2d layers.
The current DoRA implementation cannot attach to Conv2d layers.
Existing LoKr factor and memory settings are retained. See
[LoKr and optional DoRA](../docs/lokr_training.md) for checkpoint conventions and limits.

The AI assistant drawer autosaves titled chats on this toolkit server. Open
**Chat history** to search recent sessions, resume one, rename it, or delete it.
**New conversation** preserves the previous chat. Titles are generated locally
and can be renamed; duplicate titles remain separate sessions. Drafts and image
references survive reopening. Stop a running reply before changing sessions.
Save failures keep the local text and expose retry/error details; concurrent
edits offer saving a copy or confirmed reload. See [assistant sessions and
storage](../docs/ai_assistant.md) for migration, shared history, and limits.

Dataset cards show the entire first image inside their square preview area,
keeping its aspect ratio and transparency; any extra space uses the card's dark
background. They request an uncropped thumbnail bounded to 300 pixels per axis
using `?thumb=fit`, with a separate `.thumbs/<filename>.fit.png` (alpha) or
`.fit.jpg` cache. The front file server generates these previews on demand;
the Next.js image route reads the same cache variant. A standalone Next.js
server keeps its existing fallback to the uncropped original when that cache
does not exist. Ordinary `?thumb=1`
gallery thumbnails keep their existing square crop. First-image discovery
supports PNG, JPEG, WebP, GIF, BMP, and AVIF.

Dataset **AutoCaption** includes **API (OpenAI-compatible)**. Enter the API URL, optional key, optional model ID, system prompt,
and caption prompt. Leave **Model (optional)** blank for servers such as Strata that choose their own model; blank or
omitted IDs send no `model` field, with no model lookup or default substitution.
**Advanced options** contains sampling settings, timeout, and the response-token limit (4096 by
default, configurable). Click the Advanced summary or focus it and press Enter/Space to expand it. API captioning uses
its own CPU/API queue alongside training and loads no captioning model onto this toolkit's GPUs. Keys are saved
privately on the server, separately from job configurations. See the [API captioner guide](../docs/api_captioner.md) for
authentication, endpoint compatibility, supported images, and failure behavior.

The **Queue** page (`/jobs`) initially loads the 100 newest jobs by creation
time. Scrolling down to the end loads the next 100; the **Load more** button
also works with Tab and Enter/Space. Search checks every job name in the database
using all space-separated substrings, without regard to case or term order.
Matching results use the same 100-at-a-time loading. Changing the search resets
the list; polling refreshes the number of results already loaded. Selection and
bulk deletion apply to the loaded matching rows. Failed loads retain existing
rows and expose **Retry** controls.

Queued jobs can be reordered from the Queue page or dashboard, including while
a job is running and while pagination or search hides other queued jobs. Drag
the handle, or focus it and press Space, use arrow keys, then press Space again
to drop (Escape cancels). Running and stopping jobs stay pinned above the queued
rows and cannot be dragged. Hidden queued jobs retain their slots relative to
the visible jobs; reordering reads only queue IDs/positions on the server, so
the page does not fetch the rest of the training history. Handles pause while
an order is being saved. Failed saves show a persistent error and refresh the
current results without losing the loaded page depth.

Training jobs support **Fork this job** from their gear menu. It opens the job
form and creates an independent, stopped job from the latest complete checkpoint
and optimizer state. See [the fork workflow and supported layouts](../docs/job_fork.md).

**Pause** saves a checkpoint before stopping when the training job has completed
more than zero steps. At step zero it stops immediately without requesting a
checkpoint or waiting for a training step. **Save Now** remains a separate manual
checkpoint request.

Stopped, completed and failed training jobs expose **Render Samples** in the
gear menu to render configured ComfyUI samples from their latest saved LoRA,
without restarting training. Progress, checkpoint details and errors are
available from the sample status icon. See [saved-job sampling](../docs/job_samples.md)
for supported renderers, checkpoint selection and the GPU launch gate.

Dataset **Resolutions** include **Native resolution (no scaling)**. Toggle it by
clicking its label/switch, or focus the switch with Tab and press Space. Enabled
datasets batch targets by exact source width and height, disable the resolution
preset controls, and retain the preset selections for later use. See
[dataset preparation](../README.md#dataset-preparation) for alignment requirements,
unsupported transforms, and reference-image preprocessing.

Training jobs' **Sample Prompts** use the shared `TextAreaInput` for every
ordinary prompt and textual tagged field. Enter inserts a newline;
Ctrl/Cmd+Enter saves the job, as does the existing Save button. Newlines,
including blank first and last lines, survive editing, JSON/YAML config
serialization, sample prompt caching, and native or ComfyUI sample dispatch.
Tagged text is read verbatim using `tagsToObj(prompt, textTagNames)`; numeric
tags retain their JSON value parsing. Editing one tag preserves the other
text tags' whitespace and JSON-looking text values.

Training sampling and its text-embedding cache opt into
`GenerateImageConfig(preserve_prompt_whitespace=True)`. Other callers retain
their existing trimming behavior. Inline `--` sampling flags remain supported:
text flags (`--n`, `--p2`, `--n2`) consume one separating space after the flag
name and preserve the remaining text; numeric and path flags retain their
existing whitespace handling. Newlines before a flag remain part of the
preceding prompt text. Model-specific tokenization and prompt interpretation
still apply after sample dispatch.

The job-wide **Prepend Trigger Word to Sample Prompts** switch in **Sample** is off
by default (`sample.prepend_trigger_word: false`). When enabled, sampling adds
the job's `trigger_word` or phrase, then exactly `\n\n`, before each prompt.
Stored prompt text, its whitespace, and negative prompts stay unchanged. An
absent or empty trigger adds nothing. Existing `[trigger]` and `[name]`
replacement still applies; the prefix is added even if the prompt already
contains the trigger. Native sampling, its text-embedding cache, and ComfyUI
(including saved-job **Render Samples**) use the same prefix setting.

Click the switch or its label, or focus it with Tab and press Space/Enter.
Running jobs using native sampling and cached text embeddings (**Unload TE**,
**Cache Text Embeddings**, or dataset text caching) require stopping, editing,
and restarting to change the prefix or switch from ComfyUI back to native
sampling. The sample-settings save rejects those changes with a restart message;
live ComfyUI prefix edits remain available.

**Krea 2 Identity/Edit (krea2edit extension)** selects the external `krea2_edit`
training architecture. Its Model options expose **Fit References to Target**,
**Train with Two References**, and **Checkpoint Every N Transformer Blocks**.
Use one ordered **Reference Dataset** folder, or enable two-reference training
and select two. Select matching reference images separately for Comfy samples;
the source-graph preview supports one reference per sample. New selections use
the source-graph Krea 2 Edit LoRA workflow and ComfyUI renderer. Keep batch size
1, both flips off, and Unload TE off; Cache Text Embeddings is enabled for
memory savings. Native Generate is unavailable for this extension. Existing
jobs retain their saved architecture and settings. See the
[Comfy sample workflow guide](../docs/krea2_edit_comfy_samples.md).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
