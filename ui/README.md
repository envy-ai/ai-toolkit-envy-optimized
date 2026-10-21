This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

The eye icon beside a training job on the dashboard, queue, or jobs list opens
its **Samples** tab directly (`/jobs/<id>?tab=samples`). Click it or focus it with
Tab and press Enter. Job-name links still open the overview.

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

Dataset **AutoCaption** includes **API (OpenAI-compatible)**. Enter the API URL, optional key, model ID, system prompt,
and caption prompt; **Advanced options** contains sampling settings, timeout, and the response-token limit (4096 by
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
