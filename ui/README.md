This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

Training jobs support **Fork this job** from their gear menu. It opens the job
form and creates an independent, stopped job from the latest complete checkpoint
and optimizer state. See [the fork workflow and supported layouts](../docs/job_fork.md).

**Pause** saves a checkpoint before stopping when the training job has completed
more than zero steps. At step zero it stops immediately without requesting a
checkpoint or waiting for a training step. **Save Now** remains a separate manual
checkpoint request.

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
