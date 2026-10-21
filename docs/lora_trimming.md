# Adaptive LoRA trimming

Use `scripts/trim_lora.py` to reduce ordinary LoRA ranks independently on CPU.
It does not load the base model, use the GPU, download weights, execute adapter
code, or change the original checkpoint. Numerical similarity does not prove
identical generated images: compare your normal prompts after trimming.

## Preview and save

The default is a dry run with retained squared singular-value energy **0.9999**
for every layer. The report includes each old/new rank, measured error, original
storage dtypes, alpha/runtime scaling and predicted complete output size.
Metadata/header overhead can outweigh tensor savings in very small files.

```bash
conda run -n ai-toolkit python scripts/trim_lora.py output/run/adapter.safetensors
```

Choose a different energy target, or a relative Frobenius-error limit, never
both. Relative error is `||original effective update - trimmed update||F /
||original effective update||F`. Energy here is `1 - relative_error²`; the
report separately includes the ideal truncated singular-value energy.

```bash
conda run -n ai-toolkit python scripts/trim_lora.py adapter.safetensors \
  --retained-energy 0.99999 --json
conda run -n ai-toolkit python scripts/trim_lora.py adapter.safetensors \
  --relative-error 0.005 --min-rank 2 --max-rank 32
```

`--min-rank` defaults to 1. Ranks never increase. Invalid or impossible rank
bounds fail explicitly; a maximum rank never silently weakens the requested
error threshold. Zero updates can use rank 1 (or your minimum). A layer stays
bit-exact at its original rank when no smaller storage-rounded factorization
meets the threshold.

Only `--output` enables writing. The destination must be a new `.safetensors`
path; existing files and symlinks are never overwritten. The output is prepared
in its destination directory, flushed and atomically published without clobber.
Write failures propagate with a traceback and nonzero exit status.

```bash
conda run -n ai-toolkit python scripts/trim_lora.py adapter.safetensors \
  --output adapter-trimmed.safetensors
```

For reproducible preview approval, take `preview_token` from `--json`, then
provide it with `--expected-preview TOKEN` and the same source/options on save.
The token binds the full source SHA256, algorithm version and settings. A stale
token or a source modified during processing fails before output publication.
The source is rechecked before publication. Existing output files and caches
are not deleted or modified.

## Supported formats and scaling

Supported inputs are **standalone safetensors ordinary LoRAs** with matching
`.lora_down.weight` / `.lora_up.weight` or `.lora_A.weight` / `.lora_B.weight`
keys. This includes kohya-style names and toolkit-exported dotted PEFT-style
names. Linear factors and standard Conv2d down kernels with **1×1 up kernels**
are supported. Float16, bfloat16, float32 and float64 factors keep their original
dtypes, including mixed factor dtypes.

Explicit per-layer `.alpha` tensors remain byte-identical. Toolkit runtime
scale is `alpha / rank`; the trimmed factor product compensates for the new
rank so this scale change does not amplify the effective update. Without alpha,
the supported standalone convention is unit scale. Contradictory alpha metadata
or adjacent training configuration is rejected rather than guessed. `alpha=0`
is rejected because different loader paths interpret it differently.

Rejected inputs include DoRA magnitude vectors or metadata, LoHa/DoHa, LoKr,
rsLoRA, Tucker/mid factors, factor biases/trainable scalars, unknown tensors,
nonfinite values, pickle files and PEFT adapter directories/`adapter_config.json`
sidecars. DoRA normalizes adapted weights using the base model; a bound on its
low-rank direction alone is not a bound on the effective DoRA update. PEFT
directories may carry external alpha/rank patterns that a standalone output
cannot preserve automatically. Export a self-contained ordinary LoRA first.

This tool keeps original keys and user/training metadata. Existing
`sshs_model_hash` and `sshs_legacy_hash` values are recalculated using the
toolkit's additional-networks hash convention; their original values remain in
trimming provenance. Other metadata strings, including training rank/alpha
tags, remain unchanged as **historical training metadata**. Actual output ranks
are per-layer, recorded in the report and the new `aitk_lora_trimming` JSON
metadata. When that name is already present a free `_previous` suffix is used;
earlier provenance is preserved. `metadata_changes` reports every changed or
added metadata field with before/after values.

## Method and memory

For factors `B @ A`, thin QR decompositions reduce the SVD to a rank-sized
core. The tool never constructs the output-by-input dense weight update. It
uses CPU float64 for decomposition, reconstructs balanced factors and measures
their error **after storage-dtype rounding** through another small QR core.
All layer results are held as ordinary CPU factors for final serialization;
working memory scales with factor sizes/ranks, rather than full base weights.
Updates whose QR core, norm or squared energy overflows or underflows float64
are rejected before writing; finite input factors alone do not guarantee a
representable update/error calculation. Exactly zero updates remain supported.

The complete predicted file size includes metadata, tensor descriptors, padding
and payload, following the [Safetensors serialization layout](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs).
The prepared output size is checked against that prediction before publication.
There is no fixed output precision override or silent threshold adjustment.

## UI workflow

Open **Jobs → job → Overview → Checkpoints**, then click a safetensors
checkpoint's **Trim** button to open **Trim LoRA**.
Set the energy/error criterion, preview the per-layer ranks/errors and size,
then choose a new output filename and save. The UI requires a successful preview
with unchanged source/settings. Errors remain visible with details; the original
file and normal download actions remain available. Use the existing controls'
Tab/Enter keyboard behavior. See the UI README for the final interaction and
release verification. See [the recorded checks](lora_trimming_validation.md) for
synthetic reconstruction, loader, transport, browser and build evidence.

## Backend/transport contract

`toolkit.lora_trimming.trim_lora(path, *, retained_energy=None,
relative_error=None, min_rank=1, max_rank=None, output=None,
expected_preview=None, progress=None)` returns a JSON-ready report. Both
thresholds omitted selects energy 0.9999. `trim_lora_factors(down, up, *,
alpha=None, ...)` exposes the same rank policy on CPU tensors, returning
`(new_down, new_up, metrics)` without modifying its arguments.

Report schema version 1 includes:

- `algorithm`, `input_path`, `source_sha256`, `preview_token`, `options`.
- `input_bytes`, `estimated_output_bytes`, `tensor_bytes_before`,
  `tensor_bytes_after`, `written`, `output_path` (null for preview).
- `layers`: `name`, `old_rank`, `new_rank`, `relative_error`, `retained_energy`,
  `ideal_retained_energy`, `dtype_down`, `dtype_up`, `alpha` (null if absent),
  `old_scale`, `new_scale`, `old_bytes`, `new_bytes`.
- `metadata_changes` (key/before/after) and human-readable `warnings`.

CLI also accepts `--input PATH` instead of positional INPUT. `--json` emits a
compact final stdout object; progress uses stderr. The existing `/api/scripts`
transport invokes `ui_scripts/trim_lora.py`, which always emits JSON and requires
`--expected-preview` for writes. No shell or new model service is involved.
