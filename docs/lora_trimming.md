# Adaptive LoRA trimming

Use `scripts/trim_lora.py` to reduce adapter ranks independently on CPU or CUDA.
Ordinary LoRA and direction-only DoRA reduction need no base weights. Optional
DoRA baking reads matching local base weights as tensors. No operation downloads
weights, executes adapter code, or changes the original checkpoint. Numerical similarity does not prove
identical generated images: compare your normal prompts after trimming.

## Device and Python environment

`--device auto` is the default: it selects the current CUDA device when the
running Python's PyTorch reports CUDA available, and otherwise selects CPU.
Use `--device cpu` to choose CPU explicitly, or `--device cuda` /
`--device cuda:0` to require a visible CUDA device. An unavailable CUDA device or
invalid index fails with a useful error. A CUDA allocation, OOM, QR/SVD or other
runtime error is propagated; the reducer **does not retry on CPU**.

QR/core SVD, storage-rounded error measurements, LoHa/LoKr factorized expansion,
and DoRA/DoHa baking math run in float64 on the selected device. Safetensors
reading and quantized checkpoint decoding remain on CPU. Only admitted layer
weights/factors are transferred; no full model is instantiated or placed on the
GPU. Completed factor and bias outputs return to CPU for safe serialization;
working CUDA factors are released before admitting the next layer. Consumer
GPUs can have slow float64 arithmetic, so CUDA does not guarantee a speedup for
every layer size.

The JSON/text preview reports `device_requested`, canonical `device_resolved`,
and `runtime` with the Python executable, PyTorch version, CUDA availability/
version, and GPU name when selected. Requested **and resolved** device are bound
into the preview token. A preview made on CPU cannot be silently saved with
CUDA, and an `auto` preview that resolves differently must be regenerated.
CPU/CUDA numerical rounding can lead to different factors or threshold-boundary
rank decisions; each report measures its own rounded output on its selected
backend.

```bash
conda run -n ai-toolkit python scripts/trim_lora.py adapter.safetensors --device cuda:0
conda run -n ai-toolkit python scripts/trim_lora.py adapter.safetensors --device cpu
```

The UI script bridge uses the project's configured Python interpreter (the same
environment chosen by `run.sh`), with an explicit `AI_TOOLKIT_PYTHON` override
when configured. The runtime report exposes the actual interpreter rather than
guessing CUDA availability from a different shell environment.

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

`--min-rank` defaults to 1. Ordinary factor ranks never increase. Invalid or impossible rank
bounds fail explicitly; a maximum rank never silently weakens the requested
error threshold. Zero updates can use rank 1 (or your minimum). A layer stays
bit-exact at its original rank when no smaller storage-rounded factorization
meets the threshold.

## Rank policies

The default remains rounding-aware numerical energy retention. The optional
`--rank-policy POLICY --rank-value VALUE` form selects any policy below; it
cannot be combined with `--retained-energy` or `--relative-error`.

| Policy | Value | Selection |
| --- | --- | --- |
| `energy` | (0, 1], default 0.9999 | Smallest rank meeting measured squared-energy retention after dtype rounding |
| `relative_error` | [0, 1] | Smallest rank meeting measured relative Frobenius error after rounding |
| `fixed` | positive integer | Target factor rank, capped at each original factor rank |
| `sv_ratio` | at least 1 | Keep singular values strictly greater than largest singular value divided by this ratio; at least one |
| `sv_cumulative` | (0, 1] | Smallest rank retaining this fraction of the **sum of singular values** |
| `sv_fro` | (0, 1] | Smallest rank retaining this fraction of **Frobenius norm**, so the energy target is value squared |

The last four policies follow the rank-selection definitions in kohya
sd-scripts [`resize_lora.py`](https://github.com/kohya-ss/sd-scripts/blob/b1f32e5820a4eb0a7c347853efef48475d599432/networks/resize_lora.py).
They select a rank and report the resulting rounded error; they do **not** enforce
an energy/error bound. `--min-rank`/`--max-rank` explicitly constrain them. The
bounded energy/error modes instead reject a maximum that cannot meet the bound.
We use exact QR/core SVD rather than upstream's dense/randomized SVD, keep
explicit alpha unchanged with factor compensation, and do not use an absolute
small-singular-value cutoff. Exact zero updates use the minimum permitted rank.

```bash
conda run -n ai-toolkit python scripts/trim_lora.py adapter.safetensors \
  --rank-policy fixed --rank-value 32
conda run -n ai-toolkit python scripts/trim_lora.py adapter.safetensors \
  --rank-policy sv_cumulative --rank-value 0.99 --max-rank 64
```

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
The token binds the full source SHA256, algorithm version and settings, plus the
base path/content SHA256 and convention when baking DoRA. A stale
token or a source modified during processing fails before output publication.
The source is rechecked before publication. Existing output files and caches
are not deleted or modified.

## Supported formats and scaling

Supported ordinary inputs are **standalone safetensors LoRAs** with matching
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

Linear DoRA factor pairs additionally support `.dora_scale`, `.magnitude` and
PEFT `.lora_magnitude_vector[.weight|.default.weight]` output-channel magnitude
tensors. Direction-only reduction preserves the original magnitude key, shape,
dtype and values exactly. Its `metric_scope=pre_normalization_direction` covers
the low-rank direction update, **not** the final base-dependent normalized DoRA
weights. No base checkpoint is needed for this mode.

DoHa (LoHa with magnitude normalization) is supported through the explicit
base-aware baking path below. Native DoRA training in this fork supports linear
layers and explicitly rejects Conv2d; DoHa additionally trains supported
ungrouped, zero-padded Conv2d layers with flattened Hadamard factors.

Rejected inputs include rsLoRA, Tucker/CP/mid factors, unknown factor
biases/trainable scalars, unknown tensors, nonfinite values, pickle files and
PEFT adapter directories/`adapter_config.json` sidecars. Metadata-only DoRA
without a recognized magnitude is rejected. PEFT
directories may carry external alpha/rank patterns that a standalone output
cannot preserve automatically. Export a self-contained ordinary LoRA first.

### Convert LoHa or LoKr to ordinary LoRA

`--convert-to-lora` explicitly enables conversion of standard **2D** LoHa's four
`hada_w1_a/b`, `hada_w2_a/b` tensors, or LoKr's full/factored `lokr_w1` and
`lokr_w2` components. Decomposed component ranks must agree so alpha/rank scaling
is unambiguous. All-full LoKr has unit input scale. CP/Tucker/DoHa and mixed
adapter-family files are rejected. Without this option these formats fail rather
than silently changing format.

The conversion factorizes the whole Hadamard/Kronecker update exactly before
reduction; it does not trim components independently or construct the full dense
base-weight delta. Output is ordinary `.lora_down.weight`/`.lora_up.weight`
LoRA, with explicit alpha preserved and compensated. Legacy `lycoris_` module
prefixes become ordinary `lora_transformer_` prefixes. Converted factors use the
promoted floating dtype of the original components; reported error includes
output rounding. If the source omits per-layer alpha, converted output explicitly
stores alpha equal to its new rank, preserving unit scale independently of
historical global alpha/config metadata. The report and provenance retain
`original_alpha=null`. A native LyCORIS training resume requires the original file.

Conversion operates on 2D update matrices and does not infer a target's
convolution/kernel layout. Toolkit LoHa convolution adapters can also contain
flattened 2D factors; ordinary converted factors may need target-specific
convolution reshaping. A 2D layout alone does not establish universal convolution
or architecture compatibility.

For conversion, `old_rank` means the expanded ordinary factor rank;
`original_stored_rank` records the native component rank (null for all-full
LoKr), and `expanded_rank` makes the distinction explicit. Output may be larger
than input even when its factor rank is reduced; inspect the size preview.

```bash
conda run -n ai-toolkit python scripts/trim_lora.py loha.safetensors \
  --convert-to-lora --rank-policy fixed --rank-value 16
```

### Bake DoRA into ordinary LoRA before reducing

Use `--bake-dora-to-lora`, an explicit `--base-model-path` to a matching single
local floating or supported Comfy-quantized `.safetensors` checkpoint, and **explicitly choose**
`--dora-convention comfy` or `toolkit` according to how the adapter is normally
loaded. There is no inferred default: a `.dora_scale` suffix can also come from
a toolkit export and does not establish the intended consumer.

When opening the reducer from a training job's checkpoint sidebar, the base
field is prefilled from that job's `model.name_or_path` if it is an absolute
`.safetensors` path, matching the training form's ComfyUI checkpoint detection.
Repository IDs and directory paths leave it blank. The field remains editable;
edits survive toggling baking off and on until the dialog closes. Reopening
starts again from the job's configured path. Baking and convention selection
remain explicit, and the backend still validates the chosen base weights.

For direction factors `P = B @ A`, scale `s = alpha/rank` (or one when alpha is
absent), and the decoded normalization weight N, baking follows the selected
consumer's **actual implementation** at strength one:

- `comfy`: `row_scale = magnitude / (norm(N rows) + epsilon)`, matching standard
  ComfyUI float32 model patching. Epsilon is explicitly float32 epsilon
  **1.1920928955078125e-7**, reported as `normalization_epsilon`. Magnitude must
  have unambiguous `(output_rows, 1)` shape; input-axis/1D layouts are rejected.
- `toolkit`: ordinary native DoRA currently uses
  `row_scale = magnitude / norm((N + P) rows)`, with scale one on exactly-zero
  adapted rows. Its norm path receives **unscaled** `B @ A`, while its adapter
  forward applies `s * P`; this distinction matters when alpha differs from rank.

Without activation-side input transforms, the effective delta is
`row_scale * (N + s*P) - N`. With NVFP4 AWQ input scaling, native baking uses
the raw-input forward weight `F = N * pre_quant_scale` for the effective delta
`row_scale * (F + s*P) - F`, while retaining N for the normalization calculation.
The Comfy convention instead bakes in its stored-weight patch basis and leaves
the checkpoint's input transform in place. The report identifies this basis;
use the selected convention's consumer rather than interchanging AWQ-baked
adapters across different application paths.

Ordinary native scalar DoRA also multiplies the frozen layer's bias by
`row_scale`. Baking retains the exact `(row_scale-1)*base_bias` as a separate
`.diff_b` tensor. ComfyUI and the toolkit inference engine support that explicit
bias delta; a training LoRA loader or external consumer that ignores `.diff_b`
does not reproduce such a biased bake. The weight retention metric excludes the
separately preserved bias delta. The reducer recognizes and preserves validated
finite per-output-channel `.diff_b` tensors on ordinary-LoRA retrims; it does not
guess unknown adapter biases. Comfy and signed-slider conventions leave the
base bias unchanged.

Native DoRA uses linear layers with unique matching module names and exact
logical base shapes; recognized DoHa additionally supports the base-aware Conv2d
layout below. Missing weights, name collisions, unsupported quantization
or auxiliary scales, directories/sharded checkpoints and unsupported magnitude
axes fail explicitly.
There are no downloads or base-model instantiations. The base is never modified.

Supported quantized base layers use a per-layer `.comfy_quant` uint8 JSON marker:

| Marker/storage | Required scale/layout | Reconstruction |
| --- | --- | --- |
| `int8_tensorwise`, no rotation | int8 `[out,in]`, FP32 `.weight_scale` `[out]` or `[out,1]` | Per-output-row int8 scale, original input basis |
| `int8_tensorwise`, `convrot=true` | Same codes/scales; `convrot_groupsize` is 1 or a power of four dividing input width (default 256 when omitted) | The toolkit ConvRot8 quantizer undoes the stored **regular-Hadamard** input-axis rotation |
| `float8_e4m3fn` | e4m3fn or raw-byte `[out,in]` codes, one FP32 scalar `.weight_scale`; optional boolean `full_precision_matrix_mult` | The toolkit float8 quantizer reconstructs the scaled FP32 weight |
| Legacy `scaled_fp8` | Global marker plus e4m3fn/raw-byte codes and scalar FP32 `.scale_weight` | Same FP8 reconstruction |
| `nvfp4` | Packed FP4 pairs, swizzled block-16 e4m3 scales, FP32 `.weight_scale_2`, optional positive FP32 `.pre_quant_scale`; optional boolean `full_precision_matrix_mult` | Existing nibble conversion/scale unswizzling and NVFP4 decoder; AWQ normalization/forward views remain distinct |
| `asym_w4a8_int8` | Packed nibble codes, relative group scales, per-channel FP32 scales, optional 16-entry codebook/correction | Existing W4A8 decoder, including integer rounding and regular-Hadamard rotation |
| `convrot_w4a4` | Signed two's-complement nibble pairs, FP32 per-row scale, explicit rotation | Signed code unpacking followed by row scaling and regular-Hadamard rotation |

Per-row/channel/tensor scales must be finite and strictly positive; NVFP4 block
scales and W4A8 relative scales may be zero. Bare integer/float8 weights,
unmarked scales, malformed/oversized markers, inconsistent rotations/shapes and
unknown formats are rejected. The same explicitly supported configurations may
appear in `_quantization_metadata` header layers instead of tensor markers.
For FP8 and NVFP4, `full_precision_matrix_mult` may be absent, true or false;
the stored-weight decoder is the same, matching the native import path. A
non-boolean value is rejected. This does not imply activation-quantization
forward parity.
Other packed low-bit codecs are not inferred. There is no inferred packing, rotation,
padding crop or architecture conversion. FIBO2 canonical fused qkv/w13 and other
native checkpoint names match the native adapter targets directly; another key
layout must already match the adapter's exact logical shape/name.

For quantized bases, `--base-dequant-dtype float32|bfloat16|float16` chooses the
decoded weight view; default is **float32**. Decoding uses the same quantizer
`dequantize` operation as `OstrisLinear.weight`, restores the original logical
basis, then casts to the selected dtype. Choose the dtype used by training to
match that training weight view. For the usual FIBO2 bf16 training setup:

```bash
conda run -n ai-toolkit python scripts/trim_lora.py dora.safetensors \
  --bake-dora-to-lora --base-model-path fibo2_dev_int8_convrot.safetensors \
  --base-dequant-dtype bfloat16 --dora-convention toolkit \
  --rank-policy fixed --rank-value 32
```

The dtype option affects **quantized weights only**; a floating base keeps its
stored dtype. The training sidebar initializes the selector from an explicitly
available supported `train.dtype`, otherwise float32; neither checkpoint names
nor adapter metadata are used to guess it. Non-default values without baking
are rejected. The dtype is bound into the preview token. Per-layer `base_weight`
reports storage/quantization format, scale/marker keys, rotation, reconstruction
dtype, decoder version and the FP64 normalization calculation dtype. Comfy's
epsilon remains explicitly float32 epsilon. This describes static weight
reconstruction, **not** activation quantization, runtime STE behavior, or
bit-exact GPU/low-precision forward outputs.

### MOSS split QKV exports

Toolkit MOSS exports clone the shared down projection and alpha under separate
`proj.q`, `proj.k`, `proj.v` factor names, split the up projection into equal row
thirds, and keep a fused `proj.qkv.dora_scale` magnitude. The reducer recognizes
that complete layout, validates equal shared down tensors and alpha values, and
reduces/bakes it as one logical QKV layer against the matching fused QKV base.
It writes the factors back using their public split names and one shared rank;
an unbaked native DoRA retains its fused magnitude, while a baked ordinary LoRA
has no magnitude. Reports list `split_qkv_outputs`. Incomplete triples,
conflicting split/fused magnitudes or nonshared down projections fail explicitly
rather than discarding a magnitude or guessing a different adapter structure.

### DoHa baking

Use the same `--bake-dora-to-lora` option for DoHa; no additional flag is needed.

`network.type=loha` with `network.loha_dora=true` stores four 2D Hadamard factors
plus a magnitude. Baking factorizes the complete Hadamard update before applying
normalization; it does not independently trim the components or drop magnitude.
Output precision follows the promoted source-factor dtype, rather than the FP64
working expansion.

Toolkit-exported public `.dora_scale` is **already converted** by
`LoHaModule.comfy_magnitude` to Comfy's base-norm convention, so select `comfy`.
A native `.magnitude`/vector uses `toolkit`, whose DoHa denominator is the norm of
`N + scaled_Hadamard_update`, clamped to the smallest positive normal FP32 value
(FP64 when source factors are FP64). Unlike ordinary native DoRA, DoHa includes
alpha/rank in this norm and excludes base-bias rescaling. Choosing the wrong
convention for these known DoHa suffixes is rejected.

For a matching floating 4D Conv2d base, the source Hadamard factors represent a
flattened output-by-kernel matrix. Baking reconstructs the known kernel layout
from that base and writes ordinary Conv2d factors `(rank,in,kh,kw)` and
`(out,rank,1,1)`. This is explicit base-aware reshaping, not a guessed kernel.
Native DoRA Conv2d, DoKr and CP/Tucker DoHa remain unsupported because they are
not the fork's supported native training layouts.

Baking removes magnitude only after computing the actual effective delta. The
result is ordinary LoRA with `metric_scope=effective_update`; preview/provenance
record the base path, full SHA256, matched weight key and selected convention.
The base hash includes all quantization codes, scales, markers and metadata;
decoder/algorithm semantics and dtype also participate in preview identity.
The base hash is rechecked before publication. The reported metric measures the
float64 mathematical full-strength baked delta, including factor storage rounding;
it does not promise bit-exact low-precision/GPU arithmetic. Native DoRA strengths
can be nonlinear, so other strengths need not match the baked LoRA.
Originally absent alpha is explicitly written as the new rank here too, keeping
the baked output's unit scale self-contained; original explicit alpha remains
unchanged with factor compensation.

```bash
conda run -n ai-toolkit python scripts/trim_lora.py dora.safetensors \
  --bake-dora-to-lora --base-model-path base.safetensors \
  --dora-convention comfy --rank-policy sv_fro --rank-value 0.999
```

Ordinary and unbaked DoRA keep original keys and user/training metadata. Format
conversion/baking rewrites current loading instructions such as network module,
type and algorithm args to ordinary LoRA, stores the exact original metadata
snapshot in provenance, and lists every change. Existing
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
core. Ordinary reduction and factorized LoHa/LoKr conversion never construct the
output-by-input dense weight update. **DoRA baking necessarily constructs one
dense effective-delta layer and its SVD on the selected device.** Every path
uses float64 for decomposition, reconstructs balanced factors and measures
their error **after storage-dtype rounding** through another small QR core.
All layer results are held as ordinary CPU factors for final serialization;
working memory scales with factor sizes/ranks, rather than full base weights.
LoHa/LoKr expansion and DoRA baking have a conservative per-layer preallocation
admission estimate controlled by `--max-working-mb` (default **512 MiB**).
Over-budget layers are rejected before expansion/base-weight loading; raise the
explicit budget only when sufficient host memory is available. This estimate is
not an OS-enforced peak limit: library workspaces and retained CPU output factors
also consume memory. CUDA additionally checks the estimated layer working set
against current driver-free VRAM plus unused PyTorch allocator cache before
transfer/allocation. Ordinary factor QR/SVD uses a rank-sized working-set
estimate for this CUDA check; it never constructs a dense output-by-input
update. Available memory can change after admission and allocator fragmentation
can still cause an error; errors are not caught and redirected to CPU.
Quantized baking indexes headers and small marker metadata without loading the
model; the estimate includes one layer's decoder buffers and the regular
Hadamard rotation matrix before weight payloads are read or dequantized. Logical
padded rows/columns are retained and must match the adapter exactly.
Updates whose QR core, norm or squared energy overflows or underflows float64
are rejected before writing; finite input factors alone do not guarantee a
representable update/error calculation. Exactly zero updates remain supported.

The complete predicted file size includes metadata, tensor descriptors, padding
and payload, following the [Safetensors serialization layout](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs).
The prepared output size is checked against that prediction before publication.
There is no fixed output precision override or silent threshold adjustment.

## UI workflow

Open a job's **Checkpoints** list or the training console's model/checkpoint
sidebar and use the checkpoint's **Reduce** action to open the shared reducer.
Set the retention/rank policy, optional explicit conversion/baking settings,
preview the per-layer formats, metric scope, ranks/errors and size,
then choose a new output filename and save. The UI requires a successful preview
with unchanged source/settings. Errors remain visible with details; the original
file and normal download actions remain available. Use the existing controls'
Tab/Enter keyboard behavior. See the UI README for the final interaction and
release verification. See [the recorded checks](lora_trimming_validation.md) for
synthetic reconstruction, loader, transport, browser and build evidence.

## Backend/transport contract

`toolkit.lora_trimming.trim_lora(path, *, retained_energy=None,
relative_error=None, min_rank=1, max_rank=None, output=None,
expected_preview=None, progress=None, rank_policy=None, rank_value=None,
convert_to_lora=False, max_working_mb=512, bake_dora_to_lora=False,
base_model_path=None, dora_convention=None, base_dequant_dtype='float32',
device='auto')` returns a JSON-ready report. Both
thresholds omitted selects energy 0.9999. `trim_lora_factors(down, up, *,
alpha=None, device='auto', ...)` accepts CPU or CUDA tensors with the same rank policy, returning
`(new_down, new_up, metrics)` without modifying its arguments.

Report schema version 1 includes:

- `algorithm`, `input_path`, `source_sha256`, `preview_token`, `options`.
- `device_requested`, `device_resolved`, and `runtime` (`python_executable`,
  `torch_version`, `cuda_available`, `cuda_version`, `gpu_name`). Each layer also
  reports its compute device and `compute_working_bytes` estimate.
- `input_bytes`, `estimated_output_bytes`, `tensor_bytes_before`,
  `tensor_bytes_after`, `written`, `output_path` (null for preview).
- `layers`: `name`, `old_rank`, `new_rank`, `relative_error`, `retained_energy`,
  `ideal_retained_energy`, `dtype_down`, `dtype_up`, `alpha` (null if absent),
  `old_scale`, `new_scale`, `old_bytes`, `new_bytes`.
- `metadata_changes` (key/before/after) and human-readable `warnings`.
- `input_formats`, `output_formats`, `base_model` (path/SHA256/weight-decoder version or null).
- Each layer also includes `output_name`, `input_format` (including `doha` for
  magnitude-normalized Hadamard adapters), `output_format`,
  `metric_scope`, `original_stored_rank`, `expanded_rank`,
  `conversion_working_bytes`, `baked`, `base_weight_key`, `dora_convention` and
  `normalization_epsilon` (null when not applicable). `original_alpha` records
  input alpha or null independently of the output `alpha`.
- `split_qkv_outputs` lists public split output names for recognized MOSS groups;
  `baked_bias` identifies a separately retained `.diff_b` delta.
- `base_weight` is null outside baking. Otherwise it includes `decoder`, `key`,
  `shape`, `storage_dtype`, `quant_format`, `scale_key`, `marker_key`, `rotation`,
  `reconstruction_dtype`, `requested_dequant_dtype` (null for floating bases)
  and `normalization_dtype`. Expanded codecs also record `group_size` and
  `auxiliary_keys`; `forward_input_scale`, `output_basis` and
  `normalization_update` identify AWQ view/application semantics and the native
  unscaled-BA versus scaled-update norm calculation.

`trim_lora_factors` additionally accepts `rank_policy`, `rank_value` and optional
`output_dtype`; the latter supports measuring promoted-format conversions after
their selected output-dtype rounding. The default keeps each original dtype.

CLI also accepts `--input PATH` instead of positional INPUT. `--json` emits a
compact final stdout object; progress uses stderr. The existing `/api/scripts`
transport invokes `ui_scripts/trim_lora.py`, which always emits JSON and requires
`--expected-preview` for writes. No shell or new model service is involved.
