# Adapter reducer extension plan

Status: implemented and verified. Backend/CLI, shared reducer modal and
training-console entry point are integrated. Verification used only synthetic
adapters and base weights; no user checkpoints were reduced.

## Acceptance and public contract

Extend `toolkit.lora_trimming.trim_lora` and `trim_lora_factors` with
`rank_policy=None`, `rank_value=None`. Existing `retained_energy` and
`relative_error` arguments and the default energy 0.9999 retain their behavior.
Explicit policies are mutually exclusive with those legacy threshold arguments:

| Policy | Value and selection |
| --- | --- |
| `energy` | (0,1], minimum rank meeting squared-singular-value retention after output rounding |
| `relative_error` | [0,1], minimum rank meeting relative Frobenius error after rounding |
| `fixed` | positive integer target rank, bounded by each original rank |
| `sv_ratio` | ratio >=1, count singular values strictly above largest/value, at least one |
| `sv_cumulative` | (0,1], smallest rank retaining this fraction of sum of singular values |
| `sv_fro` | (0,1], smallest rank retaining this fraction of Frobenius norm (squared target for energy) |

These spectrum policies follow kohya sd-scripts
[`resize_lora.py` at b1f32e5820a4eb0a7c347853efef48475d599432](https://github.com/kohya-ss/sd-scripts/blob/b1f32e5820a4eb0a7c347853efef48475d599432/networks/resize_lora.py).
We reuse exact CPU QR/core SVD, not upstream's dense reconstruction or randomized
SVD. No absolute small-singular-value cutoff is introduced. Rank caps on spectrum
policies are explicit; only the energy/error policies promise a rounding-aware
bound and reject a cap that makes that bound impossible. No rank increases.

CLI adds `--rank-policy`, `--rank-value`, `--convert-to-lora` and
`--max-working-mb` (default 512 MiB for conversion working-memory estimate).
All settings participate in preview-token identity. New JSON fields identify
input/output formats, metric scope, original stored rank versus conversion
factor rank, and explicit conversion warnings. Existing report fields remain.

## Formats and invariants

- Ordinary LoRA keeps existing factor names, alpha, storage dtypes and semantics.
- DoRA with a recognized, shape-valid magnitude tensor keeps that tensor
  byte-identical. Reduce only direction factors, label metrics
  `pre_normalization_direction`, and state that final normalized weight/image
  error is not bounded without the base model. Metadata-only DoRA without a
  magnitude tensor is rejected. Ambiguous alpha/rsLoRA remain rejected.
- LoHa and LoKr conversion is opt-in and produces ordinary LoRA. Support
  unambiguous 2D factor layouts only. Reject Tucker/CP, unknown scalars/biases,
  DoHa, mixed conventions and ambiguous scaling rather than dropping tensors.
- LoHa `(B1 A1) * (B2 A2)` has exact factorization with columns
  `B1[:,i]*B2[:,j]` and rows `A1[i,:]*A2[j,:]`. LoKr uses
  `kron(B1,B2) @ kron(A1,A2)`; stored full Kronecker components can first use
  their own small SVD. Neither path constructs the full output-by-input delta.
- Explicit alpha/rank input scaling is absorbed/compensated in converted factors;
  output alpha and format metadata must describe the ordinary-LoRA output.
  Originally absent per-layer alpha becomes explicit output alpha=new_rank,
  preserving unit scale independently of historical metadata; provenance keeps
  original_alpha=null. Existing explicit alpha stays byte-identical.
  Preserve original format metadata inside provenance and enumerate every
  changed field. Do not mislabel original LyCORIS metadata as current format.
- Two-dimensional conversion does not infer convolution target/kernel layout;
  flattened LoHa convolution factors may require target-specific reshaping.
- Check expanded factor/core working-memory estimate before allocating them;
  reject over-budget inputs with requested/allowed bytes and an actionable error.

Existing atomic no-clobber publication, source hash rechecks, exact output-size
preview, metadata hashes and historical provenance remain mandatory. No original
files, external libraries, services or training jobs are changed.

## Added user scope: bake DoRA before reduction

`bake_dora_to_lora=False`, `base_model_path=None`, `dora_convention=None`
are additive file API options. Baking requires an explicit matching local single
safetensors base file and explicit `comfy` or `toolkit` convention. The original
floating-only support is extended below with specific quantized codecs.
The suffix alone cannot establish the intended consumer: toolkit exports can
rename native magnitude to `.dora_scale`.

For output-row magnitudes, standard ComfyUI float32 patching uses
`m/(norm(W)+epsilon(float32))`; the report records that explicit epsilon.
toolkit/native uses `m/norm(W+scaled_direction)`, with ratio one on exactly-zero
adapted rows. Compute the resulting full effective delta relative to W, then
reduce that delta as ordinary LoRA. Magnitude may be removed only after this
calculation. Reject unsupported quantized/auxiliary-scale bases, ambiguous key matching,
shape mismatches, unsupported magnitude axes and a missing explicit convention.
Perform one-layer CPU float64 baking/SVD under the same working-memory admission
budget; this opt-in path necessarily materializes a dense layer. No base-model
download or code/model instantiation is involved. Bind the full base content
hash, path and convention into preview identity, report base provenance and
matched weight key, and recheck the base before publication. Final low-precision
loader arithmetic can introduce additional rounding beyond the CPU mathematical
effective-delta metric. Bake equivalence is measured at strength one; native
DoRA strengths can be nonlinear, so other strengths are not promised equivalent.

## Ordered implementation and verification

1. Extend policy normalization and factor rank selection, keeping old calls valid.
2. Add recognized magnitude handling and explicit format/metric reporting.
3. Add opt-in factorized LoHa/LoKr conversion with layout/scaling/memory checks.
4. Expose CLI/bridge arguments and document source-derived policy differences.
5. UI owner reuses the reducer modal and adds the training-console entry point.
6. Test owner verifies known spectra, actual toolkit loader roundtrips, magnitude
   preservation, conversion equivalence, rejects, metadata, preview identity and
   no-clobber publication under the required 16 GiB/no-swap scope.

Coordinator integration complete. Production build passed; only the UI branch
was reloaded, preserving the queue worker. See
[extension validation evidence](lora_trimming_validation.md#reducer-extension-and-dora-baking).

## Quantized-base baking extension

User scope clarification: support the DoRA families this fork can actually train,
including MOSS split-QKV exports and native LoHa with DoRA (DoHa). Audit the
actual training forward, normalization, bias, and export conversions; compare
synthetic baked outputs with real loaders/forwards. Native DoRAModule currently
rejects Conv2d; DoHa supports restricted Conv2d. Do not add new training modes.
Extend existing CPU checkpoint codecs for FP8, int8/ConvRot8, NVFP4, W4A8 and
exported ConvRot W4A4, with strict layout validation. AWQ normalization and
forward weight views must remain separate. Native scaled biases require an
explicit bias delta understood by inference loaders. Keep full-strength static
weight equivalence distinct from dynamic activation quantization/low-precision
arithmetic and numerical retention distinct from visual quality.

Add a lightweight per-weight storage descriptor/CPU decoder shared helper in
`toolkit/util/checkpoint_weights.py`. Support explicitly tensor-marked Comfy
`int8_tensorwise` with FP32 per-row scales, including convrot8's self-inverse
regular-Hadamard input-axis rotation. Support marked `float8_e4m3fn` with one
FP32 scale. FP8/NVFP4's optional `full_precision_matrix_mult` accepts absent,
true or false, since the stored-weight decoding matches the native import path;
it does not establish activation-quantization parity. Reuse the toolkit quantizers'
`dequantize` implementation rather than a second quantization codec; this matches
the operation used by `OstrisLinear.weight`, without assuming that future
activation-folded codecs have the same meaning.
Validate marker JSON/dtype, code dtype, exact scale layout, rotation power-of-four
and divisibility, positive finite scales, and final logical shape/values. Never
crop padding or infer a rotation, scale, packed layout or unsupported format.

Index headers/descriptors only; admit dense baking/SVD **plus decoder/rotation
working storage** before reading/dequantizing the selected weight. Read at most
one base layer at a time. Bind decoder semantics through the reducer algorithm
version and preserve the full base-file hash (including markers/scales/metadata).
Report quantization format/rotation/storage dtype, normalization epsilon and
selected reconstruction dtype. Do not load full models, change external codecs,
download weights, or invoke GPU kernels. FIBO2 canonical qkv/w13/MLP checkpoint
names already match native adapter targets; structural split/fuse, kernel-layout
or architecture remapping remains out of scope and fails exact shape matching.

`base_dequant_dtype='float32'` optionally selects `bfloat16` or `float16` after
quantized decoding to reproduce the training weight-view cast. Floating stored
weights are unchanged. The setting participates in token identity and reports;
non-default values require baking. Normalization/SVD still use FP64 and Comfy
epsilon stays FP32. Static decoded-weight equivalence does not claim activation
quantization or bit-exact low-precision/GPU forward equivalence. Root owns the
existing-modal dtype selector/context; test owner verifies synthetic codec
exports against actual training views and the existing bake/loader path.

## Native-layout follow-up: MOSS, bias and DoHa

The initial linear pair discovery misses MOSS's actual export: separate q/k/v
factors (identical down projection and alpha, row-split up) with one fused QKV
magnitude. Recognize this complete provable layout as one logical layer,
validate all equal/shared invariants, reduce/bake jointly, and write compatible
public split factors. Preserve fused magnitude only for unbaked native DoRA.
Do not patch external models or silently discard leftover magnitude tensors.

Match the fork's actual scalar native DoRA forward: normalization receives
unscaled B@A, while the forward adapter update applies alpha/rank. Native biased
linears additionally rescale the frozen bias; retain their exact bias delta as
the supported `.diff_b` patch (ComfyUI and toolkit inference consumers), report
that requirement, and preserve validated bias patches on ordinary-LoRA retrims.
The ordinary training LoRA loader does not automatically apply `.diff_b`.
Native DoRA's constructor explicitly rejects Conv2d, so adding convolution
training is outside this request.

DoHa is a native trainable magnitude-normalized LoHa, including flattened
ungrouped/zero-padded Conv2d. Enable explicit base-aware baking of its 2D
Hadamard-factor expansion, preserving promoted source output precision and
preflighting expansion plus dense baking together. Public `.dora_scale` is
already Comfy-normalized by the export hook; native magnitude uses the scaled
Hadamard adapted-row norm with the module's FP32/FP64 tiny floor. Validate that
known convention rather than normalizing exported magnitudes twice. For a
matching 4D base, reshape baked ordinary factors using the explicit kernel
dimensions; no guessed convolution layout, CP/Tucker or DoKr support.

## CUDA computation and interpreter consistency

The earlier CPU-only scope is superseded by additive `device='auto'` on both
public reduction APIs and `--device auto|cpu|cuda[:index]`. Resolve auto to CUDA
when available in the actual process, otherwise CPU. Snapshot a canonical
explicit CUDA index and bind requested/resolved device plus algorithm version
into preview identity. Report Python executable, PyTorch/CUDA versions,
availability and GPU name. No silent CPU retry follows CUDA errors or OOM.

Move admitted per-layer factors/weights to float64 on the selected backend for
all QR/SVD, measured error, factorized conversion and baking math. Keep archive
I/O and existing quantization codecs on CPU; return outputs to CPU and release
working CUDA factors before processing the next layer. Preserve source tensor
values, dtypes/scaling/metadata and atomic no-clobber publication.

Retain explicit conversion/bake working-memory budgets and add CUDA admission
against current driver-free memory plus reusable allocator cache. Ordinary QR
admission uses a factor/core estimate, without dense base deltas. These estimates
cannot guarantee allocator workspace/fragmentation or concurrent allocations;
runtime failures remain explicit. CPU/CUDA tests compare effective updates and
rank policies within measured floating precision, not arbitrary SVD factor signs.
Coordinator owns the shared Python resolver/run.sh context and existing preview
runtime display; test owner runs scoped CPU regressions and authorized small
synthetic CUDA cases only, with no training/full-model/user checkpoint runs.
