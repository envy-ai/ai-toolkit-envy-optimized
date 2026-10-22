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
floating safetensors base file and explicit `comfy` or `toolkit` convention.
The suffix alone cannot establish the intended consumer: toolkit exports can
rename native magnitude to `.dora_scale`.

For output-row magnitudes, standard ComfyUI float32 patching uses
`m/(norm(W)+epsilon(float32))`; the report records that explicit epsilon.
toolkit/native uses `m/norm(W+scaled_direction)`, with ratio one on exactly-zero
adapted rows. Compute the resulting full effective delta relative to W, then
reduce that delta as ordinary LoRA. Magnitude may be removed only after this
calculation. Reject quantized/auxiliary-scale bases, ambiguous key matching,
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
