# Adaptive LoRA trimming plan

Status: backend implemented; integration and verification are in progress. The coordinator owns
integration, UI, release verification and commits; the test agent owns tests.

## Scope and acceptance

Add a CPU-only `toolkit.lora_trimming` module, `scripts/trim_lora.py` CLI and
thin `ui_scripts/trim_lora.py` bridge. The default operation is a dry run.
An explicit output writes a new safetensors file, atomically without replacing
an existing path. Source files and live training jobs are never modified.

For each ordinary LoRA layer choose the smallest positive rank meeting retained
squared singular-value energy 0.9999, or an explicitly selected relative
Frobenius-error limit. Optional rank bounds must be compatible with the original
rank and requested error; reject impossible limits instead of weakening the
threshold or increasing rank. Report the error after rounding to the original
factor dtypes, including all unchanged layers and exact predicted archive size.

## Mathematics and scaling

Given down A and up B, thin QR decompositions B=Qb Rb and A^T=Qa Ra reduce
SVD to the rank-sized core Rb Ra^T. Reconstruct balanced factors from its
singular triplets without forming an output-by-input delta. Measure rounded
factor error through another small-core QR of the difference factors.
Use CPU float64 for the decomposition and error calculation. Unchanged-rank
layers retain their original tensor values exactly.

Preserve explicit per-layer alpha tensors exactly. Because toolkit runtime
scale is alpha/rank, compensate reduced factor products by new_rank/old_rank;
the effective update then retains its original scale. Missing alpha in supported
standalone files means unit scale, matching the toolkit standalone PEFT loader
and ordinary Comfy LoRA convention. Do not infer external PEFT configuration.

## Supported and rejected inputs

Support standalone safetensors `.lora_down.weight`/`.lora_up.weight` and
`.lora_A.weight`/`.lora_B.weight` key pairs with their original names, including
kohya and toolkit-exported dotted PEFT-style keys. Support linear tensors and
standard Conv2d down kernels with 1x1 up kernels. Preserve original dtypes.

Reject Python pickle formats, DoRA magnitude tensors/metadata, LoHa/LoKr,
rsLoRA, Tucker/mid factors, bias/scalar or unknown tensor keys, malformed pairs,
nonfinite factors, zero/ambiguous alpha and adapter directories/PEFT sidecars.
These need different base-dependent or external scaling semantics. In particular
DoRA truncation cannot promise the same effective weight error without its base.
No model loading, network access, user-provided code or GPU work is required.

Preserve user/training metadata. Recalculate existing derived archive hashes,
retain their original values in trimming provenance and report every changed
field. Existing training rank metadata remains historical; the report and
provenance identify actual per-layer output ranks.

## Public contract and stale-preview protection

`trim_lora(path, *, retained_energy=None, relative_error=None, min_rank=1,
max_rank=None, output=None, expected_preview=None, progress=None)` returns a
JSON-ready report. Both thresholds omitted selects 0.9999 energy; specifying
both fails. `trim_lora_factors` exposes the same rank policy for tensor pairs.

The report includes schema/algorithm versions, source path/SHA256, options,
preview token, input/output byte counts, per-layer ranks/error/dtypes/scaling,
and whether an output was written. The token binds the complete source hash
and options. A write with `expected_preview` must match; the UI bridge requires
that token for every write. Recheck the source before publication and fail if
it changed. Use a temporary file in the destination directory, fsync, then an
atomic no-clobber hard link; do not use a replacing rename.

CLI: `trim_lora.py INPUT [--retained-energy X | --relative-error E]
[--min-rank N] [--max-rank N] [--output NEW.safetensors]
[--expected-preview TOKEN] [--json]`. JSON mode emits a compact final stdout
line; progress goes to stderr. Errors propagate with a traceback/nonzero exit.

## Coordinator integration (pending)

- UI: existing FilesWidget checkpoint action and shared Modal; preview first,
  then an explicit new-output save using the preview token.
- Verification: factor mathematics/rounding/scaling, real toolkit load/export,
  rejection cases, exact sizes, stale source and no-clobber races; production UI
  build and available browser coverage.
- Activation and user documentation: coordinator records final evidence and any
  service activation separately. No restart is required by the backend module.
