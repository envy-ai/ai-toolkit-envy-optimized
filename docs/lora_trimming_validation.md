# Adaptive LoRA trimming validation

## CUDA selection and launcher interpreter

Reducer algorithm `qr_core_svd_v5` adds `device=auto|cpu|cuda[:index]`.
The small synthetic CUDA check ran in the `ai-toolkit` Conda environment on
the RTX 4090 after confirming 8,758 MiB free VRAM. It passed CPU/CUDA rank and
reconstruction comparisons, ordinary factor serialization back to CPU, and
the complete DoRA bake path; CUDA peak allocated memory was below 64 MiB.
An initial fixture-only rank/index error was corrected, then the affected case
passed. No trained checkpoint, full model or training job was used.

The full CPU regression command (`unittest testing.test_lora_trimming
testing.test_checkpoint_weights`, CUDA hidden) completed **41 tests, one skipped,
no failures**;
the skipped GPU case passed separately above. Peak host memory was 837.7 MiB,
swap zero. A real CLI subprocess with CUDA hidden verified `auto -> cpu` and
reported the exact `sys.executable`; explicit unavailable CUDA rejects with an
interpreter-specific error. The resolver's three Node tests passed explicit
launcher override, invalid override rejection, active Conda priority over a
repository virtual environment, and repository selection without a usable
Conda candidate (80.6 MiB peak, zero swap).

The final scoped browser preview used the real scripts API and resolver against
an isolated synthetic-job database. It reported `auto -> cuda:0`, GPU
`NVIDIA GeForce RTX 4090`, and Python
`/home/bart/.conda/envs/ai-toolkit/bin/python`; the visible device summary and
Python hover title matched. No output copy was created. Browser verification
peaked at 361.7 MiB host memory with zero swap; Chromium itself used software
rendering. The isolated server was stopped afterward.

Standalone UI typecheck, shell syntax, diff checks and `npm run build` passed.
The build took 6 min 29 s, peaked at 14.2 GiB and used no swap. Only the UI branch
was reloaded; queue worker PID 656564 stayed running. A scoped live HTTP/asset
check passed and confirmed the device summary replaced the old CPU-only status.

All executed checks used transient systemd units/scopes capped at 16 GiB host
memory with no swap. CUDA/library failures propagate without retrying on CPU;
device selection is bound to the preview token. Source/base files remain
unchanged, and earlier format, alpha/scaling and no-clobber checks remain in
the regression suite.

## Quantized bases, MOSS and trainable DoRA families

The quantized-base extension uses reducer algorithm `qr_core_svd_v4` and weight
decoder `comfy_logical_weight_v2`. All fixtures are synthetic, CPU-only and
temporary. Verification ran inside transient systemd units/scopes with
`MemoryMax=16G`, `MemorySwapMax=0`; project Python used Conda `ai-toolkit`.

- Full `testing/test_lora_trimming.py`: **35 tests and 57 subtests passed**.
  After final test-only additions, the two affected cases passed again (two
  tests, five subtests); unaffected full-suite passes were retained. Peak host
  memory was 577.7 MiB for the full run and 530.2 MiB for the targeted run;
  swap was zero. Coverage includes actual native DoRAModule forward with
  alpha different from rank, scaled base bias, `.diff_b` through the real
  InferenceLoRA hook, MOSS fused-magnitude/split-QKV loader compatibility,
  and actual DoHa linear/Conv2d behavior and source factor dtype preservation.
- Actual quantizers/exporters verify ConvRot8, plain int8, FP8, legacy scaled
  FP8, header-described FP8, NVFP4, W4A8 and exported ConvRot W4A4 logical
  weights. Reconstruction uses the stored training view, including BF16;
  tests cover scaling/alpha, metadata, no-clobber copies, base/source hash
  preservation, precision-bound previews and pre-decode memory rejection.
- `testing/test_checkpoint_weights.py` adds W4A8 FP8/FP32 relative scales,
  optional codebook/correction, AWQ native and Comfy output bases with unequal
  alpha/rank, invalid packed shapes/configuration and nonfinite/nonpositive
  scales. Initial execution exposed two test expectation/fixture errors.
  The two affected methods passed after correction (two subtests), retaining
  the unaffected W4A8 variant and AWQ invalid-scale passes. Peak was 373 MiB,
  swap zero. Diagnostic logs are under `tmp/lora-quant-bake/`.
- `ui/tests/lora-trim-api.test.cjs`: the real scripts route and Python CLI
  passed, including the ConvRot8 BF16 reconstruction flag (446.7 MiB peak,
  zero swap). Browser tests exercise the real preview/create flow, codec and
  dtype reporting, preview invalidation after precision edits, immutable source
  files, and explicit bake/convention choices. A focused final 390 px check
  passed prefill, edit/toggle/reopen behavior and updated format/DoHa guidance;
  screenshot: `tmp/lora-trim/quantized-bake-mobile.png`. Browser peak telemetry
  was not captured; its 16 GiB/no-swap scope limits were enforced.
- Final standalone UI typecheck and `npm run build` passed. The build used
  3 min 22 s, peaked at 14.1 GiB and used no swap. Only the production UI branch
  was reloaded: the queue worker remained PID 656564. A scoped live check
  returned HTTP 200 and verified the new control/support copy in served assets.
  The isolated test server was stopped afterward.

These checks establish static full-strength weight/bias reconstruction and
loading for the documented formats. They do not establish visual quality,
bit-exact low-precision/GPU arithmetic, dynamic activation-quantized output
equivalence, or arbitrary undocumented checkpoint layouts. AWQ outputs are
consumer-specific; native biased bakes require a `.diff_b`-capable loader.

All executions use synthetic tensors/checkpoints under `tmp/lora-trim` or
temporary directories under `tmp`. No user checkpoint was trimmed, no GPU job
was launched, and no training process was stopped.

## Backend and script transport

- `testing.test_lora_trimming`: 16 tests pass, including storage-dtype subtests
  for FP16, BF16, FP32, FP64 and mixed factors. Covers measured reconstruction
  error and alpha compensation, exact/zero updates, reduced convolution ranks,
  invalid/impossible thresholds, numeric overflow/underflow, unsupported formats,
  source changes, symlink protection and competing no-clobber writers.
- Toolkit-exported A/B factors round-trip through `LoRASpecialNetwork` and its
  real `load_weights` path, with the target's per-layer rank matching the file.
  This does not claim that an old fixed-rank training configuration can resume
  a file with newly varying ranks without adjusting its network construction.
- A synthetic wide, low-rank archive verifies Unicode metadata, predicted versus
  actual file size, and both additional-networks hashes against the toolkit's
  real hash helper and Safetensors serializer. Factors stay small; a dense
  output-by-input update for that fixture would be unnecessarily large.
- `ui/tests/lora-trim-api.test.cjs`: one integration test passes through the
  actual `/api/scripts` handler and Python bridge. It checks preview-only
  behavior, required preview tokens, new-copy creation, collision protection,
  path traversal rejection and useful DoRA errors, preserving source bytes.

Python verification peaked at 566 MB; transport verification peaked at 325 MB.
Both ran in transient systemd units with `MemoryMax=16G`, `MemorySwapMax=0`,
CPU-only execution and the `ai-toolkit` Python environment.

The default-energy CLI preview for the 64-wide synthetic fixture selects rank
8 → 3: 4,368 source bytes → 2,120 predicted output bytes, relative Frobenius
error 0.0028636618 and numerical retained energy 0.9999917994. Metadata/header
cost is included. These numbers provide no visual-quality evidence.

## UI and build

The imagegen mockup is `tmp/lora-trim/ui-mockup.png`. The implementation reuses
the shared Modal and Text/Number/Select inputs, with optional input IDs for
accessible labels; the workflow is documented in `ui/README.md`.

The full UI source typecheck passes when generated `.next/types` are excluded.
The ordinary typecheck also includes pre-existing generated route-parameter
errors in inference, kill and notes routes. No type suppression was added.

`npm run build` passes with a fresh generated webpack cache and
`NODE_OPTIONS=--max-old-space-size=12288`, using the running UI's Node 25.6.0.
The capped build took 3m44s, peaked at 14 GiB and used no swap. Initial 4 GiB
and 8 GiB heap attempts failed; their logs and the old generated cache remain
under `tmp/lora-trim`. The successful build retains the existing optional
macOS temperature-sensor warning on this Linux host.

The isolated browser journey passes against the built Next server and synthetic
database on port 8676. It checks associated labels, Tab focus containment and
restoration, Enter-to-analyze, disabled busy controls/Escape, preview invalidation,
real script preview/write, unchanged source bytes, file-list refresh, collision
errors and visible DoRA rejection. Chromium 1248 runs headless with GPU disabled
inside the required 16 GiB/no-swap user scope. The completed scope did not expose
its peak-memory counter; the check was not repeated just for that telemetry.

Desktop and 390-pixel screenshots are in `tmp/lora-trim/screenshots/` and were
visually reviewed. Narrow layouts scroll vertically; the layer table scrolls
horizontally. The browser first previews the default rank 8 → 3 result above,
then changes retained energy to 99% and writes a separate synthetic copy:
1,600 bytes, rank 8 → 2, relative error 0.0280861930. These are different
settings/results, not the same measurement.

After verification, only the production UI branch was reloaded. The original
queue worker and training process remained alive with the same PIDs. The public
UI returned HTTP 200 and its new layout asset contains the trimming dialog.
The isolated test server was stopped; synthetic artifacts remain under `tmp`.

## Reproducing isolated browser verification

`playwright_scripts/prepare_lora_trim_e2e.py` creates only a synthetic SQLite
database and checkpoint folder under `tmp/lora-trim`. Run it with `ai-toolkit`
inside the same 16 GiB/no-swap systemd wrapper used for tests. Start only Next,
with `AI_TOOLKIT_DATABASE_URL=file:/absolute/repo/tmp/lora-trim/e2e.db`, on a
separate port (8676 in this run); do not launch a queue worker for the fixture.
The test server also runs in a transient unit with the same memory limits.

Run `playwright_scripts/lora-trim.e2e.cjs` in the required browser wrapper:

```bash
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
  env LORA_TRIM_BASE_URL=http://127.0.0.1:8676 \
  node playwright_scripts/lora-trim.e2e.cjs
```

Playwright must be installed for the test harness; `PLAYWRIGHT_MODULE` can point
to an existing installation instead of adding it to production dependencies.
Chromium runs headless with GPU acceleration disabled. Generated database,
checkpoint and screenshot artifacts remain under `tmp/lora-trim`.

## Reducer extension and DoRA baking

The training-console checkpoint action is now an accessible **Reduce** icon
opening the shared **Reduce adapter** modal. The extension adds six rank
policies, magnitude-preserving DoRA direction reduction, opt-in 2D LoHa/LoKr
conversion, and explicit base-aware DoRA baking. Full support boundaries and
strength-one/normalization limits are in [the guide](lora_trimming.md).

Verification used the `ai-toolkit` Conda environment and synthetic fixtures:

- The Python reducer module passed 26 tests and 43 subtests. After the final
  missing-alpha conversion fix, five affected conversion/bake tests passed with
  three subtests, including the new convert-then-reduce-again regression.
  Unaffected earlier passes were retained. Tests cover known singular spectra,
  scaling, metadata, actual toolkit LoRA/DoRA loader acceptance, both bake
  conventions, source/base immutability, hash-bound previews, memory rejection,
  unsupported layouts and no-clobber publication.
- The Node API-to-CLI integration passed 1/1, exercising all rank-policy flags,
  DoRA preview/baking, LoHa conversion and unsupported DoHa rejection through the
  real Python bridge. Its explicit-alpha fixture results remain valid after the
  missing-alpha-only fix.
- The browser journey passed against an isolated SQLite/Next server without a
  queue worker. It verified the sidebar icon, focus trapping/restoration, labeled
  fields, settings invalidation, preview/save/no-clobber, required baking base and
  convention, a baked ordinary-LoRA copy, LoHa conversion and visible DoHa errors.
  An initial harness-only Close/Cancel selector error was corrected before the
  successful run; production UI code did not change for that correction.

At default 99.99% energy, the synthetic ordinary adapter preview was 4,368 bytes
to 2,368 bytes, rank 8 to 3, relative error 0.00286366. The later 99% preview/save
produced 1,824 bytes, rank 8 to 2, relative error 0.02808619. The original DoRA
and base file hashes stayed unchanged after baking. Desktop and 390px screenshots
under `tmp/lora-trim/screenshots` were visually inspected.

Python and browser tests ran with CUDA hidden or GPU acceleration disabled;
all test commands used the prescribed 16 GiB/no-swap systemd wrappers. The
isolated server had its own matching limits. No GPU jobs, provider calls,
downloads or user-file reductions were performed.

UI source typechecking, worker compilation and `npm run build` passed. The build
ran for 3 minutes 27 seconds with a temporary 12 GiB Node heap allowance inside
the 16 GiB/no-swap unit, peaking at 14.4 GiB. The existing optional macOS sensor
warning remains on Linux; generated Next route-type errors were excluded from
the separate source check, with no new type suppression.

Only the production UI branch was reloaded; the queue worker remained PID
656564. The public UI returned HTTP 200 and served the new reducer/baking controls.
Build/typecheck/live-asset logs are under `tmp/lora-reduction`. The isolated test
server was stopped after verification. No remote push or registry publication
was performed.
