# Adaptive LoRA trimming validation

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
