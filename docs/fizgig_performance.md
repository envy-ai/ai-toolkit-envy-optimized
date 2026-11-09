# Fizgig November 9 performance ports

Reference: [Fizgig v9.3.2](https://github.com/shootthesound/Fizgig/tree/8630fca)
and its November 7–9, 2026 changes. This fork ports selected mechanisms rather
than replacing its model implementations or memory manager.

## Implemented

- Known MOSS, Krea 2 and Qwen Image 2.1 transformer block inputs receive eager
  token-axis hints before dynamic block compilation. Unknown blocks and
  `compile_dynamic: false` retain their behavior. Existing eager checkpoint
  boundaries, quantized/offloaded compile constraints and cache handling stay
  in place. This addresses shape-related warm-up; it is not a promised
  steady-state speed multiplier.
- Masked MOSS CUDA attention prefers cuDNN when enabled. Efficient/math backends
  remain eligible for shapes cuDNN cannot handle. Unmasked and split reference
  attention keep native dispatch. Krea already used cuDNN preference and
  padded sequences; its implementation was retained.
- [Optional batched Muon](muon.md), available in the existing Optimizer selector.
  Existing jobs and new-job AdamW8bit defaults are unchanged.
- [Training maps](training_maps.md), an opt-in diagnostic for supported ordinary
  image LoRA/DoRA jobs.

No text-encoder batching, lower-precision adapter arithmetic, new checkpoint
placement or automatic optimizer/preset change is part of these ports. The
existing low-memory per-prompt encoding, quantization, layer offloading and
preview memory recovery remain authoritative.

## PyTorch 2.15 trial

The active `ai-toolkit` Conda environment was inspected as Python 3.11.13,
PyTorch 2.7.0+cu126, torchvision 0.22.0+cu126, Triton 3.3.0 and bitsandbytes
0.47.0. The driver reported 580.105.08 on an RTX 4090. The repository manager
documents a newer stack than that active environment.

A separate package overlay under `/home/bart/tmp/codex/aitk-torch215-overlay`
resolved and installed PyTorch 2.15.0+cu134, torchvision 0.29.2+cu134,
Triton 3.9.0 and bitsandbytes 0.50.2. CPU imports also succeeded for the existing
torchao 0.10.0 and optimum-quanto 0.2.4. The old torchaudio binary could not load
against this Torch version (undefined `torch::autograd::deleteNode` symbol);
installing torchaudio 2.11.1+cu134 in the overlay resolved that mismatch.
Resolver metadata is in ignored `tmp/fizgig-port/torch215-resolution.json`.
No installed package in the active Conda environment was changed.

Fizgig's large reported memory reduction concerns checkpointing inside compiled
blocks compared with its earlier Torch 2.10 path. This fork keeps checkpointing
outside compiled Krea/MOSS blocks, so those numbers are not a measured benefit
here. CUDA compilation, full training, memory peaks and quality still need
qualification before adopting the new stack. The production manager pins and
installation instructions remain unchanged until that qualification.

## Verification limits

Focused CPU checks exercise Muon equations, grouping and resume, dynamic block
hints and gradients, attention mask dispatch, quantized layer offload, and
training-map persistence/state restoration. Tests run in systemd with a 16 GB
host-memory limit and no swap. Full MOSS modes collection in the old stack is
blocked by an unrelated OmniGen2 module initializing CUDA during import while
CUDA is hidden; the directly imported MOSS attention fixtures are independent
and pass. No external source was modified to bypass that issue.

Full UI typechecking still reports the pre-existing generated route-signature
errors for the kill and notes routes. The required UI build is tracked
separately from service restart. No performance or likeness claim is inferred
from these CPU checks.

The initial focused runs passed 15 Muon/dynamic-shape cases, 10 attention and
existing offload/compile cases (one additional case skipped), and 5 UI map
reader/configuration cases. Three map capture/setup/resume cases passed before
the final helper-signature, logging and early-guard assertions were added; a
final rerun was stopped after host disk I/O blocked it, so that rerun is not a
pass. The isolated Torch 2.15 overlay also passed 10 Muon cases, quantized
offload compilation, adapter offload (2 tests, 12 subtests), and prompt offload.

After build progress resumed, one final affected-case attempt used a 90-second
timeout. It timed out without pytest output (exit 124, 9.25 CPU seconds,
1.2 GB peak, zero swap), leaving the final guard/logging/signature assertions
unverified. No additional retries were made.

Browser verification passed against an identical temporary copy of the UI
source: picture/map/noise selection, step navigation and keyboard shortcuts,
opacity, manual refresh, persistent errors, empty state, and overlay alignment
at desktop and narrow widths. Requests used synthetic fixtures, the GPU was
disabled, and the preview server was stopped afterward. Screenshots are in
ignored `tmp/fizgig-port/training-maps-{desktop,narrow}.png`.

Builds at the default and 8 GB Node heap limits exhausted the heap. The project
now requires a 16 GB heap in `AGENTS.md`, and `run.sh` exports that setting.
The required `npm run build` completed successfully with that heap. It emitted
optional `macos-temperature-sensor` warnings from systeminformation. The
existing Next configuration skips type validation; the separate typecheck
reported only the pre-existing route-signature errors described above.
The production server and active trainers were not restarted.
