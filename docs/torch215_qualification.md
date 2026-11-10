# PyTorch 2.15 qualification

Scope: qualify the existing CUDA 13.4 trial stack for this Linux x86_64 RTX 4090 installation, then integrate the compatible package set into the training environment and repository installation instructions. This does not change checkpoint placement, quantization, layer offloading, per-prompt encoding, or sampling recovery to imitate another trainer.

## Plan and acceptance criteria

1. Record the baseline environment, candidate package paths, driver, free RAM/VRAM, and available local model assets. Preserve a package inventory and a reproducible rollback before replacing installed packages.
2. Run the candidate in isolation through CUDA allocation, forward/backward, compiled kernels, masked attention, and native dependency imports. The test must assert which Torch installation it actually loads.
3. Exercise the fork's quantized/offloaded training paths, optimizer updates and resume, checkpointed gradients, and preview/state restoration. Compare matching workloads with the baseline, recording timings and peak device memory without assuming a speed improvement.
4. Run a bounded training smoke test with available model assets when it fits the required host-memory limit. Distinguish tests using production blocks and tiny configurations from a complete pretrained-model run; do not infer final image quality from synthetic fixtures.
5. Resolve application compatibility issues within the requested upgrade scope. Do not patch external packages merely to obtain passing tests. If essential external behavior is incompatible, leave the current environment unchanged and record the failure.
6. Promote only a qualified package set. Keep platform-specific wheel and accelerator constraints honest; qualification on this machine does not establish compatibility on Windows, macOS, ROCm, or other GPU architectures.
7. Verify imports and a focused GPU smoke test in the installed environment, update documentation and dependency specifications, and commit the integration. Report service restarts separately.

Every test command runs in one transient systemd scope with `MemoryMax=16G`, `MemorySwapMax=0`, and a finite timeout. Tests own the GPU sequentially. Logs and temporary fixtures belong in ignored `tmp/torch215-qualification/`.

## Initial state

- Active launcher environment: `/home/bart/.conda/envs/ai-toolkit`, Torch `2.7.0+cu126`.
- Candidate overlay: `/home/bart/tmp/codex/aitk-torch215-overlay`, Torch `2.15.0+cu134`, torchvision `0.29.2+cu134`, torchaudio `2.11.1+cu134`, Triton `3.9.0`, bitsandbytes `0.50.2`.
- GPU: RTX 4090, driver `580.105.08`, 24,564 MiB total; 23,651 MiB free at preflight.
- Host: 125 GiB RAM, 114 GiB available at preflight. Existing system swap use does not change the test scope's no-swap limit.
- The repository manager currently specifies a different stack from the active Conda environment; integration must account for both entry points.

[NVIDIA's CUDA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) allows CUDA 13.x minor compatibility on driver 580 or newer, with restrictions for newer features and PTX compilation. Runtime CUDA and compilation tests are therefore required despite successful CPU imports. [PyTorch's release compatibility table](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) lists CUDA 13.2 and 13.4 builds for 2.15; the existing 13.4 overlay is the first candidate.

## Results

Candidate qualification and installed-prefix verification passed. The tested stack was installed normally into `/home/bart/.conda/envs/ai-toolkit`; new training processes from `run.sh` use PyTorch `2.15.0+cu134` without a diagnostic overlay or profile environment variable. The full 7.7 GiB Conda prefix was copied successfully before promotion; its path is recorded in ignored `tmp/torch215-qualification/rollback-prefix.txt`. Baseline/installed pip inventories, Conda inventory and pip resolution/install reports are stored beside it. Post-install `pip check` reports only the same existing `easy-dwpose` / `huggingface_hub` version mismatch as the baseline; that unrelated conflict is not part of this upgrade.

### GPU and application checks

- Both stacks passed actual Inductor CUDA forward/backward compilation and masked cuDNN attention against the math backend. The maximum attention output difference was 0.0078125 in the bfloat16 fixture.
- Both passed six GPU quantization/kernel cases: bitsandbytes NF4 and AdamW8bit, Quanto int8, torchao int8, ConvRot int8/int4 gradients, and torchaudio resampling.
- Both passed eager versus compiled/checkpointed forward, input-gradient, and parameter-gradient comparisons using the production MOSS block implementation at reduced dimensions. This was not a complete pretrained MOSS run.
- The native SD1.5 trainer used a local checkpoint and generated 256-pixel fixtures: three initial LoRA steps with checkpointing, previews and optimizer/adapter saves; a same-version resume; and a ten-step resumed comparison. The candidate also resumed the actual baseline Torch 2.7 / bitsandbytes 0.47 checkpoint, including optimizer state, and trained/saved successfully.
- Candidate torchcodec 0.15 imported and decoded an MP4 frame. Its distribution metadata is `0.15.0`; its runtime version reports `0.15.0+cu130`. The baseline torchcodec 0.9.1 cannot import against the old Torch 2.7 installation (undefined SymBool symbol); no external code was modified.
- Candidate regressions passed 38 cases plus 12 subtests: training maps including the previously unverified error/resume cases, adapter/prompt offloading and restoration, quantized/offloaded compilation, Muon, dynamic shape hints and the H3 photo recipe.
- Eleven installer-profile tests passed, covering constraints, drift detection, stale binary cleanup, managed-interpreter requirements and unchanged default/Spark contracts. Official CUDA 13.4 trio wheels also resolved for the manager's Python 3.12 target; GPU execution was tested with the active Python 3.11 interpreter.

### Bounded timing and memory comparison

| Native SD1.5 smoke measurement | Torch 2.7 / bitsandbytes 0.47 | Torch 2.15 / bitsandbytes 0.50.2 |
| --- | ---: | ---: |
| Median ordinary training iteration | 222.85 ms | 231.90 ms |
| Peak CUDA allocation | 2.15 GiB | 2.13 GiB |
| Peak CUDA reservation | 2.31 GiB | 2.28 GiB |
| Peak host memory | approximately 4.7 GiB | approximately 4.7 GiB |

The timing median uses four non-save iterations, excluding the initial aggregate/warm-up and alternating checkpoint-save iterations. The candidate was approximately 4.1% slower in this small sample. This establishes a bounded functional comparison, not a speed benefit or a statistically robust benchmark. It does not establish MOSS throughput, full-size model VRAM requirements or final likeness. Existing checkpoint/offload boundaries remain unchanged.

### Installed-prefix verification and integration

With `PYTHONPATH` unset, all six package paths and metadata were checked in the actual Conda prefix: Torch 2.15.0+cu134, torchvision 0.29.2+cu134, torchaudio 2.11.1+cu134, Triton 3.9.0, bitsandbytes 0.50.2, and torchcodec 0.15.0. Actual installed-prefix CUDA compilation and masked attention passed, followed by seven GPU quantization/codec checks. A fresh copy of the old Torch 2.7/bitsandbytes 0.47 checkpoint loaded its LoRA and optimizer at step 15, trained step 16 and saved successfully; its host peak was approximately 4.8 GiB and CUDA reservation approximately 2.28 GiB. No external package source was patched. Bitsandbytes selects its bundled CUDA 13.2 binary for this CUDA 13.4 runtime; the exercised NF4 and optimizer kernels passed.

The manager exposes the same stack through `AI_TOOLKIT_TORCH_PROFILE=torch215`; see [manager usage](../manager/README.md#opt-in-pytorch-215-cuda-profile). This is separate from the Conda launcher. Default manager pins, Docker, Windows, ARM, macOS and ROCm installation paths retain their previous versions. The opt-in profile keeps exact trio/extra constraints, rejects unsupported platforms and old managed interpreters, removes incompatible old FlashAttention/NATTEN binaries, and detects package drift. The NVIDIA driver was not changed. This qualification did not restart the UI, queue worker or any live training service.

For the tracked installer regression test:

```bash
systemd-run --wait --collect --pipe \
  -p MemoryMax=16G -p MemorySwapMax=0 -p "WorkingDirectory=$PWD" \
  timeout 120s conda run --no-capture-output -n ai-toolkit \
  python -m pytest -q testing/test_torch215_profile.py
```

Other qualification logs, fixture configs and diagnostic scripts are retained under ignored `tmp/torch215-qualification/`. `active-*` logs are installed-prefix evidence; `baseline-*` and `candidate-*` logs distinguish the original and isolated trial environments. The generated images and small SD1.5 workload are diagnostic fixtures, not a likeness evaluation.
