# Kaggle TPU ConvRot8 feasibility probe

This is the first hardware gate for the [Kaggle TPU worker plan](kaggle_tpu_worker_plan.md). It tests the installed Kaggle Torch/XLA runtime, one logical replica on every launched TPU device, a collective, and a frozen ConvRot8 projection with input and LoRA gradients. It does not load Qwen or establish full-model HBM fit, native INT8 hardware throughput, or remote job operation.

## Build and run

From this worktree, create the small allowlisted bundle:

```bash
conda run -n ai-toolkit python workers/kaggle_tpu/build_probe_bundle.py
```

The output is `tmp/kaggle_tpu_probe_bundle.zip`. Its manifest records the source commit and SHA-256 of the two bundled Python files, including uncommitted edits. It contains no model, dataset, token, database, or repository archive.

Create a private Kaggle dataset containing that ZIP. Import [`bootstrap.ipynb`](../workers/kaggle_tpu/bootstrap.ipynb) as a Kaggle notebook, attach the ZIP dataset, select a TPU accelerator, enable notebook Internet access, and run the cells. Kaggle may leave the ZIP intact or extract it; the notebook accepts either layout and verifies the exact source allowlist and hashes. It checks installed Torch and Torch/XLA versions before importing either. If they do not match, it installs the [official PyTorch/XLA 2.8.0 candidate pair](https://github.com/pytorch/xla/blob/master/README.md) into an isolated virtual environment: CPU `torch==2.8.0` and `torch_xla[tpu]==2.8.0` (which pins the compatible `libtpu` dependency). The notebook kernel's packages are not changed. This pair is a candidate until it passes on the actual Kaggle TPU.

The probe runs in a subprocess, keeping PJRT initialization out of the notebook kernel. It produces a fresh `/kaggle/working/ai_toolkit_tpu_report_<id>/diagnostic.json`, individual `rank-<ordinal>.json` files, and a neighboring `.log` file. It has a configurable 20-minute watchdog that terminates the process group and writes a timeout diagnostic. Download `diagnostic.json` and the log even if the cell raises an error. A report with `status: pass` requires one result for every logical device, a correct INT8 dot with INT32 output, an all-reduce including every rank, finite projection and LoRA gradients, compact saved ConvRot weight state, and agreement with the CPU reference at the tolerances recorded in the JSON.

The device count means **logical XLA devices**, not physical TPU chips. An INT8 dot that compiles and produces the right answer does not establish that hardware uses native INT8 throughput. Per-device HBM is reported when the installed XLA runtime supports `get_memory_info`; absence is recorded as an error field rather than estimated. The current probe does not exercise attention or measure useful Qwen samples per second.

PyTorch/XLA [documents threaded PJRT execution on TPU v2/v3](https://docs.pytorch.org/xla/master/learn/pjrt.html), which is why the notebook launches through `torch_xla.launch` and checks logical ranks. The [published r2.8 quantized-operation table](https://docs.pytorch.org/xla/release/r2.8/perf/quantized_ops.html) lists per-token W8A8 as unsupported, while [current upstream source](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/xla_quantized_matmul.py) contains a private INT8 dot route. This probe tests the installed wheel instead of assuming either description applies to the assigned Kaggle runtime.

## Next gate

After receiving a passing report, use the user's exact Qwen Image 2.1 ConvRot INT8 checkpoint, adapter configuration, and cached target/prompt/reference artifacts to implement and measure a sustained one-device training step, checkpoint save, and resume. The full runner must also qualify Qwen's block-causal attention under XLA and measure peak HBM. Only after that fit gate should the replicated all-device trainer and remote controller be integrated. If the probe fails, use its original exception and runtime versions to identify the smallest fork-side or runtime change; the worker must not substitute BF16 base weights, a W8A16 path, or model sharding.

`prepare_convrot8_scales_for_xla_` must run on CPU after all model dtype conversions and before XLA placement. It creates nonpersistent FP32 scales while leaving checkpoint byte-view scales intact. A later `model.to(dtype=...)` may change those FP32 scales and will cause the XLA forward path to reject the model. Global `XLA_USE_BF16` and `XLA_DOWNCAST_BF16` modes are rejected because they alter FP32 scale arithmetic.
