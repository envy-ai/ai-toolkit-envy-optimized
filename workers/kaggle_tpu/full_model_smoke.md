The full-model gate runs one complete frozen Qwenvy 2.1 ConvRot8 transformer
on one XLA rank, adds the original frozen Fizgig helper, and constructs the
configured signed DoRA network using AI Toolkit. It reuses
`FizgigSliderTrainer._noised_state` and `_image_loss` for independent +1/-1
noise/time samples, differential guidance, pair-difference weights, Min-SNR,
and the original half-weighted two-endpoint backward. Each backward finishes
before changing strength, preserving checkpoint recomputation semantics.

The approved optimizer substitution is explicit: ordinary AdamW uses the
saved learning rate, weight decay and toolkit epsilon of `1e-6`. Trainable adapter parameters and AdamW
moments are FP32, matching the local trainer's adapter precision; transformer
activations and the frozen helper are BF16. CPU loading remains available
under `low_vram`, and every execution layer moves to XLA. GPU compilation
settings map to XLA's own lazy compilation. Layer streaming is rejected.

Use the repository in the matching Kaggle Torch/XLA runtime that passed the
projection probe, with its Python dependencies installed. This command does
not install or replace Torch wheels. Run it in a subprocess, with the same
worker-address environment cleanup as `bootstrap.ipynb`:

```bash
env -u TPU_PROCESS_ADDRESSES -u CLOUD_TPU_TASK_ID PJRT_DEVICE=TPU \
  python workers/kaggle_tpu/full_model_smoke.py \
  --cache /kaggle/input/paired-cache/cache \
  --asset-manifest /kaggle/input/model-assets/asset-manifest.json \
  --output /kaggle/working/qwen21-full-smoke
```

The paired cache comes from `export_paired_cache.py`. The asset manifest maps
`Qwenvy2.1-v1.0.safetensors` and
`fizgig_qwen_image_2.1_training_adapter.safetensors` to their SHA-256 values;
the files sit beside it. Both must also match the cache's component identities.
The first pair is selected unless `--pair-id` chooses another. Cached tensors
retain their actual resolution. No synthetic latents or prompt are substituted.

`static_qwen.py` prepares sequence positions and rotary frequencies on CPU and
uses equivalent real-valued rotary arithmetic on the device. This initial
implementation supports only no-reference image-slider pairs. Its segmented
SDPA is dense, so it can fail the memory gate at the actual image resolution.
XLA blocks use the pinned runtime's reentrant checkpoint function, including
its RNG handling and optimization barrier; [XLA 2.8 rejects the nonreentrant
variant](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/utils/checkpoint.py).
It does not establish memory-efficient attention, peak HBM, or useful training
throughput. The diagnostic explicitly labels those limits, and reports XLA
metrics, unexpected `aten::` CPU-fallback counters, synchronized stage timings,
post-update device memory, host peak RSS, finite losses/gradients, and a direct
adapter update check. Failures exit nonzero and retain `diagnostic.json` at
the last recorded phase.

Successful execution writes `checkpoint.pt`, then restores adapter/optimizer/
CPU and XLA RNG state from it, checking every adapter tensor. This is a local
same-process checkpoint roundtrip. It does not claim fresh-process resume,
multiple-device training, inference export, durable remote storage, or the
final 40,000-update global-batch-eight run. No remote publication occurs.

Local CPU verification, run inside the required systemd 16-GiB/no-swap scope:

```bash
conda run -n ai-toolkit python workers/kaggle_tpu/check_static_qwen.py
```

The diagnostic compares tiny unmodified Qwen outputs and all signed DoRA
gradients against the static path for both strengths, padded/unpadded prompts,
and checkpointed/uncheckpointed blocks. This is CPU semantic evidence only;
full-model TPU execution must still be measured on the actual Kaggle session.
