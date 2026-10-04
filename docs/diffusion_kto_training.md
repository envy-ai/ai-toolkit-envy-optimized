# Diffusion-KTO LoRA (experimental)

Available on Qwen Image 2.1, Krea 2, Anima and Ideogram 4. This is a new
binary-feedback workflow, not DPO with a renamed label. The flow-matching
adaptation is experimental: CPU tests verify objective/replay contracts, not
actual-checkpoint image quality or VRAM fit.

## Data and form

Select **Diffusion-KTO LoRA (experimental)** as the training mode. Add ordinary
dataset folders and give each a **Liked** or **Disliked** feedback label. Each
image uses its own caption (or the folder's Default Caption). Subfolders are
scanned recursively; disabled image files are ignored. Captions should describe
the actual image, including disliked images, rather than describing the desired
replacement.

No filename matching, equal resolution, equal counts or Control Dataset 1 is
required. Aspect-ratio bucketing is retained, with each model's native alignment.
Use varied liked/disliked examples covering the subjects and styles you want to
retain. Single-class folders/jobs are accepted, with a startup warning if one
class is absent. Source-image counts are logged before dataset repeats.

The objective card contains Beta, Liked Weight, Disliked Weight and a
reference-point estimator. Weights change utility strength; folder repeats change
sampling frequency. Existing rank, alpha, quantization, checkpointing and offload
controls remain available. Latents and text embeddings must be cached.
Dataset Loss Weight applies to that folder's utility contribution, not to the
reference-point estimate. Caption and encoder identity changes invalidate text
caches; image size/mtime changes invalidate KTO latent caches. Changing only a
feedback label or utility weight does not require re-encoding the image.

## Objective and estimators

For each image, reference and policy evaluate the same caption, noisy latent and
model-native time. The reference is the base with the LoRA disabled. This version
uses unweighted per-image flow-velocity MSE as a likelihood surrogate; it does
not claim to reproduce a DDPM likelihood or the published SD 1.5 implementation.
Beta defaults to 1, not the published SD recipe's scale.

Let `g = reference_error - policy_error`, `y = +1` for liked and `-1` for disliked,
and `z = max(0, mean(g))`, detached from gradients. Each image contributes
`label_weight * (1 - sigmoid(y * beta * (g - z)))`, then its dataset loss multiplier.
The final optimizer loss is averaged by image count, including unequal bucket
batch sizes. Dataset repeats/weights do not enter the reference-point mean.

- **Batch mean** is the published default estimator: a separate detached mean
  within each accumulation batch. At batch size 1 it is a single-image estimate,
  not a multi-image reference point.
- **Pooled score window** combines examples from consecutive accumulation
  batches, including different buckets. Set Gradient Accumulation to at least
  Score Window size (minimum 2). Scoring happens at unchanged adapter weights
  before sequential gradient replay. Short final windows use their actual counts.
  No EMA, warmup or estimator state is carried across optimizer steps.

The examples choose window size 4 and Gradient Accumulation 4 for small-batch
training. These are starting configurations, not quality-tested recommendations.
Monitor the logged per-class scores/utility, saturation and reference point,
and compare fixed-seed previews at disabled/enabled strengths to detect drift.

## Memory, checkpoints and restrictions

One frozen base is shared by reference and policy; no second resident model or
reward model is loaded. Detached replay inputs and prompt objects are retained
on CPU. Only one policy activation graph is retained for backward. Reference,
policy scoring and replay restore identical prediction RNG state. Memory options
remain intact, but no actual-model memory fit is certified by CPU fixtures.

Checkpoints are permitted only after complete score/replay windows and optimizer
steps. Saving mid-window is rejected. No pending window is serialized; ordinary
model/optimizer/RNG resume starts at the next complete window. Metadata records
the surrogate version, estimator/settings and this checkpoint invariant.

Initially this mode supports ordinary LoRA and text-to-image only. DoRA/LoHa,
edit/reference-image conditioning, trainable text encoders/Anima conditioner,
pretrained or other auxiliary adapters, adapter dropout, EMA, SNR/noise offsets and
additional objectives are rejected. Use normal job resume for continuing the
same KTO run rather than loading a pretrained LoRA as a new reference.

Qwen Image 2.1 supports a frozen helper LoRA at **Model → Helper LoRA Path**
(`model.assistant_lora_path`). Both reference scoring and policy/replay use the
same helper. It is excluded from preview samples and the saved trained LoRA.

Ordinary signed-strength LoRA export and native/architecture-compatible Comfy
previews are unchanged. Ideogram uses its selected CFG reference policy for
previews; KTO scoring itself is conditional CFG 1.

## Examples and evidence

- `config/examples/train_diffusion_kto_qwen_image_21.yaml`
- `config/examples/train_diffusion_kto_krea2.yaml`
- `config/examples/train_diffusion_kto_anima.yaml`
- `config/examples/train_diffusion_kto_ideogram4.yaml`

Replace component and dataset paths before use. Constructor fixtures check all
four YAMLs with unequal image sizes/counts/stems. CPU fixtures exercise the real
optimizer hook against a full-graph oracle across all four flow profiles,
including Anima's integer prompt fields, class weights, deterministic stochastic
replay, exception cleanup and checkpoint-boundary optimizer/RNG resume. They do
not substitute for full-checkpoint gradient/export/render or quality tests.
Additional fixtures run the real tiny native transformer classes, native
schedulers, prediction/conditioning wrappers and checkpointed LoRAs on all four
models. Both initial no-op and learned-adapter optimizer updates match a
full-graph oracle while leaving the reference and Anima conditioner frozen.
Actual architecture key converters and safetensors reloads preserve predictions
at signed/fractional strengths. Installed-Comfy loading and live rendering of
these KTO exports still require their separate integration checks.

Method background: [Diffusion-KTO paper](https://arxiv.org/abs/2404.04465) and
[official SD 1.5 training code](https://github.com/jacklishufan/diffusion-kto/blob/main/train_kto_sd_v1.5.py).
