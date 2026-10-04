# SliderSpace (flow models)

Choose **SliderSpace** in the training-mode selector. Qwen Image 2.1, Krea 2,
Anima and Ideogram 4 are selectable; the new ports remain experimental pending
actual-checkpoint validation. See [cross-model contracts](cross_model_training_modes.md).
This experimental
mode discovers visual variation from provided images, generated images, or both,
and trains one ordinary LoRA per principal direction. It does not require pairs
or positive/negative attribute labels. For generated images, start with
`a spaceship`, or add several descriptions that explore the same concept.

On Krea, Anima and Ideogram, discovery banks additionally record the components
actually loaded: transformer, VAE, text encoder/tokenizer, Anima conditioner and
CLIP feature model/processor. This covers implicit default paths and selected
single-file or sharded checkpoints. Resolved component identities enter the
bank's cache directory and resume metadata; changing them requires a new job.
Earlier experimental port checkpoints without this verification are rejected
with an explicit message. Existing Qwen discovery signatures are unchanged.
Published complete checkpoints require their bank, paired optimizer and all
direction exports to remain intact. An incomplete latest published generation
does not silently fall back to a new training run.

The form follows the existing training layout. The full-width SliderSpace card
contains the image source selector, repeatable discovery folders, concept prompts,
direction count, generated image count and resolution.
Advanced discovery settings include CFG, optional negative prompt, steps, seed,
and the frozen feature model/device. These CFG and negative-prompt settings apply
to both discovery generation and teacher/student predictions. At CFG 1, the
negative prompt is retained but inactive. The Samples card independently controls
ordinary previews, including which direction and signed strength to show.

## What happens

1. Cache concept/preview text embeddings on CPU.
2. Import provided images and/or generate a reproducible discovery bank with the frozen base model.
3. Extract normalized CLIP vision features and run centered PCA on CPU.
4. Train direction LoRAs round-robin, sharing one frozen transformer.

The objective aligns the semantic feature difference between the adapted and
base image predictions with each PCA vector:
`1 - cosine(feature(adapted) - feature(base), direction)`.
The wrappers expose clean reconstruction as `x_t - (timestep / 1000) * velocity`.
The reference branch has no training graph; the student decodes differentiably
through the frozen VAE and feature encoder. The feature encoder is not trained.
This implements the mathematical SliderSpace objective independently rather than
using Fizgig pairwise training or copying the upstream training scripts.

Qwen Image 2.1 supports an optional frozen helper LoRA through **Model → Helper
LoRA Path** (`model.assistant_lora_path`). The helper stays active for generated
discovery images and both reference/student predictions. Changing its path or
file invalidates the discovery/resume signature. Native and Comfy preview samples
exclude the helper, and exports contain only the trained direction LoRAs.

The [SliderSpace paper](https://arxiv.org/abs/2502.01639) evaluated other models,
not this Qwen adaptation. Direction numbers are not attribute names; content
drift and imperfect disentanglement remain possible. Negative strengths follow
ordinary LoRA inference scaling; their quality is not guaranteed by a positive
direction-alignment objective. Multiple directions can be loaded like ordinary
LoRAs, but clean composition is not guaranteed.

## Starting settings and cost

Defaults are 4 directions, 128 discovery images at 512px, 20 generation steps,
CFG 4, rank/alpha 4, and 1,000 **total** optimizer updates (about 250 per
direction). These are conservative engineering defaults, not measured Qwen
quality recommendations. A bank must have at least `directions + 1` images and
cover every generated concept prompt. Numerically deficient PCA is an error; use fewer
directions or more varied prompts/images rather than receiving random vectors.

Discovery can take a while before the first training update. Raising discovery
count increases startup time and disk usage; raising resolution increases both
generation and differentiable-decoder memory cost. Each direction adds LoRA
factors and optimizer state, not another full Qwen model.

The default feature encoder is `openai/clip-vit-base-patch32`; the first run may
download it through Transformers. A compatible local CLIP vision checkpoint
with its image processor is supported. CPU is the default feature device to
reduce GPU residency; CUDA is optional. Saved student activations are offloaded
to CPU on CUDA, and decode/feature extraction are checkpointed. This does **not**
promise that any particular resolution fits your GPU. Existing transformer
quantization, checkpointing, low-VRAM and layer-offload controls remain available.
Only batch size 1, accumulation 1, single-process ordinary LoRA are supported.
DoRA/LoHa, shared input/output-layer training, EMA, dataset validation, other
losses, other auxiliary adapters, merge-and-reset saves and automatic Hub uploads are
not supported in this initial mode. Record-low saves are disabled because losses
from different directions are not directly comparable; use scheduled saves.

## Provided images and aspect-ratio buckets

Choose **Provided images** to skip generation entirely, or **Provided + generated
images** to combine sources. Add as many discovery image folders as needed,
using the dataset selector or a server-side path. Each folder is scanned
recursively; overlapping paths and symlink aliases to the same file are deduplicated.
Files ending in `.disabled` are ignored. All supported images are imported;
the generated discovery count is an additional count, not a limit on supplied
images. At least `directions + 1` total images are required, checked before
loading the model. Supplied images must still provide enough visual variation
for the requested PCA rank.

A matching `.txt` caption is preferred, then the folder's optional default
caption, then the first concept prompt. If none exists, conditioning is blank.
Provided-only mode does not require concept prompts, but descriptive captions
are recommended. Captions are cached on CPU and each training image uses its
own caption for both reference and student predictions. Ordinary training
datasets remain disabled; discovery folders belong inside the SliderSpace card.

**Aspect-ratio buckets** is the default provided-image sizing for newly created
UI jobs. `resolution` is an area budget (`resolution²` pixels), not a requirement
that both dimensions equal it. Bucket dimensions are multiples of 32 for Qwen/Anima
and 16 for Krea/Ideogram, retain
roughly the source aspect ratio and do not deliberately upscale small images.
EXIF orientation is respected. Mitchell resizing and a centered crop fit each
bucket; originals are never modified. Each training update uses one image's
native bucket latent shape, with the selected model's training-time profile
(Qwen latent-area shift, Krea token-count shift, Anima fixed shift 3, Ideogram fixed shift 1).
Generated discovery images remain square at the selected resolution.

The shared CLIP feature transform proportionally resizes and mean-color pads
bucketed images to its fixed input size, retaining the full image without
stretching. Discovery, reference and student features use the same transform,
and the student transform remains differentiable. Aspect ratio and padding can
still affect PCA: if dimensions correlate strongly with an attribute, a discovered
direction might partly encode composition rather than just that attribute.
The **Square crop** option restores the previous sizing/feature behavior.
Old imported configurations retain square cropping unless explicitly enabled;
YAML enables buckets with `discovery_buckets: true`.

Example supplied-image settings (replace the paths with folders on this server):

```yaml
sliderspace:
  discovery_mode: both # generated | provided | both
  discovery_buckets: true
  discovery_datasets:
    - folder_path: /datasets/spaceships
      default_caption: A spaceship
    - folder_path: /datasets/more-spaceships
  concept_prompts:
    - A spaceship exploring deep space
  num_directions: 4
  discovery_samples: 32 # Added to ALL provided images; ignored in provided mode.
  resolution: 512
```

## Cache, outputs and resume

Processed/generated PNGs, latents, feature tensors and PCA vectors live under the job's
`sliderspace_discovery/<signature>/` folder, outside the preview gallery.
Completed matching banks are reused, and an interrupted bank can finish missing
images. The signature covers concept/generation settings, model configuration,
and local model/encoder file modification times (including component folders).
Provided modes also include recursive file membership, source image/caption sizes
and nanosecond modification times, captions, folder defaults, bucket settings,
and image/feature presentation versions. Changing a source invalidates its bank;
source changes during import are rejected. Generated-only legacy cache identities
are preserved. Bucketing is inactive for generated-only banks.
Remote model IDs use their configured identity; use local checkpoints if
file-based reproducibility against upstream changes is required. Remote
repository updates are not automatically fingerprinted. A completed cache's
file sizes and modification times are verified. Changed discovery/model settings
cannot resume an existing adapter bank: use a new job name. Preview direction,
preview strength and loss weight do not invalidate discovery.

Root-level downloads are named `JOB_direction_01_STEP.safetensors`, etc., with
final versions `JOB_direction_01.safetensors`. Every direction exports regardless
of the preview selection. These are standard Qwen LoRAs with alpha, usable at
positive, negative, or zero strength in ordinary loaders. `sliderspace.json`
records the exports and explained PCA variance, not invented semantic labels.

Complete full-precision adapter-bank and paired optimizer checkpoints are saved
under `sliderspace_state/`. Resume chooses a completed bank generation, never a
single direction export. Retention removes whole scheduled generations; final
exports and the final bank are protected. Keep the discovery folder, state folder,
and job configuration when moving or backing up a resumable job.

Generated discovery uses native Qwen generation. Provided-only ingestion does
not instantiate a generation pipeline and keeps the transformer off the GPU
while encoding discovery images. Normal previews can use the existing
internal or configured ComfyUI renderer, loading only the selected direction.
Preview filenames/gallery captions identify direction and strength. Sample-only
editing permits changing preview settings without modifying discovery
or training. Use a fixed seed when comparing directions or strengths; strength
0 is the base-model reference.

Enable **Auto sampling** in the Sample card to enter one **Auto sample prompt**.
Set **Auto sample strengths** to a comma-separated list, defaulting to `-1, 1`.
Each sampling round renders the base model once, followed by every direction at
the listed strengths in order. For example, `-0.5, 0.5, 2` renders each LoRA at
those three strengths after the base image. Zero and duplicate strengths are
skipped in the list so each comparison is rendered once. All images share the
same prompt and seed, including when the seed is random. With four directions
and the default strengths, this creates nine images per round. The native and
ComfyUI renderers use one direction at a
time. Gallery labels identify the base and each signed direction.

Auto sampling uses the first sample entry; enabling it in the form retains that
entry and removes extra prompts. Disable it to return to the manual direction
and strength controls. The setting can be changed on a running job and does
not invalidate discovery caches or adapter-bank resume. YAML enables it with
`sliderspace.preview_auto: true`, `sliderspace.preview_auto_strengths: "-1, 1"`,
and one entry under `sample.samples`.

See [example configuration](../config/examples/train_sliderspace_qwen_image_21.yaml).
