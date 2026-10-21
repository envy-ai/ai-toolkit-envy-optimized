# FIBO2 Dev training

Select **FIBO2 Dev** (`model.arch: fibo2`) in the existing Model card. This is a
native FIBO2 implementation, using the non-distilled **Fibo 2.0 Dev** base.
Edit Turbo is inference-only; it is rejected as a training base. The integration
is experimental: registration, small-model fixtures and checkpoint-header checks
do not establish full-checkpoint execution, production-resolution memory fit or
learned image quality. See [validation evidence](fibo2_validation.md) for the
checks actually completed and their limits.

## Components and existing controls

Use **Base Model Source** and **ComfyUI Checkpoint Path** for a Comfy transformer
file, or **Name or Path** for a compatible native component directory/repository.
The installed local component choices are:

```yaml
model:
  arch: fibo2
  name_or_path: /home/bart/ComfyUI/models/diffusion_models/d/fibo2_dev_int8_convrot.safetensors
  text_encoder_path: /home/bart/ComfyUI/models/text_encoders/qwen3vl_4b_bf16.safetensors
  vae_path: /home/bart/ComfyUI/models/vae/flux2-vae.safetensors
  quantize: true
  qtype: convrot8
  quantize_te: false
  low_vram: true
```

**Text Encoder Safetensors Path** selects the training/native Qwen3-VL-4B encoder.
Selection checks `te_name_or_path`, then `text_encoder_path`, then
`model_kwargs.text_encoder_path`, then `Qwen/Qwen3-VL-4B-Instruct`. The processor
uses `model_kwargs.processor_path`, then a distinct explicit
`extras_name_or_path`, then the Qwen repository. These are separate from the
Comfy preview **Text Encoder** selection (`sample.comfy.text_encoder`). Local
weights still need compatible processor/tokenizer/config assets available to
the loader; a transformer file alone does not contain them.

**Image VAE Checkpoint** (`model.vae_path`) must explicitly select a compatible
FLUX.2 VAE. The same VAE packing and BatchNorm normalization boundary is used
for training, generation and edit references. FIBO2 uses packed 128-channel BCHW
latents at pixel dimensions divided by 16. Bucket dimensions must be multiples
of 16. A Qwen/MOSS encoder or a similarly shaped VAE is not interchangeable.

Reuse quantization, layer offloading, gradient checkpointing, CPU text caches,
preview optimizer offload and tiled decode controls. Keep the original ConvRot
format with `qtype: convrot8` for the local int8 checkpoint. Selecting a different
supported quantization format is an explicit conversion choice. Full finetuning
requires an unquantized base and has a substantially different memory cost;
the Hunyuan immutable shared-weight handoff is not supported here.
When a checkpoint already uses the selected quantization and the loader reports
`Checkpoint is pre-quantized (...); keeping it`, no redundant persistent module
cache is written. This applies to both the transformer and text encoder, even
with **Cache Quantized Models** enabled. Fresh quantization and conversion to a
different format can still be cached. Existing cache files are left in place.
An accuracy-recovery adapter can be used with the persistent quantized-module
cache disabled. FIBO2 rejects combining that adapter with the persistent cache
to prevent stale or doubly applied recovery weights; ordinary quantized caching
without an accuracy-recovery adapter remains available.

Architecture switching clears incompatible base/encoder/VAE/config paths,
imported component overrides, training/preview adapters and sample reference
images with the existing notice. Destination mode flags still use that
architecture's defaults. Prompt sets, paired targets,
preservation anchors, adapter ranks and memory settings are retained. Enter the
matching component paths after switching. Mouse selection and the shared
controls' normal Tab, arrow-key and Enter behavior are unchanged.

## Workflows and adapters

| Workflow | Adapter choices | Example |
| --- | --- | --- |
| Ordinary training | LoRA, DoRA, LoHa, DoHa, LoKr | [ordinary adapter](../config/examples/train_lora_fibo2.yaml) |
| Fizgig Image Slider | LoRA or signed DoRA | [matched image targets](../config/examples/train_fizgig_image_slider_fibo2.yaml) |
| Fizgig Prompt Slider | LoRA or signed DoRA | [prompt targets and practice bank](../config/examples/train_fizgig_prompt_slider_fibo2.yaml) |
| Flow-DPO | LoRA | [preferred/rejected pairs](../config/examples/train_flow_dpo_fibo2.yaml) |
| Guidance Distillation | LoRA | [full-guidance or negative-only](../config/examples/train_guidance_distillation_fibo2.yaml) |
| Diffusion-KTO | LoRA | [independent liked/disliked images](../config/examples/train_diffusion_kto_fibo2.yaml) |
| SliderSpace | LoRA direction bank | [generated/provided/combined discovery](../config/examples/train_sliderspace_fibo2.yaml) |
| Legacy Concept Slider | Existing ordinary adapter choices | [legacy concept objective](../config/examples/train_concept_slider_fibo2.yaml) |
| Full transformer finetune | Omit `network`; unquantized base | [full finetune](../config/examples/train_full_finetune_fibo2.yaml) |

For DoHa, select LoHa and enable **DoRA Weight Decomposition (DoHa)**, or set
`network.type: loha` and `network.loha_dora: true`. Specialized modes keep their
existing adapter restrictions; ordinary LoHa/DoHa/LoKr support does not broaden
them. See [shared mode rules](cross_model_training_modes.md) and the individual
mode guides for pair matching, prompt sets, anchors, discovery and KTO replay
restrictions. Legacy Concept Slider is a distinct objective from Fizgig.

An optional `model.assistant_lora_path` is a **frozen training helper**. It is
part of the training reference/policy base, excluded from previews and trained
adapter/full-model exports. No published FIBO2 Turbo training helper has been
validated; do not select Turbo as a teacher. `model.inference_lora_path` and the
Comfy preview inference adapter are **preview-only**. They do not become the
frozen training reference. Unconditional auxiliary adapters remain unsupported
in specialized modes.

Full finetuning uses the existing backend configuration with no `network`
section; there is no new Target Type control. Use an unquantized Dev checkpoint,
`quantize: false`, a frozen text encoder and ordinary optimizer/checkpoint resume.
The supplied int8 checkpoint is not an unquantized finetuning source.

## Captions and edit references

Plain prompts and complete structured JSON captions are accepted. Caption
cleaning/serialization, edit-instruction wrapping, tapped hidden states and
glyph images are part of conditioning. Automatic VGL caption expansion is not
provided. Keep complete JSON in specific prompts, multipoint targets, anchors
or dataset captions. Prepending a simple text prefix makes JSON cease to be a
standalone structured caption; the form warns without silently rewriting it.

References supply both VLM image features and clean VAE latents. Use existing
dataset and sample reference fields, in consecutive order. Ordinary training,
Flow-DPO and Guidance Distillation permit actual edit sources. In DPO,
**Control Dataset 1 is rejected output**; Controls 2/3 are actual edit sources.
In image sliders Control Dataset 1 is the negative target, never an edit source.
Fizgig, SliderSpace and KTO retain their text-to-image restrictions. Preservation
targets have their own captions and are independent of edit sources.

Each prompt retains six conditioning bundles inside one `AdvancedPromptEmbeds`
list entry. The outer list determines the sample batch size; the bundle axis
does not. The ordinary trainer's shared prediction/CFG entry point preserves
this layout, including cached embeddings and unequal prompt lengths.
When base and glyph conditioning are identical, their live tensors can share
storage. Saving the prompt cache takes independent contiguous CPU snapshots
for every key without changing those live aliases, tensor dtypes or cache
metadata. Existing caches keep the same format.

CFG negatives are text-only VLM inputs. An empty negative is the trained JSON
null caption; an edit's positive and negative predictions share its clean
reference latents. Text-to-image glyph prompts have base/glyph conditioning
variants selected at the reference noise boundary; edit prompts keep glyph
conditioning throughout. Toolkit velocity remains `noise - clean` and toolkit
time remains 0..1000. The flow profile and native schedule belong to FIBO2;
Qwen latent-area scheduling is not reused. This toolkit flow adaptation is not
a claim about the unreleased vendor training sampler.

Changing the encoder/processor/tokenizer, selected hidden layers, fonts,
caption/presentation policy, dual conditioning or target size can invalidate
text caches. VAE identity/statistics, precision and packing/presentation policy
key latent caches. Model/helper provenance also keys discovery/resume. Preview
counts, adapter rank and loss weights do not by themselves change image/text
conditioning. Restored conditioning stays on CPU until consumed.

## ComfyUI previews

Choose **FIBO2 LoRA image / edit**:
`config/comfy_templates/fibo2_lora_sample.json.njk`. Set the existing Comfy model,
encoder and VAE selectors to files relative to their respective Comfy model
folders, for example:

```yaml
sample:
  comfy:
    enabled: true
    workflow_path: config/comfy_templates/fibo2_lora_sample.json.njk
    model: d/fibo2_dev_int8_convrot.safetensors
    text_encoder: qwen3vl_4b_bf16.safetensors
    vae: flux2-vae.safetensors
    sampler: euler
    scheduler: simple
```

The graph uses installed core `CLIPLoader` type `fibo2` and `TextEncodeFibo2`.
Its positive, negative and packed latent outputs all feed the sampler. The node
owns positive glyph noise scheduling, negative encoding and shared reference
latents; the connected VAE is also used for tiled output decode. Up to three
existing ordered sample references are wired without discarding later slots.
Native training, practice, discovery and previews use image-token-dependent
`mu` (0.5 to 1.15 over 256 to 4096 packed image tokens). The installed Comfy
renderer instead retains fixed `mu=1.15`, matching the native shift at
1024×1024. Its glyph-conditioning threshold follows that installed schedule;
these renderers are not claimed numerically equivalent at other sizes.
The only additional loader is
the existing `load_lora_from_absolute_path` node from
[envy-ai/load_lora_by_index](https://github.com/envy-ai/load_lora_by_index).
Output uses core PNG `SaveImage`, so metadata saver/format controls are not used.

The installed node does not accept a switch to disable glyph images in a
structured negative caption's nonempty `text_render` entries. Those Comfy
negative prompts are rejected explicitly; use an empty/plain negative or native
previews to retain the text-only negative contract. Edit reference images still
never enter the negative VLM input, and both branches share their VAE latents.

The training adapter keeps its requested signed strength. An optional inference
adapter is loaded after it and never replaces the training helper. Prompts are
submitted individually, including when batch submission is enabled, because
reference counts and conditioning graphs can vary. Native prompt-slider practice
and SliderSpace discovery still use the native pipeline regardless of preview
renderer selection. This integration does not modify or restart ComfyUI.

### Supplied Edit Turbo workflow

To use the supplied `tmp/FIBO2Turbo.json` graph during training, enable
**Use ComfyUI Renderer** and choose **FIBO2 Edit Turbo LoRA image / edit** in
**Sample Renderer Workflow**. The tracked template is
`config/comfy_templates/fibo2_edit_turbo_lora_sample.json.njk`. The existing
selector supports mouse selection or Tab, arrow keys and Enter.

Explicitly selecting this preset fills the preview settings from the supplied
graph: model `d/fibo2_edit_turbo_int8_convrot.safetensors`, text encoder
`d/qwen3vl_4b_instruct_heretic_7refusal_int8_convrot.safetensors`, VAE
`flux2-vae.safetensors`, four steps, CFG 1, Euler and the simple scheduler.
These values remain editable. Loading an already-saved job does not reapply
the preset. Training components, prompts, seed, sample dimensions, per-sample
overrides, reference images and adapter strengths are preserved. Switching
back to the original workflow changes the template only; select the desired
preview model and sampling values explicitly.

Equivalent configuration:

```yaml
sample:
  sample_steps: 4
  guidance_scale: 1
  comfy:
    enabled: true
    workflow_path: config/comfy_templates/fibo2_edit_turbo_lora_sample.json.njk
    model: d/fibo2_edit_turbo_int8_convrot.safetensors
    text_encoder: d/qwen3vl_4b_instruct_heretic_7refusal_int8_convrot.safetensors
    vae: flux2-vae.safetensors
    sampler: euler
    scheduler: simple
```

This variant keeps the supplied graph's `VAEDecode` and `SaveImage` output,
sharing the existing FIBO2 template's current-training-LoRA, optional inference
adapter and ordered-reference wiring. The original workflow retains tiled
decoding. The configured sample dimensions replace the source graph's zero
width/height, so the training form's chosen canvas applies. The graph's example
prompt, seed and output prefix are replaced by the current sample values.

The preset selects Turbo for **preview inference**. The training base remains
FIBO2 Dev; selecting this workflow does not turn the distilled Turbo checkpoint
into a supported training base. Native previews, practice and discovery retain
their existing Dev behavior. No new ComfyUI nodes or service restart are needed.

The installed graph contract was inspected at ComfyUI
`05a7131c19e23787a568b3a5127b78b5a4037597`. Native implementation provenance is
recorded in the FIBO2 source headers and [integration plan](fibo2_training_plan.md).
Font rendering differs between vendor-configured fonts and Comfy's built-in
font; do not interpret a render from each as numerical parity. The validation
ledger separates node/schema checks from live checkpoint and image-quality
evidence.
