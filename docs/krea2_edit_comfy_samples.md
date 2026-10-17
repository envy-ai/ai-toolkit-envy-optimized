# Krea 2 edit samples through ComfyUI

Select **Krea 2 Edit LoRA (one reference)** under **Sample Renderer Workflow**
for an ordinary `diffusion_trainer` Krea 2 job. Its template is
`config/comfy_templates/krea2_edit_lora_sample.json.njk`, adapted from
`tmp/krea2_identity_edit.json`.

This workflow keeps the supplied graph's source-image VAE encoding, grounded
positive and negative encoders, `Krea2EditModelPatch`, `EmptySD3LatentImage`,
`KSampler`, and standard VAE decode. The optional inference helper loads before
the current training LoRA; both affect the diffusion model only. The text
encoder remains connected directly to both grounded encoders.

## Form mapping

| Form/configuration value | Graph input |
| --- | --- |
| Comfy Model (`sample.comfy.model`) | `UNETLoader.unet_name` |
| Comfy Text Encoder (`sample.comfy.text_encoder`) | `CLIPLoader.clip_name`, type `krea2` |
| Comfy VAE (`sample.comfy.vae`) | Standard `VAELoader.vae_name` |
| Sample prompt | Positive `Krea2EditGroundedEncode.prompt` |
| Negative Prompt (`sample.comfy.negative_prompt`) | Negative grounded encoder prompt, including an empty string |
| Width/Height, including per-sample overrides | `EmptySD3LatentImage` dimensions |
| Sample seed, Sample Steps, Guidance Scale | `KSampler.seed`, `steps`, `cfg` |
| Comfy Sampler / Scheduler | `KSampler.sampler_name` / `scheduler` |
| Current training checkpoint and path replacement settings | `load_lora_from_absolute_path.absolute_path` |
| Per-sample LoRA Scale (`network_multiplier`, default 1) | Training loader `lora_strength` |
| Comfy Inference LoRA and Inference LoRA Strength | Optional `LoraLoaderModelOnly.lora_name` / `strength_model` |
| Sample `ctrl_img` or `ctrl_img_1` | Uploaded source image, shared by both grounded encoders and the edit model patch |
| Output Format / Output Quality | `SaveImageWithMetaData.output_format` / `quality` |

The Comfy Inference LoRA field is a ComfyUI LoRA filename, for example
`d/krea2_general_edit_v2.safetensors`, rather than the separate model-level
**Inference LoRA Path** field. The running training adapter is saved and loaded
by absolute path; the graph's example training LoRA filename is not reused.
Its path must be readable from the ComfyUI host, using the path replacement
fields when the hosts use different filesystem paths.

Each sample requires exactly one existing source image. Use the existing sample
control-image picker or set `ctrl_img`/`ctrl_img_1` in the configuration. Missing
references, `ctrl_img_2`/`ctrl_img_3`, and distinct images set in both aliases
are rejected before sampling starts. **Send Prompts as Batch** submits individual
prompts for this workflow so every prompt retains its own reference and settings.
No images are created or attached automatically.

## Fixed graph options and form differences

The current form has no controls for the following inputs. The template keeps
the supplied values instead of adding controls:

- `Krea2EditModelPatch.ref_boost=2`, `ref_boost_a=2`, and `fit_mode="fit"`.
  The installed node documents `ref_boost_a` as having no effect with one reference.
- Positive `grounding_px=0` (native source resolution), negative
  `grounding_px=768`, and both `system_prompt=""` (the node's training default).
- `KSampler.denoise=1`, `EmptySD3LatentImage.batch_size=1`, loader
  `weight_dtype="default"`, and text encoder `device="default"`.
- Standard `VAEEncode`/`VAEDecode` with no tiled decode controls. This differs
  from the existing Krea text-to-image preview template's VAE Utils nodes.

The sample control form can expose more image slots for Krea 2; this graph
supports only the first. The original graph's `SaveImage` is replaced with the
existing `SaveImageWithMetaData` convention so the form's output format and
quality apply. Its metadata scope is `full`, with batch numbering and nearest
preference enabled, matching the existing renderer templates.

This graph does not implement the native `ManualSigmas` specialized-training
preview contract. Specialized flow trainers retain their existing Krea 2
workflow and edit conditioning; selecting this KSampler workflow for them
raises a descriptive error.

The supplied graph uses `comfyui-krea2edit`'s grounded encoding, fitted source
geometry, and reference boost. The existing native Krea edit wrapper documents
`ComfyUI-Krea2-Ostris-Edit` with reference tokens at `t=0`. These are distinct
edit paths; this preview workflow does not establish numerical or conditioning
parity with the trainer and does not change its training path.

Template selection preserves the saved form values. For example, the supplied
graph uses a turbo convrot model, a heretic convrot text encoder, 10 steps, and
CFG 1; a job saved with a raw unfiltered model, a heretic FP8 text encoder,
30 steps, and CFG 4 continues to use those form values. Reproducing the source
graph's settings requires explicitly choosing its model and sampling values.

## ComfyUI requirements

ComfyUI must provide the core Krea 2 loader/sampler/VAE nodes plus:

- `Krea2EditModelPatch` and `Krea2EditGroundedEncode` from the installed
  `comfyui-krea2edit` node pack.
- `load_lora_from_absolute_path` from `envy-ai/load_lora_by_index`, supporting
  an omitted CLIP input for model-only loading.
- `SaveImageWithMetaData` from `comfyui_image_metadata_extension`, using the
  schema expected by the existing sample templates.

This edit template does not require WAS, Easy Use, or VAE Utils nodes.
Changing the template or choosing a different inference helper affects previews;
it does not configure the model's training helper or edit-training architecture.
