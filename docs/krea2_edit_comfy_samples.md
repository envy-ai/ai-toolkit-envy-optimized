# Krea 2 edit samples through ComfyUI

## Experimental Edit Training models

For **Krea 2 (raw) [Edit Training]** or **Krea 2 Turbo (w/ Training Adapter)
[Edit Training]**, select **Krea 2 Experimental Edit LoRA (1–3 references)**
under **Sample Renderer Workflow**. The template is
`config/comfy_templates/krea2_ostris_edit_lora_sample.json.njk`. These model
choices run ordinary `diffusion_trainer` jobs: their UI architectures
`krea2:o_edit` / `krea2:o_edit_turbo` normalize to `krea2`, with
`model.model_kwargs.edit=true`. They are distinct from specialized flow trainers.
The model selector sets this preview template by default for newly selected edit
models. Already-saved jobs must select it explicitly; no saved sample, component,
sampler, or training setting is migrated automatically.

This template retains the source graph's `UNETLoader`, `CLIPLoader` (type
`krea2`), standard `VAELoader`/`VAEDecode`, `EmptySD3LatentImage`, `KSampler`,
model-only LoRA loaders and metadata output. Its edit conditioning instead uses
the fork's `AIToolkitKreaEditConditioning` and installed
`Krea2OstrisEditModelPatch`, matching the experimental trainer's reference-token
convention. It requires `comfy_nodes/ai_toolkit_krea_preview` to be installed in
ComfyUI; see that directory's README for manual installation while Comfy is idle.
No node installation or service restart happens automatically.

Each sample requires one to three consecutive existing reference files:
`ctrl_img`/`ctrl_img_1`, optionally `ctrl_img_2`, then `ctrl_img_3`. The first
two names are aliases for one slot and cannot specify distinct paths. Missing
references, gaps, conflicting aliases and nonexistent paths are rejected before
saving a preview checkpoint or uploading references. **Send Prompts as Batch**
uses individual requests, retaining each sample's references and settings.

Both CFG branches share one set of VAE reference encodings. The helper encodes
the configured negative prompt (including empty text) when CFG exceeds 1;
otherwise it reuses the positive conditioning. `model_kwargs.kv_cache` is passed
to the model patch (default false when absent). The node receives the model's
`vlm_max_pixels` (default `384²`), `control_image_max_pixels` (default `1024²`),
`match_target_res` (default false), sample width/height and model input dtype.
References upload as canonical RGB PNGs using the common control loader. The
background is inferred from a common Dataset Control Transparency Color, or
set explicitly with `model_kwargs.preview_control_transparent_color: [r,g,b]`;
mixed dataset colors require that explicit override. References retain their
order and use the helper's native-aligned resize/filter/precision handling.
This enables the shared RGB presentation for this Comfy preview; the ordinary
trainer's existing control-image decoding is unchanged. It does not establish
identical source presentation between the two loaders for every image format.

Prompt, negative prompt, model/component filenames, sampler, scheduler, steps,
CFG, seed, dimensions, LoRA strengths, absolute training-LoRA path and output
format/quality all come from the saved form, including per-sample overrides.
The optional **Comfy Inference LoRA** loads first; the current training adapter
loads second at `network_multiplier` (default 1). Both affect only the diffusion
model, leaving CLIP connected directly to the conditioner. **Helper LoRA Path**
under Model controls the frozen training helper and is separate from the Comfy
inference helper. Compatibility of external helper weights such as
`krea2_general_edit_v2.safetensors` with this experimental edit path has not been
established by selecting or rendering this template.

The source graph's `ref_boost`, `ref_boost_a`, `fit_mode`, `grounding_px` and
custom `system_prompt` knobs do not apply to this experimental template.
Preprocessing alignment does not establish numerical or image-quality parity
between native training and Comfy inference. This ordinary workflow uses
`KSampler`; specialized flow trainers retain their existing native
`ManualSigmas` / `SamplerCustomAdvanced` template and reject this one.

## Source graph edit workflow

Select **Krea 2 Edit LoRA (one reference)** for the separate source graph edit
path in an ordinary `diffusion_trainer` Krea 2 job. The tracked
[`krea2_edit` extension](../extensions/krea2_edit/README.md) provides the matching
training contract and appears as **Krea 2 Identity/Edit (krea2edit extension)** in the
form. Its template is
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
  For `krea2_edit`, `model_kwargs.fit_refs: false` instead selects
  `fit_mode="crop (legacy)"` to retain that explicit training geometry.
- Positive `grounding_px=0` (native source resolution), negative
  `grounding_px=768`, and both `system_prompt=""` (the node's training default).
  The `krea2_edit` training recipe jitters grounding from 384 to 768;
  native-resolution positive grounding preserves the source graph's inference
  choice and is outside that training range for larger sources. Likewise,
  `ref_boost=2` biases inference attention toward references, whereas training
  uses unboosted attention (equivalent to 1). These inference choices do not
  establish an identical training forward pass.
- `KSampler.denoise=1`, `EmptySD3LatentImage.batch_size=1`, loader
  `weight_dtype="default"`, and text encoder `device="default"`.
- Standard `VAEEncode`/`VAEDecode` with no tiled decode controls. This differs
  from the existing Krea text-to-image preview template's VAE Utils nodes.

The sample control form can expose more image slots for Krea 2; this source graph
supports only the first. The original graph's `SaveImage` is replaced with the
existing `SaveImageWithMetaData` convention so the form's output format and
quality apply. Its metadata scope is `full`, with batch numbering and nearest
preference enabled, matching the existing renderer templates.

This graph does not implement the native `ManualSigmas` specialized-training
preview contract. Specialized flow trainers retain their existing Krea 2
workflow and edit conditioning; selecting this KSampler workflow for them
raises a descriptive error.

The supplied graph uses `comfyui-krea2edit`'s grounded encoding, fitted source
geometry, and reference boost. The `krea2_edit` extension retains the upstream
trainer's corresponding grounding template, reference-before-target token order,
frame indices, and fitted fractional positions. It reuses this fork's model
loaders and memory options. Its grounding cap/jitter defaults are 768/384; cached
text embeddings freeze a single grounding scale and have a distinct cache
identity. Training and inference use different resize kernels, so geometry
agreement does not establish numerical or image-quality parity.

The existing native Krea edit wrapper documents
`ComfyUI-Krea2-Ostris-Edit` with reference tokens at `t=0`. These are distinct
edit paths. Choosing this preview for built-in `krea2` does not change its training
path. To train the source graph contract, select `arch: krea2_edit`, remove the
built-in `model_kwargs.edit` flag and start fresh weights. Do not resume a LoRA
trained under the earlier experimental contract.

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
