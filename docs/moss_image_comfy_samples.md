# MOSS-Image ComfyUI previews

MOSS-Image Base/SFT jobs, including specialized training modes, can select
**MOSS-Image LoRA image** for text-to-image previews in the existing Comfy
Workflow selector:
`config/comfy_templates/moss_image_lora_sample.json.njk`. The selector retains
its existing mouse and keyboard controls. Select component files explicitly;
the renderer rejects known incompatible bundled workflows rather than changing
an existing job's selection.

This template follows the supplied `Moss Image Simplified.json` graph. It uses
`UNETLoader`, `CLIPLoader` with type `moss_image`, standard `VAELoader`,
`EmptyLatentImage`, `KSampler`, `VAEDecode`, and core `SaveImage`. The only
additional custom node required is `load_lora_from_absolute_path` from
`envy-ai/load_lora_by_index`, which reads the current training adapter snapshot.
ComfyUI must support MOSS-Image and the selected encoder/VAE/model formats.
There is no dependency on Krea's VAE Utils, metadata saver, WAS nodes, Easy Use,
random prompt generator, date formatter, or seed helper.

| Setting | Graph input |
| --- | --- |
| `sample.comfy.model` | `UNETLoader.unet_name` |
| `sample.comfy.text_encoder` | `CLIPLoader.clip_name`, type `moss_image` |
| `sample.comfy.vae` | Standard `VAELoader.vae_name` |
| Prompt, width/height, seed, sample steps, guidance | Text encoding, latent dimensions and `KSampler` |
| `sample.comfy.sampler`, `scheduler` | `KSampler.sampler_name`, `scheduler` |
| Current training LoRA path and per-image network multiplier | Absolute-path loader and signed training strength |
| `sample.comfy.inference_lora`, `inference_lora_strength` | Optional `LoraLoaderModelOnly`, applied after the training LoRA |
| `sample.comfy.negative_prompt` | Encoded when nonempty; otherwise `ConditioningZeroOut`, as in the supplied graph |
| Sample filename | `SaveImage.filename_prefix` |

The example used `d/kromoss_int8_convrot.safetensors`, the Qwen3.5-9B encoder
`d/qwen3.5_9b_ultra_uncensored_heretic_int8_convrot.safetensors`, and
`d/flux2-vae_small.safetensors`. These are examples of ComfyUI model selectors,
not hardcoded template defaults. Qwen3-VL is a different encoder and must not
be selected for this graph. Training/native generation chooses its encoder
independently using `model.text_encoder_path` or the other documented MOSS
encoder settings; a local training file must have a valid filesystem path.
For ordinary MOSS training, changing an explicit training encoder path or its
local file invalidates cached text embeddings before the next run.

The supplied graph's optional distillation LoRA was
`d/kromoss-8-steps-dmd-lora.safetensors` at strength 1, with 8 steps, CFG 1,
sampler `dpmpp_2m_sde_gpu_sharp_fixed`, scheduler `simple`. The template reads
the job's actual values, including an explicitly configured inference strength
of zero. The inference LoRA modifies the diffusion model only. The sampled
seed is passed as an exact decimal integer through template rendering, including
values above JavaScript's safe integer range.

`SaveImage` writes PNG and its standard embedded prompt metadata. The generic
Comfy output format/quality settings do not change that node. The trainer
downloads the generated file with its returned extension. This differs from
Krea's `SaveImageWithMetaData` format/quality and JSON-sidecar behavior.

There is no bundled MOSS batch graph. With `send_prompts_as_batch: true`, the
trainer explicitly sends prompts individually, retaining each prompt's seed,
size, sampling parameters and adapter strength. Direct batch rendering of
the single-image template raises a useful error. Existing foreground/background
sampling and training offload behavior still applies.

This graph supports text-to-image previews only; edit reference images are
rejected, including when an edit-training job supplies them. Specialized MOSS
training can use this ComfyUI graph for text-to-image previews, while native
previews retain the specialized schedule and negative-branch layer skipping.
The ordinary `KSampler` graph does not reproduce those native details, so no
native/Comfy numerical parity is claimed.
The trainer validates this template against the target server's node schema
before posting a prompt; missing components or node support produce explicit
errors. This integration does not install or patch ComfyUI nodes.
