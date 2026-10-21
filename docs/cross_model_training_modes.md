# Specialized flow training: Qwen 2.1, Krea 2, Anima, Ideogram 4, MOSS-Image and FIBO2

The form uses the same workflow cards for these six architectures. Newly created
jobs use `flow_dpo` and `guidance_distillation`; the earlier `qwen_flow_dpo` and
`qwen_guidance_distillation` process IDs remain valid aliases. Existing Qwen
discovery signatures and checkpoint names are retained.

The Krea, Anima, Ideogram, MOSS and FIBO2 ports are **experimental**. CPU contract tests prove
specific code paths, not image quality or memory fit at production resolutions.
Separately approved, opt-in GPU smoke tests now exercise local Qwen and Krea
checkpoints; their configurations and results are recorded in the progress ledger.
They do not change existing jobs or restart services. Anima/Ideogram full native
checkpoint coverage and complete live Krea edit-helper coverage remain outstanding.

Live Krea ConvRot-int8 Comfy checks found that attaching even a mathematically
zero LoRA changed the output. A zero-update loader correction was tested, then
rolled back at the user's request to retain normal Comfy loader behavior for
testing. The helper does not patch that loader. Approximate pixel identity is
acceptable; measured differences remain useful diagnostics rather than proof
of learned effects or native/Comfy parity. See the progress ledger.

| Mode | Data | Adapter choices |
| --- | --- | --- |
| Fizgig Image Slider | Matched +1/-1 target folders; Control Dataset 1 is the -1 target. Multipoint folders and preservation anchors are optional. | LoRA or signed DoRA |
| Fizgig Prompt Slider | Simple prefixed prompts, specific triplets or multipoint targets; optional preservation prompts. Native practice images are generated before training. | LoRA or signed DoRA |
| Flow-DPO | Preferred targets plus matching rejected images in Control Dataset 1, sharing captions and deterministic presentation. | LoRA |
| Guidance Distillation | Ordinary bucketed images and descriptive positive captions; fixed teacher CFG/reference. | LoRA |
| Diffusion-KTO (experimental) | Individually liked/disliked image folders with independent captions and buckets; no matching pairs. Includes Qwen 2.1. | LoRA |
| SliderSpace | Generated, provided or combined discovery images; repeatable recursive folders and optional captions. | LoRA direction bank |

Saved YAML prompt sets remain shared text, not architecture-specific embeddings.
Both simplified and specific/multipoint prompts, prefixes and anchors retain the
existing load/search/save/delete behavior.

Diffusion-KTO is a new experimental flow-velocity adaptation on all six models.
Use dataset feedback labels rather than paired controls; see
[the KTO guide](diffusion_kto_training.md) for its estimator/window configuration,
restrictions and four examples. Its CPU replay evidence is not actual-model
quality certification.

Flow-DPO's Dataset Loss Weight multiplies the complete per-image preference loss
and optional preferred-image SFT term, including both sequential gradient passes.
The default is 1; zero disables that example's loss. It does not change beta,
the frozen reference or the dataset repeat count.

## Model boundaries

- FIBO2 Dev: 16-pixel alignment and packed 128-channel FLUX.2 latents. Its
  image-token count is the packed latent area. The flow adaptation interpolates
  shift 0.5 to 1.15 over 256 to 4096 image tokens; this is not a claim about an
  unreleased vendor training sampler. Qwen3-VL-4B conditioning retains six
  hidden-state bundles, real-token masks and base/glyph variants. Empty CFG
  captions use trained JSON null; edit negatives remain text-only but share
  clean reference latents. Select explicit training encoder and FLUX.2 VAE
  paths. Only non-distilled Dev is a training base; Edit Turbo is inference-only.
  See [FIBO2 setup, examples and previews](fibo2_training.md) and its separate
  [validation ledger](fibo2_validation.md).
- Krea: 16-pixel bucket alignment. Training shifts use image-token counts, not
  Qwen latent area; `schedule_mu` fixes the shift, or `schedule_min_res`,
  `schedule_max_res`, `schedule_y1`, `schedule_y2` set its endpoints. The native
  wrapper converts user CFG to Krea's pipeline offset exactly once.
- Anima: 32-pixel alignment and fixed flow shift 3. Its cached conditioning keeps
  Qwen features/masks and T5 integer IDs/masks. The learned text conditioner must
  remain frozen in specialized modes (`train_text_conditioner: false`). Ordinary
  training still has its separate conditioner-training option.
- Ideogram: 16-pixel alignment. Toolkit latents are already normalized,
  patchified 128-channel BCHW tensors; the wrapper alone reverses model time and
  negates velocity. Specialized training uses fixed shift 1, while native
  practice generation keeps the resolution-aware logit-normal schedule.
- MOSS-Image Base/SFT: 16-pixel alignment, 128-channel packed FLUX.2 latents
  normalized by the loaded VAE's BatchNorm statistics, and toolkit velocity
  `noise - clean` at timesteps 0..1000. Specialized training uses fixed shift 5
  (`mu=log(5)`), matching the toolkit MOSS training scheduler. Practice/discovery
  and native previews retain MOSS's snapped shift-5 sigma grid. Positive branches
  use all blocks; CFG negative/reference branches skip block 22 by default
  (`model_kwargs.uncond_skip_layers`, zero-based). This also applies to the
  distillation teacher's negative and blank baseline branches. CFG 1 skips the
  reference prediction. LoRA training includes block 22 by default in ordinary
  and specialized modes; the native negative CFG skip does not exclude blocks
  from the adapter. Existing saved jobs that explicitly set
  `network.network_kwargs.ignore_if_contains: ["layers.22."]` must remove that
  entry to train block 22. Architecture switching preserves explicit adapter
  filters without adding a MOSS-specific exclusion.

The MOSS form offers **SFT**, **Base**, and their experimental **Edit Training**
variants. All six modes support text-to-image Base/SFT; only Flow-DPO and Guidance
Distillation support edit references. Imported `moss_image:base`,
`moss_image:edit`, and `moss_image:base_edit` tags resolve before validation, so
their Base/SFT choice and edit/isolation flags are preserved. Explicit component
paths and memory settings take precedence. Architecture switching still asks you
to select the matching base source explicitly. MOSS text conditioning uses the
Qwen3.5-9B final hidden states, with one unpadded encoder forward per prompt;
variable prompt lengths remain separate cached tensors. All four MOSS variants
expose **Text Encoder Safetensors Path** in the training form, stored as
`model.text_encoder_path`. Encoder selection checks `model.te_name_or_path`
first, then `model.text_encoder_path`, then `model.model_kwargs.text_encoder_path`,
then the default Qwen3.5-9B encoder. A compatible local Comfy-format `.safetensors`
file (including outside the ComfyUI models folder) or
`org/repo/path/file.safetensors` reference uses the existing single-file loader,
with the Qwen3.5-9B config and vendor tokenizer. Explicit files also work with
`model.model_kwargs.use_comfy_weights: false`. Edit variants require an encoder
that includes the vision tower. This is the encoder used for training and native
previews; the separate ComfyUI sample **Text Encoder** setting
(`sample.comfy.text_encoder`) selects weights for ComfyUI sample rendering.
MOSS specialized target-latent caches also include the actual resolved VAE file
identity, precision and packed/normalized layout. Unchanged VAE inputs retain
their cache keys; replacing the VAE or its BatchNorm statistics invalidates
them. The first specialized run builds these versioned target caches instead
of trusting ordinary MOSS caches that have no VAE provenance. Ordinary MOSS
jobs with an explicit text encoder path also key cached text embeddings by
that path and local file identity, so changing the encoder rebuilds them.
Other ordinary MOSS latent caches and other models keep their existing contracts.

These objectives use one frozen base with adapters disabled for reference
predictions, not a second full model. Sequential backward/recomputation, CPU
embedding banks, transformer checkpointing, quantization and offload settings
are retained. SliderSpace additionally requires a differentiable VAE decode;
the scoped decode keeps its weights resident until checkpoint backward finishes
and restores placement afterward. Ordinary inference offload/tiling is retained.

## Ideogram guidance

In the Model card's advanced **Ideogram guidance** section choose:

- **Image-only** (default): true zero-text-token CFG. Negative text fields are
  disabled, but drafts are retained. Full guidance distillation bakes this native
  guidance into the CFG-1 student.
- **Text negative** (experimental): actually encode and evaluate the supplied
  negative in practice generation, teacher/student predictions and previews.
  Endpoint negatives remain independent of the slider's -1 conditional target.

At CFG 1 the reference branch is skipped. Ideogram negative-only distillation
requires text-negative mode and nonempty teacher negative text; it learns
`v_positive + (CFG-1)*(v_image_only-v_negative)`. Other models use the encoded
blank-caption prediction in that baseline term.

Keep complete Ideogram/FIBO2 structured JSON in specific prompts, multipoint targets,
anchors or dataset captions. A simple text prefix followed by JSON is not a
standalone structured caption. The form warns; it does not silently rewrite
strings. Native and provided Ideogram Comfy templates use the existing caption
digest helper.

## Targets versus edit references

Control Dataset 1 in image sliders/DPO is a target, not an image fed into the
model as an edit source. Krea DPO/distillation may use actual edit references
with `model_kwargs.edit: true`; DPO reserves Controls 2/3 for these. Krea
Fizgig/SliderSpace are initially text-to-image and reject `edit`/`kv_cache`.
Anima and Ideogram do not expose edit-reference conditioning here. Preservation
images remain independent examples with their own captions.

MOSS edit DPO/distillation follow the same target/source separation: DPO's
Control Dataset 1 is rejected output, and Controls 2/3 are actual sources.
Every edit dataset requires fixed source images and `model_kwargs.edit: true`.
`moss_image:edit` / `moss_image:base_edit` also default `kv_cache` and
`match_target_res` to true. The upstream Base/SFT checkpoints were trained for
text-to-image; these edit variants train the toolkit's reference-conditioning
extension. References contribute both Qwen3.5 vision tokens and clean t=0 VAE
latents. Cached positive/negative embeddings and live references share the same
EXIF-aware RGB/alpha presentation. Detached reference latents are retained only
through one accumulation, including backward; transformer K/V are recomputed for
each teacher/policy pass. `kv_cache` controls reference attention isolation in
training and branch-specific K/V caching only during native inference. Fizgig,
SliderSpace and KTO reject MOSS edit/kv_cache settings explicitly. MOSS specialized
previews currently require the native renderer; Comfy previews are rejected before
model loading. No Comfy nodes are installed or patched by this integration.

Specialized modes do not block `model.inference_lora_path` or clear it when
switching modes. Inference adapters are for previews, not training references.
The selected model loader must actually support loading the adapter; an unloaded
adapter produces an explicit generation error rather than substituting a training
helper. Krea keeps its frozen inference network inactive on CPU during training,
reference predictions, and practice/discovery generation; native previews enable
it temporarily and return it to CPU afterward. Comfy previews use their separate
`sample.comfy.inference_lora` setting, or an already distilled Comfy checkpoint.
Other auxiliary/unconditional LoRA paths remain unsupported on new-model jobs,
as do known turbo/lightning/step-distilled training checkpoints. Supporting a composite teacher
or few-step teaching is separate work. Architecture changes retain prompt sets,
paired targets, anchors, ranks and memory knobs, but clear incompatible component
paths and model-specific overrides with a notice. Select the new base explicitly.

All Qwen Image 2.1 training modes support one frozen training helper at **Model →
Helper LoRA Path** (`model.assistant_lora_path`). The helper contributes to teacher,
reference and student predictions and generated Fizgig/SliderSpace practice images.
It does not train, appear in exported LoRAs, or contribute to native/Comfy previews.
The Qwen helper accepts a local file or an `owner/repo/file` Hub path. Its strict
import preserves split/fused MLP layers and per-layer ranks/scales. With layer
offloading enabled it attaches before streaming starts, keeping the transformer
on CPU during setup to avoid a temporary full-model GPU allocation.

FIBO2 Dev supports the same optional frozen-helper separation across applicable
ordinary and specialized workflows. The helper is excluded from previews and
exports; the inference adapter remains preview-only. No FIBO2 Turbo training
helper has been validated. FIBO2 ordinary training, DPO and distillation can use
actual edit sources: DPO reserves Control Dataset 1 for rejected output and
Controls 2/3 for sources. Fizgig, SliderSpace and KTO remain text-to-image.

## Preview renderers

Practice/discovery generation remains native even when previews use Comfy.
Select a compatible workflow and component files explicitly:

FIBO2 uses [`fibo2_lora_sample.json.njk`](fibo2_training.md#comfyui-previews)
with `CLIPLoader` type `fibo2` and installed `TextEncodeFibo2`. The combined
encoder node supplies positive/negative conditioning and the packed latent,
including glyph noise scheduling and shared edit reference latents. Existing
ordered sample references and signed adapter strengths are retained. Prompts
render individually; practice/discovery generation remains native.

Ordinary MOSS-Image training can use native previews or the dedicated
[`moss_image_lora_sample.json.njk` workflow](moss_image_comfy_samples.md).
It uses MOSS's Qwen3.5 encoder type, standard VAE loader, `KSampler` and PNG
`SaveImage`, with model-only inference LoRA. Its prompts render individually
even when batch submission is enabled. The bundled Krea 2 workflows are
incompatible with MOSS's encoder, sampling graph and VAE and are rejected before
sampling. Existing compatible custom workflows remain selectable for ordinary
MOSS jobs. Specialized MOSS modes still require native previews.

- Krea: `config/comfy_templates/krea2_lora_sample.json.njk` or its Easy Use batch
  variant. Specialized modes send the native, resolution-aware sigma grid through
  `ManualSigmas` / `SamplerCustomAdvanced`, including schedule overrides. At CFG
  above 1 an empty negative is actually encoded, not zeroed; at CFG 1 no negative
  encoding is requested. Choose Euler for the closest match to native practice
  generation. Ordinary text-to-image jobs keep their existing sampler and blank-negative
  behavior. Signed/fractional preview strengths are respected in both templates.
  Specialized batch previews accept full unsigned 64-bit seeds without the
  Easy Use integer widget limit or JavaScript numeric rounding.
  Edit previews upload the actual sample references and use individual requests,
  retaining up to three references per image even when batch sending is enabled.
  They require the optional helper in `comfy_nodes/ai_toolkit_krea_preview` plus
  the installed `Krea2OstrisEditModelPatch`. See the helper's README for manual
  installation when Comfy is idle; it is not installed/restarted automatically.
  Missing helper nodes fail preflight explicitly instead of dropping references.
  The helper matches configurable VLM/reference budgets, target-area matching,
  bicubic/bilinear filters, input precision and RGB alpha-background presentation,
  and shares one VAE encoding across CFG branches. Native specialized Krea
  previews also composite transparency and avoid double-counting `ctrl_img` when
  it aliases `ctrl_img_1`. Ordinary presentation remains unchanged. A common
  Dataset Control Transparency Color is inferred for previews; mixed colors
  require explicit `model_kwargs.preview_control_transparent_color: [r, g, b]`.
  Ordinary experimental Krea Edit Training models (`krea2:o_edit` and
  `krea2:o_edit_turbo`, normalized to `krea2` plus `model_kwargs.edit=true`)
  use `config/comfy_templates/krea2_ostris_edit_lora_sample.json.njk` instead:
  the same reference conditioner/patch and preprocessing, with ordinary
  `KSampler` settings from the form. This template requires one to three
  consecutive references and retains per-image settings through individual
  requests. It does not use the source graph's boost/grounding knobs or prove
  compatibility with external inference helper weights. See
  [Krea edit samples](krea2_edit_comfy_samples.md) for the two distinct edit paths.
  CPU preprocessing checks pass and the helper is installed on this machine.
  The first live execution exposed a missing `format_encoded` initializer in
  the installed VAE Utils loader; full edit-render verification is still pending
  that compatibility fix. No references were silently dropped or loader memory
  settings bypassed.
- Anima: `config/comfy_templates/anima_lora_sample.json.njk`, Qwen3-0.6B text
  encoder (Comfy auto-detects Anima from the weights), Qwen image VAE and base
  Anima diffusion model. The template uses fixed shift 3.
- Ideogram: `config/comfy_templates/ideogram4_lora_sample.json.njk`, matching
  Ideogram/Qwen3-VL encoder and Flux-family VAE. It requires `DualModelGuider`,
  `Ideogram4Scheduler`, `EmptyFlux2LatentImage` and the absolute-path LoRA loader.
  Native image-only CFG leaves the guider's negative input unconnected; a
  `ConditioningZeroOut` node is not equivalent. Text-negative mode uses real
  encoded negatives. Wrong/ignored reference policies fail explicitly.

Anima/Ideogram previews send individual prompts even if batch sending is enabled.
The provided new templates use Comfy `SaveImage` (PNG). Supplied specialized
templates are checked against the live registry before queueing: missing nodes,
required inputs, link types/output indices, file choices and widget bounds produce
actionable errors. The registry snapshot is reused within a preview group and
invalidated after image uploads; this check has a maximum 15-second timeout.
Comfy's LoadImage dropdown lists only flat input files, while its runtime accepts
subfolders too. The client accepts exact server-confirmed input-upload receipts
for LoadImage without weakening model choices or accepting arbitrary missing paths;
Comfy still validates the file at execution time.
Ordinary jobs and user-supplied custom workflows retain their existing behavior.

For a read-only audit without loading weights or queueing any images:

```bash
conda run --no-capture-output -n ai-toolkit python testing/audit_flow_comfy_registry.py
```

Override component filenames/API URL using the command's `--help` options.
`--inference-lora` also checks the optional adapter graph branches.
`--krea-control-image` checks edit graphs using an already registered Comfy
input image, without uploading or reading its pixels; the helper must be loaded.
The audit validates registration and static schemas, not file contents, model compatibility,
custom node execution/expansion, or rendered quality. Separate opt-in smoke renders
provide limited execution evidence, not quality or native/Comfy numerical parity.
Do not install dependencies or restart Comfy during training.

## Examples and validation

`config/examples/train_{mode}_{arch}.yaml` covers each of the six modes for
`krea2`, `anima`, `ideogram4`, `moss_image`, and `fibo2`. Replace all placeholder data/component paths.
Their counts/ranks/steps are engineering examples, not measured quality advice.

MOSS CPU coverage uses contracts and tiny native transformers; no full checkpoint
or GPU run is performed while the existing training job is active. Full-resolution
memory fit, learned quality, and external renderer parity remain unverified.

Both form selectors, imports and saves use a shared UI capability table, tested
against the backend table. Server saves reject unsupported architecture/adapter/
reference combinations. Running-job sample-only edits keep the existing whitelist;
they do not change discovery folders, schedules, loss objectives or ranks.

Progress and remaining verification are recorded in
`tmp/qwen21-cross-model-implementation-progress.md`. LoHa/DoHa ordinary training
has actual tiny-architecture target discovery, installed Comfy key/adapter loader
numerical checks at signed strengths, file roundtrips, and native-target ConvRot
int4/int8 checkpoint-gradient fixtures. Krea Raw rank-2 GPU training/native preview
and signed/fractional live Comfy execution smoke tests also pass. Other-model,
normal-resolution, numerical-parity and learned-quality checks remain outstanding;
specialized adapter restrictions above are unchanged.
