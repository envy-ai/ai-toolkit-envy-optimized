# FIBO 2 training integration plan

Status: implemented with CPU/component verification and production UI build.
Full-checkpoint GPU qualification remains unverified while user jobs are active;
see [FIBO2 validation](fibo2_validation.md) for the evidence and limits. No global
dependency upgrade or ComfyUI source/service change was required.

## Acceptance and upstream boundary

Official `ostris/ai-toolkit` was fetched at
`04733873cf1da8060fce01316ec45daedbd4a918`. Its tree has no FIBO/FIBO2 training
implementation. Since the previous upstream merge at `31f4d92f`, the changes
are limited to MOSS prompt batching, lazy Qwen3.8 FLA imports, Docker FLA
installation, and version. The coordinator integrates these independently,
retaining the fork's per-prompt MOSS encoding as the default to preserve its
bounded peak memory. That integration is committed at `b8d86db5`. No requirements
pin changes are necessary for that merge. Existing CPU MOSS checks passed
(19 tests / 30 subtests), as did a fresh-process Qwen3.8 lazy FLA import check;
these are upstream-regression evidence, not FIBO execution evidence.

Add native `fibo2` training rather than aliasing FIBO1, Qwen, or MOSS. Acceptance
means every Qwen Image 2.1 workflow below has an explicit, exercised FIBO2 path;
ordinary adapters export/resume correctly; local Comfy transformer, external
text-encoder, and VAE files work; native and Comfy previews use the correct
conditioning. Existing fork features and memory behavior must remain intact.
Registration or CPU mocks alone are not evidence of full-checkpoint training,
memory fit, or learned image quality.

Training targets **Fibo 2.0 Dev**, the non-distilled base. Edit Turbo may be an
explicit inference source, but must not silently become a specialized training
teacher. No published Turbo training helper has been validated. This restriction
does not remove any existing Qwen Turbo/helper behavior.

## Pinned primary sources

Record exact revisions in vendored source headers and loader provenance, and
retain the Apache notices. Do not install an unmerged Diffusers branch globally
or import ComfyUI at toolkit runtime.

| Source | Pin and use |
| --- | --- |
| [BRIA Fibo 2.0 Dev model card](https://huggingface.co/briaai/Fibo-2.0-dev) | Architecture/component and training intent reference; resolve and record the model revision from actual loaded assets rather than assuming a moving `main`. |
| [Diffusers FIBO2 PR #14991](https://github.com/huggingface/diffusers/pull/14991) | Branch `ali/fibo2` resolved read-only to `ab96e2b58c89adb9df418af74d81cc8b1e1be074`; Apache implementation source for transformer, conditioning, image processor, and pipeline. |
| [Pinned reference transformer](https://github.com/muhammadali-afridi/diffusers/blob/ab96e2b58c89adb9df418af74d81cc8b1e1be074/src/diffusers/models/transformers/transformer_bria_fibo2.py) | Native checkpoint shape/forward contract. |
| [Pinned reference pipeline](https://github.com/muhammadali-afridi/diffusers/blob/ab96e2b58c89adb9df418af74d81cc8b1e1be074/src/diffusers/pipelines/bria_fibo2/pipeline_bria_fibo2.py) | Prompt cleaning, glyph conditioning, reference presentation, normalization, schedule, and sequential CFG. |
| [Comfy FIBO2 PR #17240](https://github.com/Comfy-Org/ComfyUI/pull/17240) | Inspected branch pin `e3699a745d513941f17ecbc9a3c58d00fc7768cb`; checkpoint/export compatibility reference only. |
| Installed ComfyUI | `/home/bart/ComfyUI`, HEAD `05a7131c19e23787a568b3a5127b78b5a4037597`; read-only execution/key-layout oracle. Its `comfy/ldm/fibo2/model.py`, `comfy/text_encoders/fibo2.py`, and `comfy_extras/nodes_fibo2.py` were inspected. Do not vendor GPL Comfy implementation code into the Apache implementation. |

The branch pins are snapshots, not claims that either PR is merged. The
implementation must compare the pinned source against installed dependency APIs
before vendoring the smallest necessary Apache modules. Resolve missing APIs
locally without changing global model behavior; escalate a required incompatible
dependency change before applying it.

## Architecture and tensor contract

The native reference defaults are 32 VAE channels with 2x2 patching, width 3584,
28 attention heads, 44 main blocks, two noise refiners, and two Perceiver layers.
Five blocks, indices 6/14/22/30/38, inject separate text bundles. Gist tokens range
from 256 to 512 in steps of 64 as text length grows from 256 to 2048. Image RoPE
planes must stay fixed independently of actual gist count. These defaults are
configuration assertions, not dimensions inferred from checkpoint filenames.

The Comfy checkpoint consumes an equivalent packed 128-channel spatial layout.
Use that layout as the toolkit's canonical transformer representation to retain
direct fused/quantized imports and Comfy adapter names. Keep patch/unpatch and
FLUX.2 BatchNorm normalization in one boundary shared by encode, decode, training,
and generation; test against the native 32-channel reference. Do not reuse MOSS
normalization/layout by assumption or apply normalization twice. Pixel bucket
alignment is 16.

The text encoder is Qwen3-VL-4B, with 15 distinct tapped hidden layers. Its six
bundles concatenate three normalized 2560-wide states each, giving 7680 features
per bundle. Preserve the exact layer selection and RMS normalization, not just
the final hidden state. Proposed `AdvancedPromptEmbeds` fields hold one tensor
per example per key: six-bundle text states, real-token mask, and optional base
states/mask without glyph images. Masks remain boolean through `.to`, cache
roundtrips, concatenation, and replay. Batch padding must not change gist count.

Prompt handling matches the trained serialization: JSON cleanup, compact output,
edit instruction wrapping, and text-render image insertion. Empty CFG input maps
to the trained JSON null caption. Edit negatives are text-only VLM inputs but
share the positive branch's clean reference latents. T2I glyph captions need two
conditioning variants selected at the reference sigma boundary; editing uses
glyph conditioning throughout. Capture only required hidden layers during TE
forward where possible, rather than retaining every layer's activations.

Reference images contribute both VLM features and clean VAE latents. Separate
reference images from rejected/negative target images in paired objectives.
Reuse EXIF/alpha presentation helpers and key the exact presentation policy.
Glyph font choice, missing-glyph behavior, image resizing, and processor grid
are part of the conditioning identity. Comfy currently uses a built-in font
where the vendor example selects fonts; this difference requires explicit
parity evidence and documented configuration, not a silent equivalence claim.

No automatic VGL expansion API/model is part of training. Accept structured
captions directly; plain prompts retain the reference's supported path. Warn
when a prefix-plus-JSON prompt ceases to be valid structured JSON, as the existing
Ideogram UI does, without rewriting user text invisibly.

## Components, paths, and checkpoint mappings

Add `extensions_built_in/diffusion_models/fibo2/` with a model wrapper, small
vendored Apache transformer/pipeline/conditioning modules, and explicit source
provenance. Register it in the existing Python extension registry, model registry,
and `extensions_built_in/diffusion_models/ui.tsx`.

Reuse configuration fields and controls:

| Configuration | Meaning |
| --- | --- |
| `model.arch: fibo2` | Unified Dev text-to-image/edit architecture. |
| `model.name_or_path` | Native component repository/directory or local/Hub Comfy transformer safetensors. |
| `model.text_encoder_path` | Explicit external encoder safetensors or compatible directory; takes precedence over defaults. Preserve an existing explicit `te_name_or_path` precedence if used by the shared loader. |
| `model.vae_path` | Explicit FLUX.2 VAE. |
| `model.extras_name_or_path` | Processor/tokenizer/config assets when a transformer file cannot supply them. |
| `model.assistant_lora_path` | Optional frozen training helper, excluded from previews and exports. |
| `model.inference_lora_path` | Preview-only adapter, separate from the training reference/helper. |
| Existing quantization/offload/cache fields | Reused without resetting user's values when switching modes. |

The coordinator found these local qualification assets; validate their headers,
resolved file identities, and supported quantization before loading:

- `/home/bart/ComfyUI/models/diffusion_models/d/fibo2_dev_int8_convrot.safetensors`
- `/home/bart/ComfyUI/models/diffusion_models/d/fibo2_edit_turbo_int8_convrot.safetensors`
- `/home/bart/ComfyUI/models/text_encoders/qwen3vl_4b_bf16.safetensors`
- `/home/bart/ComfyUI/models/vae/flux2-vae.safetensors`

Do not require downloads or move/modify these files. Additional quantized encoder
files are candidates only after format/header validation; a similarly named file
is not evidence of compatibility.

Canonical transformer keys use Comfy-style fused self-attention QKV,
cross-attention Q plus fused KV, and fused `w13`/`w2` MLPs. Comfy fused checkpoints
pad the FFN hidden width to 9728; the native reference width is 9536. Implement
independent, strict conversion for native split weights, padding only the known
zero extension. Validate every required tensor, shape, norm, and scalar; reject
unconsumed model tensors instead of partial loading. Preserve quantization
markers/scales/storage; do not dequantize or concatenate packed quantized buffers
as if they were ordinary float tensors. Test both fused and unfused input layouts.

The coordinator's read-only local Dev header probe found 1105 keys and no
safetensor metadata. Confirmed shapes include `x_embedder.weight` [3584,128],
final linear [128,3584], Perceiver queries [512,3584], fused QKV [10752,3584],
fused `w13` [19456,3584], and `w2` [3584,9728]. Its layer quantization markers
declare `int8_tensorwise`, `convrot: true`, and `convrot_groupsize: 256`.
Known `attention.comfy_attention.config` uint8 JSON selects
`comfy_kitchen_int8`; validate/consume it explicitly as execution metadata rather
than treating it as an unknown model weight. All other tensor validation remains
strict. A header probe does not establish successful import or gradient execution.

Adapters train against canonical names, including refiners and the Perceiver's
linear targets according to the existing ordinary filter contract. Export uses
the existing `diffusion_model.` prefix convention, signed DoRA/DoHa normalization,
rank/alpha metadata, and complete Comfy key consumption. Strict frozen-helper
import handles split/fused source layouts and per-layer ranks/scales; helper
factors never enter trained-adapter or full-model exports.

## Workflow and adapter parity

Enable only after the corresponding backend and UI validators agree and the
path is exercised. Preserve all current Qwen configurations and cache signatures.

| Workflow | FIBO2 implementation target | Required evidence/boundaries |
| --- | --- | --- |
| Ordinary trainer | LoRA, DoRA, LoHa, DoHa, LoKr | Actual tiny FIBO targets, quantized/checkpointed gradients, save/load/resume, signed strengths, native and Comfy adapter loading. Existing ordinary backend-only network types are unchanged; do not advertise unverified extra targets. |
| Fizgig Image Slider | LoRA / signed DoRA | Paired +1/-1 targets, multipoint folders, preservation anchors, deterministic matching and latent caches. Negative targets are not edit references. |
| Fizgig Prompt Slider | LoRA / signed DoRA | Simple/specific/multipoint prompt sets, prefixes, preservation prompts, native practice generation, resumable discovery identity. |
| Flow-DPO | LoRA | Preferred/rejected shared presentation, frozen reference, sequential gradients, dataset loss weight. Preserve target/source separation for edit training. |
| Guidance Distillation | LoRA | Full-guidance and negative-only objectives, correct blank/reference branch and independent negative text; same edit-source policy as the model wrapper. |
| Diffusion-KTO | LoRA | Independent feedback examples, class/window estimators, deterministic score/replay RNG, complete-window checkpoint invariant. Retain its existing no-edit and additional-objective restrictions. |
| SliderSpace | LoRA direction bank | Generated/provided/combined discovery, CPU feature banks, differentiable VAE decode, sequential direction training, discovery/resume identity and individual/batch exports. Retain existing batch/accumulation/objective restrictions. |
| Legacy Concept Slider | Existing ordinary adapter choices | Explicit six-bundle/dual-conditioning concatenation and guided loss. Its three/four-branch reference batching needs a bounded sequential FIBO path without changing legacy Qwen numerics. Do not equate it with Fizgig. |
| Full finetune | Existing backend no-network configuration | Unquantized trainable base, optimizer/checkpoint/RNG resume and native/Comfy full-model export. No promise of frozen-base quantization/shared immutable-weight memory savings. Not a new target-selector control. |

All six specialized modes gain FIBO2 entries in
`toolkit/training_capabilities.py` and
`ui/src/app/jobs/new/trainingCapabilities.ts`. Integrate
`toolkit/flow_training.py`, `toolkit/flow_cache_identity.py`, mode activation in
`options.tsx`/`jobConfig.ts`, and each trainer's actual prediction/bank paths.
Keep supported Qwen adapter restrictions; do not broaden LoHa/DoHa or LoKr into
specialized objectives that currently require LoRA. Test frozen helpers across
all applicable workflows rather than retaining the current Qwen-only exception.

Native edit training belongs to ordinary training and the specialized objectives
whose Qwen contract permits actual edit sources. Validate this per objective;
Fizgig/SliderSpace/KTO retain their target/source and text-to-image restrictions.
Do not add masks, extra reference slots, or unrelated editor features beyond the
existing dataset/sample reference controls merely because the vendor supports
them. Unsupported reference combinations must fail before weight allocation.

## Schedule, predictions, caching, and memory

Expose toolkit flow velocity `noise - clean` with toolkit timesteps 0..1000;
convert to model noise level 0..1 exactly once. Use FIBO's own scheduler/profile
and image-token interpretation, not Qwen's latent-area arithmetic. Read and pin
the vendor training sampler when accessible; native preview dynamic shift alone
is not proof of the vendor training distribution. If the trainer source is
unavailable, state and test the chosen flow adaptation explicitly. Keep sample
sigmas, glyph threshold, teacher, student, and replay timestep conventions aligned.

Use one frozen base with adapters disabled for reference passes. Attach frozen
helpers after quantization and before streaming, parking the base on CPU during
setup. Do not construct a second teacher transformer or cache adapted gist/KV
values across teacher/policy passes: adapters may affect those modules. Detached
VAE reference latents may be shared through one accumulation/replay scope.

Reuse component loading, CPU cache restore, atomic safetensor writes, quantized
module caching, reversible layer memory manager, gradient checkpointing, CPU
embedding banks, preview adapter/optimizer offload, tiled inference decode, and
physical-VRAM recovery gates. Preserve pinned-memory defaults and all unrelated
models' cache formats. Training residuals must be out-of-place for autograd;
Comfy inference's in-place residual/memory-compiler logic is not training code.

FIBO text-cache identity includes resolved TE/processor/tokenizer revisions and
local files, hidden-layer selection, normalization, caption cleaning/template,
font/glyph identities, dual-conditioning policy, reference ordering/presentation,
and applicable target size. Latent identity includes VAE file/statistics, dtype,
packing/normalization version, and source image presentation. Discovery/resume
also keys actual transformer provenance, flow profile, and helper identity.
Rank, preview counts, and loss weights must not unnecessarily invalidate image
latents or text embeddings. Reopened caches stay CPU-resident until consumed.

## Preview and UI integration

Reuse the current training form and shared component/path/offload controls; add
the architecture option and correct defaults rather than a parallel form.
Architecture switching keeps prompt sets, anchors, ranks, and memory knobs while
clearing incompatible component paths with the existing notice. Surface the
structured-caption limitation with the existing warning pattern.
The existing Base Model Source/Comfy checkpoint picker, Text Encoder Safetensors
Path, and Image VAE Checkpoint fields supply the approved layout; use FIBO-specific
Qwen3-VL-4B and FLUX.2 placeholders. The separate `sample.comfy.text_encoder`
selection still controls Comfy sample rendering, not the training encoder.

Native practice/discovery remains independent of preview renderer selection.
Add a dedicated `config/comfy_templates/fibo2_lora_sample.json.njk` using FIBO2
text encoding, correct model/TE/VAE files, signed adapter strength, references,
and latent output. Preserve inference LoRA separation. Start with individually
submitted prompts where graphs/reference counts vary; do not silently drop
references to satisfy a batch workflow. Validate schemas/nodes before sampling.
Do not install custom nodes, patch Comfy, or restart its server. Existing installed
FIBO support can be used read-only for qualification.

## Ordered implementation and verification

1. Coordinator completes the small upstream merge with preserved MOSS default;
   record its resulting base. Keep the FIBO change reviewable separately.
2. Add pinned architecture/conditioning code, strict checkpoint mapping, component
   selection, and model wrapper. Expose pure conversion/config helpers and small
   component-loading seams for deterministic CPU fixtures; avoid runtime test-only
   fallback paths.
3. Qualify tiny native/Comfy-layout forward equivalence, normalization, variable
   prompts/masks, glyph threshold, references and CFG before objectives/adapters.
4. Add ordinary adapter target/export/resume support and frozen helpers; exercise
   real tiny FIBO layers under float and supported ConvRot quantization/offload.
5. Integrate the six specialized trainers and legacy Concept Slider, then ordinary
   full finetune, with numerical tiny-model update/replay tests rather than only
   constructor mocks. Cover exceptions, stop cleanup, and save boundaries.
6. Add shared UI capability/defaults, dedicated Comfy previews, example YAML for
   each mode, and user documentation describing supported conditioning/settings.
7. Run affected existing Qwen/MOSS/other-model regressions; Python syntax, source
   TypeScript checks, whitespace checks, and required `ui/npm run build`. Tests
   run under the mandated 16 GB/no-swap transient systemd scope. Parent owns test
   coordination, build, commits, service activation, and existing job protection.
8. With the coordinator's training/resource authorization, validate local real
   checkpoint load, a minimal gradient/optimizer update, adapter export/resume,
   native previews and installed Comfy consumption at signed strengths. Record
   peak CPU/GPU memory and exact inputs. Reuse unaffected passes; test meaningful
   representative paths without repeatedly loading full weights for each mock
   assertion. If no GPU run occurs, explicitly leave full-checkpoint execution,
   normal-resolution fit, and learned quality unverified.

No existing optimization has yet been identified as fundamentally incompatible.
If implementation reveals one cannot be retained, stop at that concrete conflict
and ask which behavior to keep, as required by the repository instructions.
