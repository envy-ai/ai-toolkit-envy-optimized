# HunyuanImage 3.0 LoRA training implementation plan

Implementation status and measured support are tracked separately in
[hunyuan_image_3_validation.md](hunyuan_image_3_validation.md). This plan is
the original acceptance specification; it does not assert every matrix case
has been qualified.

Implement LoRA training for **HunyuanImage 3.0 Instruct, non-Distil**, covering text-to-image and editing with one to three ordered reference images. Use the existing diffusion trainer, model component loading policy, quantization backends, CPU offloader, dataset caches, LoRA network, and sampling lifecycle. Include **Base text-to-image** in the initial implementation through variant metadata and a thin holder subclass. Base does not gain edit support merely because it shares the architecture.

The primary deployment target is one 24GB GPU with approximately 128GB system RAM. Both PedroMarinhoDev's **int8 ConvRot and W4A8 checkpoints** are required deliverables. Int8 is the first integration milestone; W4A8 is part of completion, not an optional future optimization. Original Tencent BF16 shards are supported as a source for incremental quantization and reference comparisons, without requiring the entire BF16 model in RAM or VRAM.

**Implementation priority update:** the user has requested focusing on training from Comfy checkpoints. Start real-model qualification with the already installed Instruct W4A8 checkpoint, VAE, and vision encoder, then the additional Comfy formats. Preserve the shared Base architecture and incremental original-source loading design, but prioritize the working Comfy Instruct training/edit path over original BF16 and Base checkpoint qualification. Download additional checkpoints only as needed, under `/home/bart/models`, as explicitly authorized by the user.

This is an implementation specification, not a claim that complete-model training already fits. A single real-sized, random-weight MoE block completed backward on the local RTX 4090 through the existing offloader at 2.32GiB peak allocated VRAM with checkpointing. That diagnostic excludes attention, adapters, optimizer state, and the remaining blocks. Full training and quality acceptance are defined below.

## Scope and defaults

| Variant | Text-to-image LoRA | Edit LoRA | Initial targets |
| --- | --- | --- | --- |
| Instruct | Required | Required, one to three references | Attention; optional shared MLPs |
| Base | Included in the same implementation | Unsupported | Same targets |
| Instruct-Distil | Rejected during preflight | Rejected | None |

Start with rank 16, microbatch one, BF16 compute, frozen base weights, full transformer weight offloading, per-block gradient checkpointing, cached target latents, and cached conditioning. Validate 512px before 768px and 1024px. Support gradient accumulation through the existing trainer. Initial examples use the ordinary flow-matching loss and exclude VAE and vision-tower training.

Run model-specific preflight before checkpoint downloads: require LoRA and the supported target preset, reject full fine-tuning and text/vision/VAE training, validate edit inputs against the variant, and report incompatible quantization requests. Keep imports lazy so selecting another model does not load Hunyuan dependencies. Establish compatible versions inside the existing `ai-toolkit` environment; the isolated MoE diagnostic used PyTorch 2.7.0, which does not by itself validate the complete attention and quantized runtime. Do not adopt FlashInfer inference kernels as a training dependency without a verified backward path.

Routed experts remain frozen in the initial supported LoRA presets. The internal representation should allow later per-expert adapters without another weight-format migration. Reasoning/text-loss training, online prompt rewriting, MeanFlow distillation, and specialized DPO/KTO/slider objectives are outside this implementation. Ordinary image training need not produce vocabulary logits or load the separate CoT output head.

## Existing infrastructure to reuse

Paths are relative to the repository root. Proposed new files appear later.

| Concern | Existing implementation | Integration approach |
| --- | --- | --- |
| Training loop and resume | `extensions_built_in/sd_trainer/SDTrainer.py`, `jobs/process/BaseSDTrainProcess.py` | Keep optimizer, accumulation, loss reporting, checkpoints, and job lifecycle here; add a model holder rather than a new trainer. |
| Model interface | `toolkit/models/base_model.py` | Implement prompt conditioning, velocity prediction, VAE encode/decode, loss target, sampling, and LoRA conversion hooks. |
| Component loading | `toolkit/models/v2/_mixin.py`, `resolver.py`, `pool.py` | Implement the model-specific config and checkpoint mapping hooks; use `component_load_kwargs()` and `aitk_post_load()`. |
| Quantized linears | `toolkit/util/ostris_quant.py`, `convrot_quant.py`, `quantize.py` | Reuse `OstrisLinear` and ConvRot int8; add a distinct backend for the published W4A8 storage format. |
| Quantized import | `toolkit/util/comfy_quant_import.py` | Reuse ordinary 2D imports after model-specific expert-bank expansion; extend it for W4A8 metadata. |
| CPU offload | `toolkit/memory_management/manager.py`, `manager_modules.py` | Reuse linear and quantized-buffer placement, streams, and backward mechanisms. Extend only where measured lifetime or scheduling problems require it. |
| LoRA | `toolkit/lora_special.py`, `network_mixins.py` | Ordinary linear adapters and existing target filters; no new adapter implementation for the default presets. |
| Edit data and caches | `toolkit/data_loader.py`, `dataloader_mixins.py`, `data_transfer_object/data_loader.py` | Existing target/control pairing, processed-reference cache identity, dropout, and bucket handling. |
| Conditioning payload | `toolkit/advanced_prompt_embeds.py` | Carry token IDs, packed reference features, offsets, and grids in its existing tensor container and safetensors cache. |
| Edit integration examples | `extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py`, `qwen_image/qwen_image_edit_plus.py` | Reuse control-image normalization and holder lifecycle patterns, while implementing Hunyuan-specific sequence semantics. |
| Sampling and Comfy previews | `toolkit/comfy_sample.py`, `comfy_lora.py`, holder sampling hooks | Native sampler first; reuse the existing Comfy preview client and export hooks after adapter compatibility is verified. |
| Registration and UI | `extensions_built_in/diffusion_models/__init__.py`, `ui.tsx`, `toolkit/models/registry.py` | Two model entries backed by one implementation. |

Do not register these models in `FLOW_TRAINING_MODELS` merely to enable ordinary LoRA training: that registry in `toolkit/training_capabilities.py` governs specialized objectives, which need separate validation.

## Required reuse of existing memory optimizations

The following inventory is part of the implementation requirements. Adopt applicable existing mechanisms before the first complete 24GB training qualification, rather than postponing them until an OOM. **Required** means use and verify the mechanism for both Instruct and Base. **Conditional** means preserve it and exercise it whenever the named feature is enabled; it does not expand the initial training scope. An existing helper counts as reused only when the Hunyuan execution path actually reaches it.

| ID | Existing optimization and source | Hunyuan requirement |
| --- | --- | --- |
| M01 | Block-streamed quantization and per-layer extras: `toolkit/util/quantize.py`, v2 `_mixin.py` | **Required.** Quantize repeated blocks and non-block projections incrementally on the chosen quantization device. Keep the 24GB path CPU-resident between bounded operations. Preserve same-qtype no-op detection and skip transfers of non-quantizable leaves. Never move the entire unquantized remainder to GPU. Retain `keep_on_device` behavior only for configurations whose final resident model actually fits. |
| M02 | Chunked ConvRot conversion: `toolkit/util/convrot_quant.py` | **Required.** Keep the int8 stored-dtype input path and bounded row-wise float32 rotation/quantization scratch, including the existing approximately 256MB chunk policy. Give new W4A8 conversion equally bounded scratch. Do not make a model-sized or expert-bank-sized FP32 copy. |
| M03 | Quantized checkpoint and module-cache reuse: v2 `_mixin.py`, `toolkit/util/ostris_quant.py`, BaseModel cache hooks | **Required.** Cache hits bypass original checkpoint reconstruction and repeat quantization. Preserve packed codes and scales, source identity, and backend version. A prequantized load must not briefly create a BF16 master model. |
| M04 | Pinned CPU weights and per-device transfer rings: `toolkit/memory_management/manager_modules.py` | **Required.** Reuse asynchronous H2D, backward weight re-fetch, per-slot ready/free events, and the existing deeper ring controlled by `AI_TOOLKIT_OFFLOAD_DEPTH` (currently default four). Preserve device guards, `record_stream`, and allocation lifetime through backward. Tune depth against the complete peak; do not replace the ring with a new synchronous offloader. Whole-layer lookahead is a separate possible extension. |
| M05 | Correct quantized storage movement and pinning: `manager_modules.py` | **Required for supported storage types.** Move/pin actual packed buffers and scales, including inner tensor stores when wrappers are involved. Preserve parameter-replacement handling where `.data` assignment would leave underlying storage on GPU. Prevent repeat pinning and duplicate pinned copies when attaching, reloading, or sampling. |
| M06 | Packed-weight autograd: `convrot_quant.py`, `OstrisLinearLayerMemoryManager` | **Required.** Use quantized forward operations and save packed state rather than full dequantized weights or unnecessary linear inputs. Reconstruct only the projection needed for input-gradient computation. Preserve these properties in W4A8 backward and verify checkpointing bounds saved GPU state, not just module placement. |
| M07 | CPU-side embedding gathers: `EmbeddingLayerMemoryManager` | **Required.** Keep the large frozen token table on CPU and transfer looked-up rows only. Do not pin or upload the whole table on each forward. Preserve embedding scaling and tied-weight exclusions if a supported source uses them. Hunyuan must perform lookup for actual text/special-token positions without creating a second vocabulary table. |
| M08 | Managed-device reporting and resident exceptions: `manager.py`, `toolkit/util/mixed_precision.py` | **Required.** Preserve the compute-device property on offloaded modules, device-correct activations, resident norm/position/modulation exceptions, and dtype-preserving hooks where the checkpoint mixes precision. Reuse per-operation casts where needed instead of promoting the full residual stream. Avoid parent `.to()` calls that defeat placement or precision ownership. |
| M09 | Block checkpointing and gradient-compatible compilation: existing model decoder loops, `manager_modules.py` | **Required.** Implement real non-reentrant block checkpoints and replay-safe state. Keep transfer streams, temporary buffer swaps, and `record_stream` outside Dynamo tracing as the current quantized manager does. Compilation is optional and starts with `fullgraph=False`; the eager path remains supported and measured. |
| M10 | Phase-specific device presets and optimizer offload: `BaseModel`, `SDTrainer._run_with_optimizer_state_offload`, `BaseSDTrainProcess` | **Required.** During conditioning/cache preparation activate only the needed frozen components, and park optimizer state before on-demand positive/negative prompt or reference encoding. Restore in `finally`. Preserve CPU-offload-before-GPU-restore ordering so old and new phases do not overlap unnecessarily. Cached-latent runs must not have Accelerator prepare or promote an otherwise unused VAE. |
| M11 | Actual encoder destruction and host-cache release: `toolkit/unloader.py`, `MemoryManager.free`, `release_cached_memory` | **Required.** After frozen conditioning is cached, release its tensors even if a pipeline/reference still points to the component. Use direct-to-meta destruction for components being discarded, avoiding a final full CPU copy, then release transfer-ring references, pinned-host allocator cache, and reclaimable host arenas through existing helpers. Keep tokenizer/processor metadata independently available. Never destructively free the shared backbone, live training adapters, or a component still needed by a graph. |
| M12 | Avoid fork-retained model RAM: `get_safe_dataloader_num_workers` in `toolkit/data_loader.py` | **Required.** Extend the MiniMax H3 protection to these two architectures, preferably via a narrowly scoped model capability. Use `num_workers: 0` for the 24GB profile so workers cannot inherit and retain tens of GiB of model/quantization state. Keep dataset `pin_memory: false` by default; offloader weight pinning has a separate purpose and remains enabled. |
| M13 | Detached conditioning and disk-backed caches: `dataloader_mixins.py`, `AdvancedPromptEmbeds`, Krea reference-context pattern in `toolkit/flow_training.py` | **Required.** Keep dataset features detached and CPU/disk-backed, moving only the active batch to GPU. Encode a frozen reference once for the needed scope and reuse it for prompt branches/recomputation; preserve its lifetime through backward. Share only frozen image features, never adapted backbone K/V. Avoid retaining every decoded image, reference activation, or batch on GPU. |
| M14 | Tiled VAE processing and sequential image processing: Qwen/Krea holder paths and H3 VAE implementations | **Required for no-grad preprocessing and native sampling.** Use Hunyuan-compatible spatial tiling/slicing with bounded tiles where needed, process targets/references and sample outputs sequentially, and release encoder/decoder scratch and internal caches after each phase. Verify Hunyuan's overlap, normalization, precision, and image quality rather than copying another VAE's tile constants or temporal rules. |
| M15 | Comfy sampling handoff: `BaseSDTrainProcess`, `toolkit/comfy_sample.py` | **Required whenever Comfy sampling is enabled.** Release Comfy VRAM before trainer loading; offload optimizer state and training/assistant/accuracy-recovery adapters as well as the backbone before rendering. Reuse cache-preserving VRAM release when host memory permits and wait for both allocator and driver-level physical VRAM release before training resumes. Restore even after failures. Two CPU-cached 44–76GiB backbones can exceed available RAM, so include host-budget-aware unloading/reloading using existing client capabilities; GPU release alone is insufficient here. |
| M16 | Adapter/offloader composition and factorized execution: `toolkit/lora_special.py`, `network_mixins.py`, memory-manager attach/detach | **Required.** Preserve outer LoRA forward wrappers when attaching, detaching, or restoring offloading. Apply LoRA through its low-rank factors rather than constructing dense deltas during each step or preview. Inactive helper adapters stay parked if such support is enabled later. Keep the existing synchronization before CPU consumers read any asynchronously transferred gradients. |
| M17 | Bounded serialization: quantization-cache paths, `toolkit/util/streamed_safetensors.py`, adapter save hooks | **Required.** Normal checkpoints save adapters and optimizer state without materializing the base state dict. Quantized-cache writes preserve packed storage and bound additional copies. Reuse the sequential writer where appropriate, but account for its existing eager CPU-contiguous tensor collection: sequential disk writes alone do not make serialization memory-bounded. Extend that path only if profiling shows a full-copy peak. |
| M18 | Differentiable VAE checkpointing and activation offload: `toolkit/util/qwen_vae_gradient_checkpointing.py`, `SDTrainer` image-space loss paths, `training_decode_context` | **Conditional on image-space losses or another differentiable decode.** Preserve the existing non-reentrant, cache-safe VAE checkpoint pattern, `save_on_cpu` activation-offload scopes, bounded crop/patch selection, timestep eligibility, and VAE lifetime through backward. Implement Hunyuan-specific module hooks; the Qwen monkey patch itself does not apply. Keep no-grad tiled sampling separate from the validated differentiable decode path. |

The CPU-backed Nucleus MoE implementation in `toolkit/util/nucleus_moe_quant.py` is an additional reference for selected-expert and bounded expert-chunk processing. Preserve the same principle in Hunyuan, but do not copy its different weight layout, quantization scheme, or assumptions about expert input grouping. The canonical 2D expert modules allow the stronger shared linear-offload path to remain the initial implementation.

Some existing optimizations do not apply to this target: integrated-GPU unified-memory patches do not help the discrete RTX 4090; video temporal chunking and model-trained sparse attention policies cannot be transplanted into this image model; DoRA/LoHa-specific norm and factor optimizations are conditional on adding those adapter types. Preserve those shared paths without enabling them for Hunyuan by analogy. Inference-only KV/prediction caches remain excluded from training.

Use an implementation checklist keyed by M01–M18. Record the concrete call path and measured/tested evidence for each required item; for conditional items record the enabling feature and its validation status. Missing wiring is a port defect even if a reduced-size test happens to fit.

## Shared architecture and variant contract

Create one PyTorch image-generation backbone with canonical Tencent module names and a small immutable variant specification. Prefer the published Tencent model definitions as the semantic reference; isolate the image-generation path from its inference orchestration. Keep imported source notices and record the source revision. Do not depend on a running ComfyUI installation for training or change installed external packages.

Use two thin holders, proposed architecture names `hunyuan_image_3_instruct` and `hunyuan_image_3_base`, inheriting shared behavior. Their specifications select checkpoint/config/tokenizer sources, sequence template, system prompt, sampling defaults, maximum sequence length, and edit capability. Avoid scattering variant tests through attention, routing, and quantization code.

The published generation configurations select `instruct` / `en_unified` / CFG 2.5 for Instruct, and `pretrain` / no system prompt / CFG 5.0 for Base. Both specify 50 sampling steps and flow shift 3.0. Load these values from the pinned variant configuration rather than borrowing Distil defaults. Validate the effective scheduler against Tencent's code. [Instruct configuration](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct/blob/main/generation_config.json), [Base configuration](https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/generation_config.json).

The initial training prompt path uses direct image generation. Build the checkpoint's valid `image` task sequence, including required system and answer delimiters; do not call its default online `think_recaption` generator. Text-to-image and edit sampling use the same sequence builder as training.

Preflight must distinguish variant identity from tensor shapes. Base and Instruct share key layouts, so shapes cannot identify them. For known Hub sources, use repository identity; for repacks, use metadata or verified known fingerprints. An explicit holder variant resolves unidentified local fine-tunes, while conflicts with known metadata fail clearly. Reject Distil from its identity and guidance/interval embedder keys, before expensive allocation. Record the chosen variant in adapters and resume metadata.

Expose experts internally as `model.layers.N.mlp.experts.E.gate_and_up_proj` and `down_proj`, each an ordinary `nn.Linear` or `OstrisLinear`. Keep the fused QKV layout and Tencent SwiGLU ordering. This gives the existing quantizer, memory manager, and LoRA targeting real 2D modules to operate on. Preserve router arithmetic and selected-weight normalization; frozen routers still participate in gradients with respect to their input activations.

Do not wrap frozen experts in `no_grad`: adapters in earlier blocks require gradients through those experts. Use differentiable PyTorch routing, gather, expert computation, and combination. Start with the eager routed path, skipping empty experts where numerically equivalent. Grouped kernels and lookahead are later performance work within the same contracts.

## Model holder and conditioning

Implement the standard holder methods: `load_model`, `get_train_scheduler`, `get_bucket_divisibility`, `get_prompt_embeds`, `condition_noisy_latents`, `get_noise_prediction`, `get_loss_target`, `encode_images`, `decode_latents`, `get_generation_pipeline`, and `generate_single_image`. Set `is_flow_matching` and `is_transformer` and expose the repeated `model.layers` list for quantization and LoRA selection.

Hunyuan's language and image computation share the trainable backbone. There is no independent large text encoder whose final hidden states can be cached while backbone adapters change. Cache token IDs and frozen image features; recompute backbone embeddings and all adapted blocks on every training forward.

Represent conditioning with `AdvancedPromptEmbeds`. Each key contains exactly one tensor per batch item. Concatenate variable-size reference data into tensors and store offsets and shapes separately; do not put nested lists or Python slice objects into this container.

| Payload | Representation |
| --- | --- |
| Prompt/template token IDs | Integer tensor; assemble target image slots from the actual target bucket. |
| Reference VAE features | Packed latent cells, reference offsets, and latent grids. |
| Reference vision features | Packed frozen SigLIP2 features, offsets, patch grids, and any validity masks needed by NaFlex. |
| Sequence geometry | Integer ranges/types and position metadata, or deterministic reconstruction from IDs and grids. |
| Empty references | Consistent empty tensors under the same keys as edit examples. |

Put every integer and Boolean payload key in `frozen_dtype_keys`. Exercise cache save/load, `.to(dtype=...)`, concatenation, splitting, and classifier-free guidance explicitly: token IDs near 128k must never pass through BF16.

Provide a small conditioning component compatible with the existing text-encoder lifecycle. It owns the tokenizer/processor and, for edit preprocessing, frozen SigLIP2. It must not own or register the shared backbone. Give text-to-image operation a valid lightweight component even when the vision tower is not loaded. Reuse the existing caching/unloading calls; if a lifecycle assumption needs adjustment, add a narrow optional-component hook with existing-model regression coverage.

Keep VAE ownership on the holder, without registering the same VAE again inside the conditioning component. Preprocessing may call the holder's VAE encoder while the denoiser is parked. Cache frozen vision outputs before the aligner, so the backbone remains the owner of `vision_aligner`; keep the aligner frozen in the initial target presets.

Use the exact Hunyuan image VAE, wrapped as a v2 component. Its 32-channel latents and spatial factor 16 do not make another model's VAE interchangeable. Preserve normalization, scale, temporal singleton handling, encode precision, and decode behavior. Derive bucket divisibility from this variant's actual patch size; the inspected checkpoint uses patch size one. Validate the split Comfy VAE and vision files against their Tencent equivalents.

## Training objective and attention

For target latent `z`, noise `epsilon`, and noise fraction `sigma`, use the toolkit flow convention:

```text
z_sigma = (1 - sigma) * z + sigma * epsilon
target_velocity = epsilon - z
model_timestep = 1000 * sigma
loss = existing trainer reduction of (predicted_velocity - target_velocity)^2
```

Pass the trainer's actual noise fraction to the model. Do not divide by 1000 a second time, reverse the velocity, or shift an already shifted timestep. Keep noise construction and target calculation in the existing flow-training path. Verify endpoint conventions and an Euler step against the published scheduler before enabling training. [Tencent model and scheduler](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct/tree/main).

`get_noise_prediction` inserts the noisy target and clean conditioning into the joint sequence, runs the shared backbone, and returns only target-image velocity with the same shape as the target latents. Reference images get the model's clean-image timestep, zero. They receive neither target noise nor target loss. Caption-only dropout preserves edit references; full reference dropout would be a separate explicit policy.

Reproduce causal text attention, each conditioning image's joint VAE/vision attention region, target image bidirectional attention, and the exact image positions. Prefix/reference tokens must not see future target tokens. The target can attend to its valid prefix and references. Disable KV caches, Taylor/Spectrum approximations, and cross-step hidden-state reuse during training: adapted weights change and their gradients must span the complete forward.

Use PyTorch SDPA with the correct mask and verify the selected backward-capable backend. Do not assume that specifying SDPA prevents an expensive math fallback. Share compact geometry and masks across blocks; never materialize attention probabilities with shape `[batch, heads, sequence, sequence]` for diagnostics in the production path. Profile long edit sequences separately.

## Quantized checkpoint loading

Support these load paths under the v2 resolver and post-load policy:

| Source | Required behavior |
| --- | --- |
| Pedro int8 ConvRot repack | Import stored codes/scales/rotation exactly, including expert-bank slices; no full dequantization. |
| Pedro W4A8 repack | Import packed codes, group and channel scales, codebook, rotation, and supported correction metadata without changing the quantization. |
| Local toolkit quantization cache | Restore the backend, buffers, original variant, and source identity. |
| Tencent BF16 shards or BF16 repack | Instantiate on `meta`, read incrementally, quantize one projection or bounded block at a time, and release source storage promptly. |

Implement model-specific mapping from Comfy's stacked `[expert, out, in]` tensors to canonical individual experts. Slice matching quantization metadata along the expert dimension; distinguish bank-global codebooks from per-expert tables by validated shapes and metadata. Retain ordinary dense keys unchanged. Reject missing, duplicate, inconsistent, and unconsumed tensors, except an explicit allowlist for separately loaded VAE/vision/text-head components.

Use `safe_open`/mmap views and the shard index to avoid constructing a second complete state dict or stacking the full model. Extend the mixin with a narrow streaming hook only if the existing class-specific loading hooks cannot provide this. Bypass the whole-model `from_pretrained` allocation for 80B source loading. Account for pinned copies, mapped file pages, temporary float tensors, and conversion outputs when measuring host memory; keep only one primary CPU representation of each frozen weight.

For fresh quantization, use the existing exclusion hook to preserve sensitive router, normalization, embedding, and conditioning/output modules at their verified precision. Define the actual name patterns from the model inventory and compare them with the repack's quantized-module inventory. For imported checkpoints, preserve the shipped mixed precision and quantization boundaries; do not replace router precision or quantize additional tensors as an incidental load step.

The current importer handles float8, int8, and NVFP4 markers on 2D modules; it does not decode `asym_w4a8_int8`. Introduce a distinct proposed backend identifier, **`comfy_w4a8`**, in `OstrisQuantizer` resolution and the import path. Pedro's codebook-based W4A8 is not interchangeable with existing `convrot4`, `orbit4`, NVFP4, or Comfy W4A4 formats. Reuse rotation and autograd utilities only where their contracts match. [Published W4A8 converter](https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3/blob/main/tools/convert_w4a8.py).

First implement a correct per-projection dequantizer and reference linear. Then implement the production quantized forward plus a registered backward for input gradients. Match activation quantization in forward; use a documented straight-through gradient through activation quantization and the reconstructed frozen weight, including the rotation chain. Compare backward to that surrogate definition, not a numerical derivative of integer rounding. A W4-weight/BF16-activation fallback is useful for diagnostics, but must be labeled as different compute behavior and cannot pass as equivalent W4A8 validation.

The v2 post-load policy currently preserves shipped quantization only when the requested qtype matches a shipped backend. Ensure `comfy_w4a8` is registered and recognized before invoking that policy. A request to change formats must either use bounded incremental conversion or fail with a clear supported alternative; it must never expand all 80B parameters silently. Add precision ranking for W4A8 without altering existing model preferences unexpectedly. Use existing local files in place, and select the correct Instruct or Base repo explicitly.

Quantization-cache identity must include source revision/fingerprint, variant, backend and format version, relevant quantization parameters, and conversion mapping version. Reuse the existing cache machinery; avoid another model-only cache subsystem.

## Backward memory and the 24GB target

Use non-reentrant per-decoder-block checkpointing. It supports gradients from adapters even when incoming activations originate from frozen modules. The published model advertises checkpointing but its inspected decoder loop does not call checkpoint; implement and test the actual calls rather than trusting the flag.

Keep packed base weights on CPU and small trainable adapters on GPU. Reuse the existing offloader for canonical linears. Audit quantized autograd carefully: `OstrisLinearLayerMemoryManager` temporarily stages GPU buffers, and current quantized operators save those buffers for backward. Returning module attributes to CPU does not free tensors still referenced by autograd. Checkpointing must bound this retention; if profiling shows otherwise, add CPU-backed packed-weight handles and backward re-fetch in the shared quantized offload path.

Checkpoint recomputation must reproduce routing and operations. Do not cache expert selections between optimizer steps, detach selected router probabilities, prune experts, or drop tokens to force a memory pass. Async transfer buffers must remain valid until all consuming forward/backward operations complete. Preserve the existing deeper transfer ring and event scheduling from the outset; add whole-layer or additional expert lookahead only after gradient and lifetime tests pass.

Memory policy:

- Cache target latents and frozen reference features before training; release VAE and vision GPU storage for optimizer steps.
- Avoid loading the CoT head or creating vocabulary logits.
- Bound weight staging to the current work plus a small prefetch window; never dequantize a complete multi-layer expert collection.
- Keep references at an explicit pixel/token budget, with separate target and reference limits. Do not silently shrink references after an OOM.
- Count prompt, target, reference VAE, reference vision, and special tokens before allocation; enforce the variant's configured maximum sequence length.
- Run native sample CFG branches sequentially when needed and restore training placement afterward. Sampling must not leave a second resident backbone.
- Use the existing trainer's accumulation, gradient clipping, optimizer state, logging, and resume handling.

Aim for a measured process footprint below approximately 22GiB on the 24GB card, leaving room for the display and allocator variability. This is an engineering budget, not a predicted measurement. Record allocated, reserved, and total process VRAM; host RSS/pinned memory; warm seconds per optimizer step; and reference/token counts. Acceptance must include loading, preprocessing, optimizer state initialization, sampling, and resume, not just forward allocation.

## Edit datasets and cache correctness

Use the existing dataset convention: `folder_path` contains edited targets and their instruction captions; `control_path` contains an ordered list of matching source-image folders. Reference order is semantic, including phrases such as “image one.” Preserve deliberately repeated references. Normalize the legacy `control_path`/`control_path_1` aliases without deduplicating distinct reference slots by tensor equality.

Enable the existing multiple-control and control-aware prompt-cache hooks. Prefer aspect-preserving Hunyuan reference preprocessing from raw images for the initial implementation, with an explicit reference-area cap. Produce both VAE and SigLIP2 inputs from the same defined reference presentation, using each encoder's required resize/normalization. Do not reuse Qwen-specific latent packing or vision slot counts.

For spatially aligned source/target editing, coordinate any crop/flip policy across the pair. Default examples disable random flips and independently random crops, which can invalidate edit instructions. If a user enables augmentations, cache only the deterministic presented version or include the exact transformation in the cache identity. The VAE features, vision features, and sequence geometry must describe the same presentation.

Use `AdvancedPromptEmbeds` and existing per-item cache plumbing to store frozen references. The cache key/version must include ordered reference identities, the instruction/template, variant, tokenizer and encoder identities, reference resizing/crop/alpha policy, target dimensions when token geometry depends on them, VAE scale/precision, and latent sampling policy/seed. Include actual source changes, not only basenames. Reuse the existing processed-control and target-size identity hooks and extend their version payload only as needed.

Build blank-caption edit conditioning with the same references and matching geometry. A generic empty T2I payload is not an edit dropout payload. Test both cached and uncached execution. Do not fabricate zero-image references for missing files: reject incomplete edit pairs before loading model weights.

Support zero to three references in the shared forward contract from the outset. Microbatch one handles different reference sizes and counts without padding multiple long sequences. Mixed T2I/edit datasets are valid at microbatch one; larger heterogeneous batches require explicit padding and masks before being advertised. Begin 24GB examples with one 512px-area reference and validate additional references by measured total token budget.

## LoRA targets and artifacts

Default to `self_attn.qkv_proj` and `self_attn.o_proj` across all 32 blocks. Offer an explicit second preset adding `mlp.shared_mlp.gate_and_up_proj` and `mlp.shared_mlp.down_proj`. Resolve these through existing target filters or a small holder-to-filter translation, and print the matched module list and trainable count. Never let a generic all-linear default silently include every expert, router, embedding, or output head.

For rank 16, these layouts imply about 9.44M attention parameters, or 18.35M including shared MLPs. Adding both projections of every routed expert would add about 570.43M parameters. These are architecture-derived counts, not measured optimizer memory. Keep routed-expert adapters an advanced later capability unless all required training and export checks are added.

Keep adapters in the existing safetensors format and use holder conversion hooks to preserve canonical names and the established Comfy prefix where compatible. Store variant/source identity, target preset, actual matched modules, rank/alpha, quantized-base identity, and conditioning version in metadata. Resume validates these against the selected checkpoint and restores optimizer, step, and RNG through the standard trainer.

Test loading exported attention/shared-MLP adapters into the existing Pedro Comfy node. Export success alone is insufficient: verify nonzero adapter strength changes the intended modules/output and strength zero reproduces the base path. If external key discovery prevents loading, document the exact gap and provide a local conversion/integration proposal; modifying external Comfy code requires a separate explicit request under `AGENTS.md`. Native ai-toolkit sampling and adapter reload remain required independently.

Base and Instruct share adapter structure but not training identity. Do not silently treat an adapter trained on one as validated on the other. Cross-variant experiments must be explicit.

## Proposed configuration

These are planned settings, not currently runnable model entries. Existing generic keys retain their current meaning. New model-specific keys are identified below; use the existing `model_kwargs` container rather than new trainer-wide flags.

```yaml
model:
  arch: hunyuan_image_3_instruct
  name_or_path: /path/to/hunyuan_image_3_instruct_int8_convrot.safetensors
  extras_name_or_path: tencent/HunyuanImage-3.0-Instruct
  quantize: true
  qtype: convrot8
  quantize_te: false
  low_vram: true
  layer_offloading: true
  layer_offloading_transformer_percent: 1.0
  model_kwargs:
    lora_targets: attention
    reference_max_pixels: 262144

network:
  type: lora
  linear: 16
  linear_alpha: 16

train:
  batch_size: 1
  gradient_accumulation: 1
  train_unet: true
  train_text_encoder: false
  gradient_checkpointing: true
  cache_text_embeddings: true
  noise_scheduler: flowmatch
  optimizer: adamw8bit
  dtype: bf16
  lr: 0.0001

datasets:
  - folder_path: /data/edits/targets
    control_path:
      - /data/edits/source_1
    caption_ext: txt
    caption_dropout_rate: 0.05
    cache_latents_to_disk: true
    num_workers: 0
    pin_memory: false
    resolution: [512]

sample:
  sampler: flowmatch
  width: 512
  height: 512
  guidance_scale: 2.5
  sample_steps: 50
  samples:
    - prompt: Change the red jacket to blue while preserving the person and background.
      ctrl_img_1: /data/validation/source.png
```

`lora_targets` is the proposed holder preset, initially `attention` or `attention_shared_mlp`; validate conflicts with explicit network filters. `reference_max_pixels` is a proposed explicit preprocessing limit. W4A8 changes the checkpoint path and selects the new `qtype: comfy_w4a8`. Text-to-image omits controls. Base selects `hunyuan_image_3_base`, its own source/config, and CFG 5.0. Learning rate and dropout above are starting experiment values, not established quality recommendations.

Generate complete examples from the current config schema during implementation, including the normal job wrapper, step/save/sample frequencies, and explicit augmentation policy. Add Instruct T2I, Instruct edit, W4A8, and Base examples. The UI exposes target preset, reference budget, supported quantization, and existing control paths; it must not offer Distil or unsupported specialized objectives under these entries.

## Implementation sequence and acceptance

1. **Variant and forward foundation.** Add the shared backbone, holders, config/tokenizer handling, VAE/vision wrappers, and sequence builder. Validate Base and Instruct templates, time/velocity convention, exact expert/QKV mapping, reference order, and target extraction on small fixtures. Compare selected real layers and fixed latent predictions against the pinned reference. Deliver a native sample with an unadapted supported checkpoint before investigating training quality.

2. **Int8 loading and bounded backward.** Add expert-bank mapping and incremental BF16 source conversion using existing quantization and offloading. Wire actual block checkpointing and every applicable M01–M18 path before this qualification. Validate frozen-base gradients and trainable LoRA gradients, then run complete optimizer steps at 512px with the real Instruct checkpoint. Track first-step optimizer allocation and several warm steps. Record loading and host-memory peaks. Do not treat the earlier isolated block diagnostic as this milestone.

3. **W4A8 storage and autograd.** Implement its distinct backend/import/cache integration. Test dense and expert projections against the published dequantization/forward rules, including codebooks and metadata variants actually present in the released files. Validate the declared surrogate gradient and CPU-offloaded execution. Run full 512px optimizer steps and adapter reload with the real W4A8 checkpoint. Compare loss/gradient behavior and rendered quality against int8 using identical data and seeds; report format-dependent differences.

4. **Complete edit training.** Wire frozen VAE and SigLIP2 reference preprocessing through the normal cache lifecycle. Test zero, one, two, and three references; variable aspect ratios; ordered/repeated references; cached versus uncached execution; blank-caption dropout; and missing-pair rejection. Run a small paired-edit overfit test, checking that changing the instruction and changing the source each affect the output and that the target is never copied into conditioning accidentally.

5. **Base and artifact integration.** Enable the Base holder and T2I example over the same runtime, with its proper pretrain sequence and source defaults. Run a real Base optimizer step, sample, save, and resume. Validate native adapter round trips and attempt Comfy consumption of the supported 2D adapter targets. Finish the normal UI entries and example configurations; no new training-loop implementation should be needed.

6. **24GB qualification and performance.** Verify the optimization inventory and measure the matrix below with real weights, then optimize remaining observed bottlenecks. Existing applicable memory protections are prerequisites, not deferred work. New candidates include whole-layer lookahead, more efficient expert dispatch, and broader saved-activation offload if checkpoints alone leave excessive activation residency. Recheck gradients after each change. Keep correctness tests against external behavior diagnostic: do not alter external packages merely to erase measured numerical differences.

| Qualification case | Required evidence |
| --- | --- |
| Instruct T2I, int8 and W4A8, 512px | Full optimizer steps, finite gradients, adapter parameters change, stable memory, native sampling, save/reload/resume. |
| Instruct edit, each format, one 512px-area source | Same checks plus cache/uncached agreement and source-conditioned learning. |
| Instruct edit with two and three references | Full optimizer step at a declared token budget; preserve order and conditioning. Document measured resolution limits. |
| Base T2I, both formats | Same shared loading/training path, valid pretrain prompts, optimizer step and sample, save/reload/resume. |
| 768px and 1024px | Measure target and reference budgets separately. Label supported configurations from results rather than extrapolation. |
| Short learning runs | Falling loss on fixed diagnostic examples, changing adapter outputs, held-out T2I/edit samples, and comparison with the unadapted checkpoint. |
| Standard lifecycle | Initial sample, periodic sample, sample failure cleanup, checkpoint resume, and restoration of offloaded placement without progressive VRAM growth. |
| Memory optimization inventory | Evidence for all applicable M01–M18 paths, including encoder destruction, embedding lookup placement, dataloader worker policy, cache-hit loading, serialization, and optimizer/adapter handoff. |

Extend the existing relevant regression suites rather than replacing their protections: `test_qwen_prompt_memory.py`, `test_sdtrainer_prompt_offload.py`, `test_h3_low_vram_layer_streaming.py`, `test_comfy_training_offload.py`, `test_adapter_offload_restore.py`, `test_offloaded_quant_compile.py`, and the H3/Krea/Z-Image quantization-cache tests. Add Hunyuan cases for shared paths that currently special-case another architecture. These lightweight tests check lifecycle and routing; real GPU runs must separately establish VRAM, stream lifetime, and bounded host memory.

Include cold and warm loads, first optimizer allocation, several accumulated steps, cache build/unload, two sample-to-train transitions, interrupted sample cleanup, and save/resume in memory qualification. Check that CPU RSS and pinned memory stabilize as well as CUDA allocation, and that a warm cache hit does not execute the original model loader. For optional compile, compare eager/compiled gradients and confirm managed offload operations stay outside the captured graphs. For optional differentiable decode, verify gradients reach the LoRA through the decoder and that no VAE weights move before checkpoint backward completes.

Use small fixtures for shape, mapping, caching, and gradient correctness; reserve GPU integration checks for memory and real learning. Compare offloaded and resident small models with the same quantized values to distinguish implementation errors from quantization error. A LoRA initialized with zero up-projection normally has zero down-projection gradient on the first step; test gradient flow over subsequent updates rather than requiring every tensor to have a nonzero first-step gradient.

Report peak memory and timings for complete steps after warmup, number of accumulation microsteps, precision, adapter targets/rank, actual sequence lengths, and GPU/host configuration. Record numerical tolerances with measured errors. A finite loss or a decreasing toy loss alone does not establish useful edit learning.

Completion means both quantized formats and Instruct edit training work through the normal trainer, supported Base T2I passes the shared path, artifacts resume correctly, and documented 24GB configurations are backed by measurements. If 1024px or a large multi-reference case exceeds the budget, publish the validated lower-budget envelope and identify the remaining bottleneck explicitly. Do not silently drop required W4A8 or edit functionality to finish the port.

## Expected code organization

```text
extensions_built_in/diffusion_models/hunyuan_image_3/
  __init__.py
  hunyuan_image_3.py       # shared holder and thin Instruct/Base holders
  src/
    config.py             # immutable variant contract and validation
    transformer.py        # image backbone and checkpointed decoder loop
    conditioning.py       # templates, reference packing, masks, positions
    pipeline.py           # native sampling using the same prediction path

toolkit/models/v2/diffusion_models/hunyuan_image_3.py
toolkit/models/v2/vae/hunyuan_image_3.py
toolkit/models/v2/text_encoders/hunyuan_image_3.py
toolkit/util/comfy_w4a8_quant.py

testing/test_hunyuan_image_3_*.py
config/examples/train_lora_hunyuan_image_3_*.yaml
```

Keep model-specific expert-bank mappings in the Hunyuan component. Shared edits should be limited to needed quantization/import/resolver hooks, optional conditioning lifecycle support, and registration/UI. The conditioning component's text-encoder directory reflects the existing lifecycle interface; it does not imply an independent language backbone.

## Reference baseline

- [Tencent Instruct source](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct/tree/2ec2c78bee7d4b94157341fba86c4c2c7b1858b2): pinned model implementation/config used for the feasibility work. Pin and record generation/tokenizer/processor asset revisions as well when constructing fixtures.
- [Pedro Comfy implementation](https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3/tree/84ad3a3e2a54472e69e195269729f774b242d120): reference repacking, quantized formats, reference sequence construction, and inference behavior.
- [Existing Hunyuan training project](https://github.com/PhotonAISG/hunyuan-image3-finetune): useful independent flow-loss and attention-LoRA reference; its older model wrapper and memory requirements should not be copied as the integration architecture.
- Local feasibility artifacts: `/home/bart/tmp/codex/hunyuanimage3-feasibility/results.json` and `moe_diagnostic.py`. The script uses temporary downloaded reference classes; a durable integration test must vendor or fetch its pinned fixtures explicitly.

All project Python commands use `conda run -n ai-toolkit python ...`. Preserve the fork's existing memory optimizations and unrelated working-tree changes throughout implementation.
