# MiniMax-H3-Image training and previews

Select **MiniMax-H3-Image** in the image group for the official audio-free image model (`minimax_h3_image`). This is a separate architecture from the existing MiniMax-H3 video model. Selecting it for a new job initializes an experimental still-photo LoRA recipe; opening, importing or editing an existing video job does not change that job's architecture, training schedule or settings. The **Training recipe** selector can explicitly reapply the image recipe after edits. Its disabled customized-status entry reports saved settings that differ from the recipe. **Image mode (keep current settings)** clears only `model.model_kwargs.h3_photo_training`, leaving the image architecture and current training choices in place.

The example [`train_lora_minimax_h3_official_image.yaml`](../config/examples/train_lora_minimax_h3_official_image.yaml) contains the same initial settings. Set the dataset path and review the planned number of steps before training. The model partition is `image`; video `fl2va` and `ref2va` partitions are not compatible.

## Model and recipe

The loader resolves `minimax_h3_image_pruned_int8_convrot.safetensors` and `minimax_h3_image_vae_fp16.safetensors` from `Comfy-Org/MiniMax-H3-Image`, and uses the shared `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` text encoder from `Comfy-Org/MiniMax-H3` unless you provide a component override. `model.model_kwargs.dit_image_path`, `model.vae_path`, and `model.te_name_or_path` provide explicit local components. The standard `model.text_encoder_path` also works; `model_kwargs.text_encoder_path` takes precedence over it, and `te_name_or_path` takes precedence over both. The image model has no audio stream. Its transformer uses H3's existing low-memory loading and layer-streaming controls.

The example keeps the DiT's released int8 ConvRot weights (`quantize: true`, `qtype: convrot8`) and the encoder's NVFP4 quantization. Its BF16 dtype settings configure floating-point computation; the shipped compressed weights retain their quantized representation.

Compatible single-file Qwen3-VL encoder overrides can also carry int8 embeddings
with either one tensor-wide scale or one scale per vocabulary row. The native
importer preserves both layouts and dequantizes only the selected rows. Existing
quantized encoder caches retain this behavior without rebuilding the cache.

Quantized component caches use `model.quantized_model_cache_dir` (default `models/.quantized_training_cache`). Set an explicit writable directory if your usual cache location is unavailable. First-time cache creation can involve substantial disk I/O. The loader does not replace an existing cache symlink.

When ComfyUI sampling is configured, new `minimax_h3` and `minimax_h3_image`
training starts clear ComfyUI's model caches from both system RAM and VRAM
before loading the training model or large text encoder. This also unpins
ComfyUI model and LoRA caches. Other architectures keep their existing
cache-preserving VRAM release, and training preview handoffs still retain
ComfyUI's RAM cache. The shared-model provider with shared weights enabled
continues to use its lease coordinator instead of this startup clear.

H3 startup requires the installed `comfyui-cache-monitor` extension's
`model-cache`, `model-pin`, and `lora-pin` APIs. It rejects running or queued
ComfyUI work without interrupting prompts or clearing the queue. After removing
pins, it requests native `POST /api/free` with `unload_models: true` and
`free_memory: true`, then waits up to 120 seconds for the monitor's model and
LoRA lists and RAM/VRAM model-cache byte counts to become empty. Missing APIs,
failed requests, invalid responses, new queued work, or a timeout fail training
startup rather than continuing with model RAM still occupied. Retry when ComfyUI
is idle; install/enable the extension and restart ComfyUI when idle if needed.

ComfyUI acknowledges `/api/free` before its worker unloads models, resets its
executor, and collects garbage. The bounded startup check confirms observable
model caches are empty; the native API exposes no reset/garbage-collection
completion token. It does not promise zero process RSS or release unrelated
custom-node caches. Other ComfyUI clients can submit work after an idle check;
keep them idle during H3 startup because these APIs provide no exclusive queue
reservation. Existing training processes are unaffected by this code change.

A real PyTorch 2.15 run with ConvRot weights, layer streaming and block compilation
failed on its first training pass inside generated Inductor code with an undefined
`stream1`. For this configuration, use `model.compile: false` and
`model.block_compile: false`. This leaves quantization and streaming enabled;
the compiler failure has not been repaired in external PyTorch code.

The Comfy repack carries six output-head groups: the first is a complete head and the remaining groups are offsets from it. Training uses the first complete head as the initial recipe assumption; inference reconstructs each published PDD head. A converted single-file original transformer can instead retain its ordinary training head and load the official `pdd_fused_heads.safetensors` through `model.model_kwargs.pdd_fused_heads_path`, with the adjacent `pdd_config.json` or an explicit `pdd_config_path`. The loader validates the six-step metadata. Direct loading of the original sharded Diffusers directory is not supported.

The initial training recipe uses one-frame 512-pixel buckets, batch size 1, disk latent and text caches, 5% caption dropout, rank/alpha 8 LoRA on blocks 20–49, and a frozen Circlestone image training adapter. It sets Automagic v3 at `1e-6`, constant learning rate, EMA 0.98 and the 60%-clean-end shifted training density (`h3_photo_training: true`). These values are adapted from the Fizgig still-photo practice on the video-base H3 family. They are **an initial training hypothesis for the official image checkpoint**, not a validated quality or speed recommendation. The image-specific DiT, VAE and six-head PDD sampler differ from the video model, so output quality and memory use require a real training run to qualify.

The recipe preserves dataset folders and captions, sample prompt text and seeds, total training steps, model component paths, quantization, offloading, gradient checkpointing and Unload TE. It removes reference/control paths and sample size/frame/guidance overrides, and clears CFG, guidance and preservation objectives that conflict with the helper. The image model itself remains image-only whether or not the photo flag is set. The flag changes the training noise density; it does not change the official six-step inference schedule.

## Advanced form settings

The ordinary job form exposes the remaining test-training choices without
editing YAML. Expand **Advanced model settings** in Model and **Advanced target
modules** in Target by clicking the summary or focusing it with Tab and pressing
Enter/Space. Labels above text, number and select fields focus their controls.
Opening either section or loading an existing job does not write defaults.

- **DiT path (optional)** sets `model.model_kwargs.dit_image_path`.
- **Text encoder path (optional)** shows the active override, respecting
  `te_name_or_path`, then `model_kwargs.text_encoder_path`, then
  `text_encoder_path`. Editing consolidates those aliases into
  `model.te_name_or_path`; clearing removes all three overrides.
- **VAE path (optional)** shows `model.vae_path`, or an imported
  `model_kwargs.image_vae_path` alias. Editing consolidates it into `vae_path`;
  clearing removes both overrides. Blank component paths use automatic
  resolution described above.
- **Cache quantized models** controls `model.cache_quantized_models` (the
  backend default is true). **Quantized cache directory (optional)** edits
  `model.quantized_model_cache_dir`; blank retains the backend default
  `models/.quantized_training_cache` without saving an explicit default.
- **Caption token limit** edits `model.model_kwargs.max_text_length`. Blank
  uses 4096, and 0 disables the caption cap. Reference vision tokens are never
  truncated. The image text-cache identity includes this limit.
- **Cache dataset text embeddings to disk** changes
  `datasets[].cache_text_embeddings` for every dataset together. This disk
  cache is separate from Training's existing **Cache Text Embeddings** switch,
  which controls `train.cache_text_embeddings`. The backend requires every
  dataset to enable disk text caching if any does. Mixed imported choices
  remain unchanged until the switch is edited. Newly added datasets inherit a
  common existing disk-cache choice.

**Include modules** and **Exclude modules** edit
`network.network_kwargs.only_if_contains` and `ignore_if_contains`. Choose
suggested block/module chips, or type a substring and press Enter to add it.
Numbered chips such as **#20** refer to transformer blocks; the saved filter
remains `blocks.20.`. Compact labels keep block indices readable in narrow cards.
Arrow keys navigate suggestions; Backspace or a chip's remove button removes
it. Imported custom values remain visible. Clearing Include removes its key:
the photo recipe then supplies blocks 20–49, while ordinary image mode includes
all eligible modules. An imported explicit empty Include array matches no
modules and is preserved until edited. Clearing Exclude saves an empty list and
excludes nothing.

Imported process-level `only_if_contains` and `ignore_if_contains` aliases are
displayed when the corresponding nested filter is absent or empty, matching the
backend's existing normalization. Editing a filter removes its process-level
alias and writes the nested form in one update, so clearing cannot reactivate a
hidden alias. Opening the form preserves both representations unchanged.

While the photo recipe flag is enabled, Include must be unset or exactly blocks
20–49, and Exclude must be empty or contain both `token_refiner` and
`adaln_proj`. The form validates these restrictions before saving. To experiment
with other module filters, explicitly choose **Image mode (keep current
settings)** in Training recipe, then edit the chips. Filter edits never disable
the photo flag automatically. Existing optimizer, learning-rate, EMA/decay,
rank/alpha, latent-cache and layer-streaming controls remain in their usual
cards.

## Native and ComfyUI samples

Native sampling uses the official six-step PDD schedule with guidance 1 and one image frame. A new image job previews at 2048 × 2048, the official image canvas; training buckets begin at 512. These are separate settings. Native inference rejects other step counts or guidance values. The Sample card exposes up to three ordered **Reference Images** for native image editing, through the existing image-picker controls. Image references use the official plain `Picture N: ` text presentation, fixed CPU-seed-42 posterior sampling, and seeded augmentation in packed-row order. Their caches are distinct from video-reference caches. The experimental training recipe clears existing sample references when applied; you may add native preview references afterward. Large previews may need substantial memory; adjust width and height only if the native implementation and available memory permit it.

The example sets `sample.samples[].neg: ''`. If omitted, the job's empty-negative default is also accepted by the image model while the trainer caches sample prompts; actual negative text is preserved as entered, though sampling at guidance 1 does not use it.

The **MiniMax-H3-Image LoRA still image** Comfy workflow is [`minimax_h3_image_lora_sample.json.njk`](../config/comfy_templates/minimax_h3_image_lora_sample.json.njk). Select the model, image VAE and shared Qwen3-VL text encoder filenames as they appear on your ComfyUI server. The graph uses core `UNETLoader`, `CLIPLoader` with type `minimax`, `VAELoader`, the official `MiniMaxH3ImageConditioning`, `ConditioningZeroOut`, six-step `KSampler` with Euler/simple and CFG 1, `VAEDecode`, and `SaveImage`. Its single extra node is `load_lora_from_absolute_path` from `envy-ai/load_lora_by_index` for the current training LoRA. An optional inference LoRA is applied after the training LoRA. Prompt, size, exact seed, signed training strength, and the selected Comfy component names flow from the job configuration. Negative prompts are unused at guidance 1. `SaveImage` writes PNG; Comfy output format/quality fields do not control this graph.

The bundled graph is text-to-image only. It rejects sample reference images and unsupported step/guidance/frame/sampler settings before posting to ComfyUI. Batch mode sends prompts individually, retaining each prompt's settings and adapter strength. The renderer checks the target server's node schema before posting, so missing nodes or component choices report an error. Existing H3 video workflows remain separate and cannot be silently substituted for this image graph. The Comfy server must have the official image nodes and chosen model files installed; AI Toolkit does not install them.

The earlier [`MiniMax-H3 still-photo recipe`](h3_photo_training.md) trains the video-base `minimax_h3` model on image datasets. Its 20-step native sample and `fl2va` partition requirements do not apply to this official image architecture.

## Qualification status

Focused CPU fixtures verify the image VAE geometry, normalization and seeded posterior, checkpointed DiT gradients, PDD head reconstruction and update direction, frozen helper separation, and LoRA save/optimizer resume. Downloaded checkpoint headers confirm the image VAE's complete persistent key/shape layout and the DiT's audio-free six-head format and exact sigma grid. Workflow rendering and local Comfy conditioning schema checks also pass. All 28 existing H3 photo, quantization-cache and layer-streaming regression tests pass. UI tests and an isolated browser scenario verify keyboard recipe selection, saved-job preservation and desktop/narrow layouts without creating a training job. These checks do not load the complete pretrained transformer or establish training quality.

Actual-model qualification passed with the released **int8 ConvRot** DiT and NVFP4 text encoder, layer streaming, the frozen helper and Automagic v3. Two 256-pixel training updates passed with finite losses and saved adapter/optimizer files. Reloading those files and continuing from step 2 to step 3 passed. A subsequent update using a confirmed 512 × 512 source and training bucket resumed step 3 to step 4 with finite loss; all 120 learned LoRA output matrices changed and remained finite, and all 240 optimizer states advanced from step 3 to 4. Native six-step previews passed at 256 and 2048 × 2048, with finite latent and float-pixel checks before PNG conversion. The 2048 result is a coherent colored circular graphic from a minimal prompt; these short tests do not establish photo likeness or recipe quality. Real-model checks used text-to-image inputs; reference augmentation was checked with fixtures.

The GPU run covering checkpoint resume and the 2048 preview peaked at 2.39 GB CUDA allocated / 6.07 GB reserved on an RTX 4090. The confirmed 512 training run peaked at 2.37 GB allocated / 6.07 GB reserved. These are measurements of these small tests, not minimum memory requirements. The approved host-memory limit was 64 GiB with swap disabled; the cgroup reached that limit including substantial reclaimable file cache, without an OOM kill. Smaller tests retain the normal 16 GiB cap. Comfy workflow rendering/schema checks passed; a live Comfy GPU generation was not run. The production UI build passed with the required 16 GiB Node heap. After adding the advanced form controls, the web UI was reloaded and its read-only health checks passed; the training worker retained its process identity. The Comfy server was not restarted.
