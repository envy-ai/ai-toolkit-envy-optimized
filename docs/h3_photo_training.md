# MiniMax-H3 still-photo LoRA recipe

The `minimax_h3` training form has a **Training recipe** selector. Choose **Apply still-photo recipe** to change the current job. The action sets `model.model_kwargs.h3_photo_training: true`; opening or importing a job and selecting MiniMax-H3 do not apply the recipe. A photo job whose settings differ from the recipe displays **Photo mode — customized settings**. Choosing **Video mode (keep current settings)** clears the photo flag, which restores the ordinary H3 training noise schedule while leaving the other settings as they are.

This is the standard H3 architecture and training process. It does not use a separate image model or reference-distillation objective. For a complete starting config, see [`config/examples/train_lora_minimax_h3_image.yaml`](../config/examples/train_lora_minimax_h3_image.yaml). Replace the dataset path and review the model paths and training length before running it.

## What the action changes

| Setting | Photo recipe |
| --- | --- |
| Dataset | Every dataset uses 512-pixel buckets, batch size 1, one image frame, no automatic frame count, first-frame input, joint audio, or reference/control images; disk latent cache and text cache are enabled; caption dropout is 0.05. |
| LoRA | Standard rank/alpha 8 LoRA, transformer only. `network_kwargs.only_if_contains` includes exactly `blocks.20.` through `blocks.49.`. `ignore_if_contains` excludes `token_refiner` and `adaln_proj`. |
| Helper | `model.assistant_lora_path: circlestone-labs/MiniMax-H3-Image-Training-Adapter/minimax_h3_image_training_adapter.safetensors`. The frozen helper is active for training and excluded from saved LoRA weights and native samples. |
| Training | Batch and accumulation 1; flowmatch with shifted timesteps; Automagic v3, learning rate `1e-6`, constant LR; EMA 0.98. Loss targets noise, with balanced content/style, full timestep range and Min SNR weighting disabled. |
| Sampling | Native single image, 512 × 512, guidance 1, 20 steps. Existing ComfyUI sampling is switched off because the bundled H3 workflow targets video. The saved workflow selection and its component paths remain, so an image-compatible graph can be configured later. |

The photo flag changes the trainer's H3 timestep density to a shifted-uniform draw with 60% of draws below sigma 0.5. It does not change the H3 sampler's video/audio inference schedule. Clearing the flag returns the ordinary H3 training schedule. This adapts the clean-end emphasis from the Fizgig photo preset; the trainers, optimizers, and LoRA formats are not identical implementations.

The action preserves dataset folders and caption text, sample prompt text and seeds, total training steps, base checkpoint and component path overrides, quantization, CPU/GPU offloading and memory choices (including gradient checkpointing and Unload TE). It removes dataset reference paths and per-sample control, size, frame and guidance overrides so samples use the still-image defaults. It also turns off CFG, guidance and preservation objectives that conflict with the frozen image helper. Review these fields after applying the preset if your job had custom training logic. A job using the `ref2va` partition must select an `fl2va` partition before the action can apply; its component paths are left untouched.

## Supported configuration

Use `minimax_h3` with an ordinary `sd_trainer`, `ui_trainer`, or `diffusion_trainer` process, the `fl2va` or `fl2va_pruned` H3 partition, and still-image datasets. The photo flag on another architecture, training mode, or H3 reference partition is rejected at config validation. The helper path must resolve to a compatible checkpoint; a local path or the `org/repo/file.safetensors` form above is supported. The UI's **Distillation Handling Method** can change the helper, but CFG or contrastive guidance cannot be combined with this photo mode. To use those video-oriented objectives, select Video mode first.

The recipe establishes configuration and loading behavior; it does not establish image quality, speed or memory requirements for a particular dataset or GPU. A full training run and output comparison are still needed to qualify those results.

## Sources and verification

The recipe follows [Fizgig v9.3.2's H3 presets](https://github.com/shootthesound/Fizgig/blob/8630fca0c5d7932d73fe7387fd7c8a4dc99d9467/src/fizgig/families/minimax.py) and the [Circlestone image adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter). Its checkpoint header contains 208 complete rank-64 factor pairs, including the token refiner, without explicit alpha values. The strict loader defaults each missing alpha to its rank and rejects unmatched factors.

Focused CPU checks passed 16 photo-training cases, including the shifted density, native parsing of the published YAML, ordinary process types and strict helper loading with mixed ranks/alphas. The existing four H3 quantized-cache and eight layer-streaming cases also passed. Three UI configuration cases passed, covering explicit application, JSON round trips, custom settings, preservation of memory choices, and clearing incompatible references. Tests used a 16 GB host-memory limit with swap disabled. The separate Qwen adapter suite could not collect because an unrelated OmniGen2 import initializes CUDA when CUDA is hidden; no external code was changed.
