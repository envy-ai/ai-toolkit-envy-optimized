# The krea2_edit geometry contract

This document pins the *exact* reference geometry and conditioning semantics of
`krea2_edit` training, so that inference stacks and third-party trainers can verify
parity instead of guessing. The [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit)
nodes (v1.2.4+) implement the same contract; LoRAs trained against a different
geometry will misregister at inference (blur, off-center ghosting, seam artifacts).

## 1. Token sequence

```
[ text tokens | ref_1 tokens | ... | ref_N tokens | target tokens ]
```

- `N ≤ 2`: the inference nodes expose exactly two reference inputs (`source_image`,
  `source_image_b`), so the trainer refuses more than two references.
- The transformer input is one joint sequence. References are clean (un-noised) VAE
  latents; only target tokens receive flow-matching noise, and **only target tokens
  contribute to the loss** (the predicted-flow slice for refs is discarded).
- Latents are patchified 2×2 on a /8 VAE, so one token = 16×16 pixels.

## 2. 3D RoPE position ids

Each image token carries `(frame, h, w)`:

- **Target: frame 0.** References: **frame i+1** in list order (ref_1 → frame 1, …).
- Text tokens: all-zero position ids.
- Target grid: `h ∈ [0, H/16)`, `w ∈ [0, W/16)`, stride 1.

## 3. Reference fit protocol (`fit_refs`)

`model_kwargs: {fit_refs: true}` is the trainer default and corresponds to the node's
`fit_mode: "fit"` (also its default). References are AR-preserving **fitted inside**
the target token grid:

1. Compute scale `s = min(target_h/src_h, target_w/src_w)` in pixel space.
2. Token grid dims must be exact multiples of 16 px. **Do not floor-resize the scaled
   source** (`src*s → floor16`) — that squashes content by up to 15 px, and the
   misregistration peaks at the reference-band edges (renders as a doubled seam band
   on outpaints). Instead **center-crop the source first** so that the fitted axis
   lands on the /16 grid at scale `s` exactly, then resize. Zero squash by
   construction.
3. Reference tokens are placed at **fractionally centered offsets**:
   `off = (target_grid - ref_grid) / 2` — floating point, NOT `// 2`. RoPE is
   continuous; half-token positions are exact. Integer flooring places odd-gap
   references 8 px off their true center.
4. The reference grid is stride-1 (its pixels were resampled to target grid density);
   never rescale the position grid itself.

## 4. Grounded text conditioning

The instruction is encoded by Qwen3-VL-4B **together with the reference images**:

- Template: fixed system prefix (describe color/shape/size/texture/…), then the user
  turn containing one `<|vision_start|><|image_pad|><|vision_end|>` block **per
  reference, in frame order**, followed by the instruction text.
- Output: hidden states of 12 selected layers `(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)`
  stacked to `(1, L, 12, hidden)`, with the first 34 tokens (system prefix) sliced off.
- Under the fit protocol (`fit_refs: true`, the default) the grounding image is the
  reference as delivered by the loader, i.e. at its **native resolution, before the
  fit-to-target-grid resample** — the semantic path and the appearance path see
  different pixels on purpose. This is only true under the fit protocol: with
  `fit_refs: false` (legacy crop geometry) the loader hands over an already
  target-sized control tensor, so the grounding image is the cropped/resized
  reference, not the native one.
- The grounding image is then optionally downscaled: longest side capped at
  `GROUNDING_MAX_PX` (default **768**, matching the inference node's `grounding_px`)
  with per-step uniform jitter down to `GROUNDING_JITTER_MIN` (default **384**).
  Both env vars override the defaults; `0` disables the cap. The downscale kernel is
  PIL `LANCZOS` here; the inference node uses `common_upscale(..., "area")`. Jitter is
  a train-time augmentation — which is why
  **text-embedding caching must stay off** for edit datasets: a cached embedding
  freezes one grounding scale and the LoRA loses scale robustness.
- **Cache invalidation is manual.** ai-toolkit's text-embedding cache key does not
  include the grounding settings, so changing `GROUNDING_MAX_PX` /
  `GROUNDING_JITTER_MIN` does *not* invalidate an existing cache. If you train with
  `cache_text_embeddings: true` and change either value, delete the `_t_e_cache`
  folders under your dataset directories first — otherwise the run keeps feeding
  embeddings built at the previous grounding resolution and this section no longer
  describes what the model actually sees.

## 5. Objective

- Flow matching, velocity target `v = noise − clean`, MSE on target tokens only.
- `timestep_type: "weighted"` = **uniform** timestep sampling + a per-timestep loss
  weight lookup. It is not a sampling distribution (and not sigmoid/logit-normal).

## 6. Checkpoint format

LoRA keys follow the ComfyUI convention (`diffusion_model.<module>.lora_…` + per-module
`alpha`), targeting the transformer's attention/MLP linears including the text-fusion
blocks. Checkpoints load directly in ComfyUI via the standard LoRA loader.
