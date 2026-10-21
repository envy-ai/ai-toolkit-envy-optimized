# FIBO 2 validation

The integration starts at `b8d86db5`, which merges official upstream
`04733873` while retaining per-prompt MOSS encoding. The implementation plan
and pinned reference sources are in [fibo2_training_plan.md](fibo2_training_plan.md).
All test processes use the required 16 GB host-memory limit with no swap.

## Independent architecture comparison

The unchanged Apache reference transformer at
`ab96e2b58c89adb9df418af74d81cc8b1e1be074` was compared with the toolkit's
canonical fused/packed implementation using independently converted weights.
Five FP32 fixtures exercise all six conditioning bundles, unequal token masks,
zero through two references, forward evaluation, and normal/checkpointed
backward. Input, reference, conditioning, and parameter gradients agree within
the stated numerical tolerances.

- Maximum absolute output difference: `1.08e-6`.
- Maximum absolute input-gradient difference: `1.67e-6`.
- Maximum absolute parameter-gradient difference: `3.06e-5`.
- Scope peak host memory: 839 MB; CPU only.

These are small random models (width 256, five main blocks), not evidence of
full-checkpoint quality, GPU fit, or quantized execution. Temporary comparison
scripts and reports are under `tmp/fibo2/reference/`.

## Local checkpoint structure

The complete header of the local Comfy
`fibo2_dev_int8_convrot.safetensors` was compared with a default transformer
constructed on `meta`, without allocating its weight tensors. All 634 expected
model tensors matched by name and shape. There were no missing tensors, shape
differences, or unexpected model tensors. The file also contains 430 quantization
auxiliaries and 40 attention metadata entries. All recorded attention selectors
are `comfy_kitchen_int8`.

This establishes checkpoint structure only. Numerical import, training updates,
adapter round-trips, native previews, and the required UI build are recorded
below as their checks complete.

## Local FLUX.2 VAE

The local `flux2-vae.safetensors` passed strict loading of all 251 keys and a
CPU encode/decode at 64 by 64 pixels. Encoding produced packed latents of
shape `[1, 128, 4, 4]`; decoding returned finite `[1, 3, 64, 64]` pixels.
Encoding agreed with the reference's explicit packing and BatchNorm formula
to `2.38e-7` maximum absolute difference. Inverting the normalization agreed
exactly. The real running statistics were nontrivial (means approximately
`-0.0902..0.0767`, variances `2.913..3.470`, epsilon `1e-4`). Peak host memory
was 1.1 GiB. This verifies the actual VAE weights and latent boundary, without
making a large-resolution memory or image-quality claim.

## Local Qwen3-VL encoder

The local `qwen3vl_4b_bf16.safetensors` was loaded in BF16 on CPU with existing
local processor/tokenizer/config assets; no weights were downloaded. Plain-text
and JSON glyph prompts produced `[6, L, 7680]` conditioning identical to the
independent tapped-layer/RMS calculation (maximum absolute difference zero).
The cases used 20 and 102 tokens; the glyph case included one rendered image.
Glyph-enabled and base conditioning differed, as required. The production
capture hooks returned to their previous count after each encoding call.

This verifies the full encoder's conditioning path. The reported systemd
charged-memory counter does not represent its complete resident footprint:
the 8.3 GB checkpoint uses shared file-backed mappings. No small-memory fit
claim is made from that counter.

## Training, previews, and regressions

The affected CPU regression matrix passed 167 tests and 336 subtests across
23 files. Coverage includes real tiny FIBO adapter updates, signed LoRA/DoRA
checkpoint replay, ordinary adapter and full-model export/reload, all six
specialized trainer objectives, legacy Concept Slider, SliderSpace directions,
checkpoint/helper conversion, cache identity and strict cache failures, plus
existing Qwen/MOSS/flow/offload behavior.

Additional focused preview checks passed after the final preview changes:
component parking before denoising, shared references across CFG branches,
glyph-condition routing, independent Euler/schedule calculations, and rejection
of invalid canvas dimensions or step counts before allocation. The actual
Nunjucks helper rendered valid FIBO graphs for no references, three references,
and relative/absolute inference adapters; incompatible negative glyph prompts
are explicitly rejected before submitting a Comfy workflow.

The final focused conditioning/adapter run passed nine tests and seven
subtests after the last source changes. It checks raw explicit CFG-negative
text, the trained blank prompt, positive glyph rendering, and optimizer plus
export/reload parity for LoRA, DoRA, LoHa, DoHa, and LoKr.

The UI Node suites passed 21 tests. The updated capability suite separately
passed 12 tests, including architecture switching that preserves a disabled
persistent-cache choice while clearing incompatible components/references.
These counts overlap and must not be added as unique tests. Source-only UI
TypeScript, extension TypeScript, Python syntax, YAML syntax, whitespace checks,
and the final production `npm run build` passed. The generated Next.js route
type errors remain the established baseline; the production build follows the
repository's existing type-validation configuration.

## Qualification limits

Full-checkpoint CUDA training, CUDA quantized forward/backward, normal-resolution
GPU fit, and trained image quality remain unverified. The final resource check
found active Toolkit and ComfyUI jobs with about 4 GB free GPU memory. No user
job was interrupted and no Comfy model was unloaded for these checks. This is
why the user guide labels the integration experimental. CPU/synthetic evidence
and the real encoder/VAE checks do not replace that GPU qualification.
