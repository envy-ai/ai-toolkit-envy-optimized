# Official MiniMax-H3-Image implementation plan

## Scope and assumptions

Add a distinct `minimax_h3_image` architecture with native image-only loading, forward/encoding/decoding and six-step inference, plus the existing ComfyUI preview integration. Keep `minimax_h3` FL2VA/Ref2VA and its still-photo recipe available. Reuse the fork's quantization, component caches, layer streaming, per-prompt encoding and sampling/offload lifecycle.

The user authorizes the Fizgig photo recipe as the initial training hypothesis wherever it fits structurally: rank/alpha 8, blocks 20–49, frozen Circlestone helper, 60% low-noise density, Automagic v3 at 1e-6, EMA 0.98, caption dropout 0.05, batch 1 and cached stills. This is not an established recipe for the official image weights. Training resolution starts at 512; preview defaults follow the official image model rather than the video recipe.

Ordinary flow-matching training uses a single full output head. Native inference keeps the six PDD heads frozen and uses the published seven-point sigma grid. The original release's ordinary head is preferred when available; a Comfy repack's full head 0 is an explicitly documented starting assumption when that is the only available full head. The later Comfy head-bank rows are offsets from head 0, so they must not be interpreted as six independent full heads. Strict format validation must distinguish these representations.

## Ordered work and ownership

1. Backend: add the image DiT/config/loader, image-only packing and native six-step pipeline using narrow extensions of the existing H3 implementation. Preserve FP32 projection/head islands and checkpoint metadata. Register the new architecture.
2. VAE: reuse the existing CNN encoder and add the official single-frame 24-layer, 16-head, 2D-RoPE decoder and Conv2d post-quant layer. Match pixel/latent normalization and reference posterior sampling.
3. UI and Comfy: use the existing model controls and explicit preset action; add an image workflow with the official conditioning node, image VAE and six-step CFG-1 sampling. Preserve saved jobs and user paths/prompts/memory choices. Mockup precedes UI edits.
4. Integration: validate ordinary single-frame training, scope the photo recipe to its supported architectures, and retain dataloader and cache identity protections. Reject unsupported reference-training inputs explicitly; reference-image inference must either implement the official contract or report unsupported inputs.
5. Verification: strict metadata/loading fixtures, base/PDD head math, image-only packing, VAE normalization/shapes, helper freezing and LoRA train/save/resume, workflow rendering/schema checks, old H3 regressions, focused GPU checks and browser verification. Run the required UI build with a 16 GB Node heap after final UI changes.
6. Independent review and commits, with measured limits recorded in the user guide. Report service restart separately; do not restart unrelated live services or patch ComfyUI's working tree.

## Evidence and constraints

The implementation reference is ComfyUI commit `da58e396690810bed36afa9e4861e910569731a7`, a cherry-pick of upstream PR [17655](https://github.com/Comfy-Org/ComfyUI/pull/17655), plus the [official release](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image). Small release configs and the separate FP32 PDD head file are already in ignored `tmp/h3-image-assessment/`. The local ComfyUI checkout contains unrelated user changes and is read-only for this task.

Project Python uses the `ai-toolkit` Conda environment, now PyTorch 2.15.0+cu134. Tests require finite timeouts and a single 16 GB/no-swap systemd scope. Full pretrained-model qualification may need more host memory than this scope permits; establish that requirement from actual model assets before requesting a focused exception. Tiny fixtures and production-code comparisons must not be described as full-weight quality or memory qualification.

Hub file metadata establishes 20,979,314,802 bytes for the default image int8 DiT, 15,687,142,551 for the shared NVFP4 encoder, and 1,168,505,312 for the image VAE. The DiT alone exceeds the 16 GiB test cap. The user approved a one-task exception to use `MemoryMax=64G`, `MemorySwapMax=0` for full-weight load, short LoRA train/save/resume, and six-step image preview, but explicitly instructed us to wait because RAM and GPU are in use. Do not start that dependent test until the user confirms the resources are free. Other tests retain 16 GiB.

## Implementation checkpoint

Backend, image VAE, UI recipe, training guards and Comfy template are implemented. Independent review identified and corrected the native single-PIL return contract, seeded reference-noise row ordering and premature BF16 rounding before the FP32 image projection. Image reference text uses the official plain picture label while the video default remains unchanged.

Verification passed: 16 image CPU cases, 28 existing H3 regressions, five UI cases, isolated keyboard/persistence browser scenario with desktop/narrow captures, Python syntax and extension type checks, and the production UI build with a 16 GiB Node heap. The isolated preview server is stopped; the live UI and Comfy services were not restarted. Full pretrained-model GPU qualification is the remaining deferred acceptance step, under the user-approved exception and resource-wait instruction above. Its prepared short-run configuration is `tmp/h3-official-image/smoke-256.yaml`.
