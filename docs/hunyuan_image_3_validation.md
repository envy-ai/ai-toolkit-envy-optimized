# HunyuanImage 3 LoRA implementation and qualification

The implementation uses the ordinary `sd_trainer` or its UI/API
`diffusion_trainer` wrapper, existing LoRA factors,
dataset caches, v2 component policy, and asynchronous CPU weight offloader.
Both local non-distilled Instruct Comfy formats were exercised. This document
records actual evidence; the broader
[implementation plan](hunyuan_image_3_lora_plan.md) remains a specification.

For this RTX 4090 and approximately 128 GB system RAM, the recommended profile
is **Instruct ConvRot int8, offload 0.85, 512 px, rank 16 attention LoRA**. Start
with the int8 example and select `qtype: convrot8` when using the int8 checkpoint.
Measured warm edit updates were approximately 12.74 s, compared with 24.37 s
for W4A8 full offloading. The 0.85 profile preserves substantially more GPU
headroom than 0.80 for nearly the same speed. W4A8 uses less host storage;
its implementation remains available. Independent int8 T2I and full-process
resume qualification remain outstanding; the measured int8 evidence is editing.

Select `hunyuan_image_3_instruct` for T2I or editing and
`hunyuan_image_3_base` for the shared Base T2I implementation. Distil/MeanFlow,
full fine-tuning, routed-expert adapters, specialized training objectives,
differentiable VAE losses, and the Comfy preview provider are rejected.
Native previews are supported. Loading a Comfy checkpoint does not require
running the Comfy preview provider.

Use the complete `config/examples/train_lora_hunyuan_image_3_*.yaml` files.
Replace the backbone and `model.vae_path` placeholders with actual files.
Editing additionally needs `model.model_kwargs.vision_path` and one to three
ordered matching `control_path` folders. The UI exposes these asset fields.
Tokenizer assets may be a local directory containing `tokenizer.json` or the
variant's Tencent repository. Defaults pin Instruct assets to
`2ec2c78bee7d4b94157341fba86c4c2c7b1858b2` and Base assets to
`36f21fe74b65614451cc50ffd8a35a5f662dac70`.

Current local assets (replace the corresponding example fields):

| Field | Existing local asset |
| --- | --- |
| W4A8 `model.name_or_path` | `/home/bart/ComfyUI/models/diffusion_models/d/hunyuan_image_3_instruct_w4a8.safetensors` |
| Int8 `model.name_or_path` | `/d/comfy_models/diffusion_models/hunyuan_image_3_instruct_int8_convrot.safetensors` |
| `model.vae_path` | `/home/bart/ComfyUI/models/vae/d/hunyuan_image_3_vae_fp16.safetensors` |
| Edit `model.model_kwargs.vision_path` | `/home/bart/ComfyUI/models/clip_vision/hunyuan_image_3_instruct_siglip2_so400m_naflex.safetensors` |

The downloaded int8 repack is pinned to PedroMarinhoDev's Hugging Face revision
`50543c5af85fc9424f87208618f520fd28f5b882`.
The `/d` int8 path is a symlink to the same filename under
`/e/comfy_models/diffusion_models/`. Relocation was verified against SHA256
`b2fdd4f3fca0c5a864a837997ab497ab6068e517990cd2ebbc4dd6668e8a213f`.
The measured int8 runs below used the earlier
`/home/bart/models/HunyuanImage-3.0-Instruct-ComfyUI/` location on the HDD;
their configurations, artifacts and startup evidence remain historical.

The W4A8 24 GB starting configuration is BF16, rank 16 attention LoRA, microbatch 1,
full transformer offloading, block checkpointing, target-latent and frozen
conditioning caches, zero dataloader workers, and dataset pinning disabled.
Use `qtype: comfy_w4a8` for the published W4A8 checkpoint or `convrot8` for
the published int8 ConvRot checkpoint. These are distinct storage/compute
formats. The Instruct int8 examples use the measured 0.85 offload fraction,
retaining roughly 11.3 GiB of frozen weights on this GPU. Base is unqualified.
W4A8 supports published checkpoint import; fresh BF16 conversion
supports ConvRot 8, with bounded per-projection conversion.

`cache_quantized_models: false` uses a published checkpoint in place and is
the example/UI default. Optional packed module-cache reuse is exercised on
small fixtures. A complete 44–76 GiB module-cache write/warm load has not been
profiled and is not part of the measured memory envelope.

## Current support evidence

| Case | Evidence | Qualification |
| --- | --- | --- |
| Instruct W4A8 T2I, 512 px | Actual 80.775B model, two normal trainer optimizer updates, finite/nonzero saved up factors, normal adapter/optimizer saves | Functional training measured |
| Instruct W4A8 edit, one 512 px-area source | Actual VAE+SigLIP disk conditioning, cached edit updates, all 128 attention LoRA factors finite/nonzero over resumed updates, adapter/optimizer resume, periodic and final native previews | Functional edit and native lifecycle measured |
| Native 50-step zero-strength baseline | Actual checkpoint, seed 42, CFG 2.5, 512 px; visibly generates the requested centered red circle on white | Native T2I generation measured |
| Instruct int8 ConvRot edit, one 512 px-area source | Actual 76.16 GiB checkpoint, three normal optimizer updates, all 128 factor gradients finite/nonzero, adapter/optimizer saves and native preview at offload fraction 0.85 | Functional edit measured; independent int8 T2I/process resume unqualified |
| Two/three references and variable aspects | Small fixtures cover ordering, repeated slots, blank-caption preservation, geometry/mask assembly and mixed time precision | Full-model limits unqualified |
| Base T2I | Thin variant holder, pretrain template and independent source/cache identity; small template/loader contracts | Real Base weights unqualified |
| 768 px/1024 px | Configuration supports explicit target/reference budgets and sequence guards | Unqualified |
| Learning quality/overfit | Adapter updates and finite losses measured; noise/timesteps vary between optimizer steps | Useful learning and held-out quality unqualified |
| Comfy adapter consumption | Unmodified Comfy parser/factorized adapter application extracted for diagnostics consumed all 192 saved keys into 64 patches | Full Comfy render unqualified |
| Original Tencent BF16 shards | Shard-index streaming, canonical 2D mapping and incremental quantization implemented | Small loader fixtures; real shards unqualified |

Do not interpret changing stochastic training losses as an overfit result.
Two-step previews exercise the lifecycle; the non-distilled model normally
uses 50 steps. Do not extrapolate the 512 px memory results to larger buckets,
multiple references, Base checkpoints, or another GPU.

## Measured runs

Machine: RTX 4090 with 24564 MiB VRAM and approximately 125 GiB system RAM;
`ai-toolkit` environment, PyTorch 2.7.0. The separate Comfy process was idle
and its models were unloaded before the first real qualification; no active
user render was interrupted. A second CPU-cached full backbone must not be
kept alongside this trainer.

Actual local W4A8: 43.62 GiB, 1167 tensor keys, 192 quantized projection/bank
markers. Dense and expert codebooks, relative FP8 scales and channel FP32
scales remain packed. CoT output-head weights are omitted.

| Run | Training/lifecycle measurements |
| --- | --- |
| First W4A8 T2I, 512 px, rank 16 | First optimizer update 34.82 s; second 31.91 s. Training peak 3.483 GiB allocated/3.752 GiB reserved. Final 64 up factors finite/nonzero; combined norm 0.2541742. |
| W4A8 edit initial updates | Losses 0.1231/0.09086, one reference. Cache/forward cumulative peak 6.514 GiB allocated/7.273 GiB reserved. Native preview initially exposed CPU-generator handling, corrected and fixture-covered. |
| W4A8 edit resume | Actual 4325-token joint sequence, 512 px target and 512 px-area reference. Resumed update 35.55 s, up-factor norm 0.40291 then 0.51221. Periodic native two-step CFG sample 46.18 s; final sample 45.64 s. Post-sample allocation 0.113 GiB/reserved 0.381 GiB. |
| Resume loading+training+native samples | Polled peak 5.672 GiB allocated/8.029 GiB reserved; 8732 MiB driver process VRAM. Host maximum RSS 102.57 GiB while the clean checkpoint mapping was still live; steady post-load RSS 63–66 GiB. |
| Int8 edit, offload 0.85/seed 42 | First update 18.65 s; warm update intervals 12.64/12.83 s including save hooks. Actual frozen resident storage 11.301 GiB; pinned tensor payload 63.844 GiB. Peak 17.252 GiB allocated/18.564 GiB reserved, 19482 MiB process VRAM. Final up-factor norm 0.43339, all 128 gradients finite/nonzero. Native two-step edit preview 19.84 s. |
| W4A8 unadapted native baseline | 50 steps, zero adapter strength, seed 42; 2271 positive/2263 negative tokens. Sample 673.99 s; reload+sample 739.94 s. Peak 5.365 GiB allocated/6.996 GiB reserved; 7660 MiB driver process VRAM. |

The first T2I run retained a clean file-backed checkpoint mapping alongside
approximately 50.8 GiB of dirty pinned storage. Cloning only the unpinned CPU
embedding table after offloader attachment releases that mapping. Independent
`/proc` inspection of the edit run confirmed the 43.62 GiB file mapping was
absent, with 57.46 GiB RSS, 55.10 GiB dirty `/dev/zero` pinned storage and about
1.41 GiB anonymous storage. This is direct ownership evidence, not a claim
that the different run phases have identical RSS peaks.

PyTorch's pinned allocation buckets can exceed packed file size. The int8
checkpoint's routed banks alone can approach 96 GiB of pinned allocations
despite roughly 72 GiB raw bank storage. The measured int8 run deliberately
kept 11.301 GiB of frozen tensors on the GPU (offload 0.85); whole-machine
host use was approximately 101 GiB with 23 GiB available. Its maximum process
RSS was 87.09 GiB during loading, falling to roughly 46.5 GiB after preparation.
Later RSS was lower than the measured pinned tensor payload; the precise
accounting discrepancy was not isolated. Use the 63.844 GiB pinned payload and
whole-machine available RAM for host-budget evidence. Do not infer a 46 GiB host requirement or full
int8 offload fit from this result. Full-offload int8 remains unqualified.

Int8 cold startup took about 628 s before the first optimizer update, including
CPU pinning and over 94 GiB of mmap-driven disk reads from `/dev/sdb3`
(an ext4 rotational HDD). The W4A8 source resolves under `/d` on an NVMe
SSD; these startup results are not an isolated format/loading comparison. The
existing manager pins selected projections first, then moves skipped resident
projections to GPU; those late reads can revisit evicted clean pages. Startup
I/O is separate from the measured 12.6–12.8 s warm update intervals.

## Residency benchmark

The controlled W4A8 comparison uses three fresh optimizer updates on the same
cached 512 px edit example, rank 16, seed 42, no caption dropout, identical save
hooks and a final two-step native edit preview. The reported warm time is the
median of update intervals two and three, including the save hooks. The existing
manager randomly selects individual projections; the offload fraction is a
selection probability, not a guaranteed percentage of packed bytes. Actual
resident/pinned payloads are measured separately.

| W4A8 offload fraction | Frozen GPU / pinned payload | Median warm update | Combined cache/train/native peak | Native two-step preview |
| --- | --- | --- | --- | --- |
| 1.0 | 0.00055 / 42.606 GiB | 24.37 s | 6.231 GiB allocated, 7.273 GiB reserved; 7920 MiB process VRAM | 32.41 s |
| 0.75 | 10.821 / 31.786 GiB | 23.64 s | 16.648 GiB allocated, 18.297 GiB reserved; 19208 MiB process VRAM | 30.77 s |
| 0.70 | 12.964 / 29.643 GiB | 23.45 s | 18.822 GiB allocated, 20.859 GiB reserved; 21870 MiB process VRAM | 30.40 s |

The safe W4A8 examples remain at 1.0. Keeping packed projections resident avoids
transfers, while the W4A8 backend still reconstructs weights for computation;
the measured gain must account for both costs. The two warm intervals show
only about a 3.0% reduction in time, a modest result with too few measurements
to establish a precise sustained gain. Partial residency reduced pinned payload
and steady process RSS (about 48.6 GiB versus 63.1 GiB), while consuming roughly
11 GiB more GPU storage. The 0.70 run also completed all three updates with
finite/nonzero gradients
and native sampling, using another 2.14 GiB of frozen GPU storage. Its warm
median improved only 0.8% over 0.75, within the variability of these short
runs; the extra residency did not establish a useful sustained W4 speed gain.
At the 21870 MiB process peak plus 430 MiB background usage, about 1780 MiB
remained by that accounting on this GPU. Keep this aggressive profile limited to the measured
512 px/one-reference case; it is not an example default.

The full/0.75 comparison produced finite/nonzero gradients for all 128 factors.
Frozen dtype
bytes and resident/pinned totals were unchanged after conditioning destruction,
with no meta backbone tensors. Saved adapters are finite but not identical:
A-factor relative L2 difference 0.115%/maximum absolute difference 0.00012207;
B-factor relative L2 12.35%/maximum 0.00029278. Printed first losses match
(0.08353); subsequent full/partial losses were 0.07426/0.07525 and
0.09852/0.09838. Quantization and routing can amplify small update differences,
but the cause was not isolated here; full-model trajectory equivalence remains
unqualified. The resident/offloaded projection-gradient fixtures are separate
evidence. External code was not changed to enforce agreement.

The int8 warm update median was 12.74 s, approximately 1.9 times W4A8
throughput in these small runs, with substantially more host storage and cold
startup cost. This uses a different backend and is not an isolated residency
comparison; it also does not establish equal quantization or learning quality.

The additional int8 comparison preserved the earlier int8 job's 0.05 caption
dropout, seed 42, cached 512 px edit example, rank 16, three fresh updates and
two-step preview. Both included the 4325-token caption and 4317-token blank
edit branches. Increasing residency consumed the remaining GPU budget:

| Int8 offload fraction | Frozen GPU / pinned payload | Median warm update | Sampled process peak | Native two-step preview |
| --- | --- | --- | --- | --- |
| 0.85 | 11.301 / 63.844 GiB | 12.74 s | 19482 MiB | 19.84 s |
| 0.80 | 15.327 / 59.819 GiB | 12.55 s | 23162 MiB | 17.08 s |

The 0.80 run completed cache preparation, all three optimizer updates with
finite/nonzero gradients for all 128 factors, normal saves and native sampling.
Combined allocated/reserved peaks were 21.840/22.158 GiB. The independent
whole-GPU watcher recorded 23755 MiB used and only **325 MiB lowest sampled
free VRAM**; approximately 484 MiB of the advertised total was reserved by the
driver. The host watcher recorded at least 26.325 GiB available (approximately
99.379 GiB used), with maximum process RSS 109.41 GiB during loading and
approximately 78.4 GiB during training.

Another 4.026 GiB of frozen GPU storage reduced warm update time by only about
1.5% in these short runs. This demonstrates that the measured 512 px workload
can use the extra VRAM, while leaving too little margin for a general default.
Keep the Instruct int8 example at 0.85. Treat 0.80 as an aggressive measured
profile for this exact bucket/reference budget and seed-dependent module
selection; larger sequences, another selection or background GPU use remain
unqualified. No 0.82 fallback was needed and no further GPU runs were made.

Local diagnostic artifacts:

- `/home/bart/tmp/codex/h3-real-w4a8/metrics.json` and `h3-real-w4a8-run.log`.
- `/home/bart/tmp/codex/h3-real-edit-w4a8/resume-report.json` and `h3-real-edit-resume.log`.
- `/home/bart/tmp/codex/h3-int8-qualification/report.json` and `h3-int8-qualification.log`.
- W4 residency reports: `/home/bart/tmp/codex/h3-w4-bench-{full,partial}/report.json`;
  factor comparison: `/home/bart/tmp/codex/h3-w4-benchmark-comparison.json`.
- Additional residency reports: `/home/bart/tmp/codex/h3-w4-bench-070/report.json`
  and `/home/bart/tmp/codex/h3-int8-bench-080/report.json`; the int8 physical
  GPU/host headroom watchers are `h3-int8-bench-080-{gpu,host}-headroom.json`.
- Actual adapters/optimizers under each diagnostic's `output/` directory.
- Inspected native baseline: `/home/bart/tmp/codex/h3-real-edit-w4a8/baseline-50.png`;
  complete telemetry in `baseline-report.json` beside it.
- `testing/hunyuan_image_3_qualify.py` runs a real normal-trainer YAML and
  records sequence geometry, parameter-gradient hooks, adapter norms,
  process VRAM, CUDA allocations, host RSS and optional native previews.

The durable harness polls memory every 0.5 s; the additional physical-GPU and
host-budget watchers poll every 1 s. Their peak and lowest-free measurements
are sampled observations, not guaranteed absolute transient peaks or margins.
Completing cache preparation, training and native sampling demonstrates fit
for the measured workload, without establishing fit for a larger one.

The resume report's final status initially records an **extra diagnostic
preview harness** failure after the normal trainer had already released its
holder. Normal training, saving, periodic and final samples completed. The
diagnostic now performs extra previews before the existing final release.
Resume restores adapters, optimizer and step through the standard trainer.
Exact interrupted-training RNG continuation was not implemented or qualified;
sampling restores its surrounding RNG state through the existing lifecycle.
Local checkpoint identity includes resolved path, size and modification time.
Relocating an identical checkpoint changes its strict resume identity; historical
benchmark adapters/reports are preserved rather than rewritten. New jobs should
use the relocated asset path from the table above.
The earlier clean-reference time-dtype and native generator bugs were fixed;
they are covered by the small regressions.

## Numerical/reference evidence

Independent diagnostics read external reference files without changing them.
Real checkpoint projection slices match the unmodified `comfy_kitchen`
eager W4A8 arithmetic exactly: decoded INT8 and reconstructed FP32 weights,
FP32/BF16 forward results, and the declared frozen-weight surrogate input
gradient all had maximum absolute difference 0. The GPU comparison used a
64×4096 real QKV slice; the CPU comparison additionally used 2048 output rows
to cross the new row-chunk decode boundary. Gradients were finite/nonzero.
This is projection evidence, not a full denoiser prediction comparison or
proof of identical rounding to every optional Comfy CUDA/Triton backend.

W4A8 forward retains the intermediate INT8 reconstruction rounding,
normalized regular ConvRot, activation INT8 rounding, integer accumulation,
and a single FP32 output scaling. Backward is an explicitly declared
straight-through activation-quantization surrogate through the frozen
reconstructed rotated weight. It saves packed state, not dense weights or
linear inputs.

The local RMSNorm follows Tencent's FP32 normalization, cast of the normalized
activation back to input precision, then weight multiply. External Comfy's
`torch.nn.RMSNorm` differed by approximately 0.28% relative L2 in real-weight
BF16 tests (maximum absolute difference 0.03125; about 26% unequal values).
The router follows Tencent's native-precision softmax with autocast disabled,
rather than Pedro's FP32 softmax. These external differences are documented;
external code was not modified to improve agreement.

Cached VAE latents use deterministic posterior **mode**, as identified by
`vae_mean` in conditioning identity. This matches the Comfy mean-latent
convention; Tencent's full pipeline samples its posterior with a generator.
Do not claim bit-identical Tencent full-pipeline latents/generation without
accounting for that posterior noise. VAE parameters are FP32 with CUDA FP16
autocast, 32 latent channels, spatial downsample 16 and scale 0.562679178327931.

## Memory inventory call paths

| ID | Actual Hunyuan path and evidence | Remaining limit |
| --- | --- | --- |
| M01 | `HunyuanImage3Transformer.load_model` meta construction, mmap/shard reader, per-projection quantization; `.load` forwards conversion device; shipped qtype is preserved | Real BF16 conversion unqualified |
| M02 | Existing `convrot_quant`; W4 `decode_int8` bounds LUT/FP32 scratch by row chunks, and slices each expert before materialization | No fresh W4 conversion; clearly rejected |
| M03 | Existing BaseModel packed module cache, source/variant/backend/mapping identity; fixture verifies warm hit skips original loader; actual latent/text disk hits in resume | Huge packed cache write/warm load unprofiled |
| M04 | v2 post-load -> existing `MemoryManager`/quantized managers, asynchronous rings and backward re-fetch | Default depth 4 measured; other depths unqualified |
| M05 | Existing manager pins/moves actual W4 byte buffers/scales; resident/offloaded fixture verifies packed CPU/pinned placement and exact gradients | Int8 partial-offload 0.85 measured; full-offload host fit unqualified |
| M06 | Existing ConvRot custom autograd; W4 custom op saves only packed tensors and reconstructs one projection for input gradient; real LoRA gradients and checkpointed steps | Full saved-tensor trace not independently enumerated |
| M07 | Existing `EmbeddingLayerMemoryManager`; only actual text/special positions are gathered; table stays unpinned CPU; bounded one-table clone releases source mmap | No second vocabulary/CoT head |
| M08 | Existing managed `.device`, resident norm exceptions and dtype ownership; exact local RMSNorm/time casts; post-sample placement measured | Unseen original mixed-precision sources unqualified |
| M09 | Decoder loop calls non-reentrant checkpoints; fixture compares plain/checkpoint output+gradients; existing compile/offload regressions pass | Full-model compile unqualified; eager supported |
| M10 | Existing cache/generation presets, prompt/preview optimizer parking; sequential CFG, VAE/vision CPU restoration in `finally`; unused cached VAE excluded from preparation | Native provider only |
| M11 | Existing unloader directly destroys conditioning tensors to meta, preserves tokenizer/processor separately, releases host caches; lazy vision/VAE reload restores placement; optional final conditioning hook discards a reloaded unused VAE | Sampling keeps/reloads only needed frozen components |
| M12 | Holder `disable_dataloader_workers` capability -> `get_safe_dataloader_num_workers`; examples/UI workers 0, pin_memory False | Larger worker counts intentionally suppressed |
| M13 | `AdvancedPromptEmbeds` detached CPU payloads, existing disk cache hooks, ordered pixel identities, one frozen-reference scope shared by prompt branches; target IDs rebuilt from current latent bucket | No adapted backbone KV cache |
| M14 | Pinned Tencent VAE spatial tiling, explicit frozen no-grad encode/decode, singleton temporal axis, sequential targets/references/outputs; actual 512 px and isolated high-resolution encode/decode exercised | High-resolution seam/quality study unqualified |
| M15 | Comfy provider is rejected before loading | Host-aware two-backbone preview handoff not implemented/qualified |
| M16 | Existing low-rank LoRA factors and target filtering; actual training+native sampling, existing adapter/offloader restoration regressions, unmodified Comfy factorized-application diagnostic | Shared-MLP export/render full-model qualification pending |
| M17 | Normal saves contain adapters+optimizer only; packed cache writes occur while CPU-resident before attaching offloader; small packed-cache round-trip | Huge cache serialization profiling pending |
| M18 | Image-space/differentiable VAE losses rejected in preflight; ordinary detached flow-velocity target uses existing trainer | Hunyuan differentiable decode unsupported |

The Base wrapper reaches the same implementation paths, with its independent
variant identity/template. This does not establish a real Base memory or
quality result.

## Regression evidence

`testing/test_hunyuan_image_3_foundation.py` covers KV-head QKV ordering,
Base/Instruct geometry, reference causal/joint masks, real reference embedding
assembly in BF16, checkpoint gradients, canonical and banked loading,
global/per-expert codebooks, packed import/export, straight-through gradients,
conditioning disk round-trip/frozen integer dtypes, nonsquare bucket rebuild,
zero–three ordered/repeated references, blank edit branches, missing-pair/Base
preflight, resume identity, packed-cache loader bypass, native generator/CFG,
sample failure restoration, meta-vision reload and resident/offloaded gradients.

Independent combined run: 59 tests passed across this foundation and nine
existing memory/offload/cache suites. `npm run check_extensions` passed for
the UI entries. These small regressions are separate evidence from the real
checkpoint runs. Further implementation changes must rerun affected suites.

After the final loader/pool and conditioning-unload changes, 31 affected
foundation/prompt/offload/adapter tests passed, followed by all 16 then-current
foundation tests, including tokenizer mutation and actual meta-VAE destruction.
The subsequent ordinary UI trainer preflight correction passed all 17
foundation tests, explicitly accepting `sd_trainer`/`diffusion_trainer` while
rejecting specialized DPO, guidance-distillation and KTO modes.
The UI dataset follow-up passed three focused preflight regressions, including
mixed source/target extensions and exclusion of hidden thumbnail trees, hidden
files, disabled images, cache tensors and `_controls` images. Ordinary visible
subfolders still require valid source pairs.
The high-resolution UI cache pass exposed a caller with gradient mode enabled,
which disabled the wrapper's original tiling gate. Frozen encode/decode now
explicitly disable gradients; low-VRAM tiling no longer depends on caller mode.
A focused regression passed for that caller and failure cleanup. The real VAE
then encoded 848×1232 and 880×1184 targets and decoded 896×1200 while a 12 GiB
CUDA allocation simulated transformer residency. All outputs were finite with
the expected shapes; peaks including that reserve were 17.708 GiB allocated/
18.447 GiB reserved for encoding and 17.857/18.566 GiB for decoding. This is
isolated VAE fit evidence, separate from full high-resolution training.

Comfy parser diagnostic consumed all 192 adapter keys, checked all factors
against real checkpoint shapes and applied 64 factorized patches with maximum
absolute difference 0 from the native factor formula. Importing the full
Comfy runtime in the project environment is blocked by its separate
`comfy_aimdo` dependency. No external dependency/code was changed; a full
Comfy render remains unqualified.
