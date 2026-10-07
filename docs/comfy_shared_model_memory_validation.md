# Shared Hunyuan model memory implementation and validation

Status: **`runtime_qualified=false`** globally; the local Instruct int8 edit workflow
has passed two short roundtrips and one full 50-step trainer/Comfy roundtrip.
The initial implementation was completed on disk without changing the
running trainer/services or performing GPU tests. The user subsequently authorized
cancellation, deployment, service restarts, resumed training and render testing.
The coordinator stopped the trainer cleanly at checkpoint 239, backed up resume
state, installed the package into both environments and linked the Comfy mod.
The first resumed 50-step attempt completed sampling but failed at decode; a
cached-wrapper fix passed two short live roundtrips in both directions, including
reuse of the cached Comfy model. The final full 50-step retry produced its image,
resumed training and saved the subsequent step-241 checkpoint and optimizer state.
The job is now **stopped at checkpoint 241 as requested**, with the trainer PID
gone, queue stopped and broker owner/queue/fault cleared after verified recovery.

The initial supported configuration is one local GPU, frozen-base LoRA training,
HunyuanImage 3 **Instruct int8 ConvRot**, with strict shared attachment. Ordinary
loading and sampling remain the default when shared configuration is absent.
W4A8, Base, full fine-tuning, base merging, multi-GPU clones and an ordinary
two-backbone Hunyuan Comfy preview are rejected or remain outside this release.

## Implementation delivered

| Area | Implemented behavior |
|---|---|
| Shared CPU storage | Independently installable, standard-library broker; bounded checkpoint reads into a sealed memfd or explicit local-file fallback; descriptor/shape/dtype/extent/provenance checks; read-only client mappings and bank/expert aliases. |
| Ownership and transfers | Sidecar CPU-master ownership survives Parameter/view wrapping; writes, full CPU clones, private pinning and raw writable exports rejected; bounded two-buffer pinned staging with at most 512 MiB per arena owner; optional experimental read-only host-registration probe. |
| Trainer loader/offloader | Existing streamed H3 loader and int8 importer accept descriptor tensors; strict zero-copy scales/codes/bias; retained frozen CPU masters; explicit suspension/resumption preserves resident placement, private state and existing computation paths. |
| Trainer boundary | Shared handoff occurs after a verified successful optimizer/EMA update, gradient clearing and graph/batch release. Waiting retains UI `running` status, does not advance training, and polls cancellation. Auxiliary modules, optimizer/EMA and private tensor roots are parked; surviving CUDA allocations prevent acknowledgment. |
| Comfy mod | All Comfy integration is in `integrations/ComfyUI-AITK-SharedModels`; meta construction and direct binding cover quantized and dense weights; source/signature compatibility checks; package-owned operations and non-dynamic patcher preserve base aliases and reject unsafe copies. No installed Comfy/Pedro sources were edited. |
| Whole-prompt lease | Version-checked reversible execution wrapper gates every prompt before node GPU work. Lease covers cached loaders, VAE/vision conditioning, sampling, decode and cleanup. Inflight acquisition and cleanup are drained on async cancellation; failed quiescence retains ownership. |
| Adapter snapshots | Immutable safetensors generations with base/variant/rank/alpha/step/hash; real H3 fused projection key mapping; private factorized patches; latest resolved once per prompt; queued/inflight references protect generations from pruning. |
| Scheduled previews | Shared provider publishes an explicit generation, releases trainer ownership before submit/wait, builds text/edit API workflows, retains existing sample progress/timing/output handling and uses targeted prompt cancellation. Euler/simple and encoder-provided unconditional conditioning are the initial sampling contract. |
| Failure handling | No lease release by timeout, heartbeat or disconnect. Dead-owner recovery requires exact process identity and separate verified CUDA release. Builder failures discard incomplete generations; surviving clients retain their mappings after another client exits. |

Comfy AIMDO/dynamic packing and dynamic lookahead are **not used**. Bank-resident
quantized expert execution and the parent's activation-quantization dispatch are
preserved, but actual kernel execution and throughput require qualification.
The small real INT8 activation path and full-model smoke described below now
provide initial execution evidence; throughput remains an explicit limitation.
Small CPU conditioning projections may cast to a private temporary, capped at
64 MiB per projection and accounted in `TensorOwner.cpu_temporary_peak_bytes`.
GPU LoRA projection temporaries are capped at 512 MiB per projection. These are
separate from the 512 MiB pinned-staging budget.

## Completed CPU evidence

The final new-fixture run used the project's Conda environment with CUDA hidden
and BLAS/Torch CPU concurrency restricted:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n ai-toolkit python testing/test_shared_model_memory.py
```

Final complete fixture run after the runtime inference-wrapper fixes:
**32 tests passed in 6.163 seconds**. Five broker tests are deliberately inherited
by the execution fixture, so this is 32 executed tests and 27 distinct test
methods. The earlier run after UUID normalization, UI request deferral,
private-cache parking and startup lease guarding passed 30 tests in 5.264
seconds. Coverage includes:

- Sealed/read-only arena descriptors, source symlink identity, invalid extents,
  bank/expert offsets, Parameter/slice lifetime, write rejection, rejected NumPy
  exports and read-only storage accounting proxy ownership.
- Real CPU Comfy Kitchen quantized wrapper detach/Parameter behavior, scale
  aliasing and preserved ConvRot parameters, loaded directly without unrelated
  CUDA-initializing plugin imports; strict H3 importer codes/scales/bias aliases.
- READY store attachment through descriptor passing, common backing inode,
  cold-start builder exclusion, failed builder, incompatible metadata, importer
  exit/reattachment, lease epochs, stale acknowledgments, timeout/cancellation,
  disconnected-owner failure, immutable snapshot hashes/references/pruning.
- Whole-prompt failure/reentrancy/finally cleanup, cleanup failure retaining the
  owner, cancellation while a worker grants ownership, and cancellation during
  asynchronous cleanup.
- Validated physical UUID normalization and exclusion when one participant uses
  NVML's `GPU-` prefix and another uses Torch's bare UUID.
- UI render-next-step requests take the shared deferred boundary path without
  discarding accumulated gradients; ordinary UI sampling retains its prior path.
- Optimizer-owned quantization caches and scheduler tensor attributes follow exact
  device parking/restoration in a CPU fixture with simulated device metadata.
- Shared startup avoids an out-of-band Comfy unload while another prompt owns
  the lease; ordinary startup release retains its prior behavior.
- Tiny actual CPU Adam training with accumulated gradients, optimizer update,
  explicit park/resume, unchanged adapter/optimizer/RNG state and injected
  retained-allocation rejection. CUDA methods are mocked in this CPU fixture;
  it establishes state bookkeeping, not GPU quiescence or device restoration.
- Saved adapter serialization keys to Comfy targets, explicit snapshot workflow
  substitution for text and image references, bounded conditioning dtype casts
  and prompt-specific interrupt payloads.
- Shared CPU aliases and quantized wrappers remain coherent across repeated
  inference-execution/ordinary-cleanup Parameter cycles; bounded GPU patch/set
  disables inference mode and gradients, avoiding Comfy's hidden scale-copy path.
- Snapshot fingerprinting with an unresolved linked MODEL requests fresh
  execution without resolving a generation prematurely.

Existing H3 foundation, Comfy training offload and adapter offload regressions
also passed: **29 tests, one GPU test skipped**. Normal collection is blocked by
an unrelated eager diffusion-model registry importing an OmniGen2 Triton module
that queries CUDA at import time. The isolated runner bypassed only that registry
while loading the real tested modules. No external code was changed to collect
the tests. Review-session runner/log: `/tmp/run_shared_existing_checks.py` and
`/tmp/shared-existing-checks.log`.

These CPU fixtures use tiny tensors and multiple clients in the trainer environment.
They do not establish separate-process physical-memory measurements, the actual
Python 3.11/3.12 bridge, a complete installed Comfy model construction, rendering,
or any CUDA transfer/cleanup behavior.

## Authorized tiny CUDA prerequisite

With the trainer stopped and no Comfy prompts queued, one isolated ai-toolkit
CUDA process acquired the broker lease and tested a tiny sealed descriptor arena.
It passed int8-code and FP32-scale asynchronous H2D transfer, copy into an existing
GPU destination, and GPU-to-CPU restoration of a Parameter using the installed
Comfy Kitchen base wrapper. Restoration kept the original CPU code/scale pointers
and ConvRot metadata. Peak Torch CUDA allocation was **3,584 bytes**; the test
explicitly configured two **1 MiB** pinned staging buffers and staged 72 bytes.
After synchronization/cleanup, Torch allocation was zero and the broker accepted
the quiescence acknowledgment. Review-session script:
`/tmp/aitk_shared_tiny_cuda_check.py`.

This used a fixture layout with the real Comfy Kitchen base tensor class. It did
not execute real int8 matmul kernels, host registration, the default 512 MiB
staging configuration, full expert banks or the independent Comfy environment.
Both inspected Torch runtimes use the same bare physical UUID; code now validates
and canonicalizes both that form and NVML's case-insensitive `GPU-` prefix.

The actual Instruct arena was then built successfully: **76.1624 GiB**, sealed
memfd, **238.80 seconds**, with **37.84 GiB MemAvailable** immediately after
construction. The resumed trainer attached the store, loaded its saved adapter
and optimizer, and completed the next optimizer update. Handoff initially failed
closed on 16 KiB of scheduler allocations. Failure diagnostics identified
`timesteps`, `sigmas`, `alphas_cumprod` and `betas`; their owner is now included in
exact park/restore. No acknowledgment was issued while those tensors remained.
The coordinator verified failed worker death and CUDA resource release before
broker recovery, retaining the completed arena for retries.

After scheduler parking was fixed, the trainer's real suspension passed the zero
allocation audit and acknowledged quiescence. The first workflow preflight found
three required Pedro image-encoder settings; the template now supplies
`vit_strength=1`, `latent_strength=1` and `reference_vit_padding=true`, and text
conversion removes those image-only inputs. The exact 896×1200, 50-step, CFG 2.5,
seed-42 edit graph passed validation against live Comfy `/object_info`. Its API
graph and submission payload are `/tmp/hunyuan_shared_live_workflow.json` and
`/tmp/hunyuan_shared_live_prompt_payload.json`, with the explicit step-240 snapshot
and `extra_pnginfo.workflow`.

The first direct Comfy execution constructed and attached the full shared model
and completed image conditioning, then found a hidden CPU scale clone in Comfy's
quantized LoRA copying path. The mod's patcher now constructs the bounded private
GPU temporary directly from descriptor leaves before conversion and LoRA merging,
preserving normal factorized patch/requantization semantics. No Comfy Kitchen or
Comfy core source was changed. Two targeted CPU tests cover that transfer contract
and the corrected text/edit template; the existing complete 30-test result above
predates this additional patcher regression.

Comfy's cache fingerprint prepass intentionally omits linked MODEL inputs. The
snapshot node now explicitly returns a fresh-execution fingerprint when that
input is unavailable, preserving the previous exception fallback's behavior
without its warning. Execution still validates the resolved MODEL and pins one
snapshot generation. One focused CPU regression passed in **0.002 seconds**; this
on-disk warning fix was not loaded into the active render's Comfy process.

An authorized tiny probe in the actual **Comfy Torch 2.11.0+cu130** environment
passed **real INT8 ConvRot LoRA GPU patch/requantization**, activation-quantized
forward execution, a nonzero adapter effect and exact immutable CPU code/scale
pointer restoration. Peak Torch CUDA allocation was **8,617,984 bytes**. After
cleanup, Torch CUDA allocation was zero and the broker accepted quiescence.
Review-session script: `/tmp/aitk_shared_comfy_lora_cuda_check.py`. This qualifies
the small projection path.

A direct full-model **one-step edit smoke test** then succeeded with the real
shared Instruct checkpoint and step-240 training adapter: 896×1200, CFG 2.5,
seed 42, the same uploaded lineart reference. Comfy history confirmed an image;
the prompt reported **38.18 seconds**, including conditioning/model preparation,
one sampling step (**19.93 seconds**) and decode. Prompt ID:
`d067f09b-c5ff-4250-aa58-b25572e30445`. The coordinator verified broker owner/fault
were both null after cleanup and Comfy idle driver usage was **572 MiB**. This is
smoke evidence, not a 50-step quality/performance baseline. The resumed trainer's
requested 50-step next-step preview then exercised the full sampling path.

That preview, prompt `6967c26f-ecbd-4063-b616-75356c1d7fc9`, completed **50/50
sampling steps in 853.8 seconds** (17.07 seconds/step average), then failed during
VAE decode's partial backbone unload with `Cannot set version_counter for
inference tensor`. The trainer measured **863.15 seconds** waiting for zero of
one output images and exited with the error. This is failed-attempt timing, not
a successful render duration. Its exact trainer workflow is
`output/hunyuan-image3-instruct-lineart-remover-500/shared_preview_api_workflow.json`;
snapshot `631d405b-5ea5-42b3-b6a2-f968c7a2b31a`.

An actual Comfy Torch 2.11 CPU reproduction isolated the cause: node execution
created inference-mode shared tensor wrappers; ordinary finally cleanup rebuilt
a normal outer wrapper around inference inner leaves; a later inference-mode
Parameter detach could not assign its version counter. The shared bridge now
creates immutable mapped leaves and quantized alias metadata outside inference
mode in every execution context. This preserves the same pointers/read-only
pages and does not allocate a private backbone. The exact reproduction passes;
three targeted CPU tests passed in **1.798 seconds**, including three repeated
inference-execution/ordinary-cleanup Parameter cycles with unchanged code/scale
pointers. Review script: `/tmp/aitk_shared_inference_repro.py`. Live repeated
trainer/Comfy cycles and another full 50-step render are the next gates.

A subsequent short live test exposed the corresponding GPU LoRA construction
case before sampling: inference-mode requantization and Parameter wrapping also
need coherent metadata. The policy is now restricted to immutable CPU aliases;
the mod's bounded GPU patch/set runs with inference mode disabled and gradients
disabled, and frozen module placement rewraps Parameters in that same context.
The actual Comfy GPU probe then passed **three** cycles of inference-mode
patch/load/activation-quantized forward, decode-style backup restoration inside
inference mode and ordinary final master restoration. Peak GPU allocation stayed
**8,617,984 bytes**, followed by zero allocated bytes and broker quiescence ACK.
The focused CPU setter regression now asserts that full patch/set preserves
normal tensor metadata even when invoked by an inference-mode caller.

The first actual trainer-requested **one-step** retry then succeeded: prompt
`b2bbd2b8-4a66-4dd8-90be-fedf2f17f2e1`, **33.17 seconds** measured by the trainer's
sample provider, output
`1791339009158__000000240_0_00001_.png`. After releasing trainer ownership for
conditioning/sampling/decode, Comfy cleaned up and returned the lease; the trainer
resumed and completed the next update, advancing to **step 241** with UI status
Training. This verifies one real handoff in both directions for this job. A
second cached-model next-step preview also succeeded: prompt
`785e9cf2-ffc5-4f74-9dfd-07ef929b0d92`, **39.10 seconds** measured by the trainer's
sample provider, followed by a completed training update to **step 243**, loss
`8.362e-02`. Both directions of the handoff therefore succeeded twice in the live
job. A successful full 50-step render remains pending. The user requested
stopping training after the full working render and verification of both handoff
directions; final state will be recorded after that authorized stop.

The coordinator then stopped the short-test job to restore the 50-step setting.
The UI stop interrupts the loop rather than taking the normal completed-job
cleanup path. `on_error` intentionally returns early for stop/interrupt; its
traceback may still retain batch/graph tensors or accumulated gradients. The
trainer disconnected without a quiescence acknowledgment, so the broker
correctly retained a fault instead of allowing a concurrent owner. The
coordinator verified the exact trainer PID had exited and had no NVML GPU
allocation before explicit broker recovery. No stop-time audit was bypassed.
Normal completion already calls coordinator `close()` with suspension and a
strict zero-allocation audit. Graceful mid-loop stop cleanup remains a separate
lifecycle enhancement; until then, an interrupted owner requires verified
post-exit recovery.

The final **50-step** retry succeeded: prompt
`0c90f084-08be-4feb-923c-b537a7a02a59`, snapshot
`42731b69-aad1-481c-b9a7-bfb14d68edb0`, 896×1200, CFG 2.5, seed 42,
Euler/simple and the same lineart reference. The trainer's sample provider
reported **866.14 seconds (14 minutes 26.14 seconds)** for **1/1** images.
Comfy history's execution-start/success timestamps measured **859.51 seconds**,
excluding remaining cleanup/download/provider overhead. The downloaded output is
`output/hunyuan-image3-instruct-lineart-remover-500/samples/1791339293747__000000240_0.png`.
Visual inspection confirmed a valid 896×1200 image with the reference scene
preserved and softened lineart. After Comfy released ownership, training resumed
and completed **update 241**, finite loss **`1.137e-01`**, then saved
`hunyuan-image3-instruct-lineart-remover-500_000000241.safetensors`. This qualifies
the requested full render and both handoff directions for this job/configuration.
It does not qualify other formats, error/cancel cases, environments or performance
equivalence. The measured provider duration is **1.93×** the archived native AITK
448.87-second preview; transfer scheduling remains an explicit optimization gap.

Final state was verified after the user-requested stop: API job status **stopped**,
step **241**, `save_now=false`, `sample_now=false`; training queue
`is_running=false`. Trainer PID **684049** was absent and had no NVML GPU
allocation. The remaining Comfy process used **530 MiB** of driver VRAM.
After verifying trainer death and GPU release, explicit broker recovery returned
**epoch 20**, **owner null**, **queue empty**, **fault null**. The step-241 adapter
checkpoint and optimizer state are saved. The actual full API workflow and
complete Comfy history are preserved in the job output directory as
`shared_preview_api_workflow.json` and `shared_preview_history.json`.

During the earlier active 50-step preview that later failed at decode, broker
diagnostics reported **30.64 GiB
MemAvailable**, with the same **76.1624 GiB** committed shared arena attached by
both trainer and Comfy. Trainer RSS was 81.17 GiB, but PSS was 43.57 GiB and its
private clean/dirty total was 5.97 GiB: RSS includes shared mapped pages and is not
a second physical backbone allocation. The broker could not read Comfy's smaps,
so aggregate PSS remains incomplete. The captured report is
`/tmp/hunyuan_shared_render_memory.json`.
During the final successful render, read-only monitoring observed approximately
**29 GiB MemAvailable** and **22,207 MiB** driver VRAM use. These are observed
samples, not aggregate process-memory or peak-allocation measurements.

The shared render began at approximately **20 seconds per sampling step**.
Archived native AITK previews of the same 50-step job reported **447.64** and
**448.87 seconds** total. These are different render backends and no controlled
ordinary-Comfy baseline has been measured. Read-only source review confirms that
shared Linear inherits Comfy's activation quantization, MoE uses the parent's
fast expert dispatch, image passes transfer each bank once per resident context,
and sparse decode slices the expert before transfer. Native AITK also executes
separate positive/negative CFG predictions. There is no identified extra CFG pass
or repeated whole-bank image transfer to correct immediately. Shared fallback
staging adds a CPU copy of every transferred byte and enqueues H2D on the compute
stream; staging-buffer reuse synchronizes that stream. Disabling Pedro's dynamic
layer lookahead therefore removes transfer/compute overlap. A package-owned
bounded offload-stream/prefetch path requires profiling and separate validation;
enabling incompatible AIMDO packing is not a supported workaround.

## Qualification still required in an authorized idle window

| Gate | Required evidence | Current result |
|---|---|---|
| Actual two environments | Independent Python 3.11 trainer/Python 3.12 Comfy descriptor attach, expert/scale aliases, importer process exit/restart and aggregate PSS/private bytes | Both runtimes attached the real arena and Comfy rendered; trainer PSS/private bytes measured. Aggregate PSS/importer restart validation remains incomplete |
| CUDA transfer | Read-only host-registration capability, asynchronous correctness/event lifetime and unregister; bounded pinned fallback H2D/copy-into correctness, allocation bounds and throughput | Tiny fallback H2D/copy-into passed in trainer runtime; real small ConvRot LoRA/activation path passed in Comfy. Registration/default-budget/full-bank throughput checks remain pending |
| Real Comfy constructor | Installed Pedro/Comfy source baseline, dense/quantized direct binding, clone/unpatch/cache teardown, CPU vision alignment and real activation-quantized expert dispatch | Full descriptor model/edit conditioning/sampling/decode succeeded in smoke, two trainer-driven cached cycles and the final 50-step render. Wider configurations and controlled numerical comparison remain pending |
| Full backing arena | Exact 76.16 GiB checkpoint construction, memfd/cgroup or local-filesystem capacity, committed/resident shared bytes and at least 12 GiB `MemAvailable` under the real workload | Actual 76.1624 GiB sealed memfd built in 238.80 s; 37.84 GiB MemAvailable after build and 30.64 GiB during the active render. Aggregate PSS remains incomplete |
| GPU quiescence | Trainer and Comfy release every resident/private tensor, auxiliary model, stream/cast/staging/cache and library workspace; allocator and driver VRAM agreement; cancellation and decode-error cleanup | Actual trainer suspension and Comfy cleanup returned ownership across two short cycles and the full 50-step cycle. Earlier decode error cleanup left broker idle; abrupt trainer stop requires verified dead-owner recovery. Wider cancel/decode-error cases remain pending |
| Handoff correctness | Repeated trainer → Comfy → trainer cycles, exact resident restoration, optimizer/EMA/RNG/data/scheduler equivalence, finite nonzero gradients, no skipped updates and no warm checkpoint reads | Tiny CPU state fixture plus two short cached cycles and a full 50-step trainer → Comfy → trainer cycle, each followed by a finite-loss update; final update241 checkpoint saved. Full state equivalence remains pending |
| Adapter rendering | Chosen generation, nonzero adapter effect, zero-strength agreement within the same Comfy backend, edit/text outputs and failure/cancel handling | Small actual ConvRot LoRA probe showed nonzero effect; full 50-step edit rendered an explicit step-240 snapshot and passed visual inspection. Controlled zero-strength/text/failure comparisons remain pending |
| Scheduled/UI behavior | Several scheduled preview cycles, normal output/UI progress, stop while waiting, latest cache invalidation and queue occupancy while status is running | Real UI next-step requests completed two short previews and one 50-step preview with explicit snapshots, per-image progress, downloaded outputs and resumed training. Scheduled cadence/latest/stop-while-waiting cases remain pending |
| Performance/headroom | Ordinary/shared Comfy identical-input timing, trainer warm-step timing, host-copy cost, peak physical RAM/VRAM and about 1 GiB GPU headroom where achievable | Successful full preview866.14s, 1.93× archived native448.87s; measured host headroom30.64GiB. Controlled ordinary-Comfy comparison and transfer profiling remain pending |

The zero-CUDA-allocation audit deliberately fails closed. It may expose runtime
caches or private roots not represented by the tiny fixtures. Guarded cuBLAS
workspace clearing is included after synchronization, without relaxing the audit;
the installed CUDA runtime and other retained library/graph allocations must be
checked later. A surviving owner or failed cleanup requires explicit recovery,
not concurrent ownership or an automatic restart.

Setup instructions are in the package and custom-node READMEs. The example YAML
is a fragment for an existing-style Instruct int8 LoRA configuration. Deployment
and resumed-job configuration during this session were separately authorized;
further production rollout should account for the remaining qualification gates.
