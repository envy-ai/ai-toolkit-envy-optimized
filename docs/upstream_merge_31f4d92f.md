# Upstream merge through 31f4d92f

The merge starts from fork recovery commit `a5e4aeb7` and integrates upstream
`origin/main` at `31f4d92f`. Existing training processes keep their imported
Python modules; no job restart, configuration, cache, output, Comfy template or
Nunjucks renderer update is part of this merge.

## Conflict resolutions

- `extensions_built_in/diffusion_models/__init__.py`: register both fork Hunyuan
  models and upstream Flux.3/MOSS models, retaining both Iris implementations.
- `extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py`: add upstream
  Turbo scheduling and attach its frozen training adapter after quantization and
  before layer offloading. Reuse the fork's strict helper importer, preserving
  split/fused MLP conversion, per-layer ranks/scales, complete weight validation
  and external text-encoder support. The helper accepts local files or the
  upstream `owner/repo/file` Hub form. The shared helper's optional `is_ara`
  argument lets offloading wrap the helper's base call without streaming its
  frozen factors or bypassing the adapter. While streaming attachment is
  deferred, CPU parking prevents a temporary full-transformer GPU allocation,
  including when layer offloading is enabled without `low_vram`.
- `extensions_built_in/diffusion_models/ui.tsx` and `toolkit/models/registry.py`:
  keep fork `iris` defaults/controls and add upstream `iris3b` separately.
- `extensions_built_in/sd_trainer/SDTrainer.py`: retain frequency-filtered pixel
  loss and add upstream band loss.
- `toolkit/config_modules.py`: retain fork fields and disabled-by-default pinned
  memory; add upstream pooled datasets and batch loading threads.
- `toolkit/data_loader.py`: scan each pooled source's actual path while retaining
  recursive KTO `_controls` exclusion and fork per-item cache/reference flags.
- `toolkit/util/comfy_quant_import.py`: keep strict shared-storage import and add
  rotated-row embedding imports, which dequantize only selected rows.
- `toolkit/util/convrot_quant.py`: use upstream's three-value Triton helper and
  update all call sites; keep `libdevice` published in module globals for older
  Triton versions, alongside the fork's XLA work. The fork-only AdamConvRot
  optimizer also unpacks the updated three-value helper.
- `ui/src/app/jobs/new/SimpleJob.tsx`: retain specialized-mode controls and the
  fork's dataset layout; add Pool Datasets in the existing Advanced card.

## Memory and active-job review

The automatic merges retain CPU latent restore, atomic safetensor cache writes
and validation, processed-reference/caption-dropout cache identity, Qwen target
size cache keys and bucketed references, tiled VAE decode with the training
gradient exception, shared model component storage, adapter/optimizer preview
offload and the physical-VRAM recovery gate. Upstream adds lazy uint8 pixel
materialization and parks the text encoder after fresh embedding caching.

Runtime-read Comfy templates and `ui/scripts/render_comfy_template.mjs` are
unchanged from the recovery commit. The existing Qwen sprite-rotation job is not
restarted or reconfigured. GPU training or live rendering validation would
compete with that job and is deliberately not part of these CPU checks.

## Verification

- Python AST parsing succeeds for all 62 changed Python files.
- `git diff --check` succeeds.
- Extension TypeScript check succeeds:
  `cd ui && ./node_modules/.bin/tsc --noEmit -p ../tsconfig.json`.
- Full UI TypeScript checking still reports existing Next route signature errors
  from generated `.next/types` for `api/inference/[...path]`,
  `api/jobs/[jobID]/kill` and `api/jobs/[jobID]/notes`. Those route sources are
  unchanged by the merge; no unrelated route fixes are included.
- Focused CPU diagnostics run under a transient systemd unit with
  `MemoryMax=16G` and `MemorySwapMax=0`, with CUDA hidden. Legacy fake file-item
  fixtures are updated for upstream's per-source `dataset_config` and
  `dataloader_transforms` contract; production DTOs already supply both.
- Final focused result: 102 CPU test cases pass, plus one expected shared-model
  skip. The separate Iris CPU runner passes 18/18. The model-loading registry
  listing succeeds and includes `moss_image` and `iris3b` without loading model
  weights. Unaffected passing cases were retained; only failing fixture modules
  were rerun after the fixture corrections.
