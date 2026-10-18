# Exporting the Qwen 2.1 Fizgig paired cache

`workers/kaggle_tpu/export_paired_cache.py` stages the cached training inputs for
the local `qwen2.1-anime-34_copy-diffguidance` job. Despite its name, this job is a
Fizgig image slider with signed DoRA and differential guidance. Its BEFORE
images are the -1 training targets, not edit-reference conditioning.

The exporter reads the job database in read-only mode, uses the existing
`FileItemDTO`, caption handling and bucket calculations, and applies the normal
repeat/flip expansion. It does not start or modify the source job. Hidden
thumbnail directories are excluded. Source images, model weights, database
contents, credentials, old adapters and optimizer state are not copied into the
export. No remote upload occurs.

Inspect missing cache paths without loading any model weights:

```bash
conda run --no-capture-output -n ai-toolkit python \
  workers/kaggle_tpu/export_paired_cache.py --audit-only
```

Fizgig normally stores negative latents only in trainer RAM. Generate absent
paired latents locally and publish a complete export with:

```bash
conda run --no-capture-output -n ai-toolkit python \
  workers/kaggle_tpu/export_paired_cache.py \
  --prepare-missing --device cuda --seed 42 \
  --output tmp/kaggle_qwen21_pairs
```

Preparation loads only the Qwen VAE through the same component loader/source
selection as ordinary Qwen training. It uses the existing paired image loader,
including RGB conversion, EXIF handling, flips, Mitchell resizing and the
positive item's crop geometry. Qwen's existing `encode_images` supplies opaque
alpha, samples the VAE posterior and normalizes latent statistics. Missing
positive caches use the ordinary image loader and cache metadata. Negative
caches are stored separately under `tmp/kaggle_paired_negative_cache` with a
content/presentation identity. Existing cache files are reused, and existing
invalid caches fail validation. This command can use the configured model cache
or download the VAE/config through the normal loader if absent.

Without `--prepare-missing`, every latent and exact text-conditioning cache must
already exist. Missing text conditioning always raises an error: prepare it
through AI Toolkit's normal text-cache workflow first. The exporter never picks
an older or approximately matching cache. Cached text embeddings and their
native integer/bool masks are copied verbatim. No encoder is loaded.

`--config PATH` accepts a JSON/YAML process or complete training job instead of
the database. `--database PATH --job NAME` chooses another compatible source.
The profile intentionally rejects stochastic preprocessing, caption
randomization, extra edit references, masks, preservation anchors, multipoint
training and multiple resolutions per dataset. It supports deterministic x/y
flips and configured repeats. This is a scoped staging tool, not a general
dataset converter.

The destination must not exist. All output is written into a temporary sibling
directory and renamed only after validation completes. A failure removes that
partial export; completed local latent caches remain reusable.

## Export contract

`manifest.json` has `schema: 1`, `kind: qwen21_fizgig_paired_cache`, and an explicit
`initialization: fresh_adapter` policy. It records the source configuration hash,
Git commit, relevant working-file hashes, model/helper/encoder component file
hashes, preserved objective/network settings, preparation seed, unique pairs,
repeat/flip exposure IDs, and a checksummed file inventory. Existing pretrained
adapter/start-step fields are removed from the exported fresh-run profile.
Component file paths become logical `component:<field>` references; the actual
large assets must be staged separately by the later training launcher.

Each pair has source-relative image names and SHA-256 hashes, dataset index,
crop/resize/flip geometry, caption and loss multiplier. Its `latents` reference
points to a content-addressed safetensors file containing `positive` and
`negative` tensors, each `(64, crop_height/16, crop_width/16)`. Its `conditioning`
reference points to the original `AdvancedPromptEmbeds` serialization with
`text_embeds`, `attention_mask` and `image_slot_mask`. This image-pair profile
requires no edit-image slots. All tensors must be finite and structurally valid.

The source job has 230 original image pairs, 255 unique presentations and 355
weighted exposures: 25 pairs repeated three times with horizontal-flip copies,
plus 205 pairs without flips. Exposure order is canonical; the replicated runner
owns deterministic global shuffling, rank partitioning and remainder policy.
Its approved batch is eight pairs per global update for 40,000 updates.

Legacy positive cache metadata identifies architecture, source file and
presentation, but does not identify the VAE weight hash. The exporter records
this limitation rather than claiming full legacy VAE provenance. Hashes identify
the exact exported tensors. Fresh negative posterior samples also need not be
bit-identical to a previous local training session's in-memory samples.

## Verification

The focused CPU suite covers profile rejection, real DTO pair/crop/flip/repeat
mapping and hidden-folder exclusion, missing caches, content-addressed tensor
publication, negative preparation through paired pixels, and incompatible cache
metadata. It runs in the required 16 GB/no-swap transient scope:

```bash
systemd-run --wait --collect --pipe \
  -p MemoryMax=16G -p MemorySwapMax=0 -p "WorkingDirectory=$PWD" --uid=bart \
  conda run -n ai-toolkit python -m unittest \
  testing.test_kaggle_paired_cache_export -v
```

The six focused tests passed on 2026-10-16 with 626 MB peak host memory and no
swap. The actual local VAE preparation/export also completed: 255 missing BEFORE
presentations were generated on the local RTX 4090 and exported with the existing
AFTER/text caches into `tmp/kaggle_qwen21_pairs`. Independent integrity validation
verified all 257 tensor files (267,578,010 bytes), 255 pair IDs and 355 exposures;
the dataset pair counts are 50 and 205 after deterministic flip expansion.
Evidence is in `tmp/kaggle-paired-cache-export.log`,
`tmp/kaggle-paired-cache-tests.log` and `tmp/kaggle-paired-cache-integrity.log`.
Nothing has been uploaded to Kaggle. This is exporter evidence; full-model TPU fit, attention performance and
all-device training remain separate gates described in
[the worker plan](kaggle_tpu_worker_plan.md).
