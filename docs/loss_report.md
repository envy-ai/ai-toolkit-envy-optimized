# Per-image loss reports

In **Training** settings, **Loss reporting** is on by default for every training
mode. Turn that switch off to disable recording when starting or resuming
training. **Record noise RNG state** remains off by default. The job's **Loss Report**
tab (Material `assessment` icon) sits beside **Loss Graph**. Existing scalar losses
cannot be linked to images retroactively.

The trainer records inputs after a successful training step, with available
step/microbatch/item indices, objective losses, weights, timesteps, noise
statistics, captions, crop/flips and model settings. Ordinary diffusion training
records its per-image objective vector. Shared auxiliary terms or aggregate
clipping are labeled **primary component**. CFG distillation records per-image
MSE and teacher-correction RMS; DPO records pair objectives on preferred/rejected
inputs; KTO records its per-image utility loss. Other objectives associate all
inputs with the **shared step** loss, leaving independent image losses blank.
Shared-step entries appear in step reports and are excluded from individual
loss reports. The attribution column makes those differences visible.

The stored step loss is the trainer's actual reported loss, including special
reductions such as KTO windows with uneven microbatches. Logging does not add
model passes, decode latents, copy dataset images, reseed, or change gradients.
Only the main process records its local images when using multiple processes.

SliderSpace records its selected discovery image and direction. Fizgig records
its selected practice image, paired/multi-point targets and preservation anchors.
Prompt-slider practice images are saved as they are already generated, under
`loss_report_practice` in the job folder. Pure prompt objectives without a source
image have prompt-only records.

Records share the existing job `loss_log.db` through deduplicated metadata,
per-exposure rows and optional per-microbatch RNG snapshots. SQLite WAL writes
are buffered. Noise RNG snapshots are optional and use additional storage;
CFG distillation snapshots precede its timestep/noise draws. Other modes can
record state before the training hook; reproducing their noise additionally
requires the same preprocessing and draw sequence. Reproducing predictions needs the exact
checkpoint, helper weights and cached conditioning/latents.

The report compares either **step losses** (showing every image in those steps)
or **individual weighted image losses** against the mean of the preceding
logged training steps. The current step is excluded. Defaults are a 100-step
window, at least 20 prior finite losses, and a strict threshold of **3×** the
mean. Shorter windows require that many prior finite losses. **Top X spikes**
selects the biggest ratios above the mean: X distinct steps in step mode, or X
exposures in image mode. Nonfinite losses and positive losses over a zero mean
are flagged separately and ranked first.

Start/end step bounds are optional and inclusive. Baselines are calculated
before applying those bounds, so narrowing the report does not change a
step's classification. Table view contains dataset-relative paths and loss
metadata without images or captions. Grid view loads dataset thumbnails.
Click an entry to zoom its original image, inspect the recorded caption and
full metadata, download its record (including RNG state when enabled), or open
its step on the graph, with a yellow marker and a zoomed view. Applied filters
and the table/grid choice are remembered per job. Lightbox arrows navigate the
current page; controls at
the first/last entry continue through report pages.

The preview is the current original dataset image, not a saved training crop.
Recorded crop/flip settings describe its training presentation. Missing or moved
images retain their metadata. Captions come from training records, not current
sidecar files. A high-loss image can simply be harder, differently weighted, or
sampled at a difficult noise level; a spike alone is not a reason to remove it.

Resuming replaces exposures from the resumed step onward. Deleting graph loss
rows removes their example records and unused metadata as well. The UI must be
rebuilt/restarted to display this feature; running trainers must be restarted or
resumed with the new setting to begin recording.

YAML configuration:

```yaml
logging:
  log_every: 1
  use_ui_logger: true
  record_training_examples: true
  record_training_rng: false
```

Local input records remain available even when scalar logging uses WandB. Explicit
`record_training_examples: false` remains respected when importing old configs.
