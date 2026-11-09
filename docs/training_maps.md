# Training maps

Training maps are an optional spatial diagnostic for ordinary still-image Qwen Image, Qwen Image 2.1, Krea 2, and MOSS-Image text-to-image LoRA/DoRA jobs. Enable them with:

```yaml
logging:
  training_maps: true
  training_maps_every: 250
  training_maps_images: 3
```

The switch defaults to `false`. One baseline is captured before the first update. Once at least `training_maps_every` training loop steps have elapsed, the next successful optimizer update triggers a map; gradient accumulation microsteps do not trigger captures. A map capture runs two no-gradient model forwards per selected image and noise level, so enabling it adds noticeable time and temporary GPU memory use. The selected images are evenly spread through sorted training paths, excluding regularization images. Their processed resize/crop/flip presentations and chosen captions are frozen at setup and saved as PNGs. The image, latent, conditioning and noise are identical at later captures. Noise levels are 0.25, 0.5 and 0.75, selected from the configured training scheduler for each image size.

Each map cell reports an unweighted channel-mean squared prediction error (`error`), the change from the baseline error (`improvement`, positive means better), and the mean absolute change in prediction between adapter strength 1 and 0 (`effect`). These values are diagnostic prediction measures; they are not the weighted training objective or per-image Loss Report values. Baseline frames have no improvement map. Trigger attention is not part of this first version.

Files live in the job's `training_maps` directory. `manifest.json` records the probes, levels, step list, color scales and any persistent error with traceback. `probe-N.png` is the exact processed image used for each probe. `step-NNNNNNNNN.json` contains one frame of map grids. Resuming keeps prior frames only when the frozen inputs and relevant model/scheduler configuration match; it removes frames beyond the resumed step. A fresh start clears prior frames. A map setup/capture failure stops only map recording and remains visible in the Training Maps tab; training continues.

Maps currently require a single-process flowmatch job with an unmerged LoRA or DoRA network, ordinary MSE/noise training, no edit/reference or auxiliary adapter path, and no random image crop/augmentation or token shuffling. Unloading the text encoder requires cached text embeddings. Unsupported configurations are rejected with an explicit error when maps are enabled. The ordinary training path with maps disabled does not preload probes or run extra model forwards.

In the Training Maps tab, select a picture, map and noise level. Green means
improvement and magenta regression; blue → yellow → red indicates increasing
error or adapter effect. The same scale is used across recorded steps for each
picture, level and map. Use Previous/Next, the step slider or Play; focus the
viewer for Left/Right stepping and Space play/pause. Map opacity reveals the
crop underneath. Refresh loads newly saved frames manually; playback advances
only through saved steps. Source and caption details are expandable.

Enabling maps on an existing run starts its baseline at that resumed step.
Masked training displays unweighted spatial error, including the background.
Noise offset, blended blur noise and optimal noise pairing are unsupported.
CPU fixtures cover numerical maps, cached BF16 conditioning, state restoration,
errors and resume behavior; full-model VRAM, overhead and image quality remain
unmeasured. Restart/resume trainers to record maps and rebuild/restart the UI
to display the tab.
