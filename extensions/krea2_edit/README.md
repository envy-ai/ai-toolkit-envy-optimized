# Krea 2 source-graph edit training

This tracked extension integrates [lbouaraba/krea2edit-trainer](https://github.com/lbouaraba/krea2edit-trainer)
at commit `56bf37b892c5b71643ed08a51d052f0d684c9968` (Apache-2.0).
`krea2_edit.py` is adapted from that revision; `docs/GEOMETRY.md`, the example
configs, `LICENSE`, and the original `NOTICE` preserve the upstream references.
No model weights or nested Git repository are included.

The fork reuses its built-in Krea transformer, text encoder, VAE, quantization,
layer offloading, cache loading, and helper LoRA support. It does not vendor a
second older copy of those loaders. The upstream geometry and grounded instruction
encoding remain in the edit subclass: clean references precede target tokens,
reference frames start at 1, target frame is 0, and fitted references use fractional
center offsets. See [the geometry contract](docs/GEOMETRY.md).

Select **Krea 2 Identity/Edit (krea2edit extension)** in the job form or set
`model.arch: krea2_edit`. This is a separate contract from built-in `krea2` with
`model_kwargs.edit: true`; remove that flag when migrating. `kv_cache: true` is
also specific to the built-in edit contract and is rejected here. Start fresh
adapter weights when switching contracts; an experimental edit checkpoint does
not become compatible by changing its architecture string.

## Training configuration

- Use `model_kwargs.fit_refs: true` (default), matching Comfy `fit_mode: fit`.
  Different source and target aspect ratios are supported; references are fitted
  to the target grid without stretching. `false` selects the upstream legacy crop
  contract; this fork's source-graph preview then selects `crop (legacy)`.
- Supply `control_path` as an ordered list with one or two source directories.
  Every target must have a stem-matched source. The loader looks directly inside
  each source directory, even when targets were discovered recursively. For a
  nested layout such as `targets/training`, `targets/training64` and corresponding
  source leaves, use separate dataset entries pairing each target leaf with its
  matching source leaf. A root-to-root declaration with only nested source files
  fails the startup pairing audit.
- Use batch size 1, `flip_x: false`, `flip_y: false`, and
  `train.unload_text_encoder: false`. Keep `full_size_control_images: true` so fit
  geometry receives native source dimensions. Targets-only dataset entries remain
  valid for text-to-image regularization.
- The upstream recipe uses flow matching and `timestep_type: weighted`.
  Rank 64–128 is its suggested starting range. These are training choices;
  changing architecture does not automatically rewrite saved hyperparameters.
- `cache_text_embeddings: true` remains supported to save memory, but freezes one
  randomly selected grounding scale instead of retaining per-step jitter. The
  defaults are longest-side cap 768 and jitter minimum 384, overridable by
  `GROUNDING_MAX_PX` / `GROUNDING_JITTER_MIN`. This fork includes those values,
  architecture, fit mode and multi-reference mode in embedding cache identity.
  Cached grounding uses the same decoded controls and alpha background as live
  training, including caption dropout. Two cached references require
  `model_kwargs.multi_ref: true`.
- `checkpoint_every_n: 1` checkpoints every transformer block when gradient
  checkpointing is enabled. A positive integer greater than 1 checkpoints only
  every Nth block, trading memory for speed. Default behavior is unchanged.

The existing convrot transformer and text-encoder options, low-VRAM mode, tiled
VAE, layer offloading, assistant helper, and separate inference helper are
inherited. The assistant is merged into the training base once; the separate
inference helper network is inactive during training. Configuring the same file
for both does not apply it twice during training. A standalone Comfy preview needs
its own `sample.comfy.inference_lora` filename because Comfy loads the original
base model independently.

## Samples

Native and shared-model previews use different conditioning and are rejected.
Ordinary ComfyUI rendering is supported with `sample.comfy.enabled: true`,
`provider: ordinary` and
`workflow_path: config/comfy_templates/krea2_edit_lora_sample.json.njk`.
Alternatively, set `train.disable_sampling: true`.

The current source template takes exactly one existing reference per sample in
`ctrl_img`/`ctrl_img_1`. It preserves the graph's grounded encoders, edit model
patch, ordinary KSampler, helper-before-training-LoRA loading, and form sampling
settings. Training supports two references, but this preview template does not;
evaluate two-reference checkpoints in a corresponding Comfy graph separately.
See [the renderer contract](../../docs/krea2_edit_comfy_samples.md) for mappings
and required Comfy nodes. Geometry compatibility does not imply numerical pixel
or image-quality parity: upstream documents different resize kernels between its
training path and Comfy nodes.

An explicit `KREA2_COMPILE=1` request compiles blocks lazily after LoRA attachment;
compile failures propagate instead of silently selecting eager execution.
