# AI Toolkit Krea edit preview helper

Optional V3 ComfyUI extension for this fork's specialized Krea Flow-DPO and
guidance-distillation previews. It requires Comfy's Krea encoder and the existing
`ComfyUI-Krea2-Ostris-Edit` model patch. No additional pip dependencies are needed.

Copy this directory into your ComfyUI `custom_nodes` directory (or symlink it),
then restart Comfy when no training or sampling task is using it. This repository
does **not** install the extension or restart Comfy automatically. Until it is
loaded, edit previews report a missing-node error; ordinary/text-to-image jobs
do not need it. On another machine, copy the whole directory, not only nodes.py.

Edit references require a working VAE encoder as well as a decoder. Older
`ComfyUI-VAE-Utils` releases that reimplement VAE initialization may omit
`format_encoded`, which current Comfy's inherited encoder expects (normally
`None` for the Qwen image VAE). This is a loader compatibility error, not a reason
to discard references or bypass the configured tiling/offload behavior.

This helper does not change Comfy's LoRA loader. Quantized adapter application
may change pixels even with a mathematically zero update; preview checks should
measure those differences rather than assume exact identity or native parity.

The node shares one set of VAE reference encodings between both CFG branches,
skips negative text encoding at CFG <= 1, and matches the trainer's configurable
VLM/reference budgets, target-size matching, filters, model input precision and
RGB alpha-composite background. Preprocessing parity does not establish identical
VAE posteriors or image quality across native and Comfy implementations. The
native VAE samples its posterior, while Comfy may use a deterministic encoder.
Do not claim bit-identical rendered images.

Node registration/schema discovery resolves no encoder/model-manager imports or
CUDA device. Comfy's own Krea template is imported lazily when the node executes.
The CPU V3 tests use the installed `ComfyNode`, `Schema` and `NodeOutput` classes,
with unrelated video implementations isolated for ai-toolkit's older PyAV;
they do not establish live extension/encoder compatibility. No dependency upgrade
or installation is performed by these tests:

```bash
conda run --no-capture-output -n ai-toolkit python -m unittest testing.test_krea_preview_v3
```

Specialized Krea edit previews upload canonical RGB PNGs decoded with the same
versioned loader as raw training controls. This handles indexed transparency,
16-bit grayscale, EXIF orientation and the first frame of animated images
consistently, without changing source files. Prepared uploads use content-specific
names; changing the source or alpha background cannot overwrite a queued reference.
The presentation version/source identity invalidates only affected specialized
Krea text/reference caches. Ordinary and legacy Qwen loaders remain unchanged.
