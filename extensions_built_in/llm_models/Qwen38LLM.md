# Qwen3.8 dependency

Qwen3.8 uses the optional `flash-linear-attention` package (`fla`) for its
Gated DeltaNet layers. Install it in the `ai-toolkit` Conda environment before
loading a `qwen38` model. It is intentionally not a dependency of ordinary
diffusion training jobs. Extension discovery imports the Qwen3.8 registration
without loading its model implementation, so a missing `fla` package cannot
prevent unrelated jobs from starting.
