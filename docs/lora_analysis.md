# AI Toolkit LoRA analysis

`scripts/analyze_lora.py` ranks the layers changed by an AI Toolkit LoRA. It
reports each functional layer type separately so, for example, attention-query
projections are not ranked directly against MLP output projections.

```bash
conda run -n ai-toolkit python scripts/analyze_lora.py \
  output/my_run/my_run.safetensors
```

The default report shows the top 20 layers in each type. Use `--top` to change
that limit, or write every result to JSON:

```bash
conda run -n ai-toolkit python scripts/analyze_lora.py \
  output/my_run/my_run.safetensors \
  --top 50 \
  --json output/my_run/lora-analysis.json
```

## Base-model comparison

When matching local base weights are available, layers are ranked by relative
Frobenius change:

```text
||effective LoRA update||F / ||base weight||F
```

Otherwise they are ranked by update RMS, which normalizes for the number of
elements in the effective weight update. The report always includes both the
Frobenius norm and RMS.

The analyzer looks for a base model in this order:

1. `--base PATH_OR_REPO`
2. Exact base-model metadata in the LoRA
3. `model.name_or_path` in the adjacent `config.yaml` or `.job_config.json`
4. An already-cached Hugging Face snapshot
5. Safetensors under AI Toolkit's local `models/` directory

It never downloads a model. Additional local model directories can be supplied
with repeatable `--base-search-root` arguments. Use `--no-base` for a fast,
LoRA-only analysis.

```bash
conda run -n ai-toolkit python scripts/analyze_lora.py adapter.safetensors \
  --base models/diffusion_models/base.safetensors
```

Float base weights and integer weights with a recognized per-row
`weight_scale` are supported. DoRA magnitude vectors are included in the
effective change when matching base weights are available. Without base
weights, the DoRA report can measure only its low-rank direction update.

AI Toolkit's whole-module `.pt` quantization caches are not loaded by the
analyzer: they are Python pickles without a random-access weight manifest and
loading one would construct the entire cached module. Point `--base` at the
source or repacked safetensors instead. The analyzer can compare quantized
safetensors directly without loading the complete model into memory.
