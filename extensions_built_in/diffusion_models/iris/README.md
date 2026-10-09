# Iris 3B

Select **Iris 3B (pixel space)** in the UI, or set `model.arch: iris`.
The released model is [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b).
See [the training example](../../../config/examples/train_lora_iris_3b.yaml).

`model.name_or_path` accepts native/Comfy `.safetensors`, a local model folder
containing `model.safetensors` and `config.yaml`, a Hub repository, or a Hub
file reference such as `speridlabs/iris-3b/model.safetensors`. Single files use
the released 3B architecture by default. `model.model_kwargs.config_path`
can supply a different architecture YAML. Set `checkpoint_filename` to choose
a filename inside a folder or repository.

The local `/d/comfy_models/diffusion_models/iris-3b/model.safetensors` and
`model_fp16.safetensors` match all 639 expected tensors. Prefix-wrapped
`diffusion_model.*` and `model.diffusion_model.*` checkpoints also load.
Prequantized Comfy tensors use the existing Comfy quantization importer;
FP8 and int8/ConvRot formats follow the toolkit's shared backend support.
Choose the matching `qtype` to preserve shipped quantization; a different
requested type invokes the usual dequantization/requantization policy.
The depth and upscaler variants have different conditioning and are outside
this text-to-image integration.

The frozen encoder is the language tower of **Qwen3-VL-4B-Instruct**. The
vision tower and LM head are never materialized. The exact Iris chat template,
300-token caption/suffix window, attention mask and twelve hidden-state taps
are preserved. Overlong captions are truncated before the suffix, with a
warning. `model.text_encoder_path` (or `model.model_kwargs.text_encoder_path`)
can point to the original Comfy Qwen3-VL safetensors or a compatible quantized
4B text encoder. For a single-file encoder, `tokenizer_path` defaults to the
original Qwen repository. Changing encoder weights changes conditioning;
the original Qwen encoder is the reference behavior.

Images are RGB in `[-1, 1]`; there is no learned VAE. The existing identity VAE
lets dataset bucketing, pixel caching and sample handling work unchanged.
The model sees shifted flow time in `[0, 1000]`, and trains against `noise - image`.
The default sigmoid training distribution uses Iris's fixed shift of four.
Image dimensions must be divisible by 16. Train only the diffusion model;
text encoder fine-tuning is not part of this integration.

LoRA training uses the existing adapter network, optimizer, accumulation,
resume, save and loss-report machinery. With `save_format: diffusers`, saved
adapters use `diffusion_model.<native module>.lora_A/B.weight` plus alpha,
which the normal Comfy LoRA loader recognizes. Attention, MLP, text adapter,
shared modulation and pixel refinement linears are targetable; use the normal
network include/exclude filters to narrow scope. Inference LoRAs load as a
separate frozen adapter and are active only during sampling.

## Memory and sampling

The integration reuses single-file mmap loading, meta construction, per-block
quantization, persistent quantized transformer/text caches, pinned layer
offloading, text-embedding disk caching and text-encoder unloading. Input,
timestep, shared modulation, pixel input/output and text-layer-pooling
projections are excluded from fresh quantization. Shared modulation cores
stay resident when layer offloading is active. `low_vram` parks the transformer
during text encoding and parks the encoder afterward.

`train.gradient_checkpointing` covers the patch trunk, pixel refiner, text
adapter and full-resolution input/output heads.
`model.model_kwargs.activation_checkpointing` selects `full` (default),
`selective_op`, `selective_layer` or `none`; `ac_selective_every` controls the
selective-layer interval. `activation_offload: true` additionally stores saved
autograd tensors in pinned host memory. The existing `compile`/`block_compile`
options use the model's block lists. Start with batch one, cached/unloaded text
conditioning and 512px images; actual GPU peak memory depends on rank,
resolution and quantization, and has not yet been measured for this port.

Internal previews use the released DPM-Solver++ order-two algorithm, shift-four
time grid, sequential CFG and FP32 integration. The released default is 100
steps at CFG 3. A progress bar reports each sample's denoising steps.
`sample.sampler: flowmatch` selects this model's native preview implementation.

For external Comfy previews, choose
`config/comfy_templates/iris_lora_sample.json.njk`, use `dpmpp_2m`, and select
Iris model/Qwen text encoder files. The workflow requires the installed
ComfyUI-Iris nodes and the existing absolute-path LoRA loader. No VAE is used.

## Validation and attribution

Run `conda run -n ai-toolkit python testing/run_iris_cpu_tests.py` for CPU
diagnostics without initializing unrelated CUDA-only extension imports.

CPU tests cover actual LoRA backward/update/save/reload, all native checkpoint
policies, an int8 Comfy checkpoint backward pass, native and wrapped checkpoint
loading, exact HF versus single-file text-tower loading, prompt cache/batch
handling, cache restoration, the installed Comfy LoRA parser and native sampler
agreement. The Comfy parser test executes its unchanged parser bodies with GPU
management omitted; it does not render images. The local full-size
checkpoint headers match the architecture. Full-size GPU training and external
Comfy rendering still need a smoke run when the GPU is available.

Architecture, attention, embeddings, checkpointing and solver code in `src/`
are adapted from [speridlabs/iris-3b](https://github.com/speridlabs/iris-3b),
revision `a8d15239dea469aba042cfa56ca3bb4e450d5ebc`, under Apache-2.0.
Original LICENSE and NOTICE are included. Imports are scoped to this extension,
and checkpoint regions are extended to the text adapter and pixel heads;
the toolkit wrapper supplies loading, adapters, offloading and training hooks.
