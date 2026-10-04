# LoHa and optional DoRA (DoHa)

Choose **LoHa (LyCORIS)** in the ordinary training form's **Target Type** selector.
Enable **DoRA Weight Decomposition (DoHa)** to also train row magnitudes. Linear
rank, alpha, optional convolution rank/alpha, pretrained checkpoints, and layer
learning-rate multipliers work as usual. Start with rank 8–16: LoHa has four
factor matrices instead of LoRA's two, and a potentially rank-squared update.

```yaml
network:
  type: loha
  linear: 16
  linear_alpha: 16
  loha_dora: true   # false (default) for plain LoHa
  network_kwargs:
    loha_chunk_size: 128
```

LoHa uses `(W1a @ W1b) * (W2a @ W2b) * alpha / rank`. One factor starts at
zero, making an untrained adapter a no-op. Dense products are reconstructed in
output-row chunks during forward and backward, not retained in the autograd
graph. The base forward remains untouched, including ConvRot int4/int8 and
layer streaming; merging into the frozen base is disabled. DoHa additionally
needs detached base/adapted row norms and may transiently dequantize one base
layer. It does not keep a dense copy of the base model.
Saving a merged full model (`merge_network_on_save`) is not supported for LoHa.

DoHa uses a detached adapted-weight norm at strength +1; other strengths
interpolate the complete +1 weight delta, keeping bias unchanged. Checkpoints
contain standard `hada_w1_a/b`, `hada_w2_a/b`, `alpha`, and optionally `dora_scale`
keys. On export/resume, DoRA magnitudes are converted between training's
adapted-weight normalization and the local ComfyUI loader's base-weight
normalization so signed/fractional strengths agree. This conversion targets
ComfyUI's float32 patching; other loaders must implement the same normalization
to reproduce DoHa exactly. Plain LoHa follows the usual LyCORIS layout.

Linear and ungrouped, zero-padded Conv2d layers are supported; convolution
factors are flattened, not Tucker-decomposed. A pretrained adapter must use
matching factor shapes/rank. Loading DoHa with its DoRA checkbox off is rejected
instead of silently discarding magnitudes. Loading plain LoHa with DoRA enabled
initializes the magnitudes to preserve the loaded plain-LoHa behavior.

This does not upgrade the pinned LyCORIS package or alter existing LoCon,
LoRA, LoKr, DoRA, quantization, or cache paths. DPO, guidance-distillation, and
Fizgig training retain their adapter restrictions: Fizgig supports LoRA/DoRA;
DPO, guidance-distillation and SliderSpace support LoRA only. Krea/Anima/Ideogram
ordinary LoHa/DoHa target/export paths now have CPU fixtures using their actual
tiny transformer classes (including Anima's optionally trainable conditioner).
The installed Comfy key mapper/LoHa loader consumes all exported keys, matches
toolkit outputs at strengths -1, 0, 0.5 and 1, and file reloads reproduce outputs.
Native transformer targets also pass ConvRot int4/int8 checkpointed-gradient
fixtures. Full tiny-model predictions now also exercise the actual layer memory
manager with float/ConvRot int4/int8 bases, checkpointed learned LoHa/DoHa factors,
Anima conditioner adaptation on/off, and signed pretrained file reloads. Adapter
chains remain intact when streaming is attached, detached or reattached; no dense
base model copy is made by these lifecycle operations. CPU ConvRot's STE training
and no-hardware inference have different activation-quantization numerics, so
no-op inference checks compare both adapters on the same inference path.
Those CPU fixtures alone do not certify full-model GPU execution. Separately
approved Krea Raw rank-2 ConvRot-int8/offloaded LoHa/DoHa training, native previews
and signed/fractional live Comfy loading/rendering smoke tests now pass. These
short 256px runs do not certify production-resolution memory, learned quality or
native/Comfy numerical parity. Full-checkpoint Anima/Ideogram and broader
pretrained/resume coverage remain unverified.
See [cross-model training](cross_model_training_modes.md).
