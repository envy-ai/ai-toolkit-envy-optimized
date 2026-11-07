# LoKr and optional DoRA (DoKr)

Choose **LoKr** in the ordinary training form's **Target Type** selector. Enable
**DoRA Weight Decomposition (DoKr)** to train an output-row magnitude in addition
to LoKr's Kronecker factors. The simple form retains LoKr's existing factor
selector and full-rank default. YAML supports the existing linear rank/alpha,
optional convolution rank/alpha, pretrained checkpoint, and layer learning-rate
settings.

```yaml
network:
  type: lokr
  linear: 16
  linear_alpha: 16
  lokr_factor: 8
  lokr_dora: true  # false (default) for plain LoKr
  network_kwargs:
    lokr_norm_chunk_size: 128
```

DoKr keeps the native LoKr factorized Linear forward, including the frozen base
module's quantized or streamed forward. Its DoRA denominator is the detached norm
of each output row in the complete strength +1 adapted weight. Row norms rebuild
LoKr factors in bounded output-row chunks without retaining a full Linear
Kronecker product in the gradient graph. A quantized base layer may be
transiently dequantized while measuring its rows; no dense copy of the whole base
model is kept. Conv2d and CP/Tucker LoKr factors retain LoKr's existing dense
Kronecker update path. DoKr Conv2d supports groups=1 and zero padding mode.

At other strengths, DoKr interpolates the **complete** +1 weight delta from
the frozen base, leaving bias unchanged. This includes negative and fractional
strengths used in previews. Exported checkpoints contain standard `lokr_w1`
and `lokr_w2` full or factored keys, `alpha`, and a `dora_scale` per adapted
layer. The public magnitude is converted from the training adapted-row norm to
the local ComfyUI loader's base-row norm (with its float32 epsilon). Loading a
public checkpoint reverses the conversion. Internal state dictionaries retain
the `magnitude` parameter. Loading plain LoKr with DoKr enabled initializes
magnitudes to preserve the loaded plain update; loading a DoKr checkpoint with
the option disabled raises an error instead of discarding magnitudes.
When both Kronecker factors are stored as full matrices, their scale is one,
matching the local ComfyUI loader regardless of a checkpoint's nominal alpha.

The attached network supports native training previews; ComfyUI can load the
exported adapter. The separate native inference LoRA loader rejects DoKr
checkpoints because its standalone hook and merge paths do not apply the DoRA
weight decomposition. Saving a merged full model with
`merge_network_on_save` is unsupported. The LoRA trimming utility does not bake
DoKr into LoRA. Specialized DPO, guidance-distillation, SliderSpace, KTO, and
Fizgig adapter restrictions remain unchanged. The pinned LyCORIS package is
unchanged.

CPU verification covers dense-reference gradients, full/factored/CP layouts,
ConvRot int4/int8 with layer offloading and checkpointed backward, save/resume,
and the installed ComfyUI LoKr loader at signed and fractional strengths. These
small fixtures do not measure full-checkpoint GPU memory or training quality.
