# Adapted from Tencent HunyuanImage-3.0 and PedroMarinhoDev/ComfyUI-HunyuanImage3.
# Tencent source revision: 2ec2c78bee7d4b94157341fba86c4c2c7b1858b2.
# Pedro source revision: 84ad3a3e2a54472e69e195269729f774b242d120.
# See NOTICE and LICENSE files in this directory.
import torch

def build_input_embeddings(model, sequence, latents, timestep,
                           cond_latent=None, cond_vit=None):
    """Embed text/special IDs and replace image slots with their trainable projections.

    Target patches use the noisy timestep; frozen reference patches use clean
    time zero and VAE-scaled latents. Reference vision slots use the shared
    aligner. Ordered references remain part of every adapted backbone forward.
    """
    from .transformer import _as_list

    ids = sequence["ids"]
    # Image positions are replaced below. Gather only actual text/special-token rows
    # from the CPU vocabulary, avoiding duplicate image-placeholder lookups.
    slots = ids != model.config.image_token_id
    rows = model.model.wte(ids[slots].unsqueeze(0))
    embeds = rows.new_zeros((1, len(ids), model.config.hidden_size))
    embeds[:, slots] = rows
    dtype, device = embeds.dtype, ids.device

    if latents is not None:
        image_emb, token_h, token_w = model.patch_embed(latents.to(dtype), model.time_embed(timestep))
        assert (token_h, token_w) == (sequence["token_height"], sequence["token_width"]), \
            f"patch_embed grid {token_h}x{token_w} != sequence grid"
        embeds[0, sequence["image_slice"]] = image_emb[0].to(dtype)
        embeds[0, sequence["timestep_position"]] = model.timestep_emb(timestep)[0].to(dtype)

    blocks = sequence.get("cond_blocks", [])
    cond_latents, cond_vits = _as_list(cond_latent), _as_list(cond_vit)
    if len(cond_latents) != len(blocks) or len(cond_vits) != len(blocks):
        raise ValueError(f"this sequence conditions on {len(blocks)} images but {len(cond_latents)} "
                         f"latents and {len(cond_vits)} tower outputs arrived")
    if blocks:
        clean = torch.zeros(1, dtype=torch.float32, device=device)
        clean_time, clean_timestep = model.time_embed(clean), model.timestep_emb(clean)[0].to(dtype)
    for block, latent, vit in zip(blocks, cond_latents, cond_vits):
        # a `latent_dim == 3` VAE encodes with a one-frame time axis
        latent = latent[:, :, 0] if latent.dim() == 5 else latent
        cond_emb, cond_h, cond_w = model.patch_embed(latent.to(device=device, dtype=dtype), clean_time)
        assert (cond_h, cond_w) == (block["token_height"], block["token_width"]), \
            f"conditioning patch_embed grid {cond_h}x{cond_w} != sequence grid"
        embeds[0, block["vae_slice"]] = cond_emb[0].to(dtype)
        embeds[0, block["timestep_position"]] = clean_timestep
        aligned = model.vision_aligner(vit.to(device=device, dtype=dtype))
        embeds[0, block["vit_slice"]] = aligned.reshape(-1, embeds.shape[-1])
    return embeds


def build_attention_mask(sequence, seq_len, dtype, device):
    """Causal joint sequence, bidirectional within each individual image block.

    Earlier references cannot attend to later references or the target. Every
    row retains its diagonal, so no attention row is entirely masked.
    """
    keep = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=0)
    for full_attention in sequence["full_attention_slices"]:
        keep[full_attention, full_attention] = True
    return torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device).masked_fill(~keep, float("-inf"))



def rebuild_target_ids(ids, latent_shape, tokenizer, variant):
    """Caches own the frozen prefix; each forward owns target geometry.

    Replace the target metadata/placeholder span using the actual latent bucket.
    Cached blank prompts and text-only samples therefore work at every bucket
    without encoding frozen references again or caching adapted backbone states.
    """
    from .tokenizer import ImageGeometry, _Writer, _token_ids
    ids = ids.detach().to(device='cpu', dtype=torch.long)
    image_id = tokenizer.token_to_id('<img>')
    flags = ids == image_id
    starts = torch.where(flags & ~torch.cat([torch.zeros(1, dtype=torch.bool), flags[:-1]]))[0]
    if not len(starts):
        raise ValueError('Conditioning has no target image slot')
    start = int(starts[-1])
    end = start
    while end < len(ids) and int(ids[end]) == image_id:
        end += 1
    height, width = latent_shape
    geometry = ImageGeometry((height * 16, width * 16), extra_rows=variant.supports_edit)
    writer = _Writer(tokenizer, _token_ids(tokenizer), extra_rows=variant.supports_edit)
    writer.image_meta(geometry)
    replacement = torch.tensor(writer.tokens + [image_id] * (height * width), dtype=torch.long)
    rebuilt = torch.cat([ids[:start - 4], replacement, ids[end:]])
    if len(rebuilt) > variant.max_positions:
        raise ValueError(f'Total prompt/reference/target sequence {len(rebuilt)} exceeds {variant.max_positions}')
    return rebuilt
