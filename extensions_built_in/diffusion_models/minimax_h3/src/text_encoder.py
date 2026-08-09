"""Qwen3-VL conditioning for MiniMax-H3.

MiniMax-H3 conditions on the **unnormalized** ``hidden_states[50]`` of its
Qwen3-VL-32B conditioner (``hidden_states[0]`` is the embedding output, so
this is the output of decoder layer 49, before the final norm). The LM head
and layers 50..63 are never used, which lets the loader truncate the stack.

The presentation is raw tokens — no chat template, no special tokens:

  - t2va: the verbatim prompt.
  - fl2va: per keyframe, a ``"<Picture i>: "`` label plus a vision block
    (``<|vision_start|>`` + one ``<|image_pad|>`` per merged vision patch +
    ``<|vision_end|>``), then the verbatim prompt. Vision-block rows are
    tagged as *video* (0) rather than text (1) — the transformer's AdaLN
    modality selection keys off these tags.
  - ref2va: reference images, video soundtracks/videos, and standalone audio
    are presented in request order as ``<Picture i>`` / ``<Video j>`` /
    ``<Audio k>``. Qwen sees videos at 2 fps in temporal pairs with timestamp
    labels; audio itself never enters Qwen.
"""

from typing import Dict, List, Optional

import torch

from .packing import TEXT_TAG, VIDEO_TAG

TEXT_ENCODER_LAYER = 50


@torch.no_grad()
def encode_minimax_h3_prompt(
    text_encoder,  # transformers Qwen3VLForConditionalGeneration
    tokenizer,  # Qwen2TokenizerFast
    processor,  # Qwen3VLProcessor (needed only when keyframes are present)
    prompt: str,
    keyframes: Optional[List] = None,  # PIL images already on the target canvas
    reference_items: Optional[List[Dict]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_length: Optional[int] = None,  # cap on PROMPT tokens (vision blocks are never cut)
):
    """Encode ONE prompt (with optional keyframes) into MiniMax-H3 conditioning.

    Returns (embeds (L, 5120), token_tags (L,) long). The embeds come from
    ``hidden_states[50]`` unnormalized. A stack truncated to exactly 50 layers
    also works ONLY if the final ``model.norm`` has been replaced with an
    Identity (transformers applies the final norm to the last entry of
    ``hidden_states``); the loader in minimax_h3.py does exactly that.
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers < TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 needs at least {TEXT_ENCODER_LAYER} Qwen3-VL decoder "
            f"layers to read hidden_states[{TEXT_ENCODER_LAYER}], got {num_layers}"
        )
    if device is None:
        device = text_encoder.device

    if keyframes and reference_items:
        raise ValueError("MiniMax H3 prompt encoding cannot mix FL2VA keyframes and Ref2VA references")

    pixel_values, image_grid_thw = None, None
    pixel_values_videos, video_grid_thw = None, None
    token_ids: List[int] = []
    token_tags: List[int] = []
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_pad = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    def append_text(text: str):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.extend(ids)
        token_tags.extend([TEXT_TAG] * len(ids))

    def append_vision(pad_token: int, count: int):
        ids = [vision_start] + [pad_token] * count + [vision_end]
        token_ids.extend(ids)
        token_tags.extend([VIDEO_TAG] * len(ids))

    if keyframes:
        vision = processor.image_processor(images=keyframes, return_tensors="pt")
        pixel_values = vision["pixel_values"]
        image_grid_thw = vision["image_grid_thw"]
        merge = processor.image_processor.merge_size**2
        for i in range(len(keyframes)):
            num_image_tokens = int(image_grid_thw[i].prod()) // merge
            append_text(f"<Picture {i + 1}>: ")
            append_vision(image_pad, num_image_tokens)

    if reference_items:
        images = [item["data"] for item in reference_items if item["type"] == "image"]
        videos = [item["data"] for item in reference_items if item["type"] == "video"]
        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
        if videos:
            # The dataloader has already sampled each reference at 2 fps and
            # supplies a list of PIL frames. Transformers rejects a second
            # sampling pass for that input form, so preserve those exact
            # frames (and the timestamps constructed alongside them).
            vision = processor.video_processor(
                videos=videos,
                do_sample_frames=False,
                return_tensors="pt",
            )
            pixel_values_videos = vision["pixel_values_videos"]
            video_grid_thw = vision["video_grid_thw"]

        counters = {"image": 0, "audio": 0, "video": 0}
        image_index = 0
        video_index = 0
        image_merge = processor.image_processor.merge_size**2
        video_merge = processor.video_processor.merge_size**2
        for item in reference_items:
            kind = item["type"]
            counters[kind] += 1
            if kind == "image":
                append_text(f"<Picture {counters['image']}>: ")
                count = int(image_grid_thw[image_index].prod()) // image_merge
                append_vision(image_pad, count)
                image_index += 1
            elif kind == "audio":
                append_text(f"<Audio {counters['audio']}>: ")
            elif kind == "video":
                append_text(f"<Video {counters['video']}>: ")
                grid = video_grid_thw[video_index]
                block_count = int(grid[0])
                tokens_per_block = int(grid[1] * grid[2]) // video_merge
                timestamps = list(item.get("timestamps", []))
                if not timestamps:
                    timestamps = [i / 2.0 for i in range(block_count * 2)]
                if len(timestamps) % 2:
                    timestamps.append(timestamps[-1])
                while len(timestamps) < block_count * 2:
                    timestamps.append(timestamps[-1])
                for block in range(block_count):
                    start = block * 2
                    block_time = (timestamps[start] + timestamps[start + 1]) / 2.0
                    append_text(f"<{block_time:.1f} seconds>")
                    append_vision(video_pad, tokens_per_block)
                video_index += 1
            else:
                raise ValueError(f"Unsupported MiniMax H3 reference item {kind!r}")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if max_length is not None and max_length > 0:
        # the cap applies to the caption only; a keyframe's vision block is
        # structural conditioning and cannot be truncated without corrupting it
        prompt_ids = prompt_ids[:max_length]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    if len(token_ids) == 0:
        # empty (unconditional) prompt: a single pad token keeps the sequence
        # non-degenerate; the model was not trained with CFG so this is only
        # ever a fallback
        token_ids = [tokenizer.pad_token_id or 0]
        token_tags = [TEXT_TAG]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
    )

    # call the inner .model directly: the LM head's vocab projection is dead
    # weight here and hidden_states[50] is all that is consumed
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None
        if pixel_values is None
        else pixel_values.to(device, text_encoder.dtype),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
        pixel_values_videos=None
        if pixel_values_videos is None
        else pixel_values_videos.to(device, text_encoder.dtype),
        video_grid_thw=None
        if video_grid_thw is None
        else video_grid_thw.to(device),
        use_cache=False,
        output_hidden_states=False,
    )
    # The loader truncates the LM to exactly 50 layers and replaces its final
    # norm with Identity, so last_hidden_state is the required unnormalized
    # layer-49 output. Requesting every hidden-state snapshot would otherwise
    # retain roughly 51 copies of a vision-heavy sequence during prompt cache.
    embeds = outputs.last_hidden_state[0]
    if dtype is not None:
        embeds = embeds.to(dtype)
    return embeds, torch.tensor(token_tags, dtype=torch.long)
