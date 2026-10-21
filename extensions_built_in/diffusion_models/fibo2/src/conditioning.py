# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0; http://www.apache.org/licenses/LICENSE-2.0
"""FIBO2 prompt/embedding contract, adapted from the pinned Apache reference.

Source: muhammadali-afridi/diffusers@ab96e2b58c89adb9df418af74d81cc8b1e1be074,
pipeline_bria_fibo2.py. No language-model calls, downloads, or device moves here.
"""
import json
import math

import torch
from PIL import Image

from .text_images import draw_text_image

TEMPLATE = '<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
VISION_MARKER = '<|vision_start|><|image_pad|><|vision_end|>'
NULL_CAPTION = '{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}'
BUNDLE_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TAP_LAYERS = tuple(sorted({i for bundle in BUNDLE_LAYERS for i in bundle}))


def parse_prompt(prompt):
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop('edit_instruction', None)
    if 'structured_caption' in caption:
        caption = caption['structured_caption']
        if not isinstance(caption, dict):
            return caption or None, instruction
    original_keys = set(caption)
    def clean(value):
        def empty(item):
            return item is None or (isinstance(item, (str, dict, list)) and not item) or (isinstance(item, float) and math.isnan(item))
        if isinstance(value, dict):
            value = {key: clean(item) for key, item in value.items() if key not in ('aesthetic_score', 'preference_score')}
            return {key: item for key, item in value.items() if not empty(item)}
        if isinstance(value, list):
            return [item for item in map(clean, value) if not empty(item)]
        return value
    caption = clean(caption)
    if original_keys == {'short_description', 'text_render'}:
        caption = {'short_description': caption.get('short_description') or '', 'text_render': caption.get('text_render') or []}
    return caption or None, instruction


def prompt_inputs(prompt, reference_images=(), *, text_images=True, fonts=(), is_unconditional=False):
    """Return the exact chat string + PIL images, references first then glyphs.

    CFG prompts omit all VLM images, including edit references; their VAE
    latents are supplied separately to both prediction branches. Explicit CFG
    text is wrapped unchanged, matching the reference's direct negative encode.
    """
    if not prompt.strip():
        return TEMPLATE.format(NULL_CAPTION), []
    if is_unconditional:
        return TEMPLATE.format(prompt), []
    caption, instruction = parse_prompt(prompt)
    if reference_images and instruction is None:
        caption, instruction = None, prompt
    glyphs = []
    if isinstance(caption, dict):
        for entry in caption.get('text_render') or []:
            if isinstance(entry, dict) and isinstance(entry.get('text'), str) and entry['text']:
                entry['image_tokens'] = VISION_MARKER if text_images else ''
                if text_images:
                    glyphs.append(draw_text_image(entry['text'], fonts))
    if reference_images:
        content = {'image': VISION_MARKER} if len(reference_images) == 1 else {
            f'image_{i+1}': VISION_MARKER for i in range(len(reference_images))}
        if caption:
            content['structured_caption'] = caption
        content['edit_instruction'] = instruction
    else:
        content = caption if caption is not None else {}
    text = content if isinstance(content, str) else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    references = []
    for image in reference_images:
        image = image.convert('RGB').copy()
        image.thumbnail((768, 768), Image.Resampling.BICUBIC)
        references.append(image)
    return TEMPLATE.format(text), references + glyphs


def hidden_state_bundles(hidden_states, layers=BUNDLE_LAYERS):
    """One encoder's tapped states -> B,6,L,7680, RMS normalization in fp32."""
    normalized = {}
    for index in sorted({i for bundle in layers for i in bundle}):
        value = hidden_states[index]
        fp = value.float()
        normalized[index] = (fp * torch.rsqrt(fp.square().mean(-1, keepdim=True) + 1e-6)).to(value.dtype)
    return torch.stack([torch.cat([normalized[i] for i in bundle], -1) for bundle in layers], 1)


def pack_latents(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//2, 2, w//2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c*4, h//2, w//2)


def unpack_latents(x):
    b, c, h, w = x.shape
    return x.reshape(b, c//4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c//4, h*2, w*2)


def flow_mu(height, width):
    """Packed latent H*W is the image-token count, without another /4."""
    return .5 + (height * width - 256) * (.65 / (4096 - 256))
