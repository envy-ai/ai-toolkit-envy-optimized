"""Standalone Krea reference preprocessing matching the toolkit wrapper.

Comfy IMAGE is BHWC [0,1]; MASK is inverted opacity. No model imports, placement
changes or GPU discovery happen here. Defaults preserve the existing trainer.
"""
import math

import torch
import torch.nn.functional as F


DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


def composite_reference(image, mask, background):
    if image.ndim != 4 or image.shape[-1] != 3 or image.shape[0] != 1:
        raise ValueError('Krea edit previews require one RGB image per reference slot')
    if mask is None:
        return image
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape != image.shape[:3]:
        # LoadImage returns a default 64x64 zero mask for images without alpha.
        if not torch.any(mask):
            return image
        raise ValueError('Reference alpha mask does not match its image')
    # Match PIL paste's integer alpha compositing, including half rounding.
    rgb = (image.float().clamp(0, 1) * 255).round().to(torch.int32)
    alpha = ((1 - mask.float().clamp(0, 1)) * 255).round().to(torch.int32).unsqueeze(-1)
    bg = torch.tensor(background, device=image.device, dtype=torch.int32)
    return ((rgb * alpha + bg * (255 - alpha) + 127) // 255).float() / 255


def prepare_reference(image, *, dtype, vlm_max_pixels, control_image_max_pixels, match_target_res, width, height):
    """Return VLM BHWC and VAE BHWC pixels with native rounding/filter order."""
    if dtype not in DTYPES or min(vlm_max_pixels, control_image_max_pixels, width, height) <= 0:
        raise ValueError('Invalid Krea reference precision or pixel budget')
    image = image.movedim(-1, 1).to(dtype=DTYPES[dtype])
    h, w = image.shape[-2:]
    scale = min(1., math.sqrt(vlm_max_pixels / (h * w)))
    nh, nw = max(round(h * scale), 28), max(round(w * scale), 28)
    vlm = image.float()
    if (nh, nw) != (h, w):
        vlm = F.interpolate(vlm, size=(nh, nw), mode='bicubic', antialias=True).clamp(0, 1)
    budget = width * height if match_target_res else control_image_max_pixels
    if match_target_res or h * w > budget:
        ratio = h / w
        new_h = math.sqrt(budget * ratio)
        new_w = new_h / ratio
    else:
        new_h, new_w = float(h), float(w)
    new_h = max(16, int(round(new_h / 16)) * 16)
    new_w = max(16, int(round(new_w / 16)) * 16)
    vae = image
    if (new_h, new_w) != (h, w):
        vae = F.interpolate(vae, size=(new_h, new_w), mode='bilinear')
    # The native wrapper computes this normalization in the model precision.
    # Invert it in float32 before Comfy's VAE.encode, which normalizes again.
    # Multiplication by two is exact: Comfy receives the same normalized pixels.
    vae = ((vae * 2 - 1).float() + 1) / 2
    return vlm.movedim(1, -1), vae.movedim(1, -1)
