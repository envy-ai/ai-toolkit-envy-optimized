"""Explicit target-aligned reference presentation for native FIBO2 training."""
import math
from functools import lru_cache
from PIL import Image, ImageOps


@lru_cache(maxsize=256)
def trained_size(width, height, area=1024*1024):
    aspect = width/height
    sizes = [(w, h) for w in range(64, 2*area//64, 16)
             for h in range(max(64, 16*math.ceil(.9*area/w/16)), 16*math.floor(1.1*area/w/16)+1, 16)]
    return min(sizes, key=lambda size: (round(1-min(size[0]/size[1]/aspect, aspect*size[1]/size[0]), 9),
                                      abs(size[0]*size[1]-area), -size[0]))


def prepare_references(images, target_size=None):
    """Same source crops feed VAE and VLM: one target crop, multiple own buckets."""
    result = []
    for image in images:
        image = ImageOps.exif_transpose(image).convert('RGB')
        width, height = target_size if len(images) == 1 and target_size else trained_size(image.width, image.height)
        scale = max(width/image.width, height/image.height)
        resized = image.resize((math.ceil(image.width*scale), math.ceil(image.height*scale)), Image.Resampling.LANCZOS)
        left, top = (resized.width-width)//2, (resized.height-height)//2
        result.append(resized.crop((left, top, left+width, top+height)))
    return result
