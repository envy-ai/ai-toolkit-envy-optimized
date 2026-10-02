"""Image resampling helpers used by dataset preprocessing."""

from typing import Tuple

import numpy as np
from PIL import Image

MITCHELL_RESIZE_VERSION = "mitchell_netravali_b_c_1_3_v1"


def _mitchell_netravali(x: np.ndarray) -> np.ndarray:
    """Evaluate the Mitchell-Netravali filter with B = C = 1/3."""
    x = np.abs(x)
    x2 = x * x
    x3 = x2 * x

    inner = (7.0 * x3 - 12.0 * x2 + 16.0 / 3.0) / 6.0
    outer = (-7.0 / 3.0 * x3 + 12.0 * x2 - 20.0 * x + 32.0 / 3.0) / 6.0
    return np.where(x < 1.0, inner, np.where(x < 2.0, outer, 0.0))


def _axis_contributions(input_size: int, output_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return source indices and normalized weights for one resampling axis."""
    scale = input_size / output_size
    filter_scale = max(1.0, scale)
    support = 2.0 * filter_scale
    centers = (np.arange(output_size, dtype=np.float64) + 0.5) * scale - 0.5

    # Two spare samples cover both fractional support boundaries.
    tap_count = int(np.ceil(2.0 * support)) + 2
    first = np.floor(centers - support).astype(np.int64)
    source_indices = first[:, None] + np.arange(tap_count, dtype=np.int64)[None, :]
    distances = (centers[:, None] - source_indices) / filter_scale
    weights = _mitchell_netravali(distances) / filter_scale

    # Edge extension matches the behavior expected from image resize filters.
    source_indices = np.clip(source_indices, 0, input_size - 1)
    weights /= weights.sum(axis=1, keepdims=True)
    return source_indices, weights


def _resample_axis(array: np.ndarray, output_size: int, axis: int) -> np.ndarray:
    input_size = array.shape[axis]
    if input_size == output_size:
        return array

    indices, weights = _axis_contributions(input_size, output_size)
    source = np.moveaxis(array, axis, 0)
    output = np.zeros((output_size, *source.shape[1:]), dtype=np.float64)
    weight_shape = (output_size,) + (1,) * (source.ndim - 1)

    # Apply every tap in the widened antialiasing kernel. Iterating here changes
    # only peak memory use; it does not truncate or approximate the filter.
    for tap in range(indices.shape[1]):
        output += source[indices[:, tap]] * weights[:, tap].reshape(weight_shape)

    return np.moveaxis(output, 0, axis)


def resize_mitchell(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize a PIL image with an antialiased Mitchell-Netravali filter.

    Pillow does not expose Mitchell-Netravali as a built-in resampling mode.
    This performs separable filtering and widens the kernel during downscaling
    to suppress aliasing.
    """
    output_width, output_height = size
    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"resize dimensions must be positive, got {size}")
    if image.size == size:
        return image.copy()

    if image.mode not in ("L", "LA", "RGB", "RGBA"):
        raise ValueError(f"Mitchell resize does not support PIL mode {image.mode!r}")

    pixels = np.asarray(image, dtype=np.float64)
    pixels = _resample_axis(pixels, output_width, axis=1)
    pixels = _resample_axis(pixels, output_height, axis=0)
    pixels = np.clip(np.floor(pixels + 0.5), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)
