"""
Grayscale conversion.

AITEX images are already single-channel 8-bit greyscale fabric scans, but
this module provides an explicit, documented conversion step so the
pipeline is correct even if a 3-channel image is supplied, and so the
DIP report can point to a concrete "image representation" stage.

Mathematical basis
-------------------
For an RGB image, standard luminance-preserving conversion is used:

    Gray = 0.299 R + 0.587 G + 0.114 B      (ITU-R BT.601)

This weighting reflects the human eye's differing sensitivity to each
channel and is the conventional first step before any texture analysis,
since LBP/GLCM/Gabor are all defined on scalar (single-channel) images.
"""
from __future__ import annotations

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel float32 grayscale.

    If `image` is already 2-D, it is returned as float32 unchanged
    (AITEX's native case). If 3-D (H, W, 3), BT.601 luminance weights
    are applied.
    """
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[-1] == 3:
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return (image.astype(np.float32) * weights).sum(axis=-1)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")
