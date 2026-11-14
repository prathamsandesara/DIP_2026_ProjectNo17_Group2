"""
Intensity normalization.

Fabric scans can vary in overall brightness/contrast across rolls due to
illumination drift in the acquisition line. Texture descriptors such as
GLCM (which quantizes intensity into gray levels) and local statistics
(mean/variance) are sensitive to absolute intensity scale, so images are
normalized to a common range before texture analysis. This also ensures
adaptive parameters learned from training images generalize to test
images acquired under slightly different illumination.
"""
from __future__ import annotations

import numpy as np


def min_max_normalize(image: np.ndarray, out_range=(0.0, 255.0)) -> np.ndarray:
    """Linearly rescale intensities to `out_range`.

        I' = (I - min(I)) / (max(I) - min(I)) * (hi - lo) + lo
    """
    lo, hi = out_range
    img_min, img_max = float(np.min(image)), float(np.max(image))
    if img_max - img_min < 1e-12:
        return np.full_like(image, lo, dtype=np.float32)
    return ((image - img_min) / (img_max - img_min) * (hi - lo) + lo).astype(np.float32)


def zscore_normalize(image: np.ndarray) -> np.ndarray:
    """Standardize to zero mean and unit variance: Z = (I - mu) / sigma.

    Useful before computing deviation maps, since it puts different
    feature channels (LBP/GLCM/Gabor responses) on comparable scales.
    """
    mu, sigma = float(np.mean(image)), float(np.std(image))
    if sigma < 1e-12:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - mu) / sigma).astype(np.float32)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convenience wrapper: normalize to [0, 255] and cast to uint8, for
    saving/visualization or for feeding into gray-level-quantized
    descriptors like GLCM that expect an 8-bit range."""
    return min_max_normalize(image, out_range=(0.0, 255.0)).astype(np.uint8)
