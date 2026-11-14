"""
Contrast / histogram-based enhancement.

Subtle fabric defects (thin bars, faint stains) can have very low
intensity contrast against the surrounding weave. Enhancing local
contrast increases the separability of defect and non-defect texture
*before* descriptor extraction, which in turn makes the texture-deviation
map (src/detection/deviation_map.py) more discriminative.

CLAHE (Contrast Limited Adaptive Histogram Equalization) is used, rather
than global histogram equalization, because:

  - Global histogram equalization redistributes intensities using the
    WHOLE image's histogram, which can over-amplify large uniform fabric
    regions and under-enhance small local defects.
  - CLAHE operates on small tiles (local neighborhoods) and clips the
    histogram to limit noise amplification -- appropriate for images
    that already have fairly uniform overall texture (fabric) but
    localized subtle anomalies (defects).

CLAHE is applied conditionally (configurable) since some fabric types
already have sufficient contrast and additional enhancement is not
always beneficial (see LIMITATIONS.md).
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def clahe_enhance(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE. Expects/returns float32 in [0, 255]."""
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    if cv2 is not None:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(img_u8).astype(np.float32)
    else:  # pragma: no cover - fallback: skimage
        from skimage import exposure

        out = exposure.equalize_adapthist(img_u8 / 255.0, clip_limit=clip_limit / 40.0)
        return (out * 255.0).astype(np.float32)


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization: I' = CDF(I) * 255.

    Provided for comparison/ablation with CLAHE; not used by default
    because it is less suitable for locally-textured fabric images
    (see module docstring).
    """
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255.0 / (cdf.max() - cdf.min() + 1e-12)
    equalized = cdf_normalized[img_u8]
    return equalized.astype(np.float32)


def enhance(image: np.ndarray, method: str = "clahe", clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Dispatch to the configured enhancement method: "clahe", "hist_eq", or "none"."""
    if method == "none":
        return image.astype(np.float32)
    if method == "clahe":
        return clahe_enhance(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    if method == "hist_eq":
        return histogram_equalize(image)
    raise ValueError(f"Unknown enhancement method: {method}")
