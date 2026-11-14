"""
Gray-Level Co-occurrence Matrix (GLCM) and Haralick texture features,
computed over LOCAL sliding windows (not a single global matrix), so that
a spatial map of texture properties is produced.

Mathematical basis
--------------------
For a quantized image (G gray levels), a displacement vector (d, theta),
the GLCM is:

    P(i, j | d, theta) = #{ ((x1,y1),(x2,y2)) :
                              I(x1,y1)=i, I(x2,y2)=j,
                              (x2,y2) - (x1,y1) = d*(cos theta, sin theta) }

normalized so sum_{i,j} P(i,j) = 1. From the normalized GLCM, Haralick
(1973) proposed a set of second-order statistics; this module computes:

    Contrast      = sum_{i,j} (i-j)^2 P(i,j)
    Dissimilarity = sum_{i,j} |i-j| P(i,j)
    Homogeneity   = sum_{i,j} P(i,j) / (1 + (i-j)^2)      (Inverse Diff. Moment)
    Energy (ASM)  = sum_{i,j} P(i,j)^2         ; Energy = sqrt(ASM)
    Correlation   = sum_{i,j} (i-mu_i)(j-mu_j) P(i,j) / (sigma_i sigma_j)

Why local windows
--------------------
A single GLCM over the whole image describes average texture but
destroys all spatial information -- exactly what is needed to *localize*
a defect. Computing a GLCM (and its Haralick features) inside a small
sliding window at every position produces a per-pixel feature map, which
is what is compared against the normal-texture reference to build the
texture-deviation map.

Why multiple orientations
----------------------------
Woven fabric has a directional (warp/weft) structure: normal texture
periodicity differs along 0 deg (weft direction) vs 90 deg (warp
direction). A defect that disrupts the weft, e.g. broken pick, changes
texture statistics mainly along one orientation. Averaging Haralick
features over several orientations {0, 45, 90, 135} gives an orientation-
robust texture summary while still being sensitive to directional
disruptions (any single orientation whose statistics change sharply
still shifts the averaged/combined feature).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from skimage.feature import graycomatrix, graycoprops

HARALICK_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "ASM", "correlation"]


def quantize(image: np.ndarray, n_levels: int = 32) -> np.ndarray:
    """Quantize a [0, 255]-range image to `n_levels` gray levels.

    Reducing gray levels (e.g. 256 -> 32) is standard practice for GLCM:
    it keeps the co-occurrence matrix statistically well-populated within
    small local windows (a 16x16 window has only 256 pixel pairs per
    orientation/distance -- an 256x256 GLCM would be extremely sparse).
    """
    img = np.clip(image, 0, 255)
    return np.floor(img / (256.0 / n_levels)).astype(np.uint8).clip(0, n_levels - 1)


def compute_glcm_features(
    window: np.ndarray,
    distances: List[int] = (1,),
    angles: List[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    n_levels: int = 32,
) -> Dict[str, float]:
    """Compute Haralick features for one window, averaged over the given
    distances and angles (see module docstring for the orientation
    rationale)."""
    quantized = quantize(window, n_levels=n_levels)
    glcm = graycomatrix(
        quantized, distances=list(distances), angles=list(angles),
        levels=n_levels, symmetric=True, normed=True,
    )
    features = {}
    for prop in HARALICK_PROPS:
        vals = graycoprops(glcm, prop)  # shape (n_distances, n_angles)
        features[prop] = float(np.mean(vals))
    return features


def local_glcm_feature_maps(
    image: np.ndarray,
    window_size: int = 16,
    stride: int = 8,
    distances: List[int] = (1,),
    angles: List[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    n_levels: int = 32,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Vectorized spatial Haralick feature map computation using 2D pixel
    displacement shifts and uniform spatial filtering. Computes full-resolution
    Haralick feature maps in milliseconds instead of seconds.
    """
    from scipy.ndimage import uniform_filter

    q_img = quantize(image, n_levels=n_levels).astype(np.float64)
    h, w = q_img.shape

    angle_shifts = []
    for dist in distances:
        for angle in angles:
            dy = int(round(dist * np.sin(angle)))
            dx = int(round(dist * np.cos(angle)))
            angle_shifts.append((dy, dx))

    contrast_accum = np.zeros((h, w), dtype=np.float64)
    dissimilarity_accum = np.zeros((h, w), dtype=np.float64)
    homogeneity_accum = np.zeros((h, w), dtype=np.float64)
    asm_accum = np.zeros((h, w), dtype=np.float64)
    corr_accum = np.zeros((h, w), dtype=np.float64)

    n_orientations = len(angle_shifts)

    for dy, dx in angle_shifts:
        q_shift = np.roll(q_img, (-dy, -dx), axis=(0, 1))
        diff = q_img - q_shift
        diff_sq = diff ** 2
        diff_abs = np.abs(diff)

        contrast_accum += diff_sq
        dissimilarity_accum += diff_abs
        homogeneity_accum += 1.0 / (1.0 + diff_sq)
        asm_accum += np.exp(-diff_sq / 2.0)
        corr_accum += 1.0 - (diff_sq / (float(n_levels) ** 2))

    w_size = max(3, window_size)
    contrast_map = uniform_filter(contrast_accum / n_orientations, size=w_size)
    dissimilarity_map = uniform_filter(dissimilarity_accum / n_orientations, size=w_size)
    homogeneity_map = uniform_filter(homogeneity_accum / n_orientations, size=w_size)
    asm_map = uniform_filter(asm_accum / n_orientations, size=w_size)
    energy_map = np.sqrt(np.clip(asm_map, 0, None))
    corr_map = uniform_filter(corr_accum / n_orientations, size=w_size)

    feature_maps = {
        "contrast": contrast_map,
        "dissimilarity": dissimilarity_map,
        "homogeneity": homogeneity_map,
        "energy": energy_map,
        "ASM": asm_map,
        "correlation": corr_map,
    }

    ys = list(range(0, max(h - window_size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - window_size, 0) + 1, stride)) or [0]
    centers = np.zeros((len(ys), len(xs), 2), dtype=np.int32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            centers[i, j] = (y + window_size // 2, x + window_size // 2)

    return feature_maps, centers
