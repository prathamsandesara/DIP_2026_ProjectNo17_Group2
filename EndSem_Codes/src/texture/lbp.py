"""
Local Binary Patterns (LBP).

Mathematical definition
------------------------
For a center pixel g_c at (x, y) with P sampling points evenly spaced on a
circle of radius R around it, the LBP code is:

    LBP(P, R) = sum_{p=0}^{P-1} s(g_p - g_c) * 2^p
    s(z) = 1 if z >= 0 else 0

where g_p is the (bilinearly interpolated, for non-integer offsets) gray
value at sampling point p. This encodes, for every pixel, the local
*relative* intensity pattern of its circular neighborhood as a P-bit
binary number -- invariant to monotonic illumination shifts, which is
important because fabric scans can have mild brightness drift.

Uniform patterns
------------------
A pattern is "uniform" if it has at most 2 bitwise 0/1 transitions when
read circularly (e.g. 00011110 has 2 transitions; 01010000 has 4).
Uniform patterns correspond to primitive micro-structures (edges, spots,
flat regions) and dominate natural/woven textures; non-uniform patterns
are collapsed into a single bin. This (a) reduces the histogram from 2^P
to P+2 bins, and (b) suppresses noise-driven high-frequency patterns,
which matters for fabric where the majority of texture is
uniform-pattern weave structure and non-uniform patterns are more likely
to indicate genuine irregularities (candidate defects).

Why LBP for fabric texture
----------------------------
Woven fabric has a highly regular, repetitive local micro-texture. LBP
captures exactly this kind of local micro-pattern statistics cheaply and
is robust to the monotonic illumination variation typical of line-scan
fabric imaging. A local histogram of LBP codes over a window is therefore
a compact, classical descriptor of "what normal weave looks like" at a
given scale (radius).
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
from skimage.feature import local_binary_pattern


def compute_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8, method: str = "uniform") -> np.ndarray:
    """Compute the per-pixel LBP code map.

    Parameters
    ----------
    image : 2-D array (grayscale, any numeric range)
    radius : int
        Circular sampling radius R (pixels). Larger R captures coarser
        texture scale.
    n_points : int
        Number of circularly sampled neighbors P. Typically 8*radius.
    method : str
        skimage LBP variant: "uniform" (recommended, see module docstring),
        "default", "ror" (rotation invariant), "var" (rotation-invariant
        variance measure).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return local_binary_pattern(image.astype(np.uint8), P=n_points, R=radius, method=method)


def lbp_histogram(lbp_map: np.ndarray, n_points: int = 8, method: str = "uniform") -> np.ndarray:
    """Global histogram of LBP codes, normalized to sum to 1 (a probability
    distribution over local micro-patterns)."""
    if method == "uniform":
        n_bins = n_points + 2
    else:
        n_bins = int(lbp_map.max()) + 1
    hist, _ = np.histogram(lbp_map.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


def local_lbp_histograms(
    lbp_map: np.ndarray, window_size: int = 16, stride: int = 8, n_points: int = 8, method: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized sliding-window LBP histogram computation using 2D indicator
    masks and spatial uniform filtering. Computes full-resolution LBP bin
    distributions in milliseconds instead of seconds.
    """
    from scipy.ndimage import uniform_filter

    n_bins = n_points + 2 if method == "uniform" else int(lbp_map.max()) + 1
    h, w = lbp_map.shape
    w_size = max(3, window_size)

    bin_maps = np.zeros((h, w, n_bins), dtype=np.float64)
    for b in range(n_bins):
        mask = (lbp_map == b).astype(np.float64)
        bin_maps[..., b] = uniform_filter(mask, size=w_size)

    ys = list(range(0, max(h - window_size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - window_size, 0) + 1, stride)) or [0]

    histograms = bin_maps[np.ix_(ys, xs)]
    centers = np.zeros((len(ys), len(xs), 2), dtype=np.int32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            centers[i, j] = (y + window_size // 2, x + window_size // 2)

    return histograms, centers


def lbp_map_to_pixelwise_feature(
    histograms: np.ndarray, centers: np.ndarray, image_shape: Tuple[int, int], window_size: int
) -> np.ndarray:
    """Upsample or return full-resolution (H, W, n_bins) LBP feature maps."""
    from scipy.ndimage import zoom

    h, w = image_shape
    n_bins = histograms.shape[-1]
    ny, nx = histograms.shape[:2]
    zoom_y = h / ny if ny else 1.0
    zoom_x = w / nx if nx else 1.0
    res = np.zeros((h, w, n_bins), dtype=np.float64)
    for b in range(n_bins):
        res[..., b] = zoom(histograms[..., b], (zoom_y, zoom_x), order=0)[:h, :w]
    return res
