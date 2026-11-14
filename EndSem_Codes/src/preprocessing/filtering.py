"""
Spatial-domain noise reduction filters.

ECE501 Module 3 — Image Restoration / Noise-Only Restoration
-------------------------------------------------------------
Fabric scans exhibit sensor noise, illumination speckle, and occasional
impulsive (dust/scanner) noise. This module provides the full set of
filters covered in ECE501 Module 3:

  Gaussian    (low-pass, attenuates additive Gaussian noise)
  Median      (order-statistic, edge-preserving, kills salt-and-pepper)
  Geometric   (mean filter for multiplicative/speckle noise -- Module 3:
               arithmetic/geometric/harmonic mean filters)
  Alpha-Trimmed (trims d/2 extremes from each neighborhood then averages;
               d=0 -> arithmetic mean, d=n-1 -> median; best all-around
               when noise type is unknown -- Module 3: order-statistic
               filters, alpha-trimmed mean)

References: Gonzalez & Woods, Sec. 5.3 (mean filters), 5.4 (order-statistic).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import generic_filter


def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply 2-D Gaussian smoothing. `sigma` is in pixels."""
    return ndimage.gaussian_filter(image.astype(np.float32), sigma=sigma)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a `size x size` median filter. `size` should be odd."""
    if size % 2 == 0:
        raise ValueError("median filter size must be odd")
    return ndimage.median_filter(image.astype(np.float32), size=size)


def geometric_mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """ECE501 Module 3: Geometric mean filter.

    f_hat(x,y) = prod(neighborhood)^(1/mn)
               = exp( (1/mn) * sum(log(s(x,y))) )

    Better than arithmetic mean for multiplicative/speckle noise (common in
    line-scan fabric imaging) because it does not amplify isolated outliers
    as strongly. Computed efficiently via log -> uniform_filter -> exp.
    """
    if size % 2 == 0:
        raise ValueError("filter size must be odd")
    img = np.clip(image, 1e-6, None).astype(np.float64)
    log_img = np.log(img)
    log_mean = ndimage.uniform_filter(log_img, size=size)
    return np.exp(log_mean).astype(np.float32)


def alpha_trimmed_mean_filter(image: np.ndarray, size: int = 3, d: int = 4) -> np.ndarray:
    """ECE501 Module 3: Alpha-trimmed mean filter.

    For each size×size neighborhood, drop the d/2 lowest and d/2 highest
    pixel values, then average the remaining (size^2 - d) values.

        d = 0           → arithmetic mean filter
        d = size^2 - 1  → median filter

    This makes it the best general-purpose filter when noise type is unknown
    (combines the smoothing of mean filters with the outlier rejection of
    median filters). n = size^2 pixels per neighborhood.
    """
    if size % 2 == 0:
        raise ValueError("filter size must be odd")
    n = size * size
    d = int(np.clip(d, 0, n - 1))
    half_d = d // 2

    def _alpha_trimmed(neighborhood: np.ndarray) -> float:
        sorted_vals = np.sort(neighborhood)
        trimmed = sorted_vals[half_d: n - half_d] if half_d > 0 else sorted_vals
        return float(np.mean(trimmed))

    return generic_filter(
        image.astype(np.float64), _alpha_trimmed, size=size
    ).astype(np.float32)


def denoise(
    image: np.ndarray,
    method: str = "gaussian",
    gaussian_sigma: float = 1.0,
    median_size: int = 3,
    geom_size: int = 3,
    alpha_size: int = 3,
    alpha_d: int = 4,
) -> np.ndarray:
    """Dispatch to the configured denoising method.

    Methods (ECE501 Module 3):
      gaussian        -- Gaussian low-pass (additive noise)
      median          -- Median filter (salt-and-pepper)
      geometric_mean  -- Geometric mean (multiplicative/speckle noise)
      alpha_trimmed   -- Alpha-trimmed mean (unknown noise type) [DEFAULT RECOMMENDATION]
      both            -- median then Gaussian
      none            -- no filtering
    """
    if method == "none":
        return image.astype(np.float32)
    if method == "gaussian":
        return gaussian_filter(image, sigma=gaussian_sigma)
    if method == "median":
        return median_filter(image, size=median_size)
    if method == "geometric_mean":
        return geometric_mean_filter(image, size=geom_size)
    if method == "alpha_trimmed":
        return alpha_trimmed_mean_filter(image, size=alpha_size, d=alpha_d)
    if method == "both":
        return gaussian_filter(median_filter(image, size=median_size), sigma=gaussian_sigma)
    raise ValueError(f"Unknown denoising method: {method}")
