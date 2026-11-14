"""
Thresholding / segmentation of the texture-deviation map into a binary
defect mask, plus (optional) local/adaptive thresholding for spatially
varying deviation baselines.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def global_threshold(deviation_map: np.ndarray, threshold: float) -> np.ndarray:
    """Binary segmentation: pixel is a defect candidate if its fused
    deviation exceeds the (train-derived) global threshold."""
    return (deviation_map > threshold).astype(np.uint8)


def local_adaptive_threshold(deviation_map: np.ndarray, window_size: int = 25, k: float = 2.0) -> np.ndarray:
    """Local thresholding: pixel is a defect candidate if it exceeds the
    LOCAL mean deviation (over a window) by more than k local standard
    deviations. Useful when the deviation baseline itself drifts slowly
    across the image (e.g. mild illumination gradient not fully removed
    by preprocessing)."""
    dev = deviation_map.astype(np.float64)
    local_mean = uniform_filter(dev, size=window_size)
    local_mean_sq = uniform_filter(dev ** 2, size=window_size)
    local_std = np.sqrt(np.clip(local_mean_sq - local_mean ** 2, 0, None))
    return (dev > (local_mean + k * local_std)).astype(np.uint8)


def segment(deviation_map: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Dispatch: method = "global" (kwargs: threshold) or "local"
    (kwargs: window_size, k)."""
    if method == "global":
        return global_threshold(deviation_map, kwargs["threshold"])
    if method == "local":
        return local_adaptive_threshold(deviation_map, kwargs.get("window_size", 25), kwargs.get("k", 2.0))
    raise ValueError(f"Unknown segmentation method: {method}")
