"""
Threshold selection for the texture-deviation map, estimated ONLY from
TRAIN/VALIDATION deviation statistics -- never from test ground truth
(REQUIREMENT: "the final threshold must NOT be chosen using the test
ground-truth mask").

Supported classical methods
------------------------------
- Otsu's method (maximizes between-class variance) applied to the
  deviation-score histogram pooled over training images.
- Mean + k*sigma: threshold = mean(D_train) + k * std(D_train), a
  standard "outlier beyond k standard deviations" rule for anomaly maps.
- Percentile-based: threshold = the p-th percentile of pooled training
  deviation scores (e.g. the 99th percentile), i.e. calibrate the
  threshold to a chosen "false alarm budget" on defect-free data.
- Median + k*MAD: robust alternative to mean/std that is less sensitive
  to any residual outliers/noise in the training deviation scores;
  MAD = median(|D - median(D)|), scaled by 1.4826 to approximate sigma
  for a Gaussian-like distribution.

The method actually used is selected by `configs/*.yaml` (`threshold.method`)
and validated against the VAL split's pixel-level F1 (still no test-mask
access) to (optionally) pick k / percentile among a small candidate grid --
see scripts/run_fixed.py / run_adaptive.py.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.filters import threshold_otsu


def otsu_threshold(deviation_scores: np.ndarray) -> float:
    values = deviation_scores.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0 or np.all(values == values[0]):
        return float(np.max(deviation_scores)) if deviation_scores.size else 0.0
    return float(threshold_otsu(values))


def mean_k_std_threshold(deviation_scores: np.ndarray, k: float = 3.0) -> float:
    values = deviation_scores.ravel()
    return float(np.mean(values) + k * np.std(values))


def percentile_threshold(deviation_scores: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(deviation_scores.ravel(), percentile))


def median_mad_threshold(deviation_scores: np.ndarray, k: float = 5.0) -> float:
    values = deviation_scores.ravel()
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    robust_sigma = 1.4826 * mad
    return float(median + k * robust_sigma)


def select_threshold(pooled_train_deviation: np.ndarray, config: Dict) -> float:
    """Dispatch to the configured threshold-selection method.

    `config` is the `threshold` section of the YAML config, e.g.:
        method: percentile
        k: 3.0
        percentile: 99.0
    """
    method = config.get("method", "percentile")
    if method == "otsu":
        return otsu_threshold(pooled_train_deviation)
    if method == "mean_k_std":
        return mean_k_std_threshold(pooled_train_deviation, k=config.get("k", 3.0))
    if method == "percentile":
        return percentile_threshold(pooled_train_deviation, percentile=config.get("percentile", 99.0))
    if method == "median_mad":
        return median_mad_threshold(pooled_train_deviation, k=config.get("k", 5.0))
    raise ValueError(f"Unknown threshold method: {method}")
