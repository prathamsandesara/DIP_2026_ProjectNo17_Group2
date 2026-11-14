"""
Texture-deviation map computation.

Per-feature deviation
------------------------
Every flattened feature map F_k(x,y) is Z-scored against its
NormalTextureModel reference statistics:

    Z_k(x, y) = (F_k(x, y) - mu_k) / sigma_k

|Z_k(x,y)| is the per-pixel, per-feature deviation from normal texture,
already unit-free and comparable across differently-scaled features
(LBP histogram bin counts, GLCM contrast, Gabor energy, ...) -- this
implements the "normalize the individual components before fusion" step
required by the project spec.

Per-descriptor aggregation
------------------------------
Within one descriptor family, the per-feature deviations are combined by
their mean absolute Z-score:

    LBP_deviation(x,y)   = mean_k |Z_k(x,y)|      for k in LBP features
    GLCM_deviation(x,y)  = mean_k |Z_k(x,y)|      for k in GLCM features
    Gabor_deviation(x,y) = mean_k |Z_k(x,y)|      for k in Gabor features

Feature fusion
------------------
The final combined deviation map is a weighted sum:

    D(x,y) = w1 * LBP_deviation(x,y) + w2 * GLCM_deviation(x,y) + w3 * Gabor_deviation(x,y)

with w1+w2+w3 = 1 (configs/*.yaml: `feature_weights`). Equal weights
(1/3 each) are the default, reflecting no prior assumption about which
descriptor is more reliable; EXPERIMENTS.md's ablation study empirically
investigates whether unequal/descriptor-specific weighting is justified.

Multi-scale combination
---------------------------
D is computed independently per scale (small/medium/large) using that
scale's own features, then the three scale maps are combined by
element-wise maximum:

    D_final(x,y) = max(D_small(x,y), D_medium(x,y), D_large(x,y))

Max-combination is used (rather than averaging) because a genuine defect
is expected to produce a strong deviation at AT LEAST one scale that
matches its physical size, and averaging across scales would dilute a
strong single-scale signal with two scales at which the defect may be
too small/large to disrupt the local texture statistics.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.signal import wiener as _wiener

from src.detection.normal_texture import NormalTextureModel


def compute_zscore_deviation(feature_map: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.abs((feature_map - mean) / std)


def per_descriptor_deviation(
    flat_features: Dict[str, np.ndarray], model: NormalTextureModel, descriptor_prefix: str
) -> np.ndarray:
    """Mean |Z-score| over all feature keys whose feature-name part starts
    with `descriptor_prefix` ("lbp", "glcm", or "gabor")."""
    z_maps = []
    for key, fmap in flat_features.items():
        feature_part = key.split("__", 1)[1]
        if feature_part.startswith(descriptor_prefix):
            mean = model.means.get(key, 0.0)
            std = model.stds.get(key, 1.0)
            z_maps.append(compute_zscore_deviation(fmap, mean, std))
    if not z_maps:
        shape = next(iter(flat_features.values())).shape
        return np.zeros(shape, dtype=np.float64)
    return np.mean(np.stack(z_maps, axis=0), axis=0)


def normalize_deviation(deviation: np.ndarray) -> np.ndarray:
    """Min-max normalize a deviation map to [0, 1] for fusion, so that
    descriptors with naturally larger Z-score ranges do not dominate the
    weighted sum purely due to scale."""
    d_min, d_max = float(np.min(deviation)), float(np.max(deviation))
    if d_max - d_min < 1e-12:
        return np.zeros_like(deviation)
    return (deviation - d_min) / (d_max - d_min)


def stable_scale_normalize(deviation: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Normalize a per-descriptor deviation map by dividing by its p-th
    percentile value (default 95th), so that the result has a stable [0, ~1]
    range regardless of descriptor family.

    FIX (mid-sem bug): this percentile MUST be computed once from TRAINING
    normal images and reused as a fixed constant for every image at
    inference (see `stable_scale_normalize_with_const` below and
    `fit_pipeline` in src/pipeline.py, which computes and stores these
    constants in `FittedPipeline.norm_constants`). Computing the percentile
    from the image currently being processed (what this function alone
    does) makes EVERY image -- defective or not -- normalize to the same
    [0, ~1] range by construction, defeating the whole point of a
    deviation-based detector: every image would look "equally anomalous".
    This function is kept only as a fallback for callers (tests, ad-hoc
    analysis) that have no training-derived constant available; the real
    pipeline always goes through `stable_scale_normalize_with_const`.
    """
    p = float(np.percentile(deviation, percentile))
    if p < 1e-12:
        return deviation  # constant map -- leave as zeros, don't amplify noise
    return deviation / p


def stable_scale_normalize_with_const(deviation: np.ndarray, norm_const: float) -> np.ndarray:
    """Same idea as `stable_scale_normalize`, but divides by a FIXED
    constant (the descriptor's 95th-percentile deviation, pooled over
    TRAINING normal images -- computed once in `fit_pipeline` and reused
    for every image at inference). This is what makes the resulting
    normalized deviation scores -- and therefore the calibrated threshold
    -- actually comparable across images."""
    if norm_const < 1e-12:
        return deviation
    return deviation / norm_const


def fuse_descriptor_deviations(
    lbp_dev: np.ndarray, glcm_dev: np.ndarray, gabor_dev: np.ndarray, weights: Dict[str, float]
) -> np.ndarray:
    """D = w1*LBP + w2*GLCM + w3*Gabor fused in raw mean-|Z-score| units.

    NOTE: per-image min-max normalization was intentionally removed here.
    Normalizing each image's deviation map to [0,1] independently before
    fusion makes the training-derived percentile threshold incomparable to
    test-image deviation scores (every image, defective or not, ends up
    with the same [0,1] range, so the threshold is always near the maximum
    and fires on nothing). Keeping raw Z-score units ensures that the 99th-
    percentile threshold calibrated on defect-free training images is
    actually meaningful when applied to test images.
    """
    w1, w2, w3 = weights.get("lbp", 1 / 3), weights.get("glcm", 1 / 3), weights.get("gabor", 1 / 3)
    total = w1 + w2 + w3
    w1, w2, w3 = w1 / total, w2 / total, w3 / total
    return (
        w1 * lbp_dev
        + w2 * glcm_dev
        + w3 * gabor_dev
    )


def compute_scale_deviation(
    flat_features: Dict[str, np.ndarray], model: NormalTextureModel, weights: Dict[str, float],
    descriptors: List[str] = ("lbp", "glcm", "gabor"),
    norm_constants: Dict[str, float] = None,
) -> Dict[str, np.ndarray]:
    """Compute per-descriptor deviations and their fused deviation map for
    one texture scale. Returns a dict with keys "lbp", "glcm", "gabor",
    "fused" (missing descriptors, per the ablation `descriptors` list,
    are returned as all-zero maps so shapes stay consistent).

    Descriptor normalization: before fusion, each descriptor's mean-|Z-score|
    map is divided by its own 95th-percentile value (stable_scale_normalize).
    This removes the scale difference between descriptor families (LBP bins
    in [0,1] vs. locally-windowed Gabor energy) so the configured weights
    (lbp: 1/3, glcm: 1/3, gabor: 1/3) actually have equal effect on the
    fused map. Crucially, this is NOT per-image min-max; pixels with
    deviations above the 95th percentile (genuine defect candidates) will
    score >1.0 after normalization, so the threshold retains discriminative
    power."""
    shape = next(iter(flat_features.values())).shape
    zero = np.zeros(shape, dtype=np.float64)

    lbp_dev = per_descriptor_deviation(flat_features, model, "lbp") if "lbp" in descriptors else zero
    glcm_dev = per_descriptor_deviation(flat_features, model, "glcm") if "glcm" in descriptors else zero
    gabor_dev = per_descriptor_deviation(flat_features, model, "gabor") if "gabor" in descriptors else zero

    # Apply stable per-descriptor normalization so each family contributes
    # proportionally to its configured weight, not proportionally to its
    # raw Z-score magnitude. When `norm_constants` (TRAINING-derived, fixed)
    # is supplied, use it -- this is the path every real pipeline run takes.
    # Falling back to the image's own percentile is only for isolated/test
    # usage; see `stable_scale_normalize`'s docstring.
    norm_constants = norm_constants or {}
    if "lbp" in descriptors:
        lbp_dev = (stable_scale_normalize_with_const(lbp_dev, norm_constants["lbp"])
                   if "lbp" in norm_constants else stable_scale_normalize(lbp_dev))
    if "glcm" in descriptors:
        glcm_dev = (stable_scale_normalize_with_const(glcm_dev, norm_constants["glcm"])
                    if "glcm" in norm_constants else stable_scale_normalize(glcm_dev))
    if "gabor" in descriptors:
        gabor_dev = (stable_scale_normalize_with_const(gabor_dev, norm_constants["gabor"])
                     if "gabor" in norm_constants else stable_scale_normalize(gabor_dev))

    active_weights = {k: (v if k in descriptors else 0.0) for k, v in weights.items()}
    fused = fuse_descriptor_deviations(lbp_dev, glcm_dev, gabor_dev, active_weights)

    return {"lbp": lbp_dev, "glcm": glcm_dev, "gabor": gabor_dev, "fused": fused}


def compute_training_norm_constants(
    per_image_flat_features: List[Dict[str, np.ndarray]], model: NormalTextureModel,
    descriptors: List[str] = ("lbp", "glcm", "gabor"), percentile: float = 95.0,
) -> Dict[str, float]:
    """Pool the RAW (pre-normalization) per-descriptor mean-|Z-score| maps
    over every TRAINING (normal) image and return each descriptor's
    `percentile`-th value -- the fixed normalization constant used by
    `stable_scale_normalize_with_const` for every image at inference
    time (see `fit_pipeline` in src/pipeline.py). Computed once from
    training data only; never touches test images."""
    pooled: Dict[str, List[np.ndarray]] = {d: [] for d in descriptors}
    for flat in per_image_flat_features:
        for d in descriptors:
            pooled[d].append(per_descriptor_deviation(flat, model, d).ravel())
    return {
        d: float(np.percentile(np.concatenate(vals), percentile)) if vals else 0.0
        for d, vals in pooled.items()
    }


def combine_scales_max(scale_deviation_maps: List[np.ndarray]) -> np.ndarray:
    """D_final = max over scales of the fused deviation map at each scale."""
    return np.max(np.stack(scale_deviation_maps, axis=0), axis=0)


def combine_scales_weighted(
    scale_deviation_maps: List[np.ndarray],
    weights: List[float] = (0.2, 0.5, 0.3),
) -> np.ndarray:
    """Weighted sum of scale deviation maps [small, medium, large].

    Gives medium-scale more influence (0.5) since it best matches the
    weave period, while small-scale LBP (noisier) gets a lower weight.
    This reduces False Positives from small-scale noise spikes dominating.
    """
    total = sum(weights)
    result = np.zeros_like(scale_deviation_maps[0], dtype=np.float64)
    for w, m in zip(weights, scale_deviation_maps):
        result += (w / total) * m.astype(np.float64)
    return result.astype(scale_deviation_maps[0].dtype)


def smooth_deviation_wiener(deviation: np.ndarray, mysize: int = 5) -> np.ndarray:
    """ECE501 Module 3 — Wiener (MMSE) filter applied to the deviation map.

    Scipy's wiener() is a local MMSE filter that adaptively smooths flat
    background regions (where deviation is nearly constant) while preserving
    sharp peaks (genuine defect blobs). This reduces impulsive noise in the
    deviation map before thresholding, directly complementing the
    morphological post-processing step.

    Reference: Gonzalez & Woods Sec. 5.8 (Wiener filter).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = _wiener(deviation.astype(np.float64), mysize=(mysize, mysize))
    return np.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0).astype(deviation.dtype)
