"""
Multi-scale texture analysis orchestration.

Defines the FIXED multi-scale parameter set (Pipeline A) and provides the
generic machinery for running LBP/GLCM/Gabor at multiple scales and
combining (e.g. concatenating/averaging) their per-pixel outputs, shared
by both the fixed and adaptive pipelines.

Scale rationale (fixed pipeline)
-----------------------------------
Three scales are used, following common multi-resolution texture-analysis
practice:

  - small  : captures fine yarn-level texture / thin defects (thin bar,
             broken end): small LBP radius, small GLCM window.
  - medium : captures the weave repetition unit (typical pitch of woven
             fabric): matches GLCM window size to a few weave periods.
  - large  : captures coarse/large-area defects (stains, holes, netting):
             larger LBP radius and GLCM window.

For the FIXED pipeline these three scales are literal constants (see
configs/fixed.yaml), identical for every fabric type. For the ADAPTIVE
pipeline (src/adaptive/scale_estimation.py), the analogous three scales
are *derived* from each fabric type's own measured normal-texture
periodicity/statistics rather than fixed a priori -- see that module and
METHODOLOGY.md for the exact procedure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.texture.gabor import apply_gabor_bank, build_gabor_bank, local_gabor_energy
from src.texture.glcm import local_glcm_feature_maps
from src.texture.lbp import compute_lbp, local_lbp_histograms, lbp_map_to_pixelwise_feature

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


@dataclass
class ScaleParams:
    """One texture scale's parameter set."""

    name: str
    lbp_radius: int
    lbp_points: int
    glcm_window: int
    glcm_distance: int
    gabor_frequency: float
    gabor_sigma: float


def default_fixed_scales(config: Dict) -> List[ScaleParams]:
    """Build the three fixed scales from `configs/fixed.yaml`'s
    `texture_scales.fixed` section."""
    scales = []
    for entry in config["texture_scales"]["fixed"]:
        scales.append(ScaleParams(
            name=entry["name"],
            lbp_radius=entry["lbp_radius"],
            lbp_points=entry["lbp_points"],
            glcm_window=entry["glcm_window"],
            glcm_distance=entry["glcm_distance"],
            gabor_frequency=entry["gabor_frequency"],
            gabor_sigma=entry["gabor_sigma"],
        ))
    return scales


def extract_scale_features(
    image: np.ndarray,
    scale: ScaleParams,
    glcm_angles: List[float],
    glcm_levels: int,
    gabor_orientations: List[float],
    lbp_stride: int = 4,
) -> Dict[str, np.ndarray]:
    """Extract LBP, GLCM (Haralick), and Gabor pixelwise feature maps for
    one texture scale. Returns a flat dict of full-resolution feature maps
    keyed by descriptive names (e.g. "lbp_hist_bin_3", "glcm_contrast",
    "gabor_energy")."""
    features: Dict[str, np.ndarray] = {}

    # Fix: cast to uint8 before LBP. skimage's local_binary_pattern on float
    # images produces random codes when adjacent pixel values differ only by
    # floating-point rounding (the emitted UserWarning was genuine). Integer
    # dtype makes the s(g_p - g_c) comparisons fully deterministic.
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)

    lbp_map = compute_lbp(image_u8, radius=scale.lbp_radius, n_points=scale.lbp_points, method="uniform")
    histograms, centers = local_lbp_histograms(
        lbp_map, window_size=max(scale.glcm_window, scale.lbp_radius * 4),
        # Fix: stride=4 instead of 8. stride=8 on e.g. a 256x256 image gives
        # only a ~30x30 histogram grid that zoom-upsamples to large constant
        # blocks, destroying spatial localization of small defects. stride=4
        # doubles the grid resolution for better defect localization.
        stride=lbp_stride, n_points=scale.lbp_points, method="uniform",
    )
    lbp_feature = lbp_map_to_pixelwise_feature(histograms, centers, image.shape, scale.glcm_window)
    for b in range(lbp_feature.shape[-1]):
        features[f"lbp_bin_{b}"] = lbp_feature[..., b]

    # Fix: also pass uint8 to GLCM so quantize() operates on integer values.
    glcm_maps, _ = local_glcm_feature_maps(
        image_u8, window_size=scale.glcm_window, stride=max(scale.glcm_window // 2, 4),
        distances=[scale.glcm_distance], angles=glcm_angles, n_levels=glcm_levels,
    )
    for prop, fmap in glcm_maps.items():
        features[f"glcm_{prop}"] = fmap

    bank = build_gabor_bank(
        frequencies=[scale.gabor_frequency], orientations=gabor_orientations,
        sigma_x=scale.gabor_sigma, sigma_y=scale.gabor_sigma,
    )
    gabor_responses = apply_gabor_bank(image, bank)
    features["gabor_energy"] = local_gabor_energy(gabor_responses["energy"], window_size=scale.glcm_window)

    # ECE501 Module 5 Part 2: Canny edge density map
    # Sobel/Canny edge detection taught in Week 12. Defects always disrupt
    # the regular weave edge pattern. Local density of Canny edges within a
    # window gives a compact per-pixel "edge irregularity" feature.
    if _HAS_CV2:
        edges = _cv2.Canny(image_u8, threshold1=20, threshold2=60)
    else:
        from skimage.feature import canny as skimage_canny
        edges = skimage_canny(image_u8.astype(float) / 255.0,
                              sigma=1.0, low_threshold=0.1, high_threshold=0.2).astype(np.uint8) * 255
    from scipy.ndimage import uniform_filter as _uf
    features["edge_density"] = _uf(edges.astype(np.float64), size=scale.glcm_window)

    return features


def extract_multiscale_features(
    image: np.ndarray,
    scales: List[ScaleParams],
    glcm_angles: List[float],
    glcm_levels: int,
    gabor_orientations: List[float],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run `extract_scale_features` at every scale. Returns
    {scale_name: {feature_name: map}}."""
    return {
        scale.name: extract_scale_features(image, scale, glcm_angles, glcm_levels, gabor_orientations)
        for scale in scales
    }
