"""
Preprocessing stage: grayscale -> denoise -> normalize -> enhance.

`preprocess_image` composes the individual modules (grayscale, filtering,
normalization, enhancement) into the single ordered pipeline described in
PIPELINE.md and DIP_CONCEPTS.md, driven entirely by the `preprocessing`
section of the YAML config so no thresholds are hard-coded.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.preprocessing.alignment import (
    estimate_fabric_skew_angle,
    align_fabric_image,
    align_mask,
    inject_synthetic_skew,
)
from src.preprocessing.frequency_filters import (
    detect_harmonic_peaks_2d,
    synthesize_butterworth_notch_filter,
    suppress_weave_pattern_2d,
    butterworth_band_reject,
    notch_reject,
    apply_band_reject_to_deviation,
)
from src.preprocessing.enhancement import enhance
from src.preprocessing.filtering import denoise
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.normalization import min_max_normalize


def preprocess_image(image: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run the full preprocessing pipeline and return the final image plus
    every intermediate stage (for visualization / the 16-panel figure).

    `config` is the `preprocessing` section of the YAML config, e.g.:
        denoise:
          method: gaussian
          gaussian_sigma: 1.0
          median_size: 3
        normalization:
          out_range: [0, 255]
        enhancement:
          method: clahe
          clip_limit: 2.0
          tile_grid_size: 8
    """
    stages: Dict[str, np.ndarray] = {}

    gray = to_grayscale(image)
    stages["grayscale"] = gray

    denoise_cfg = config.get("denoise", {})
    denoised = denoise(
        gray,
        method=denoise_cfg.get("method", "gaussian"),
        gaussian_sigma=denoise_cfg.get("gaussian_sigma", 1.0),
        median_size=denoise_cfg.get("median_size", 3),
        geom_size=denoise_cfg.get("geom_size", 3),
        alpha_size=denoise_cfg.get("alpha_size", 3),
        alpha_d=denoise_cfg.get("alpha_d", 4),
    )
    stages["denoised"] = denoised

    norm_cfg = config.get("normalization", {})
    normalized = min_max_normalize(denoised, out_range=tuple(norm_cfg.get("out_range", [0, 255])))
    stages["normalized"] = normalized

    enh_cfg = config.get("enhancement", {})
    enhanced = enhance(
        normalized,
        method=enh_cfg.get("method", "clahe"),
        clip_limit=enh_cfg.get("clip_limit", 2.0),
        tile_grid_size=enh_cfg.get("tile_grid_size", 8),
    )
    stages["enhanced"] = enhanced

    return enhanced, stages
