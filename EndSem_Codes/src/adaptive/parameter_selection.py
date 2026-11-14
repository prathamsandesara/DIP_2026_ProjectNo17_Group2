"""
Maps measured normal-texture characteristics (scale_estimation.py) to
concrete, per-fabric-type texture-analysis parameters (LBP radius/points,
GLCM window/distance, Gabor frequency/sigma) for THREE adaptive scales
(small/medium/large), mirroring the fixed pipeline's structure so the two
are directly comparable (EXPERIMENTS.md, RESULTS.md).

Mapping rules (documented explicitly, not a black box)
----------------------------------------------------------
Let P = mean dominant weave period (pixels), estimated from
autocorrelation; V = variance-stabilization window; f_dom = dominant
Gabor frequency.

  medium scale  <-> matched directly to the fabric's own periodicity:
      glcm_window   = round_to_odd(clip(P, min_window, max_window))
      glcm_distance = round(P / 4), at least 1
      lbp_radius    = round(clip(P / 4, min_radius, max_radius))
      lbp_points    = 8 * lbp_radius
      gabor_frequency = f_dom
      gabor_sigma   = glcm_window / 2

  small scale   <-> a fixed fraction of the medium scale (captures
                    finer sub-period structure such as individual yarns):
      glcm_window   = round_to_odd(medium.glcm_window * small_scale_ratio)
      lbp_radius    = max(1, round(medium.lbp_radius * small_scale_ratio))
      gabor_frequency = medium.gabor_frequency / small_scale_ratio  (higher freq)

  large scale   <-> a fixed multiple of the medium scale (captures
                    coarse/large-area defects):
      glcm_window   = round_to_odd(medium.glcm_window * large_scale_ratio)
      lbp_radius    = round(medium.lbp_radius * large_scale_ratio)
      gabor_frequency = medium.gabor_frequency / large_scale_ratio (lower freq)

All ratios/bounds are configurable in `configs/adaptive.yaml`
(`adaptive.parameter_mapping`), so the mapping can be tuned without
touching code, while the *procedure* (measure -> map) stays fixed and
mathematically explainable, satisfying the requirement that "adaptive"
not simply mean manually-chosen-per-fabric parameters.
"""
from __future__ import annotations

from typing import Dict, List

from src.adaptive.scale_estimation import TextureCharacteristics
from src.texture.texture_scale import ScaleParams


def _round_to_odd(x: float) -> int:
    n = int(round(x))
    return n if n % 2 == 1 else n + 1


def select_adaptive_scales(tc: TextureCharacteristics, mapping_config: Dict) -> List[ScaleParams]:
    """Derive [small, medium, large] `ScaleParams` for one fabric type from
    its measured `TextureCharacteristics`, per the rules documented above.
    """
    min_window = mapping_config.get("min_glcm_window", 7)
    max_window = mapping_config.get("max_glcm_window", 33)
    min_radius = mapping_config.get("min_lbp_radius", 1)
    max_radius = mapping_config.get("max_lbp_radius", 8)
    small_ratio = mapping_config.get("small_scale_ratio", 0.5)
    large_ratio = mapping_config.get("large_scale_ratio", 2.0)

    period = max((tc.mean_period_x + tc.mean_period_y) / 2.0, 2.0)

    medium_window = _round_to_odd(min(max(period, min_window), max_window))
    medium_distance = max(1, int(round(period / 4)))
    medium_radius = int(min(max(round(period / 4), min_radius), max_radius))
    medium_points = 8 * medium_radius
    medium_freq = tc.dominant_gabor_frequency
    medium_sigma = medium_window / 2.0

    medium = ScaleParams(
        name="medium", lbp_radius=medium_radius, lbp_points=medium_points,
        glcm_window=medium_window, glcm_distance=medium_distance,
        gabor_frequency=medium_freq, gabor_sigma=medium_sigma,
    )

    small_window = _round_to_odd(max(medium_window * small_ratio, min_window))
    small_radius = max(1, int(round(medium_radius * small_ratio)))
    small = ScaleParams(
        name="small", lbp_radius=small_radius, lbp_points=8 * small_radius,
        glcm_window=small_window, glcm_distance=max(1, int(round(medium_distance * small_ratio))),
        gabor_frequency=min(medium_freq / max(small_ratio, 1e-3), 0.45),
        gabor_sigma=small_window / 2.0,
    )

    large_window = _round_to_odd(min(medium_window * large_ratio, max_window))
    large_radius = min(max_radius, int(round(medium_radius * large_ratio)))
    large_radius = max(large_radius, small_radius + 1)
    large = ScaleParams(
        name="large", lbp_radius=large_radius, lbp_points=8 * large_radius,
        glcm_window=large_window, glcm_distance=max(1, int(round(medium_distance * large_ratio))),
        gabor_frequency=max(medium_freq / large_ratio, 0.02),
        gabor_sigma=large_window / 2.0,
    )

    return [small, medium, large]


def scale_params_to_dict(scale: ScaleParams) -> Dict:
    return {
        "name": scale.name,
        "lbp_radius": scale.lbp_radius,
        "lbp_points": scale.lbp_points,
        "glcm_window": scale.glcm_window,
        "glcm_distance": scale.glcm_distance,
        "gabor_frequency": scale.gabor_frequency,
        "gabor_sigma": scale.gabor_sigma,
    }
