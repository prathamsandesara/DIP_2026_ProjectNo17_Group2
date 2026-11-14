"""
Feature-map bookkeeping (flattening the {scale: {feature: map}} structure
into a flat {"scale__feature": map} dict) and per-descriptor grouping,
used by both normal-texture fitting and deviation computation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def flatten_multiscale(multiscale_features: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """{"small": {"lbp_bin_0": arr, ...}, "medium": {...}} ->
    {"small__lbp_bin_0": arr, ...}"""
    flat = {}
    for scale_name, features in multiscale_features.items():
        for feat_name, fmap in features.items():
            flat[f"{scale_name}__{feat_name}"] = fmap
    return flat


def group_by_descriptor(flat_features: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
    """Group flattened feature keys by descriptor family: lbp / glcm / gabor.
    Used by the ablation framework (LBP-only, GLCM-only, Gabor-only, and
    their combinations)."""
    groups: Dict[str, List[str]] = {"lbp": [], "glcm": [], "gabor": []}
    for key in flat_features:
        feature_part = key.split("__", 1)[1]
        if feature_part.startswith("lbp"):
            groups["lbp"].append(key)
        elif feature_part.startswith("glcm"):
            groups["glcm"].append(key)
        elif feature_part.startswith("gabor"):
            groups["gabor"].append(key)
    return groups


def select_descriptors(flat_features: Dict[str, np.ndarray], descriptors: List[str]) -> Dict[str, np.ndarray]:
    """Keep only feature keys belonging to the requested descriptor
    families (subset of {"lbp", "glcm", "gabor"}) -- powers the ablation
    study (EXPERIMENTS.md section on descriptor combinations)."""
    groups = group_by_descriptor(flat_features)
    keep_keys = set()
    for d in descriptors:
        keep_keys.update(groups.get(d, []))
    return {k: v for k, v in flat_features.items() if k in keep_keys}
