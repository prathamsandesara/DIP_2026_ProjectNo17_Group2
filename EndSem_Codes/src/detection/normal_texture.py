"""
Normal-texture reference model.

For each fabric type, and for each texture feature map produced by
`src.texture.texture_scale.extract_multiscale_features` on DEFECT-FREE
TRAIN images, we estimate per-pixel-location-independent reference
statistics: mean and standard deviation of that feature's value pool,
computed over all training defect-free images of that fabric type
(pooling all pixels, since fabric texture is treated as spatially
stationary -- a reasonable classical assumption for a regularly woven
material).

    mu_k  = mean over all training pixels of feature k
    sigma_k = std over all training pixels of feature k

These (mu_k, sigma_k) pairs define the "normal texture reference" used by
`deviation_map.py` to Z-score any test feature map:

    Z_k(x, y) = (F_k(x, y) - mu_k) / sigma_k

Only defect-free TRAIN images may be passed to `fit_normal_texture_model`
(see splitter.normal_texture_training_samples for the leakage-safe
selection).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class NormalTextureModel:
    """Reference statistics for one fabric type, one feature-map key."""

    fabric_id: str
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    n_images: int = 0


def fit_normal_texture_model(fabric_id: str, feature_maps_list: List[Dict[str, np.ndarray]]) -> NormalTextureModel:
    """Fit a `NormalTextureModel` by pooling pixel values of every feature
    map across all provided (defect-free, TRAIN) images.

    `feature_maps_list` : list of {feature_name: 2D array} dicts, one per
    training image (as produced by extract_scale_features, flattened
    across scales beforehand -- see feature_fusion.flatten_multiscale).
    """
    if not feature_maps_list:
        raise ValueError(f"No feature maps supplied to fit normal texture model for fabric {fabric_id}")

    pooled: Dict[str, List[np.ndarray]] = {}
    for feature_maps in feature_maps_list:
        for key, fmap in feature_maps.items():
            pooled.setdefault(key, []).append(fmap.ravel())

    means, stds = {}, {}
    for key, arrays in pooled.items():
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        means[key] = float(np.mean(values)) if values.size else 0.0
        stds[key] = float(np.std(values)) if values.size else 1.0
        if stds[key] < 1e-8:
            stds[key] = 1e-8  # guard against division by zero for constant features

    return NormalTextureModel(fabric_id=fabric_id, means=means, stds=stds, n_images=len(feature_maps_list))


def model_to_dict(model: NormalTextureModel) -> Dict:
    return {
        "fabric_id": model.fabric_id,
        "n_images": model.n_images,
        "means": model.means,
        "stds": model.stds,
    }


def model_from_dict(data: Dict) -> NormalTextureModel:
    return NormalTextureModel(
        fabric_id=data["fabric_id"], means=data["means"], stds=data["stds"], n_images=data["n_images"],
    )
