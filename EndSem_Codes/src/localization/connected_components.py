"""
Connected-component analysis for defect localization and confidence
scoring.

For each 8-connected component of the final binary defect mask, computes
geometric descriptors (area, bounding box, centroid, width, height,
aspect ratio) and deviation-based statistics (mean/max deviation inside
the region), then a per-region confidence/reliability score.

Confidence score
-------------------
A simple, interpretable formulation combining region deviation strength,
size (very large OR very tiny outlier regions are down-weighted -- tiny
regions are more likely to be residual noise; anomalously large regions
relative to the image may indicate a segmentation/illumination artifact
rather than a genuine localized defect) and descriptor agreement:

    confidence = clip( a * mean_deviation_norm
                       + b * max_deviation_norm
                       + c * agreement_norm
                       - penalty(area), 0, 1 )

where mean/max_deviation_norm are the region's mean/max fused deviation
score normalized by the global 99th-percentile deviation observed on
TRAIN data (so confidence is comparable across images), agreement_norm
measures how consistently LBP/GLCM/Gabor deviations agree in this region
(1 - normalized variance across the three per-descriptor mean
deviations: high agreement across descriptors is a more reliable
detection than one descriptor spiking alone), and penalty(area) softly
discounts implausibly small regions (a, b, c, area bounds configurable in
`configs/*.yaml`: `confidence`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import ndimage


@dataclass
class DefectRegion:
    label: int
    area: int
    bbox: tuple  # (x, y, w, h)
    centroid: tuple  # (row, col)
    width: int
    height: int
    aspect_ratio: float
    mean_deviation: float
    max_deviation: float
    confidence: float
    # ECE501 Module 7: Region descriptors (Week 13)
    compactness: float = 0.0   # perimeter^2 / (4*pi*area); = 1 for circle
    circularity: float = 0.0   # 1 / compactness; = 1 for perfect circle
    eccentricity: float = 0.0  # 0 = circle, 1 = line segment
    extra: Dict = field(default_factory=dict)


def _region_agreement(lbp_dev: np.ndarray, glcm_dev: np.ndarray, gabor_dev: np.ndarray, region_mask: np.ndarray) -> float:
    means = [float(np.mean(d[region_mask])) if region_mask.any() else 0.0 for d in (lbp_dev, glcm_dev, gabor_dev)]
    means = np.array(means)
    if means.mean() < 1e-8:
        return 0.0
    cv = float(np.std(means) / (means.mean() + 1e-8))
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def extract_defect_regions(
    binary_mask: np.ndarray,
    fused_deviation: np.ndarray,
    lbp_dev: np.ndarray,
    glcm_dev: np.ndarray,
    gabor_dev: np.ndarray,
    train_deviation_p99: float,
    confidence_weights: Dict[str, float] = None,
    min_area: int = 1,
    raw_mask: np.ndarray = None,
):
    """Run connected-component labeling on `binary_mask` and compute
    geometric + deviation + confidence statistics for each region.

    `train_deviation_p99` : the 99th percentile of fused deviation scores
    observed on TRAIN data -- used purely as a normalization constant so
    confidence scores are comparable across images (NOT derived from the
    test image itself).

    Returns `(regions, filtered_mask)`. FIX (mid-sem bug): regions that
    fail the confidence-based "Intelligent Defect-Region Validation" below
    are excluded from `regions`, but until this fix `final_mask` (what
    gets saved to disk and scored by evaluate.py) was built from the RAW
    `binary_mask` and never had those rejected pixels removed -- so the
    filtering had no effect on the actual predicted mask. `filtered_mask`
    here zeroes out every pixel belonging to a rejected component, so the
    saved mask matches the validated region list.
    """
    weights = confidence_weights or {"a": 0.4, "b": 0.3, "c": 0.3}
    labeled, n_components = ndimage.label(binary_mask.astype(bool), structure=np.ones((3, 3)))
    regions = []
    surviving_labels = []
    norm_const = max(train_deviation_p99, 1e-6)

    for lbl in range(1, n_components + 1):
        region_mask = labeled == lbl
        area = int(region_mask.sum())
        if area < min_area:
            continue

        ys, xs = np.where(region_mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        width, height = int(x1 - x0 + 1), int(y1 - y0 + 1)
        centroid = (float(ys.mean()), float(xs.mean()))
        aspect_ratio = width / height if height > 0 else 0.0

        # ECE501 Module 7: Boundary & region descriptors (Week 13 Session B)
        # Compactness = perimeter^2 / (4*pi*area)  (= 1 for a perfect circle)
        # Circularity = 1 / compactness            (= 1 for a perfect circle)
        # Eccentricity via skimage regionprops (0 = circle, 1 = line)
        try:
            from skimage.measure import regionprops, label as sk_label, perimeter as sk_perimeter
            # perimeter of region boundary
            region_only = (labeled == lbl).astype(np.uint8)
            perim = float(sk_perimeter(region_only))
            compactness = (perim ** 2) / (4 * np.pi * area + 1e-8)
            circularity = float(np.clip(1.0 / (compactness + 1e-8), 0, 1))
            props = regionprops(region_only)[0]
            eccentricity = float(props.eccentricity)
        except Exception:
            compactness = circularity = eccentricity = 0.0

        if raw_mask is not None:
            active_mask = region_mask & raw_mask.astype(bool)
            if not active_mask.any():
                active_mask = region_mask
        else:
            active_mask = region_mask

        mean_dev = float(np.mean(fused_deviation[active_mask]))
        max_dev = float(np.max(fused_deviation[active_mask]))

        mean_dev_norm = float(np.clip(mean_dev / norm_const, 0, 1))
        max_dev_norm = float(np.clip(max_dev / norm_const, 0, 1))
        agreement = _region_agreement(lbp_dev, glcm_dev, gabor_dev, region_mask)

        confidence = weights["a"] * mean_dev_norm + weights["b"] * max_dev_norm + weights["c"] * agreement
        
        # Intelligent Defect-Region Validation
        # Simple baseline penalties
        if area < 50:
            confidence -= 0.4
        elif area < 150:
            confidence -= 0.15
            
        ratio = max_dev / mean_dev if mean_dev > 0 else 1.0
        
        # We only penalize flat noise blobs if their absolute deviation is below a threshold
        # Niblack noise is almost always flat (ratio < 1.3)
        if mean_dev < 2.2 and ratio < 1.3:
            confidence -= 1.0
            
        # And if it's very large but weak, it's a lighting shadow
        if area > 1000 and mean_dev < 2.0:
            confidence -= 1.0
            
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Reject regions that failed validation
        if confidence < 0.1:
            continue

        regions.append(DefectRegion(
            label=lbl, area=area, bbox=(int(x0), int(y0), width, height),
            centroid=centroid, width=width, height=height, aspect_ratio=aspect_ratio,
            mean_deviation=mean_dev, max_deviation=max_dev, confidence=confidence,
            compactness=compactness, circularity=circularity, eccentricity=eccentricity,
        ))
        surviving_labels.append(lbl)

    filtered_mask = np.isin(labeled, surviving_labels) if surviving_labels else np.zeros_like(binary_mask, dtype=bool)
    return regions, filtered_mask.astype(binary_mask.dtype)


def regions_to_dicts(regions: List[DefectRegion]) -> List[Dict]:
    return [
        {
            "label": r.label, "area": r.area, "bbox": r.bbox, "centroid": r.centroid,
            "width": r.width, "height": r.height, "aspect_ratio": r.aspect_ratio,
            "mean_deviation": r.mean_deviation, "max_deviation": r.max_deviation,
            "confidence": r.confidence,
        }
        for r in regions
    ]


def image_level_confidence(regions: List[DefectRegion]) -> float:
    """Image-level defect confidence = confidence of the strongest region
    (0 if no region detected)."""
    if not regions:
        return 0.0
    return max(r.confidence for r in regions)
