"""
Defect-type-wise performance breakdown, analogous to fabric_metrics.py but
grouped by the 12 AITEX defect types (defect-free images are excluded
here since they carry no defect-type label; they are still included in
the overall/image-level metrics computed elsewhere).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from src.evaluation.pixel_metrics import PixelMetrics, aggregate_pixel_metrics


def compute_defect_type_wise_metrics(
    defect_types: List[Optional[str]], per_image_metrics: List[PixelMetrics]
) -> Dict[str, PixelMetrics]:
    grouped: Dict[str, List[PixelMetrics]] = defaultdict(list)
    for dtype, m in zip(defect_types, per_image_metrics):
        if dtype is None:
            continue
        grouped[dtype].append(m)
    return {dtype: aggregate_pixel_metrics(metrics) for dtype, metrics in grouped.items()}


def defect_metrics_to_table(metrics_by_defect: Dict[str, PixelMetrics]) -> List[Dict]:
    return [
        {
            "defect_type": dtype, "iou": m.iou, "precision": m.precision,
            "recall": m.recall, "f1": m.f1, "dice": m.dice,
            "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
        }
        for dtype, m in sorted(metrics_by_defect.items())
    ]
