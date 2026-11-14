"""
Fabric-type-wise performance breakdown: aggregates pixel-level metrics
separately for every fabric type, so per-fabric texture difficulty can be
compared (RESULTS.md: "fabric-wise results").
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.evaluation.pixel_metrics import PixelMetrics, aggregate_pixel_metrics


def compute_fabric_wise_metrics(
    fabric_ids: List[str], per_image_metrics: List[PixelMetrics]
) -> Dict[str, PixelMetrics]:
    """`fabric_ids[i]` and `per_image_metrics[i]` must correspond to the
    same image (same ordering)."""
    grouped: Dict[str, List[PixelMetrics]] = defaultdict(list)
    for fid, m in zip(fabric_ids, per_image_metrics):
        grouped[fid].append(m)
    return {fid: aggregate_pixel_metrics(metrics) for fid, metrics in grouped.items()}


def fabric_metrics_to_table(metrics_by_fabric: Dict[str, PixelMetrics]) -> List[Dict]:
    return [
        {
            "fabric_id": fid, "iou": m.iou, "precision": m.precision,
            "recall": m.recall, "f1": m.f1, "dice": m.dice,
            "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
        }
        for fid, m in sorted(metrics_by_fabric.items())
    ]
