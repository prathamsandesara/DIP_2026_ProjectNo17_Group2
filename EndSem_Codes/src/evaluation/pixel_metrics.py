"""
Pixel-level evaluation metrics: TP/TN/FP/FN, IoU, Precision, Recall,
F1-score, Dice coefficient.

Definitions (pred, gt are binary {0,1} masks of identical shape):

    TP = sum(pred == 1 & gt == 1)
    TN = sum(pred == 0 & gt == 0)
    FP = sum(pred == 1 & gt == 0)
    FN = sum(pred == 0 & gt == 1)

    IoU        = TP / (TP + FP + FN)                    (Jaccard index)
    Precision  = TP / (TP + FP)
    Recall     = TP / (TP + FN)
    F1         = 2 * Precision * Recall / (Precision + Recall)
    Dice       = 2*TP / (2*TP + FP + FN)   [equivalent to F1 for binary masks]

Empty ground-truth handling
------------------------------
When gt has zero positive pixels (a defect-free image), IoU/Precision/
Recall/F1 are ill-defined for the "positive class" in the usual sense. We
follow the convention: if gt has no positives AND pred has no positives,
all metrics = 1.0 (perfect agreement on "no defect"); if gt has no
positives but pred has positives, metrics = 0.0 (all predictions are
false positives). This is applied consistently and explicitly rather than
silently producing NaN, and is documented so aggregate averages are
interpretable (see EVALUATION section of RESULTS.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PixelMetrics:
    tp: int
    tn: int
    fp: int
    fn: int
    iou: float
    precision: float
    recall: float
    f1: float
    dice: float


def compute_pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> PixelMetrics:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    gt_empty = (tp + fn) == 0
    pred_empty = (tp + fp) == 0

    if gt_empty and pred_empty:
        iou = precision = recall = f1 = dice = 1.0
    elif gt_empty and not pred_empty:
        iou = precision = recall = f1 = dice = 0.0
    else:
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return PixelMetrics(tp=tp, tn=tn, fp=fp, fn=fn, iou=iou, precision=precision, recall=recall, f1=f1, dice=dice)


def aggregate_pixel_metrics(metrics_list) -> PixelMetrics:
    """Aggregate a list of per-image PixelMetrics by summing raw counts
    and recomputing derived metrics from the totals (micro-averaging),
    which is more statistically stable than averaging per-image ratios
    when defect area varies widely across images."""
    tp = sum(m.tp for m in metrics_list)
    tn = sum(m.tn for m in metrics_list)
    fp = sum(m.fp for m in metrics_list)
    fn = sum(m.fn for m in metrics_list)

    gt_empty = (tp + fn) == 0
    pred_empty = (tp + fp) == 0
    if gt_empty and pred_empty:
        iou = precision = recall = f1 = dice = 1.0
    elif gt_empty and not pred_empty:
        iou = precision = recall = f1 = dice = 0.0
    else:
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return PixelMetrics(tp=tp, tn=tn, fp=fp, fn=fn, iou=iou, precision=precision, recall=recall, f1=f1, dice=dice)
