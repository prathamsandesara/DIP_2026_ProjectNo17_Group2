"""
Image-level (defect / no-defect classification) evaluation metrics.

An image is classified "defective" if its predicted binary mask contains
at least one foreground pixel (or, optionally, at least one connected
component above a minimum area -- configurable); ground truth is
"defective" if its mask contains at least one positive pixel. Standard
binary classification metrics and confusion matrix are then computed over
the resulting per-image labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ImageMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]]


def image_is_defective(mask: np.ndarray, min_area: int = 1) -> bool:
    return int(np.sum(mask)) >= min_area


def compute_auc(scores: List[float], labels: List[bool]) -> float:
    """Image-level ROC AUC from continuous detection scores (here, each
    image's `image_confidence` -- the strongest surviving region's
    confidence, 0 if no region detected) against the binary GT label.

    Implemented directly as the Mann-Whitney U statistic (rank-sum)
    rather than pulling in scikit-learn as a new dependency:
        AUC = P(score_positive > score_negative) + 0.5 * P(tie)
    computed exactly via one sort, O(n log n). Equivalent to the
    trapezoidal-rule ROC AUC for any score/label pairing.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # average ranks for ties
    i = 0
    rank = 1.0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        rank += (j - i + 1)
        i = j + 1

    rank_sum_pos = float(ranks[labels].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_image_metrics(pred_labels: List[bool], gt_labels: List[bool]) -> ImageMetrics:
    pred = np.array(pred_labels, dtype=bool)
    gt = np.array(gt_labels, dtype=bool)

    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cm = np.array([[tn, fp], [fn, tp]])
    return ImageMetrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1, confusion_matrix=cm)
