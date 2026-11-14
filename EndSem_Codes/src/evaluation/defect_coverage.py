"""
Defect-level coverage via a circular-shift permutation test.

For each defective test image, the predicted mask is circularly shifted
by ~300 random (dx, dy) offsets (wrapping around image borders) and the
pixel overlap with the (fixed) ground-truth mask is recomputed each time.
This builds a null distribution of "how much overlap would we see if the
predicted blob were in a random position, but kept its size/shape" --
a classical, assumption-free significance test (no parametric model of
the overlap distribution is needed).

The observed (unshifted) overlap is compared against this null: the
empirical p-value is the fraction of shifted overlaps that are >= the
observed overlap. A defect is counted as "covered" (a hit) if this
p-value is below `alpha` (default 0.05), i.e. the detector's predicted
mask overlaps the true defect location far more than chance placement
would. The defect-level hit-rate is the fraction of defective images
that pass this test.

This is deliberately simple and classical (no learned model), matching
the rest of the pipeline's design philosophy, and is reused from a
prior analysis pass on this project (see scratch/).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CoverageResult:
    sample_id: str
    observed_overlap: int
    p_value: float
    hit: bool


def circular_shift_overlap_test(
    pred_mask: np.ndarray, gt_mask: np.ndarray, n_shifts: int = 300, seed: int = 0,
) -> "tuple[int, float]":
    """Return (observed_overlap_pixels, empirical_p_value) for one image."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    observed = int(np.sum(pred & gt))

    if not gt.any():
        return observed, 1.0  # nothing to localize -- test is not meaningful

    rng = np.random.default_rng(seed)
    h, w = pred.shape
    null_overlaps = np.empty(n_shifts, dtype=np.int64)
    for i in range(n_shifts):
        dy = int(rng.integers(0, h))
        dx = int(rng.integers(0, w))
        shifted = np.roll(np.roll(pred, dy, axis=0), dx, axis=1)
        null_overlaps[i] = np.sum(shifted & gt)

    p_value = float(np.mean(null_overlaps >= observed))
    return observed, p_value


def compute_defect_coverage(
    sample_ids: List[str], pred_masks: List[np.ndarray], gt_masks: List[np.ndarray],
    is_defective: List[bool], n_shifts: int = 300, alpha: float = 0.05, seed: int = 0,
) -> "tuple[List[CoverageResult], float]":
    """Run the circular-shift test on every DEFECTIVE sample (defect-free
    images have nothing to localize, so they are skipped) and return the
    per-image results plus the overall defect-level hit-rate."""
    results = []
    for i, sid in enumerate(sample_ids):
        if not is_defective[i]:
            continue
        observed, p = circular_shift_overlap_test(pred_masks[i], gt_masks[i], n_shifts=n_shifts, seed=seed + i)
        results.append(CoverageResult(sample_id=sid, observed_overlap=observed, p_value=p, hit=p < alpha))
    hit_rate = float(np.mean([r.hit for r in results])) if results else 0.0
    return results, hit_rate
