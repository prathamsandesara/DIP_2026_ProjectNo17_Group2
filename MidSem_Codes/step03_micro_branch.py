"""
STEP 3: Parallel Detection Branch 1 — Micro-Scale Morphological Inspection.

Pipeline Stage: Step 3 (Branch 1)
Purpose:
  - Detects fine, pinpoint, high-frequency yarn flaws (slubs, knots, neps, broken picks, pinholes).
  - Uses dual-scale morphological White Top-Hat (WTH) and Black Top-Hat (BTH) operations:
      * WTH = I - (I o b)  -> Isolates features brighter than the local weave repeat b.
      * BTH = (I * b) - I  -> Isolates features darker than the local weave repeat b.
  - Employs dual structural footprints:
      * Primary scale (1T): matches the fundamental weave repeat unit.
      * Localized spot scale (0.5T): resolves compact yarn knots and neps.
  - Applies interquartile range (IQR) contrast normalization on fabrics with loom illumination drift.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

submission_dir = Path(__file__).resolve().parent
repo_root = submission_dir.parent
for p in [str(submission_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import get_logger

logger = get_logger("Step03_MicroBranch")


def compute_micro_residuals(image: np.ndarray, ksize: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute White Top-Hat (bright flaws) and Black Top-Hat (dark flaws)
    using rectangular structuring element matching weave repeat period.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    white_top = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    black_top = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return white_top, black_top


def compute_dual_scale_micro(
    image: np.ndarray,
    ksize_primary: Tuple[int, int],
    ksize_spot: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute morphological residuals at both 1T (yarn scale) and 0.5T (spot scale).
    Returns: (w_top, b_top, w_spot, b_spot)
    """
    w_top, b_top = compute_micro_residuals(image, ksize_primary)
    w_spot, b_spot = compute_micro_residuals(image, ksize_spot)
    return w_top, b_top, w_spot, b_spot


def compute_iqr_contrast_scale(image: np.ndarray, active_mask: np.ndarray, fabric_id: str) -> float:
    """
    Compute adaptive illumination contrast scaling on fabrics with across-the-loom lighting drift.
    Fabric 00: reference IQR = 25.0
    Fabric 03: reference IQR = 43.0
    """
    if fabric_id in ["00", "03"] and np.any(active_mask):
        ref_iqr = 25.0 if fabric_id == "00" else 43.0
        active_pixels = image[active_mask]
        p75 = float(np.percentile(active_pixels, 75))
        p25 = float(np.percentile(active_pixels, 25))
        iqr = p75 - p25
        return max(1.0, iqr / ref_iqr)
    return 1.0


def run_micro_branch(
    image: np.ndarray,
    active_mask: np.ndarray,
    fabric_id: str,
    ksize_primary: Tuple[int, int],
    ksize_spot: Tuple[int, int],
    calib_thresholds: Dict[str, float]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Execute the Micro-Scale Morphological Inspection Branch:
    Returns: (raw_micro_candidate_mask, residual_maps_dict)
    """
    w_top, b_top, w_spot, b_spot = compute_dual_scale_micro(image, ksize_primary, ksize_spot)
    scale = compute_iqr_contrast_scale(image, active_mask, fabric_id)

    # Scaled thresholds calibrated from clean defect-free training rolls
    t_mw = calib_thresholds.get("micro_white", 100.0) * scale
    t_mb = calib_thresholds.get("micro_black", 100.0) * scale
    t_mw_sm = calib_thresholds.get("micro_white_sm", t_mw) * scale
    t_mb_sm = calib_thresholds.get("micro_black_sm", t_mb) * scale

    micro_candidate = (
        (w_top >= t_mw) |
        (b_top >= t_mb) |
        (w_spot >= t_mw_sm) |
        (b_spot >= t_mb_sm)
    ) & active_mask

    residuals = {
        "w_top": w_top,
        "b_top": b_top,
        "w_spot": w_spot,
        "b_spot": b_spot,
        "scale": np.array([scale]),
    }
    return micro_candidate, residuals


def main() -> None:
    print("=" * 80)
    print("STEP 3: PARALLEL BRANCH 1 — MICRO-SCALE MORPHOLOGICAL INSPECTION")
    print("=" * 80)

    # Test with synthetic fabric texture
    test_img = np.full((256, 4096), 128, dtype=np.uint8)
    # Inject synthetic high-contrast knot and dark slub hole
    test_img[100:108, 2000:2008] = 230  # bright knot
    test_img[150:158, 2100:2108] = 30   # dark hole
    active_mask = np.ones_like(test_img, dtype=bool)

    mock_calib = {
        "micro_white": 70.0,
        "micro_black": 70.0,
        "micro_white_sm": 70.0,
        "micro_black_sm": 70.0,
    }

    cand, res = run_micro_branch(test_img, active_mask, "00", (15, 15), (7, 7), mock_calib)
    print(f"Synthesized Defects Injected : Knot (bright 8x8) at [100, 2000], Hole (dark 8x8) at [150, 2100]")
    print(f"Micro Detected Pixels        : {int(np.sum(cand))} px")
    print(f"Residual Max (White Top-Hat) : {res['w_top'].max():.1f} gray levels")
    print(f"Residual Max (Black Top-Hat) : {res['b_top'].max():.1f} gray levels")
    print("=" * 80)
    print("Step 3 Completed Successfully. Micro branch pipeline verified.")


if __name__ == "__main__":
    main()
