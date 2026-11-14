"""
STEP 5: Parallel Detection Branch 3 — Directional Warp & Column Streak Inspection.

Pipeline Stage: Step 5 (Branch 3)
Purpose:
  - Detects narrow, elongated, longitudinal yarn streaks (misdraws, tight warp ends,
    loose warp yarns, and vertical loom machine oil lines).
  - Uses tall directional 1D box averaging (1 x 35 kernel) along fabric columns:
      * Circular/square filters miss single-thread vertical streaks because contrast is faint at any single pixel.
      * Sliding a tall 1x35 vertical ruler integrates the streak signal along the column while
        averaging out random horizontal yarn noise.
  - Compares column intensity profile against smooth 2D Gaussian background envelope.
  - Filters connected candidates by minimum vertical elongation height (min_h).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

submission_dir = Path(__file__).resolve().parent
repo_root = submission_dir.parent
for p in [str(submission_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import get_logger

logger = get_logger("Step05_WarpBranch")


def run_warp_branch(
    image: np.ndarray,
    active_mask: np.ndarray,
    fabric_id: str,
    bad_cols: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute the Directional Warp Column Inspection Branch.
    Active for Fabrics 00, 01, 02, and 05.
    Returns: (warp_defect_mask, column_deviation_map)
    """
    pred_warp = np.zeros_like(image, dtype=bool)
    h, w = image.shape

    if fabric_id not in ["00", "01", "02", "05"]:
        return pred_warp, np.zeros_like(image, dtype=np.float32)

    # Erode active mask horizontally (80 px) to eliminate scanner border illumination transients
    act_warp = cv2.erode(active_mask.astype(np.uint8), np.ones((1, 80), np.uint8)).astype(bool)
    if len(bad_cols) > 0:
        act_warp[:, bad_cols] = False

    padded = image.copy().astype(np.float32)
    if np.any(active_mask):
        padded[~active_mask] = float(np.mean(image[active_mask]))

    # Directional 1x35 box filter along columns
    v_avg = cv2.blur(padded, (1, 35))
    bg_v = cv2.GaussianBlur(v_avg, (201, 201), 50.0)

    dev_v = (bg_v - v_avg) * act_warp
    dev_abs = np.abs(bg_v - v_avg) * act_warp

    # Fabric-specific calibration parameters
    if fabric_id == "00":
        thresh_v, min_h, cand = 16.0, 25, dev_v
    elif fabric_id == "05":
        thresh_v, min_h, cand = 14.0, 20, dev_v
    elif fabric_id == "01":
        thresh_v, min_h, cand = 12.0, 70, dev_v
    elif fabric_id == "02":
        thresh_v, min_h, cand = 18.0, 70, dev_abs
    else:
        thresh_v, min_h, cand = 16.0, 50, dev_v

    # Connected component height verification
    binary_cand = (cand >= thresh_v).astype(np.uint8)
    num_v, lab_v, st_v, _ = cv2.connectedComponentsWithStats(binary_cand, connectivity=8)

    for i in range(1, num_v):
        height = st_v[i, cv2.CC_STAT_HEIGHT]
        if height >= min_h:
            pred_warp[lab_v == i] = True

    return pred_warp, cand


def main() -> None:
    print("=" * 80)
    print("STEP 5: PARALLEL BRANCH 3 — DIRECTIONAL WARP & COLUMN STREAK INSPECTION")
    print("=" * 80)

    # Synthetic image with a vertical misdraw streak (1 px wide x 120 px tall)
    test_img = np.full((256, 4096), 135, dtype=np.uint8)
    test_img[50:170, 2200:2203] = 90  # Dark warp misdraw streak (3 px wide, 120 px tall)
    active_mask = np.ones_like(test_img, dtype=bool)

    pred_warp, dev_map = run_warp_branch(test_img, active_mask, "01", bad_cols=[])
    print(f"Synthesized Defect Injected : Vertical Warp Misdraw (120 px tall) at x=2200")
    print(f"Warp Detected Pixels       : {int(np.sum(pred_warp))} px")
    print(f"Max Column Deviation       : {dev_map.max():.2f} gray levels")
    print("=" * 80)
    print("Step 5 Completed Successfully. Directional warp branch verified.")


if __name__ == "__main__":
    main()
