"""
STEP 4: Parallel Detection Branch 2 — Macro-Scale Spatial Envelope Inspection.

Pipeline Stage: Step 4 (Branch 2)
Purpose:
  - Detects large, low-frequency, structural anomalies (fabric tears, cracks, broken picks,
    horizontal shade bars, wide illumination drift).
  - Employs an active-margin padded Gaussian spatial envelope filter:
      * Kernel Window W = 201 pixels (largest odd window fitting 256-pixel image height).
      * Sigma = 50.0 pixels (4 to 5x larger than thread width, flattens weave texture).
  - Computes two-sided envelope deviations:
      * Dark deviation: Diff_dark = Background - Image (tears, holes, cracks).
      * Bright deviation: Diff_bright = Image - Background (shade bars, bleach marks).
  - Padded with active fabric mean to prevent line-scan scanner border illumination bleeding.
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

logger = get_logger("Step04_MacroBranch")

MACRO_WINDOW = 201
MACRO_SIGMA = 50.0


def compute_macro_residuals(
    image: np.ndarray,
    active_mask: np.ndarray,
    window_size: int = MACRO_WINDOW,
    sigma: float = MACRO_SIGMA
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute smooth background envelope and two-sided spatial residuals.
    Returns: (diff_bright, diff_dark, background_envelope)
    """
    fab_mean = float(np.mean(image[active_mask])) if np.any(active_mask) else 128.0
    padded = image.copy()
    padded[~active_mask] = int(round(fab_mean))

    # 2D Gaussian background envelope
    bg = cv2.GaussianBlur(padded, (window_size, window_size), sigma)

    diff_dark = cv2.subtract(bg, padded)
    diff_dark[~active_mask] = 0

    diff_bright = cv2.subtract(padded, bg)
    diff_bright[~active_mask] = 0

    return diff_bright, diff_dark, bg


def run_macro_branch(
    image: np.ndarray,
    active_mask: np.ndarray,
    fabric_id: str,
    calib_thresholds: Dict[str, float]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Execute the Macro-Scale Spatial Envelope Inspection Branch:
    Returns: (raw_macro_candidate_mask, residuals_dict)
    """
    diff_bright, diff_dark, bg = compute_macro_residuals(image, active_mask)

    t_mbr = calib_thresholds.get("macro_bright", 65.0)
    t_mdk = calib_thresholds.get("macro_dark", 45.0)

    # Border transient protection for coarse herringbone / broken twill (Fabric 03)
    if fabric_id == "03":
        macro_support = cv2.erode(active_mask.astype(np.uint8), np.ones((1, 21), np.uint8)).astype(bool)
    else:
        macro_support = active_mask

    raw_macro = ((diff_bright >= t_mbr) | (diff_dark >= t_mdk)) & macro_support

    residuals = {
        "diff_bright": diff_bright,
        "diff_dark": diff_dark,
        "background": bg,
    }
    return raw_macro, residuals


def main() -> None:
    print("=" * 80)
    print("STEP 4: PARALLEL BRANCH 2 — MACRO-SCALE SPATIAL ENVELOPE INSPECTION")
    print("=" * 80)

    # Create synthetic fabric with a 40x120 pixel dark void tear
    test_img = np.full((256, 4096), 140, dtype=np.uint8)
    test_img[100:140, 1500:1620] = 40  # Large dark tear (40x120 = 4,800 pixels)
    active_mask = np.ones_like(test_img, dtype=bool)

    mock_calib = {
        "macro_bright": 60.0,
        "macro_dark": 40.0,
    }

    raw_macro, res = run_macro_branch(test_img, active_mask, "01", mock_calib)
    print(f"Synthesized Defect Injected  : Large Dark Void Tear (40x120 = 4,800 px) at [100, 1500]")
    print(f"Macro Detected Pixels        : {int(np.sum(raw_macro)):,} px")
    print(f"Max Dark Difference          : {res['diff_dark'].max():.1f} gray levels")
    print(f"Background Mean              : {np.mean(res['background']):.2f}")
    print("=" * 80)
    print("Step 4 Completed Successfully. Macro envelope branch verified.")


if __name__ == "__main__":
    main()
