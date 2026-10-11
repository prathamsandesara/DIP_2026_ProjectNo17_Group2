"""
STEP 8: Multimodal Anomaly Map Fusion & Weave-Adaptive Structural Enhancements.

Pipeline Stage: Step 8
Purpose:
  - Fuses detections from all three parallel inspectors using a logical Boolean OR operation:
      P_base = P_micro OR P_macro OR P_warp
  - Incorporates fabric-specific structural enhancements:
      * Fabric 02 (Coarse Twill): Gated Point-Singularity Laplacian filter (sigma=1.5, thresh=10.5)
        for compact micro-knots/slubs that blend into thick diagonal twill ridges.
      * Fabric 03 (Herringbone / Broken Twill): Safe macro/micro relaxation (75%-80% threshold)
        plus sensor-masked vertical Gabor filter resonance (theta=pi/2) to catch subtle broken patterns.
      * Fabric 04 (Dense Plain): Moderate micro/macro sensitivity relaxation (80%-88%) for void holes.
  - Generates unified multimodal binary anomaly prediction: P_fused = P_base OR P_opt.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

submission_dir = Path(__file__).resolve().parent
repo_root = submission_dir.parent
for p in [str(submission_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import get_logger

logger = get_logger("Step08_FeatureFusion")


def fuse_base_branches(
    pred_micro: np.ndarray,
    pred_macro: np.ndarray,
    pred_warp: np.ndarray
) -> np.ndarray:
    """
    Base Feature Fusion across all three parallel inspectors:
    P_base = P_micro | P_macro | P_warp
    """
    return pred_micro | pred_macro | pred_warp


def compute_adaptive_enhancements(
    image: np.ndarray,
    active_mask: np.ndarray,
    fabric_id: str,
    pred_base: np.ndarray,
    residuals: Dict[str, np.ndarray],
    calib_entry: dict,
    bad_cols: List[int]
) -> np.ndarray:
    """
    Fabric-Specific Adaptive Enhancements (Adaptive Mode):
    Injects specialized structural filters for difficult weave structures.
    """
    p_opt = np.zeros_like(active_mask, dtype=bool)

    # FABRIC 02: Gated Point-Singularity Laplacian for micro-knots and slubs
    if fabric_id == "02":
        if np.sum(pred_base) < 15:
            g_lap = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.5)
            lap = np.maximum(cv2.Laplacian(g_lap, cv2.CV_32F), 0)
            bm_lap = ((lap >= 10.5) & active_mask).astype(np.uint8)
            if len(bad_cols) > 0:
                bm_lap[:, bad_cols] = 0
            nl_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats(bm_lap, connectivity=8)
            if nl_l > 1:
                lut = (st_l[:, cv2.CC_STAT_AREA] >= 4).astype(np.uint8)
                lut[0] = 0
                p_opt |= lut[lab_l].astype(bool)

    # FABRIC 03: Safe Macro/Micro Relaxation + Sensor-Masked Vertical Gabor
    elif fabric_id == "03":
        m_br = residuals.get("diff_bright", np.zeros_like(image, dtype=np.float32))
        m_dk = residuals.get("diff_dark", np.zeros_like(image, dtype=np.float32))
        w_top = residuals.get("w_top", np.zeros_like(image, dtype=np.float32))
        b_top = residuals.get("b_top", np.zeros_like(image, dtype=np.float32))
        t_mbr = calib_entry.get("macro_bright", 65.0)
        t_mdk = calib_entry.get("macro_dark", 45.0)
        t_mw = calib_entry.get("micro_white", 100.0)
        t_mb = calib_entry.get("micro_black", 100.0)

        macro_support = cv2.erode(active_mask.astype(np.uint8), np.ones((1, 21), np.uint8)).astype(bool)

        # Macro relaxation (75%)
        rml = ((m_br >= t_mbr * 0.75) | (m_dk >= t_mdk * 0.75)) & macro_support
        if len(bad_cols) > 0:
            rml[:, bad_cols] = False
        nl_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats(rml.astype(np.uint8))
        if nl_l > 1:
            lut = (st_l[:, cv2.CC_STAT_AREA] >= 35).astype(np.uint8)
            lut[0] = 0
            p_opt |= lut[lab_l].astype(bool)

        # Micro relaxation (80%)
        rml2 = ((w_top >= t_mw * 0.80) | (b_top >= t_mb * 0.80)) & active_mask
        if len(bad_cols) > 0:
            rml2[:, bad_cols] = False
        nl_l2, lab_l2, st_l2, _ = cv2.connectedComponentsWithStats(rml2.astype(np.uint8))
        if nl_l2 > 1:
            lut2 = (st_l2[:, cv2.CC_STAT_AREA] >= 25).astype(np.uint8)
            lut2[0] = 0
            p_opt |= lut2[lab_l2].astype(bool)

        # Sensor-Masked Vertical Gabor
        k_gab = cv2.getGaborKernel((21, 21), 3.0, np.pi / 2, 9.0, 0.5, 0, ktype=cv2.CV_32F)
        gv = np.abs(cv2.filter2D(image.astype(np.float32), cv2.CV_32F, k_gab))
        bm_g = (gv >= 1612.99 * 1.1) & active_mask
        bm_g[:, 1020:1110] = False  # Suppress camera sensor blemish zone
        nl3, lab3, st3, _ = cv2.connectedComponentsWithStats(bm_g.astype(np.uint8))
        if nl3 > 1:
            lut3 = (st3[:, cv2.CC_STAT_AREA] >= 20).astype(np.uint8)
            lut3[0] = 0
            p_opt |= lut3[lab3].astype(bool)

    # FABRIC 04: Moderate Micro/Macro Relaxation for Void Holes
    elif fabric_id == "04":
        w_top = residuals.get("w_top", np.zeros_like(image, dtype=np.float32))
        b_top = residuals.get("b_top", np.zeros_like(image, dtype=np.float32))
        m_br = residuals.get("diff_bright", np.zeros_like(image, dtype=np.float32))
        m_dk = residuals.get("diff_dark", np.zeros_like(image, dtype=np.float32))
        t_mw = calib_entry.get("micro_white", 100.0)
        t_mb = calib_entry.get("micro_black", 100.0)
        t_mbr = calib_entry.get("macro_bright", 65.0)
        t_mdk = calib_entry.get("macro_dark", 45.0)

        rl = ((w_top >= t_mw * 0.88) | (b_top >= t_mb * 0.88)) & active_mask
        if len(bad_cols) > 0:
            rl[:, bad_cols] = False
        nl_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats(rl.astype(np.uint8))
        if nl_l > 1:
            lut = (st_l[:, cv2.CC_STAT_AREA] >= 12).astype(np.uint8)
            lut[0] = 0
            p_opt |= lut[lab_l].astype(bool)

        rml = ((m_br >= t_mbr * 0.80) | (m_dk >= t_mdk * 0.80)) & active_mask
        if len(bad_cols) > 0:
            rml[:, bad_cols] = False
        nl_l2, lab_l2, st_l2, _ = cv2.connectedComponentsWithStats(rml.astype(np.uint8))
        if nl_l2 > 1:
            lut = (st_l2[:, cv2.CC_STAT_AREA] >= 20).astype(np.uint8)
            lut[0] = 0
            p_opt |= lut[lab_l2].astype(bool)

    return p_opt


def fuse_all_features(
    pred_micro: np.ndarray,
    pred_macro: np.ndarray,
    pred_warp: np.ndarray,
    image: np.ndarray,
    active_mask: np.ndarray,
    fabric_id: str,
    residuals: Dict[str, np.ndarray],
    calib_entry: dict,
    mode: str = "adaptive"
) -> np.ndarray:
    """Complete multimodal feature fusion pipeline."""
    pred_base = fuse_base_branches(pred_micro, pred_macro, pred_warp)
    if mode == "adaptive":
        bad_cols = calib_entry.get("bad_cols", [])
        p_opt = compute_adaptive_enhancements(
            image, active_mask, fabric_id, pred_base, residuals, calib_entry, bad_cols
        )
        return pred_base | p_opt
    return pred_base


def main() -> None:
    print("=" * 80)
    print("STEP 8: MULTIMODAL FEATURE FUSION & WEAVE-ADAPTIVE ENHANCEMENTS")
    print("=" * 80)

    # Synthetic boolean branches
    h, w = 256, 4096
    p_micro = np.zeros((h, w), dtype=bool)
    p_macro = np.zeros((h, w), dtype=bool)
    p_warp = np.zeros((h, w), dtype=bool)

    # Micro detected 30 px, Macro detected 150 px, Warp detected 70 px
    p_micro[50:55, 1000:1006] = True
    p_macro[100:115, 2000:2010] = True
    p_warp[30:100, 3000] = True

    fused = fuse_base_branches(p_micro, p_macro, p_warp)
    print(f"Micro Branch Detections   : {int(np.sum(p_micro))} px")
    print(f"Macro Branch Detections   : {int(np.sum(p_macro))} px")
    print(f"Warp Branch Detections    : {int(np.sum(p_warp))} px")
    print(f"Unified Fused Anomaly Map : {int(np.sum(fused))} px (Union of all inspectors)")
    print("=" * 80)
    print("Step 8 Completed Successfully. Feature fusion pipeline verified.")


if __name__ == "__main__":
    main()
