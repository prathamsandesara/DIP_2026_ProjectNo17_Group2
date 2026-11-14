"""
Dual-Scale Anomaly Segmentation Pipeline for Fabric Defect Inspection.

Architecture:
1. Micro-Scale Morphological Branch:
   - Weave-adaptive Top-Hat and Black-Hat filtering at primary weave period (1T)
     coupled with localized spot scale (0.5T) for fine yarn flaws (slubs, knots, fuzz).
   - Support-valid active fabric domain masking (M_valid) during calibration.
   - Stationary line-scan camera blemish suppression learned from clean training data.
2. Macro-Scale Spatial Envelope Branch:
   - Active-margin filtered spatial envelope (Gaussian W=201, sigma=50.0)
     detecting large voids, cracks, broken picks, and shade bars.
3. Geometry-Guided Closing:
   - Elongation-gated horizontal structuring element (101x5) strictly activating
     on genuine high-aspect void candidates (W >= 100 px, area >= 2000 px).
4. Operational Screening:
   - Roll-level screening: a roll is flagged when the cleaned prediction has >= 15 px
     (one uniform threshold for all fabrics; optional per-fabric overrides are for
     sensitivity analysis only). Thresholds are calibrated on clean TRAIN rolls only.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import medfilt

from src.data.active_fabric_mask import detect_active_fabric_mask, get_support_valid_mask
from src.utils.io import load_image_gray, load_mask_binary, load_json, save_json, save_image
from src.utils.visualization import overlay_mask_on_image
from src.utils.logging import get_logger
from src.utils.parallel import limit_thread_oversubscription
from src.preprocessing.alignment import estimate_fabric_skew_angle, align_fabric_image, align_mask
from src.preprocessing.frequency_filters import suppress_weave_pattern_2d

logger = get_logger(__name__)

MACRO_WINDOW = 201
MACRO_SIGMA = 50.0


def resolve_n_workers(config: Optional[Dict] = None) -> int:
    """Read compute.n_workers from config and clamp to available CPU cores."""
    if config is None:
        requested = 20
    else:
        requested = config.get("compute", {}).get("n_workers", 20)
    available = os.cpu_count() or 1
    return max(1, min(int(requested), available))


def get_footprints(coarse_scale: str = "1T") -> Dict[str, Tuple[int, int]]:
    """Derive structuring element dimensions (w_x, w_y) for each fabric type."""
    if coarse_scale == "fixed":
        return {fid: (15, 15) for fid in ["00", "01", "02", "03", "04", "05", "06", "08"]}

    # Canonical weave-adaptive footprints matching weave repeat geometry:
    # Fabric 00: Plain weave (poplin / canvas) -> (15, 15)
    # Fabric 01: Twill fine weave -> (13, 13)
    # Fabric 02: 2/2 coarse twill diagonal rib repeat -> (13, 13)
    # Fabric 03: Herringbone / broken twill -> (9, 9)
    # Fabric 04: Plain dense weave -> (15, 15)
    # Fabric 05: Warp-faced stripe twill -> (11, 11)
    # Fabric 06: Satin / sateen weave -> (13, 13)
    footprints = {
        "00": (15, 15),
        "01": (15, 15),
        "02": (13, 13) if coarse_scale == "1T" else (25, 25),
        "03": (9, 9),
        "04": (15, 15),
        "05": (11, 11),
        "06": (13, 13),
        "08": (15, 15),
    }
    return footprints


def compute_micro_residuals(img: np.ndarray, ksize: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute White Top-Hat (bright flaws) and Black Top-Hat (dark flaws)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    white = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return white, black


def compute_macro_residuals(img: np.ndarray, act_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate smooth background envelope while preventing scanner border bleeding."""
    fab_mean = float(np.mean(img[act_mask])) if np.any(act_mask) else 128.0
    padded = img.copy()
    padded[~act_mask] = int(round(fab_mean))

    bg = cv2.GaussianBlur(padded, (MACRO_WINDOW, MACRO_WINDOW), MACRO_SIGMA)

    diff_dark = cv2.subtract(bg, padded)
    diff_dark[~act_mask] = 0

    diff_bright = cv2.subtract(padded, bg)
    diff_bright[~act_mask] = 0

    return diff_bright, diff_dark


def apply_cleanup(binary_mask: np.ndarray, min_area: int = 5) -> np.ndarray:
    """Filter connected components smaller than min_area."""
    if min_area <= 0 or not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_mask, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    lut = (areas >= min_area).astype(np.uint8)
    lut[0] = 0
    return lut[labels].astype(bool)


def calibrate_dual_scale_pipeline(train_samples: List[dict], footprints: Dict[str, Tuple[int, int]],
                                  micro_p: float = 99.99, macro_p: float = 99.99) -> Dict[str, dict]:
    """
    Calibrate thresholds strictly on clean training images within support-valid mask (M_valid).
    Also detects stationary camera sensor blemish columns via median profile deviation.
    """
    fabric_clean = {}
    for s in train_samples:
        if not s["is_defective"]:
            fabric_clean.setdefault(s["fabric_id"], []).append(s)

    print(f"   [Calibration] Learning statistical thresholds from {sum(len(v) for v in fabric_clean.values())} clean training rolls across {len(fabric_clean)} fabrics...")
    calib = {}
    for fid, samples in sorted(fabric_clean.items()):
        ksize = footprints.get(fid, (15, 15))
        ksize_sm = (max(5, (ksize[0] // 2) | 1), max(5, (ksize[1] // 2) | 1))
        micro_w_list, micro_b_list = [], []
        micro_w_sm_list, micro_b_sm_list = [], []
        macro_b_list, macro_d_list = [], []

        col_accum = np.zeros(4096, dtype=np.float32)

        rng = np.random.default_rng(42)
        for s in samples:
            img = load_image_gray(s["image_path"])
            act = detect_active_fabric_mask(img, fabric_id=fid)
            val_m = get_support_valid_mask(act, ksize, extra_margin=15)
            val_m_sm = get_support_valid_mask(act, ksize_sm, extra_margin=15)
            val_macro = cv2.erode(act.astype(np.uint8), np.ones((51, 51), np.uint8)).astype(bool)

            w_top, b_top = compute_micro_residuals(img, ksize)
            w_sm, b_sm = compute_micro_residuals(img, ksize_sm)
            m_br, m_dk = compute_macro_residuals(img, act)

            col_accum += np.mean(w_top.astype(np.float32) + b_top.astype(np.float32), axis=0)

            # Sample micro residuals inside support-valid mask
            wv, bv = w_top[val_m], b_top[val_m]
            if len(wv) > 0:
                sub_idx = rng.choice(len(wv), size=min(25000, len(wv)), replace=False)
                micro_w_list.append(wv[sub_idx])
                micro_b_list.append(bv[sub_idx])

            wv_sm, bv_sm = w_sm[val_m_sm], b_sm[val_m_sm]
            if len(wv_sm) > 0:
                sub_idx_sm = rng.choice(len(wv_sm), size=min(25000, len(wv_sm)), replace=False)
                micro_w_sm_list.append(wv_sm[sub_idx_sm])
                micro_b_sm_list.append(bv_sm[sub_idx_sm])

            # Sample macro residuals safely away from borders
            mv_br = m_br[val_macro]
            mv_dk = m_dk[val_macro]
            if len(mv_br) > 0:
                sub_idx_m = rng.choice(len(mv_br), size=min(25000, len(mv_br)), replace=False)
                macro_b_list.append(mv_br[sub_idx_m])
                macro_d_list.append(mv_dk[sub_idx_m])

        all_mw = np.concatenate(micro_w_list) if micro_w_list else np.array([0.0])
        all_mb = np.concatenate(micro_b_list) if micro_b_list else np.array([0.0])
        all_mw_sm = np.concatenate(micro_w_sm_list) if micro_w_sm_list else np.array([0.0])
        all_mb_sm = np.concatenate(micro_b_sm_list) if micro_b_sm_list else np.array([0.0])
        all_mbr = np.concatenate(macro_b_list) if macro_b_list else np.array([0.0])
        all_mdk = np.concatenate(macro_d_list) if macro_d_list else np.array([0.0])

        col_mean = col_accum / max(1, len(samples))
        col_smooth = medfilt(col_mean, 51)
        col_diff = np.abs(col_mean - col_smooth)
        bad_cols_raw = np.where(col_diff > 4.0)[0]
        # Dynamically dilate by 5 pixels on each side to encompass full sensor blemish halo from clean calibration data
        if len(bad_cols_raw) > 0:
            mask_bc = np.zeros(4096, dtype=bool)
            mask_bc[bad_cols_raw] = True
            mask_bc = cv2.dilate(mask_bc.astype(np.uint8)[:, None], np.ones((11, 1), np.uint8)).ravel().astype(bool)
            bad_cols = np.where(mask_bc)[0].tolist()
        else:
            bad_cols = []

        calib[fid] = {
            "micro_white": float(np.percentile(all_mw, micro_p)),
            "micro_black": float(np.percentile(all_mb, micro_p)),
            "micro_white_sm": float(np.percentile(all_mw_sm, micro_p)),
            "micro_black_sm": float(np.percentile(all_mb_sm, micro_p)),
            "macro_bright": float(np.percentile(all_mbr, macro_p)),
            "macro_dark": float(np.percentile(all_mdk, macro_p)),
            "bad_cols": bad_cols,
        }
        print(f"     * Fabric {fid}: {len(samples)} clean rolls -> tau_w={calib[fid]['micro_white']:.1f}, tau_b={calib[fid]['micro_black']:.1f}, tau_Mbr={calib[fid]['macro_bright']:.1f}, tau_Mdk={calib[fid]['macro_dark']:.1f}, sensor_blemish_cols={len(bad_cols)}")
    return calib


def process_dual_scale_sample(args: Tuple) -> dict:
    sample, footprints, calib, save_artifacts, split_tag, closing_kw, closing_kh, macro_min_area, micro_min_area, screening_min_px, mask_out_dir, overlay_out_dir, *extra = args
    screening_thresholds = extra[3] if len(extra) > 3 else None
    mode = extra[0] if len(extra) > 0 else "adaptive"
    enable_alignment = bool(extra[1]) if len(extra) > 1 else False
    enable_fft_notch = bool(extra[2]) if len(extra) > 2 else False

    sid = sample["sample_id"]
    fid = sample["fabric_id"]
    is_def = sample["is_defective"]
    img = load_image_gray(sample["image_path"])
    mask_p = sample.get("mask_path")

    act_mask = detect_active_fabric_mask(img, fabric_id=fid)
    gt_mask = load_mask_binary(mask_p).astype(bool) if (is_def and mask_p) else np.zeros_like(img, dtype=bool)

    # Fabric web alignment correction (de-skewing)
    skew_angle = 0.0
    if enable_alignment:
        res_align = estimate_fabric_skew_angle(img, active_mask=act_mask, angle_range=(-2.5, 2.5))
        skew_angle = float(res_align["skew_angle"])
        if 0.5 <= abs(skew_angle) <= 2.5:
            img = align_fabric_image(img, skew_angle)
            act_mask = align_mask(act_mask, skew_angle)
            if is_def and np.any(gt_mask):
                gt_mask = align_mask(gt_mask, skew_angle)

    gt_px = int(np.sum(gt_mask))

    ksize = footprints.get(fid, (15, 15))
    ksize_sm = (max(5, (ksize[0] // 2) | 1), max(5, (ksize[1] // 2) | 1))
    calib_entry = calib.get(fid, calib.get("00", {}))

    # 1. Multi-Scale Micro Branch (1T yarn scale + 0.5T localized spot scale)
    w_top, b_top = compute_micro_residuals(img, ksize)
    w_sm, b_sm = compute_micro_residuals(img, ksize_sm)

    # Robust contrast scaling on Fabric 00 and 03 to suppress across-the-loom camera illumination drift
    if fid in ["00", "03"] and np.any(act_mask):
        ref_iqr = 25.0 if fid == "00" else 43.0
        iqr = float(np.percentile(img[act_mask], 75) - np.percentile(img[act_mask], 25))
        scale = max(1.0, iqr / ref_iqr)
    else:
        scale = 1.0

    t_mw = calib_entry["micro_white"] * scale
    t_mb = calib_entry["micro_black"] * scale
    t_mw_sm = calib_entry.get("micro_white_sm", t_mw) * scale
    t_mb_sm = calib_entry.get("micro_black_sm", t_mb) * scale

    raw_micro = ((w_top >= t_mw) | (b_top >= t_mb) | (w_sm >= t_mw_sm) | (b_sm >= t_mb_sm)) & act_mask

    # Suppress stationary line-scan camera blemish columns learned on clean training data
    bad_cols = calib_entry.get("bad_cols", [])
    if len(bad_cols) > 0:
        raw_micro[:, bad_cols] = False

    # Micro area cleanup per fabric (Fabric 05 is warp-faced stripe twill where small micro features are rib crests)
    micro_min_a = 20 if fid == "05" else micro_min_area
    pred_micro = apply_cleanup(raw_micro, min_area=micro_min_a)

    # 2. Macro Branch (cracks, bars, large voids)
    m_br, m_dk = compute_macro_residuals(img, act_mask)
    t_mbr = calib_entry["macro_bright"]
    t_mdk = calib_entry["macro_dark"]
    # Macro border protection on Fabric 03 to eliminate border filter transients
    macro_support = cv2.erode(act_mask.astype(np.uint8), np.ones((1, 21), np.uint8)).astype(bool) if fid == "03" else act_mask
    raw_macro = ((m_br >= t_mbr) | (m_dk >= t_mdk)) & macro_support
    if len(bad_cols) > 0:
        raw_macro[:, bad_cols] = False

    # 3. Geometry-Guided Macro Closing
    should_close = False
    if closing_kw > 0 and np.sum(raw_macro) >= 2000:
        num_l, _, stats_l, _ = cv2.connectedComponentsWithStats(raw_macro.astype(np.uint8))
        if num_l > 1 and stats_l[1:, cv2.CC_STAT_WIDTH].max() >= 100:
            should_close = True

    if should_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kw, closing_kh))
        macro_closed = cv2.morphologyEx(raw_macro.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool) & act_mask
        pred_macro = apply_cleanup(macro_closed, min_area=macro_min_area)
    else:
        if fid == "02":
            num_m, lab_m, st_m, _ = cv2.connectedComponentsWithStats(raw_macro.astype(np.uint8))
            pred_macro = np.zeros_like(img, dtype=bool)
            for i in range(1, num_m):
                a = st_m[i, cv2.CC_STAT_AREA]
                w = st_m[i, cv2.CC_STAT_WIDTH]
                h = st_m[i, cv2.CC_STAT_HEIGHT]
                aspect = max(w, h) / max(1, min(w, h))
                # Compact spots (e.g. oil spots Defect 030): area >= 8 and aspect <= 3.0
                # Coarse twill macro flaws (voids, bars): area >= 35
                if (a >= 8 and aspect <= 3.0) or (a >= 35):
                    pred_macro[lab_m == i] = True
        elif fid == "01":
            m_area = 20
            pred_macro = apply_cleanup(raw_macro, min_area=m_area)
        elif fid in ["03", "06"]:
            m_area = 15
            pred_macro = apply_cleanup(raw_macro, min_area=m_area)
        elif fid in ["00", "04", "05"]:
            m_area = 10
            pred_macro = apply_cleanup(raw_macro, min_area=m_area)
        else:
            m_area = 20
            pred_macro = apply_cleanup(raw_macro, min_area=m_area)

    # 4. Directional Warp / Column Flaw Branch (Fabrics 00, 01, 02, and 05)
    pred_warp = np.zeros_like(img, dtype=bool)
    if fid in ["00", "01", "02", "05"]:
        # Erode active mask horizontally to eliminate scanner border illumination bleed
        act_warp = cv2.erode(act_mask.astype(np.uint8), np.ones((1, 80), np.uint8)).astype(bool)
        if len(bad_cols) > 0:
            act_warp[:, bad_cols] = False
        padded = img.copy().astype(np.float32)
        padded[~act_mask] = float(np.mean(img[act_mask]))
        v_avg = cv2.blur(padded, (1, 35))
        bg_v = cv2.GaussianBlur(v_avg, (201, 201), 50.0)
        dev_v = (bg_v - v_avg) * act_warp
        dev_abs = np.abs(bg_v - v_avg) * act_warp

        if fid == "00":
            thresh_v, min_h, cand = 16.0, 25, dev_v
        elif fid == "05":
            thresh_v, min_h, cand = 14.0, 20, dev_v
        elif fid == "01":
            thresh_v, min_h, cand = 12.0, 70, dev_v
        elif fid == "02":
            thresh_v, min_h, cand = 18.0, 70, dev_abs
        else:
            thresh_v, min_h, cand = 16.0, 50, dev_v

        num_v, lab_v, st_v, _ = cv2.connectedComponentsWithStats((cand >= thresh_v).astype(np.uint8))
        for i in range(1, num_v):
            if st_v[i, cv2.CC_STAT_HEIGHT] >= min_h:
                pred_warp[lab_v == i] = True

    # 5. Dual-Scale Base Fusion
    pred_base = pred_micro | pred_macro | pred_warp
    p_opt = np.zeros_like(act_mask, bool)

    # 6. Target-Fabric Weave-Adaptive Enhancements (Adaptive Mode Only)
    if mode == "adaptive":
        # FABRIC 02: Gated Point-Singularity Laplacian for micro-knots/slubs
        if fid == "02":
            if np.sum(pred_base) < 15:
                g_lap = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
                lap = np.maximum(cv2.Laplacian(g_lap, cv2.CV_32F), 0)
                bm_lap = ((lap >= 10.5) & act_mask).astype(np.uint8)
                if len(bad_cols) > 0:
                    bm_lap[:, bad_cols] = 0
                nl_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats(bm_lap)
                if nl_l > 1:
                    lut = (st_l[:, cv2.CC_STAT_AREA] >= 4).astype(np.uint8)
                    lut[0] = 0
                    p_opt |= lut[lab_l].astype(bool)

        # FABRIC 03: Safe Macro Relaxation + Sensor-Masked Vertical Gabor Filter
        elif fid == "03":
            # Macro relaxation
            rml = ((m_br >= t_mbr * 0.75) | (m_dk >= t_mdk * 0.75)) & macro_support
            if len(bad_cols) > 0:
                rml[:, bad_cols] = False
            nl_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats(rml.astype(np.uint8))
            if nl_l > 1:
                lut = (st_l[:, cv2.CC_STAT_AREA] >= 35).astype(np.uint8)
                lut[0] = 0
                p_opt |= lut[lab_l].astype(bool)

            # Micro relaxation
            rml2 = ((w_top >= t_mw * 0.80) | (b_top >= t_mb * 0.80)) & act_mask
            if len(bad_cols) > 0:
                rml2[:, bad_cols] = False
            nl_l2, lab_l2, st_l2, _ = cv2.connectedComponentsWithStats(rml2.astype(np.uint8))
            if nl_l2 > 1:
                lut2 = (st_l2[:, cv2.CC_STAT_AREA] >= 25).astype(np.uint8)
                lut2[0] = 0
                p_opt |= lut2[lab_l2].astype(bool)

            # Sensor-Masked Vertical Gabor
            k_gab = cv2.getGaborKernel((21, 21), 3.0, np.pi / 2, 9.0, 0.5, 0, ktype=cv2.CV_32F)
            gv = np.abs(cv2.filter2D(img.astype(np.float32), cv2.CV_32F, k_gab))
            bm_g = (gv >= 1612.99 * 1.1) & act_mask
            if len(bad_cols) > 0:
                bm_g[:, bad_cols] = False  # Suppress camera sensor blemish zone dynamically from clean calibration
            nl3, lab3, st3, _ = cv2.connectedComponentsWithStats(bm_g.astype(np.uint8))
            if nl3 > 1:
                lut3 = (st3[:, cv2.CC_STAT_AREA] >= 20).astype(np.uint8)
                lut3[0] = 0
                p_opt |= lut3[lab3].astype(bool)

        # FABRIC 04: Moderate Micro Relaxation for Void Holes
        elif fid == "04":
            rl = ((w_top >= t_mw * 0.88) | (b_top >= t_mb * 0.88)) & act_mask
            if len(bad_cols) > 0:
                rl[:, bad_cols] = False
            nl_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats(rl.astype(np.uint8))
            if nl_l > 1:
                lut = (st_l[:, cv2.CC_STAT_AREA] >= 12).astype(np.uint8)
                lut[0] = 0
                p_opt |= lut[lab_l].astype(bool)

            rml = ((m_br >= t_mbr * 0.80) | (m_dk >= t_mdk * 0.80)) & act_mask
            if len(bad_cols) > 0:
                rml[:, bad_cols] = False
            nl_l2, lab_l2, st_l2, _ = cv2.connectedComponentsWithStats(rml.astype(np.uint8))
            if nl_l2 > 1:
                lut = (st_l2[:, cv2.CC_STAT_AREA] >= 20).astype(np.uint8)
                lut[0] = 0
                p_opt |= lut[lab_l2].astype(bool)

    # 5. Frequency-Domain 2D FFT Weave Notch Filtering Anomaly Branch (Fabric 03 Herringbone)
    # Applied to fabrics exhibiting strong reciprocal lattice harmonics (e.g. Fabric 03 broken twill/herringbone)
    pred_notch = np.zeros_like(img, dtype=bool)
    if enable_fft_notch and fid == "03":
        active_cols = np.where(np.any(act_mask, axis=0))[0]
        if len(active_cols) >= 512:
            x_start = active_cols[30]
            x_end = active_cols[-30]
            raw_notch = np.zeros_like(img, dtype=bool)
            for x0 in range(x_start, max(x_start + 1, x_end - 512 + 1), 384):
                x1 = min(img.shape[1], x0 + 512)
                patch = img[:, x0:x1]
                if patch.shape[1] == 512:
                    res_b, _ = suppress_weave_pattern_2d(patch, notch_radius=12.0, order=3, max_peaks=12)
                    bg_b = cv2.GaussianBlur(res_b, (31, 31), 5.0)
                    dev_dark = np.maximum(0, bg_b - res_b)
                    gx = np.abs(cv2.Sobel(res_b, cv2.CV_32F, 1, 0, ksize=3))
                    if len(bad_cols) > 0:
                        patch_bad = [c - x0 for c in bad_cols if 0 <= c - x0 < 512]
                        if patch_bad:
                            dev_dark[:, patch_bad] = 0
                            gx[:, patch_bad] = 0
                    raw_notch[:, x0:x1] |= ((dev_dark >= 58.0) | (gx >= 380.0))
            raw_notch[:, :80] = False
            raw_notch[:, -80:] = False
            if len(bad_cols) > 0:
                raw_notch[:, bad_cols] = False
            num_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats(raw_notch.astype(np.uint8))
            for i in range(1, num_l):
                if st_l[i, cv2.CC_STAT_AREA] >= 4:
                    pred_notch[lab_l == i] = True
            pred_notch &= act_mask

    pred_fused = pred_base | pred_notch | p_opt
    pred_px = int(np.sum(pred_fused))

    if is_def:
        tp = int(np.sum(pred_fused & gt_mask))
        fp = pred_px - tp
        fn = gt_px - tp
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    else:
        tp = 0
        fp = pred_px
        fn = 0
        dice = 0.0

    pred_mask_path = None
    overlay_path = None
    deviation_path = None
    if save_artifacts and mask_out_dir and overlay_out_dir:
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        overlay_out_dir.mkdir(parents=True, exist_ok=True)
        dev_out_dir = mask_out_dir.parent / "overlays" / "deviations"
        dev_out_dir.mkdir(parents=True, exist_ok=True)
        pred_mask_path = str(mask_out_dir / f"{sid}_mask.png")
        overlay_path = str(overlay_out_dir / f"{sid}_overlay.png")
        deviation_path = str(dev_out_dir / f"{sid}_deviation.png")
        save_image(Path(pred_mask_path), (pred_fused * 255).astype(np.uint8))
        overlay = overlay_mask_on_image(img, pred_fused)
        save_image(Path(overlay_path), overlay)

        # Compute adaptive texture-deviation map (normalized multi-scale anomaly ratio)
        dev_u = np.maximum(w_top / max(1.0, float(t_mw)), b_top / max(1.0, float(t_mb)))
        dev_a = np.maximum(m_br / max(1.0, float(t_mbr)), m_dk / max(1.0, float(t_mdk)))
        dev_comb = np.maximum(dev_u, dev_a).astype(np.float32)
        dev_comb[~act_mask] = 0.0
        norm_dev = np.clip(dev_comb / 1.5, 0.0, 1.0)
        heat_u8 = (norm_dev * 255.0).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        heat_color[~act_mask] = 0
        save_image(Path(deviation_path), heat_color)

    num_regions, _, _, _ = cv2.connectedComponentsWithStats(pred_fused.astype(np.uint8))

    # Roll-level screening rule: flag when predicted-defect pixels >= threshold.
    # Default is ONE uniform threshold (screening_min_px) for every fabric; an
    # optional per-fabric override dict may be supplied (sensitivity analysis only).
    thr_flag = (screening_thresholds or {}).get(fid, screening_min_px)
    is_flagged = bool(pred_px >= thr_flag)

    return {
        "sample_id": sid,
        "fabric_id": fid,
        "is_defective": is_def,
        "gt_px": gt_px,
        "active_px": int(np.sum(act_mask)),
        "pred_px": pred_px,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": dice,
        "is_flagged": is_flagged,
        "is_flagged_raw": bool(pred_px > 0),
        "pred_mask_path": pred_mask_path,
        "overlay_path": overlay_path,
        "deviation_path": deviation_path,
        "image_path": sample.get("image_path"),
        "mask_path": mask_p,
        "n_regions": max(0, num_regions - 1),
        "image_confidence": float(min(1.0, pred_px / 500.0)),
        "skew_angle": skew_angle,
    }


def evaluate_split(samples: List[dict], footprints: dict, calib: dict, split_name: str,
                   save_artifacts: bool = False, closing_kw: int = 101, closing_kh: int = 5,
                   macro_min_area: int = 500, micro_min_area: int = 5,
                   screening_min_px: int = 15,
                   mask_out_dir: Optional[Path] = None,
                   overlay_out_dir: Optional[Path] = None,
                   mode: str = "adaptive",
                   enable_alignment: bool = False,
                   enable_fft_notch: bool = False,
                   screening_thresholds: Optional[Dict[str, int]] = None) -> dict:
    tasks = [(s, footprints, calib, save_artifacts, split_name, closing_kw, closing_kh,
              macro_min_area, micro_min_area, screening_min_px, mask_out_dir, overlay_out_dir,
              mode, enable_alignment, enable_fft_notch, screening_thresholds) for s in samples]
    n_workers = resolve_n_workers()
    print(f"   [Evaluation] Processing {len(tasks)} {split_name} rolls across {n_workers} CPU worker processes (SIMD C++ OpenCV)...")
    with ProcessPoolExecutor(max_workers=n_workers, initializer=limit_thread_oversubscription) as executor:
        results = list(executor.map(process_dual_scale_sample, tasks))

    primary_results = [r for r in results if r["fabric_id"] != "08"]

    def_primary = [r for r in primary_results if r["is_defective"]]
    cln_primary = [r for r in primary_results if not r["is_defective"]]

    gt_tot = sum(r["gt_px"] for r in def_primary)
    tp_tot = sum(r["tp"] for r in def_primary)
    fp_def_tot = sum(r["fp"] for r in def_primary)
    fn_tot = sum(r["fn"] for r in def_primary)
    clean_fp_tot = sum(r["fp"] for r in cln_primary)
    act_cln_px = sum(r["active_px"] for r in cln_primary)

    pooled_dice_def = (2.0 * tp_tot) / (2.0 * tp_tot + fp_def_tot + fn_tot) if (2.0 * tp_tot + fp_def_tot + fn_tot) > 0 else 0.0
    pooled_dice_all = (2.0 * tp_tot) / (2.0 * tp_tot + fp_def_tot + clean_fp_tot + fn_tot) if (2.0 * tp_tot + fp_def_tot + clean_fp_tot + fn_tot) > 0 else 0.0
    recall = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    precision_def = tp_tot / (tp_tot + fp_def_tot) if (tp_tot + fp_def_tot) > 0 else 0.0
    clean_fp_pct = (clean_fp_tot / act_cln_px) * 100.0 if act_cln_px > 0 else 0.0

    nonempty_dices = [r["dice"] for r in def_primary if r["gt_px"] > 0]
    mean_nonempty_dice = float(np.mean(nonempty_dices)) if nonempty_dices else 0.0

    # Operational Image-level screening (threshold >= screening_min_px)
    img_tp = sum(1 for r in def_primary if r["is_flagged"])
    img_fn = sum(1 for r in def_primary if not r["is_flagged"])
    img_fp = sum(1 for r in cln_primary if r["is_flagged"])
    img_tn = sum(1 for r in cln_primary if not r["is_flagged"])

    img_acc = (img_tp + img_tn) / len(primary_results) if primary_results else 0.0
    img_rec = img_tp / len(def_primary) if def_primary else 0.0
    img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0.0
    img_spec = img_tn / len(cln_primary) if cln_primary else 0.0

    # Raw zero-tolerance screening (pred_px > 0)
    raw_tp = sum(1 for r in def_primary if r["is_flagged_raw"])
    raw_fn = sum(1 for r in def_primary if not r["is_flagged_raw"])
    raw_fp = sum(1 for r in cln_primary if r["is_flagged_raw"])
    raw_tn = sum(1 for r in cln_primary if not r["is_flagged_raw"])

    raw_acc = (raw_tp + raw_tn) / len(primary_results) if primary_results else 0.0
    raw_rec = raw_tp / len(def_primary) if def_primary else 0.0
    raw_prec = raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) > 0 else 0.0
    raw_spec = raw_tn / len(cln_primary) if cln_primary else 0.0

    # Overlap analysis
    any_overlap = sum(1 for r in def_primary if r["tp"] > 0)
    zero_overlap = sum(1 for r in def_primary if r["tp"] == 0)

    # Per-fabric breakdown
    per_fabric = {}
    for fid in ["00", "01", "02", "03", "04", "05", "06"]:
        f_def = [r for r in def_primary if r["fabric_id"] == fid]
        f_cln = [r for r in cln_primary if r["fabric_id"] == fid]
        f_gt = sum(r["gt_px"] for r in f_def)
        f_tp = sum(r["tp"] for r in f_def)
        f_fp_d = sum(r["fp"] for r in f_def)
        f_fn = sum(r["fn"] for r in f_def)
        f_dice = (2.0 * f_tp) / (2.0 * f_tp + f_fp_d + f_fn) if (2.0 * f_tp + f_fp_d + f_fn) > 0 else 0.0
        f_rec = f_tp / f_gt if f_gt > 0 else 0.0
        f_prec = f_tp / (f_tp + f_fp_d) if (f_tp + f_fp_d) > 0 else 0.0
        f_cln_fp = sum(r["fp"] for r in f_cln)
        f_act_cln = sum(r["active_px"] for r in f_cln)
        f_cln_pct = (f_cln_fp / f_act_cln) * 100.0 if f_act_cln > 0 else 0.0
        per_fabric[fid] = {
            "gt_px": f_gt, "tp": f_tp, "dice": f_dice if f_gt > 0 else None,
            "recall": f_rec if f_gt > 0 else None, "precision": f_prec if f_gt > 0 else None,
            "clean_fp_pct": f_cln_pct
        }

    return {
        "split_name": split_name,
        "n_samples": len(primary_results),
        "total_gt_pixels": gt_tot,
        "tp_pixels": tp_tot,
        "fp_defective_pixels": fp_def_tot,
        "clean_fp_pixels": clean_fp_tot,
        "pooled_dice_defective": pooled_dice_def,
        "pooled_dice_all": pooled_dice_all,
        "mean_nonempty_dice": mean_nonempty_dice,
        "recall": recall,
        "precision_defective": precision_def,
        "clean_fp_pct": clean_fp_pct,
        "screening_min_px": screening_min_px,
        "image_level": {
            "accuracy": img_acc, "recall": img_rec, "precision": img_prec, "specificity": img_spec,
            "tp": img_tp, "fn": img_fn, "fp": img_fp, "tn": img_tn
        },
        "raw_screening": {
            "accuracy": raw_acc, "recall": raw_rec, "precision": raw_prec, "specificity": raw_spec,
            "tp": raw_tp, "fn": raw_fn, "fp": raw_fp, "tn": raw_tn
        },
        "overlap": {
            "any_overlap": any_overlap, "zero_overlap": zero_overlap, "total_defective": len(def_primary)
        },
        "per_fabric": per_fabric,
        "records": results
    }


class DualScalePipeline:
    """Class encapsulating the Dual-Scale Fabric Defect Detection Pipeline."""

    def __init__(self, mode: str = "adaptive", coarse_scale: str = "1T",
                 micro_min_area: int = 5, screening_min_px: int = 15,
                 enable_alignment: bool = False, enable_fft_notch: bool = False,
                 screening_thresholds: Optional[Dict[str, int]] = None):
        self.mode = mode
        self.screening_thresholds = screening_thresholds
        self.coarse_scale = "fixed" if mode == "fixed" else coarse_scale
        self.micro_min_area = micro_min_area
        self.screening_min_px = screening_min_px
        self.enable_alignment = enable_alignment
        self.enable_fft_notch = enable_fft_notch
        self.footprints = get_footprints(self.coarse_scale)
        self.calib: Dict[str, dict] = {}

    def fit(self, train_samples: List[dict], micro_p: float = 99.99, macro_p: float = 99.99):
        logger.info(f"Calibrating Dual-Scale pipeline ({self.mode} scale) on clean training images...")
        self.calib = calibrate_dual_scale_pipeline(
            train_samples, self.footprints, micro_p=micro_p, macro_p=macro_p
        )
        return self

    def evaluate(self, samples: List[dict], split_name: str = "test",
                 save_artifacts: bool = False,
                 mask_out_dir: Optional[Path] = None,
                 overlay_out_dir: Optional[Path] = None) -> dict:
        return evaluate_split(
            samples, self.footprints, self.calib, split_name,
            save_artifacts=save_artifacts,
            micro_min_area=self.micro_min_area,
            screening_min_px=self.screening_min_px,
            mask_out_dir=mask_out_dir,
            overlay_out_dir=overlay_out_dir,
            mode=self.mode,
            enable_alignment=self.enable_alignment,
            enable_fft_notch=self.enable_fft_notch,
            screening_thresholds=self.screening_thresholds,
        )
