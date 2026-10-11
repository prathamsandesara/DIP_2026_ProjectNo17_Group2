"""
STEP 12: Unified Master End-to-End Multi-Branch Pipeline Orchestrator.

Pipeline Stage: Step 12 (Complete End-to-End Orchestrator)
Purpose:
  - Connects all pipeline modules into a unified, high-performance, parallel inspection engine.
  - Architecture:
      [Raw Fabric Scan (256 x 4096)]
                    │
                    ▼
      [Step 1: Preprocessing & Active Fabric Domain Masking]
                    │
                    ├──────────────────────┬──────────────────────┐
                    ▼                      ▼                      ▼
           [Branch 1: Micro]      [Branch 2: Macro]      [Branch 3: Warp]
           (Top-Hat at 1T & 0.5T) (Gaussian W=201, s=50) (1x35 Box Profiling)
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                                           ▼
                    [Step 8: Multimodal Feature Fusion & Weave Adaptations]
                                           │
                                           ▼
                    [Step 9: Morphological Cleanup & Void Bridge Closing]
                                           │
                                           ▼
                    [Step 10: Connected Component Localization & BBoxes]
                                           │
                                           ▼
                    [Step 11: Operational Screening Decision & Evaluation]
                    (Area >= 15 px? -> Defective Loom Stop vs Clean Pass)
  - Executes across all 81 held-out test rolls and prints complete benchmark results.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

submission_dir = Path(__file__).resolve().parent
repo_root = submission_dir.parent
for p in [str(submission_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from step00_dataset_setup import load_one_class_split
from step01_preprocessing import detect_active_fabric_mask
from step02_pitch_and_scale_estimation import get_adaptive_footprints, get_dual_scale_kernels
from step03_micro_branch import compute_dual_scale_micro, compute_iqr_contrast_scale
from step04_macro_branch import compute_macro_residuals
from step05_warp_branch import run_warp_branch
from step07_calibration_and_normal_model import calibrate_all_fabrics
from step08_multimodal_feature_fusion import compute_adaptive_enhancements, fuse_base_branches
from step09_morphological_cleanup import apply_macro_geometry_closing, filter_small_components
from step10_localization_and_bounding_boxes import extract_defect_regions
from step11_operational_decision_and_screening import (
    aggregate_operational_metrics,
    evaluate_single_roll,
    print_evaluation_summary_table,
)
from utils import (
    get_logger,
    limit_thread_oversubscription,
    load_image_gray,
    load_mask_binary,
    resolve_n_workers,
)

logger = get_logger("Step12_EndToEndPipeline")


def inspect_single_fabric_roll(args: Tuple) -> dict:
    """Worker function inspecting a single fabric roll across all pipeline branches."""
    sample, footprints, calib, mode, screening_min_px = args

    sid = sample["sample_id"]
    fid = sample["fabric_id"]
    is_def = sample["is_defective"]
    img_path = sample["image_path"]
    mask_path = sample.get("mask_path")

    # Load raw grayscale scan and ground-truth mask
    img = load_image_gray(img_path)
    gt_mask = load_mask_binary(mask_path) if (is_def and mask_path) else np.zeros_like(img, dtype=bool)

    # Step 1: Active Fabric Mask
    act_mask = detect_active_fabric_mask(img, fabric_id=fid)

    # Step 2: Footprints
    ksize_primary = footprints.get(fid, (15, 15))
    _, ksize_spot = get_dual_scale_kernels(ksize_primary)
    calib_entry = calib.get(fid, calib.get("00", {}))
    bad_cols = calib_entry.get("bad_cols", [])

    # Step 3: Micro Branch
    w_top, b_top, w_sm, b_sm = compute_dual_scale_micro(img, ksize_primary, ksize_spot)
    scale = compute_iqr_contrast_scale(img, act_mask, fid)

    t_mw = calib_entry.get("micro_white", 100.0) * scale
    t_mb = calib_entry.get("micro_black", 100.0) * scale
    t_mw_sm = calib_entry.get("micro_white_sm", t_mw) * scale
    t_mb_sm = calib_entry.get("micro_black_sm", t_mb) * scale

    raw_micro = (
        (w_top >= t_mw) | (b_top >= t_mb) | (w_sm >= t_mw_sm) | (b_sm >= t_mb_sm)
    ) & act_mask

    if len(bad_cols) > 0:
        raw_micro[:, bad_cols] = False

    micro_min_a = 20 if fid == "05" else 5
    pred_micro = filter_small_components(raw_micro, min_area=micro_min_a)

    # Step 4: Macro Branch
    diff_bright, diff_dark, _ = compute_macro_residuals(img, act_mask)
    t_mbr = calib_entry.get("macro_bright", 65.0)
    t_mdk = calib_entry.get("macro_dark", 45.0)

    macro_support = cv2.erode(act_mask.astype(np.uint8), np.ones((1, 21), np.uint8)).astype(bool) if fid == "03" else act_mask
    raw_macro = ((diff_bright >= t_mbr) | (diff_dark >= t_mdk)) & macro_support
    if len(bad_cols) > 0:
        raw_macro[:, bad_cols] = False

    # Step 9 (Macro sub-cleanup & geometry closing)
    should_close = False
    if np.sum(raw_macro) >= 2000:
        num_l, _, stats_l, _ = cv2.connectedComponentsWithStats(raw_macro.astype(np.uint8))
        if num_l > 1 and stats_l[1:, cv2.CC_STAT_WIDTH].max() >= 100:
            should_close = True

    if should_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 5))
        macro_closed = cv2.morphologyEx(raw_macro.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool) & act_mask
        pred_macro = filter_small_components(macro_closed, min_area=500)
    else:
        if fid == "02":
            num_m, lab_m, st_m, _ = cv2.connectedComponentsWithStats(raw_macro.astype(np.uint8))
            pred_macro = np.zeros_like(img, dtype=bool)
            for i in range(1, num_m):
                a = st_m[i, cv2.CC_STAT_AREA]
                w = st_m[i, cv2.CC_STAT_WIDTH]
                h = st_m[i, cv2.CC_STAT_HEIGHT]
                aspect = max(w, h) / max(1, min(w, h))
                if (a >= 8 and aspect <= 3.0) or (a >= 35):
                    pred_macro[lab_m == i] = True
        elif fid == "01":
            pred_macro = filter_small_components(raw_macro, min_area=20)
        elif fid in ["03", "06"]:
            pred_macro = filter_small_components(raw_macro, min_area=15)
        elif fid in ["00", "04", "05"]:
            pred_macro = filter_small_components(raw_macro, min_area=10)
        else:
            pred_macro = filter_small_components(raw_macro, min_area=20)

    # Step 5: Warp Branch
    pred_warp, _ = run_warp_branch(img, act_mask, fid, bad_cols)

    # Step 8: Multimodal Fusion & Adaptive Enhancements
    pred_base = fuse_base_branches(pred_micro, pred_macro, pred_warp)
    residuals_dict = {
        "diff_bright": diff_bright,
        "diff_dark": diff_dark,
        "w_top": w_top,
        "b_top": b_top,
    }

    if mode == "adaptive":
        p_opt = compute_adaptive_enhancements(
            img, act_mask, fid, pred_base, residuals_dict, calib_entry, bad_cols
        )
        pred_fused = pred_base | p_opt
    else:
        pred_fused = pred_base

    # Step 10: Localization
    regions = extract_defect_regions(pred_fused, min_area=5)

    # Step 11: Screening Decision & Metrics
    screening_thr_map = {"00": 10, "01": 5, "02": 10} if mode == "adaptive" else {}
    roll_thresh = screening_thr_map.get(fid, screening_min_px)

    eval_record = evaluate_single_roll(
        pred_fused, gt_mask, act_mask, is_defective_gt=is_def, screening_thresh=roll_thresh
    )
    eval_record.update({
        "sample_id": sid,
        "fabric_id": fid,
        "n_defect_regions": len(regions),
    })

    return eval_record


def run_pipeline(
    mode: str = "adaptive",
    screening_min_px: int = 15,
    coarse_scale: str = "1T"
) -> dict:
    """
    Run the complete Multi-Branch Pipeline from Step 0 to Final Results.
    """
    print("\n" + "=" * 90)
    print(f"STARTING MASTER MULTI-BRANCH INSPECTION PIPELINE (MODE: {mode.upper()})")
    print("=" * 90)

    # Step 0: Ingest Splits
    train_samples = load_one_class_split("train")
    test_samples = load_one_class_split("test")

    print(f"Ingested Train Split : {len(train_samples)} pristine rolls (One-Class Reference)")
    print(f"Ingested Test Split  : {len(test_samples)} unseen rolls ({sum(1 for s in test_samples if s['is_defective'])} Defective, {sum(1 for s in test_samples if not s['is_defective'])} Clean)")

    # Step 2: Adaptive Footprints
    footprints = get_adaptive_footprints(mode=mode, coarse_scale=coarse_scale)

    # Step 7: Calibration
    print("\nCalibrating P99.99 thresholds & sensor scratch columns on clean training data...")
    t0_calib = time.time()
    calib = calibrate_all_fabrics(train_samples, footprints)
    print(f"Calibration completed in {time.time() - t0_calib:.2f}s across {len(calib)} fabric types.")

    # Execute Parallel Inspection across Test Split
    print(f"\nInspecting {len(test_samples)} test rolls across parallel processing threads...")
    tasks = [(s, footprints, calib, mode, screening_min_px) for s in test_samples]

    n_workers = resolve_n_workers(12)
    t0_eval = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=limit_thread_oversubscription) as executor:
        test_results = list(executor.map(inspect_single_fabric_roll, tasks))
    print(f"Inspection completed in {time.time() - t0_eval:.2f}s ({len(test_results) / (time.time() - t0_eval):.1f} rolls/sec).")

    # Step 11: Aggregate & Display
    metrics = aggregate_operational_metrics(test_results)
    print_evaluation_summary_table(
        metrics,
        title=f"FINAL MULTI-BRANCH PIPELINE TEST RESULTS (MODE: {mode.upper()})"
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="adaptive", choices=["adaptive", "fixed"], help="Pipeline mode")
    parser.add_argument("--screening-min-px", type=int, default=15, help="Image-level screening threshold in pixels")
    parser.add_argument("--coarse-scale", default="1T", choices=["1T", "2T"], help="Scale for coarse twill")
    args = parser.parse_args()

    run_pipeline(mode=args.mode, screening_min_px=args.screening_min_px, coarse_scale=args.coarse_scale)


if __name__ == "__main__":
    main()
