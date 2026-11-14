"""
Fair Classical Thresholding Baselines Benchmark
ECE501 Digital Image Processing - End-Semester Project

Evaluates classical baseline methods on all 81 held-out test rolls inside the active fabric mask:
1. Naive Global Otsu (Raw Image including conveyor borders)
2. Fair Global Otsu (Masked strictly inside Active Fabric)
3. Gaussian Residual (Calibrated on 85 clean training rolls at 99.9th percentile)
4. Gaussian Residual (Calibrated on 85 clean training rolls at 99.99th percentile)
5. Single-Scale Morphological Top-Hat (Calibrated on 85 clean training rolls at 99.9th percentile)
6. Single-Scale Morphological Top-Hat (Calibrated on 85 clean training rolls at 99.99th percentile)

Exports:
- outputs/metrics/classical_baselines_benchmark.json
- outputs/tables/classical_baselines_table.csv
"""

import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SUPER = _ROOT.parent
if str(_SUPER) not in sys.path:
    sys.path.insert(0, str(_SUPER))

from src.data.splitter import load_split
from src.utils.io import load_image_gray, load_mask_binary, save_json
from src.data.active_fabric_mask import detect_active_fabric_mask


def calibrate_thresholds_on_clean_train(train_samples: List[dict]) -> Dict[str, float]:
    """Calibrate baseline thresholds on the 85 clean training rolls (leakage-free)."""
    print(f"Calibrating baseline thresholds across {len(train_samples)} clean training rolls...")
    gauss_residuals = []
    tophat_residuals = []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    for s in train_samples:
        img = load_image_gray(s["image_path"])
        fid = s["fabric_id"]
        act_mask = detect_active_fabric_mask(img, fabric_id=fid)
        if not np.any(act_mask):
            continue

        # Gaussian background subtraction
        bg = cv2.GaussianBlur(img.astype(np.float32), (51, 51), 15.0)
        res_g = np.abs(img.astype(np.float32) - bg)[act_mask]
        # Subsample to keep memory bounded
        if len(res_g) > 20000:
            step = len(res_g) // 10000
            gauss_residuals.extend(res_g[::step])
        else:
            gauss_residuals.extend(res_g)

        # Single-scale top-hat + black-hat morphology
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        w_top = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, k)
        b_top = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, k)
        res_th = np.maximum(w_top, b_top)[act_mask]
        if len(res_th) > 20000:
            step = len(res_th) // 10000
            tophat_residuals.extend(res_th[::step])
        else:
            tophat_residuals.extend(res_th)

    gauss_arr = np.array(gauss_residuals, dtype=np.float32)
    tophat_arr = np.array(tophat_residuals, dtype=np.float32)

    calib = {
        "gauss_p99_9": float(np.percentile(gauss_arr, 99.9)),
        "gauss_p99_99": float(np.percentile(gauss_arr, 99.99)),
        "tophat_p99_9": float(np.percentile(tophat_arr, 99.9)),
        "tophat_p99_99": float(np.percentile(tophat_arr, 99.99)),
    }
    print(f"  -> Learned Gaussian thresholds: 99.9% = {calib['gauss_p99_9']:.2f}, 99.99% = {calib['gauss_p99_99']:.2f}")
    print(f"  -> Learned Top-Hat thresholds:  99.9% = {calib['tophat_p99_9']:.2f}, 99.99% = {calib['tophat_p99_99']:.2f}")
    return calib


def evaluate_baselines():
    splits_dir = Path("data/splits_oneclass")
    train_samples = load_split(splits_dir, "train")
    test_samples = load_split(splits_dir, "test")
    total_rolls = len(test_samples)
    screening_min_px = 15  # same uniform roll-screening threshold as the main pipeline (DualScalePipeline default)

    calib = calibrate_thresholds_on_clean_train(train_samples)

    print(f"\nEvaluating Classical Baselines on all {total_rolls} held-out test rolls...")

    baselines = [
        {"name": "1. Naive Global Otsu (Raw Strip)", "type": "otsu_raw", "tau": None},
        {"name": "2. Fair Global Otsu (Active Mask)", "type": "otsu_masked", "tau": None},
        {"name": "3. Gaussian Residual (Clean 99.9%ile)", "type": "gauss", "tau": calib["gauss_p99_9"]},
        {"name": "4. Gaussian Residual (Clean 99.99%ile)", "type": "gauss", "tau": calib["gauss_p99_99"]},
        {"name": "5. Top-Hat Morphology (Clean 99.9%ile)", "type": "tophat", "tau": calib["tophat_p99_9"]},
        {"name": "6. Top-Hat Morphology (Clean 99.99%ile)", "type": "tophat", "tau": calib["tophat_p99_99"]},
    ]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    results = []

    for bl in baselines:
        b_type = bl["type"]
        tau = bl["tau"]
        tp, fn, fp, tn = 0, 0, 0, 0
        tp_px_tot, fp_px_tot, fn_px_tot = 0, 0, 0
        dice_defective_list = []
        t0 = time.perf_counter()

        for s in test_samples:
            img = load_image_gray(s["image_path"])
            fid = s["fabric_id"]
            is_def = s["is_defective"]
            gt = load_mask_binary(s["mask_path"]).astype(bool) if is_def and s.get("mask_path") else np.zeros_like(img, bool)
            act_mask = detect_active_fabric_mask(img, fabric_id=fid)

            pred = np.zeros_like(img, dtype=bool)
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)

            if b_type == "otsu_raw":
                # Naive Otsu across the entire unmasked strip
                _, th = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                pred = th.astype(bool)

            elif b_type == "otsu_masked":
                # Otsu computed strictly over active fabric pixels
                active_px = img_u8[act_mask]
                if len(active_px) > 0:
                    val_th, _ = cv2.threshold(active_px, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    pred = (img_u8 < val_th) & act_mask

            elif b_type == "gauss":
                # Background subtraction via Gaussian blur with threshold calibrated on clean train
                bg = cv2.GaussianBlur(img.astype(np.float32), (51, 51), 15.0)
                res = np.abs(img.astype(np.float32) - bg)
                pred = (res > tau) & act_mask

            elif b_type == "tophat":
                # Single-scale morphological top-hat with threshold calibrated on clean train
                w_top = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, k)
                b_top = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, k)
                res = np.maximum(w_top, b_top)
                pred = (res > tau) & act_mask

            # Screening classification based on minimum area threshold
            flagged = int(np.sum(pred)) >= screening_min_px
            if is_def:
                if flagged:
                    tp += 1
                else:
                    fn += 1
                # Spatial Dice on defective rolls
                gt_sum = np.sum(gt)
                pr_sum = np.sum(pred)
                inter = np.sum(pred & gt)
                dice = (2.0 * inter) / (gt_sum + pr_sum) if (gt_sum + pr_sum) > 0 else 0.0
                dice_defective_list.append(dice)
                tp_px_tot += inter
                fp_px_tot += (pr_sum - inter)
                fn_px_tot += (gt_sum - inter)
            else:
                if flagged:
                    fp += 1
                else:
                    tn += 1

        elapsed = time.perf_counter() - t0
        acc = (tp + tn) / total_rolls * 100.0
        rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
        mean_dice = float(np.mean(dice_defective_list)) * 100.0 if dice_defective_list else 0.0
        pooled_dice = (2.0 * tp_px_tot) / (2.0 * tp_px_tot + fp_px_tot + fn_px_tot) * 100.0 if (2.0 * tp_px_tot + fp_px_tot + fn_px_tot) > 0 else 0.0

        results.append({
            "name": bl["name"],
            "accuracy_pct": round(acc, 2),
            "recall_pct": round(rec, 2),
            "precision_pct": round(prec, 2),
            "specificity_pct": round(spec, 2),
            "pooled_dice_pct": round(pooled_dice, 2),
            "mean_dice_pct": round(mean_dice, 2),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "runtime_s": round(elapsed, 2)
        })
        print(f"  {bl['name']:<42}: Acc={acc:>6.2f}%, Rec={rec:>6.2f}%, Prec={prec:>6.2f}%, Spec={spec:>6.2f}%, PooledDice={pooled_dice:>5.2f}%, MeanDice={mean_dice:>5.2f}% (FP={fp}, TN={tn})")

    # Export metrics JSON
    out_json = Path("outputs/metrics/classical_baselines_benchmark.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_json, {
        "total_test_rolls": total_rolls,
        "defective_rolls": 53,
        "clean_rolls": 28,
        "calibration": calib,
        "baselines": results
    })
    print(f"\nSaved baselines JSON to: {out_json}")

    # Export CSV table
    out_csv = Path("outputs/tables/classical_baselines_table.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Baseline Method", "Accuracy (%)", "Recall (%)", "Precision (%)", "Specificity (%)", "Pooled Dice (%)", "Mean Dice (%)", "TP", "FN", "FP", "TN", "Runtime (s)"])
        for r in results:
            writer.writerow([r["name"], r["accuracy_pct"], r["recall_pct"], r["precision_pct"], r["specificity_pct"], r["pooled_dice_pct"], r["mean_dice_pct"], r["tp"], r["fn"], r["fp"], r["tn"], r["runtime_s"]])
    print(f"Saved baselines CSV to: {out_csv}")


if __name__ == "__main__":
    evaluate_baselines()
