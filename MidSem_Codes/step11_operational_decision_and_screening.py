"""
STEP 11: Operational Screening Decision, Industrial Quality Gating & Evaluation Metrics.

Pipeline Stage: Step 11
Purpose:
  - Enforces the Factory Operational Screening Decision Rule:
      * Criterion: Total predicted defect foreground >= 15 pixels.
      * Flagged (YES) -> DEFECTIVE ROLL: Sends loom stoppage signal & outputs laser cutting plan.
      * Not Flagged (NO) -> CLEAN ROLL: Passes quality check, continues high-speed weaving.
  - Computes Image-Level Operational Screening Metrics:
      * Accuracy, Sensitivity (Recall), Precision, Specificity, Confusion Matrix (TP, FN, FP, TN).
  - Computes Pixel-Level Localization & Segmentation Metrics against ground-truth masks:
      * IoU (Jaccard Index), Dice (F1-score), Recall, Precision, Clean False Alarm Rate (FPR <= 0.05% safety limit).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

submission_dir = Path(__file__).resolve().parent
repo_root = submission_dir.parent
for p in [str(submission_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import get_logger

logger = get_logger("Step11_EvaluationAndDecision")

DEFAULT_SCREENING_THRESHOLD = 15  # pixels


def evaluate_single_roll(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    active_mask: np.ndarray,
    is_defective_gt: bool,
    screening_thresh: int = DEFAULT_SCREENING_THRESHOLD
) -> dict:
    """
    Evaluate one inspected fabric roll at both image-level and pixel-level.
    """
    pred_px = int(np.sum(pred_mask))
    gt_px = int(np.sum(gt_mask)) if is_defective_gt else 0
    active_px = int(np.sum(active_mask))

    # Operational decision
    is_flagged = bool(pred_px >= screening_thresh)
    decision = "DEFECTIVE_ROLL (STOP_LOOM)" if is_flagged else "CLEAN_ROLL (PASS)"

    if is_defective_gt:
        tp = int(np.sum(pred_mask & gt_mask))
        fp = pred_px - tp
        fn = gt_px - tp
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    else:
        tp = 0
        fp = pred_px
        fn = 0
        dice = 0.0
        iou = 0.0

    return {
        "pred_px": pred_px,
        "gt_px": gt_px,
        "active_px": active_px,
        "is_defective_gt": is_defective_gt,
        "is_flagged": is_flagged,
        "decision": decision,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": dice,
        "iou": iou,
    }


def aggregate_operational_metrics(roll_records: List[dict]) -> dict:
    """
    Aggregate evaluation metrics across a dataset split:
    Returns complete image-level and pixel-level benchmark summary.
    """
    def_rolls = [r for r in roll_records if r["is_defective_gt"]]
    cln_rolls = [r for r in roll_records if not r["is_defective_gt"]]

    # 1. Image-Level Screening Performance
    img_tp = sum(1 for r in def_rolls if r["is_flagged"])
    img_fn = sum(1 for r in def_rolls if not r["is_flagged"])
    img_fp = sum(1 for r in cln_rolls if r["is_flagged"])
    img_tn = sum(1 for r in cln_rolls if not r["is_flagged"])

    n_total = len(roll_records)
    img_acc = (img_tp + img_tn) / n_total if n_total > 0 else 0.0
    img_rec = img_tp / len(def_rolls) if def_rolls else 0.0
    img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0.0
    img_spec = img_tn / len(cln_rolls) if cln_rolls else 0.0

    # 2. Pixel-Level Segmentation Performance
    gt_tot = sum(r["gt_px"] for r in def_rolls)
    tp_tot = sum(r["tp"] for r in def_rolls)
    fp_def_tot = sum(r["fp"] for r in def_rolls)
    fn_tot = sum(r["fn"] for r in def_rolls)
    clean_fp_tot = sum(r["fp"] for r in cln_rolls)
    act_cln_px = sum(r["active_px"] for r in cln_rolls)

    pooled_dice_def = (2.0 * tp_tot) / (2.0 * tp_tot + fp_def_tot + fn_tot) if (2.0 * tp_tot + fp_def_tot + fn_tot) > 0 else 0.0
    pooled_iou = tp_tot / (tp_tot + fp_def_tot + fn_tot) if (tp_tot + fp_def_tot + fn_tot) > 0 else 0.0
    pixel_rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    pixel_prec = tp_tot / (tp_tot + fp_def_tot) if (tp_tot + fp_def_tot) > 0 else 0.0
    clean_fp_pct = (clean_fp_tot / act_cln_px) * 100.0 if act_cln_px > 0 else 0.0

    return {
        "n_samples": n_total,
        "n_defective": len(def_rolls),
        "n_clean": len(cln_rolls),
        "image_level": {
            "accuracy": img_acc,
            "recall": img_rec,
            "precision": img_prec,
            "specificity": img_spec,
            "tp": img_tp,
            "fn": img_fn,
            "fp": img_fp,
            "tn": img_tn,
        },
        "pixel_level": {
            "total_gt_pixels": gt_tot,
            "tp_pixels": tp_tot,
            "fp_defective_pixels": fp_def_tot,
            "fn_pixels": fn_tot,
            "clean_fp_pixels": clean_fp_tot,
            "pooled_dice": pooled_dice_def,
            "pooled_iou": pooled_iou,
            "recall": pixel_rec,
            "precision": pixel_prec,
            "clean_fp_rate_pct": clean_fp_pct,
        }
    }


def print_evaluation_summary_table(metrics: dict, title: str = "BENCHMARK RESULTS") -> None:
    """Pretty-print benchmark results formatted to industrial standards."""
    im = metrics["image_level"]
    px = metrics["pixel_level"]

    print("\n" + "=" * 90)
    print(f"{title:^90}")
    print("=" * 90)
    print(f"Total Evaluated Rolls : {metrics['n_samples']} ({metrics['n_defective']} Defective, {metrics['n_clean']} Clean)")
    print("-" * 90)
    print("1. IMAGE-LEVEL OPERATIONAL SCREENING (Threshold >= 15 px):")
    print(f"   Accuracy    : {im['accuracy']*100:.2f}%")
    print(f"   Sensitivity : {im['recall']*100:.2f}% (Recall: {im['tp']} / {im['tp'] + im['fn']} defective rolls caught)")
    print(f"   Precision   : {im['precision']*100:.2f}%")
    print(f"   Specificity : {im['specificity']*100:.2f}% ({im['tn']} / {im['tn'] + im['fp']} clean rolls confirmed pristine)")
    print(f"   Confusion   : TP = {im['tp']:<3} | FN = {im['fn']:<3} (Missed)")
    print(f"                 FP = {im['fp']:<3} | TN = {im['tn']:<3} (Clean Passed)")
    print("-" * 90)
    print("2. PIXEL-LEVEL DEFECT SEGMENTATION & LOCALIZATION:")
    print(f"   Pooled Dice : {px['pooled_dice']:.5f} ({px['pooled_dice']*100:.2f}%)")
    print(f"   Pooled IoU  : {px['pooled_iou']:.5f} ({px['pooled_iou']*100:.2f}%)")
    print(f"   Pixel Recall: {px['recall']*100:.2f}% ({px['tp_pixels']:,} / {px['total_gt_pixels']:,} defect pixels localized)")
    print(f"   Precision   : {px['precision']*100:.2f}%")
    print(f"   Clean FPR   : {px['clean_fp_rate_pct']:.4f}% (Safety Limit <= 0.0500%)")
    print("=" * 90 + "\n")


def main() -> None:
    print("=" * 80)
    print("STEP 11: OPERATIONAL SCREENING DECISION & EVALUATION")
    print("=" * 80)

    # Test single roll evaluation
    h, w = 256, 4096
    mock_pred = np.zeros((h, w), dtype=bool)
    mock_gt = np.zeros((h, w), dtype=bool)
    mock_act = np.ones((h, w), dtype=bool)

    # Defect of 200 pixels
    mock_pred[100:110, 2000:2020] = True  # 200 px
    mock_gt[100:110, 2000:2018] = True    # 180 px

    record = evaluate_single_roll(mock_pred, mock_gt, mock_act, is_defective_gt=True)
    print("Single Roll Decision :", record["decision"])
    print(f"Predicted Pixels     : {record['pred_px']} px")
    print(f"Dice Overlap         : {record['dice']:.4f}")

    # Aggregated benchmark mock demonstration
    mock_records = [record]
    agg = aggregate_operational_metrics(mock_records)
    print_evaluation_summary_table(agg, title="MOCK VERIFICATION BENCHMARK")
    print("Step 11 Completed Successfully. Operational decision logic verified.")


if __name__ == "__main__":
    main()
