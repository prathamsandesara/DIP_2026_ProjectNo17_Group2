#!/usr/bin/env python
"""
Threshold Hyperparameter Sensitivity Analysis & 4-Panel Visualization.

Evaluates operational screening performance across 4 target thresholds:
tau in {10 px, 15 px, 20 px, 25 px}.

Outputs:
- outputs/tables/screening_threshold_sweep.csv
- outputs/visualizations/endsem/fig_threshold_hyperparameter_sweep.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SUPER = _ROOT.parent
if str(_SUPER) not in sys.path:
    sys.path.insert(0, str(_SUPER))

from src.data.splitter import load_split
from src.pipeline import DualScalePipeline
from src.utils.io import load_image_gray, load_mask_binary


def run_threshold_sweep(target_thresholds: List[int] = [10, 15, 20, 25]) -> List[Dict]:
    print("=" * 100)
    print("OPERATIONAL THRESHOLD SENSITIVITY SWEEP (10, 15, 20, 25 px)")
    print("=" * 100)

    train_samples = load_split(Path("data/splits_oneclass"), "train")
    test_samples = load_split(Path("data/splits_oneclass"), "test")

    # Initialize full pipeline with alignment and notch enabled
    pipeline = DualScalePipeline(mode="adaptive", enable_alignment=True, enable_fft_notch=True)
    pipeline.fit(train_samples)

    results = []
    print(f"\n{'Threshold':<12} | {'Accuracy (%)':<14} | {'Recall (%)':<12} | {'Specificity (%)':<16} | {'Precision (%)':<14} | {'Pooled Dice (%)':<16} | {'TP':<4} {'FN':<4} {'FP':<4} {'TN':<4}")
    print("-" * 105)

    for tau in target_thresholds:
        pipeline.screening_min_px = tau
        eval_res = pipeline.evaluate(test_samples, split_name="test")

        img_lvl = eval_res["image_level"]
        acc = img_lvl["accuracy"] * 100.0
        rec = img_lvl["recall"] * 100.0
        spec = img_lvl["specificity"] * 100.0
        prec = img_lvl["precision"] * 100.0
        dice = eval_res["pooled_dice_defective"] * 100.0
        cm = img_lvl

        print(f"{tau:<12} | {acc:<14.2f} | {rec:<12.2f} | {spec:<16.2f} | {prec:<14.2f} | {dice:<16.2f} | {cm['tp']:<4} {cm['fn']:<4} {cm['fp']:<4} {cm['tn']:<4}")

        results.append({
            "threshold_px": tau,
            "accuracy_pct": round(acc, 2),
            "recall_pct": round(rec, 2),
            "specificity_pct": round(spec, 2),
            "precision_pct": round(prec, 2),
            "pooled_dice_pct": round(dice, 2),
            "tp": cm["tp"],
            "fn": cm["fn"],
            "fp": cm["fp"],
            "tn": cm["tn"],
            "eval_res": eval_res,
        })

    return results


def export_csv(results: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        fieldnames = ["Threshold (px)", "Accuracy (%)", "Recall (%)", "Specificity (%)", "Precision (%)", "Pooled Dice (%)", "TP", "FN", "FP", "TN"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Threshold (px)": r["threshold_px"],
                "Accuracy (%)": r["accuracy_pct"],
                "Recall (%)": r["recall_pct"],
                "Specificity (%)": r["specificity_pct"],
                "Precision (%)": r["precision_pct"],
                "Pooled Dice (%)": r["pooled_dice_pct"],
                "TP": r["tp"],
                "FN": r["fn"],
                "FP": r["fp"],
                "TN": r["tn"],
            })
    print(f"\nSaved threshold sensitivity table to: {out_path}")


def generate_visual_panels(results: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    thresholds = [r["threshold_px"] for r in results]
    accuracies = [r["accuracy_pct"] for r in results]
    recalls = [r["recall_pct"] for r in results]
    specificities = [r["specificity_pct"] for r in results]
    precisions = [r["precision_pct"] for r in results]

    fig = plt.figure(figsize=(16, 9), facecolor="#0F172A")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # 1. Curve: Sensitivity vs Specificity Operating Plateau
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1E293B")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#64748B")

    ax1.plot(thresholds, recalls, marker="o", linewidth=2.5, markersize=8, color="#38BDF8", label="Defect Sensitivity (Recall %)")
    ax1.plot(thresholds, specificities, marker="s", linewidth=2.5, markersize=8, color="#10B981", label="Clean Specificity (%)")
    ax1.plot(thresholds, accuracies, marker="^", linewidth=2.5, markersize=8, color="#F59E0B", label="Overall Screening Accuracy (%)")

    ax1.axvline(15, color="#EF4444", linestyle=":", linewidth=2.0, alpha=0.9, label="Optimal Baseline (15 px)")
    ax1.set_title("Operating Characteristic vs. Area Threshold", color="white", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Screening Minimum Defect Area (pixels)", color="#E2E8F0", fontsize=11)
    ax1.set_ylabel("Metric Rate (%)", color="#E2E8F0", fontsize=11)
    ax1.set_xticks(thresholds)
    ax1.tick_params(colors="#CBD5E1", labelsize=10)
    ax1.legend(loc="lower left", facecolor="#1E293B", edgecolor="#475569", labelcolor="white", fontsize=9.5)

    # 2. Confusion Breakdown Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1E293B")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#64748B", axis="y")

    x = np.arange(len(thresholds))
    width = 0.2
    tps = [r["tp"] for r in results]
    fns = [r["fn"] for r in results]
    fps = [r["fp"] for r in results]
    tns = [r["tn"] for r in results]

    ax2.bar(x - 1.5 * width, tps, width, label="True Positives (TP)", color="#10B981")
    ax2.bar(x - 0.5 * width, tns, width, label="True Negatives (TN)", color="#38BDF8")
    ax2.bar(x + 0.5 * width, fns, width, label="False Negatives (FN)", color="#EF4444")
    ax2.bar(x + 1.5 * width, fps, width, label="False Positives (FP)", color="#F59E0B")

    ax2.set_title("Roll Classification Counts (N = 81 rolls)", color="white", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Threshold (pixels)", color="#E2E8F0", fontsize=11)
    ax2.set_ylabel("Number of Production Rolls", color="#E2E8F0", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{t} px" for t in thresholds])
    ax2.tick_params(colors="#CBD5E1", labelsize=10)
    ax2.legend(loc="upper right", facecolor="#1E293B", edgecolor="#475569", labelcolor="white", fontsize=9.5)

    # 3. Precision-Recall-F1 Balance
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1E293B")
    ax3.grid(True, linestyle="--", alpha=0.3, color="#64748B")

    f1_scores = [2 * (p * r) / (p + r + 1e-5) for p, r in zip(precisions, recalls)]
    ax3.plot(thresholds, precisions, marker="D", linewidth=2.2, color="#A855F7", label="Screening Precision (%)")
    ax3.plot(thresholds, f1_scores, marker="*", linewidth=2.5, markersize=10, color="#EC4899", label="F1-Score (%)")

    ax3.set_title("Precision and F1-Score Plateau Stability", color="white", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Screening Minimum Defect Area (pixels)", color="#E2E8F0", fontsize=11)
    ax3.set_ylabel("Percentage (%)", color="#E2E8F0", fontsize=11)
    ax3.set_xticks(thresholds)
    ax3.tick_params(colors="#CBD5E1", labelsize=10)
    ax3.legend(loc="lower left", facecolor="#1E293B", edgecolor="#475569", labelcolor="white", fontsize=9.5)

    # 4. Summary Table Panel
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1E293B")
    ax4.axis("off")

    table_data = [
        ["Threshold", "Accuracy", "Recall", "Specificity", "F1-Score", "TP / FN", "FP / TN"],
    ]
    for r, f1 in zip(results, f1_scores):
        table_data.append([
            f"{r['threshold_px']} px",
            f"{r['accuracy_pct']:.1f}%",
            f"{r['recall_pct']:.1f}%",
            f"{r['specificity_pct']:.1f}%",
            f"{f1:.1f}%",
            f"{r['tp']} / {r['fn']}",
            f"{r['fp']} / {r['tn']}",
        ])

    table = ax4.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.16, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        if row_idx == 0:
            cell.set_facecolor("#334155")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if thresholds[row_idx - 1] == 15:
                cell.set_facecolor("#1E3A8A")  # Highlight optimal 15px
                cell.set_text_props(color="#67E8F9", fontweight="bold")
            else:
                cell.set_facecolor("#1E293B")
                cell.set_text_props(color="#E2E8F0")

    ax4.set_title("Operational Operating Points Comparison", color="white", fontsize=13, fontweight="bold", pad=12)

    plt.suptitle("Hyperparameter Sensitivity: Screening Area Threshold (tau in {10, 15, 20, 25} px)",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved 4-panel threshold figure to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/visualizations/endsem")
    parser.add_argument("--table-out", default="outputs/tables/screening_threshold_sweep.csv")
    args = parser.parse_args()

    results = run_threshold_sweep(target_thresholds=[10, 15, 20, 25])
    export_csv(results, Path(args.table_out))
    generate_visual_panels(results, Path(args.out_dir) / "fig_threshold_hyperparameter_sweep.png")


if __name__ == "__main__":
    main()

