#!/usr/bin/env python
"""
Phase 3 Integration Benchmark & Multi-Stage Ablation Verification.

ECE501 Digital Image Processing - End-Semester Project
Evaluates the Dual-Scale Anomaly Segmentation Pipeline across 4 systematic configurations:
1. Baseline: dual-scale morphological detector (no alignment, no notch)
2. + Phase 1: Fabric Web Alignment Correction (Radon Angle Optimization & De-skewing)
3. + Phase 2: Frequency-Domain Filtering (Automated 2D FFT Cross-Notch Suppression)
4. Phase 3 Integrated: Full End-to-End DIP Pipeline

Outputs:
- outputs/metrics/phase3_integration_benchmark.json
- outputs/visualizations/endsem/fig_phase3_ablation_comparison.png
- outputs/visualizations/endsem/fig_phase3_confusion_matrices.png
- outputs/visualizations/endsem/fig_recovered_rolls_gallery.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SUPER = _ROOT.parent
if str(_SUPER) not in sys.path:
    sys.path.insert(0, str(_SUPER))

from src.data.splitter import load_split
from src.pipeline import DualScalePipeline
from src.preprocessing.alignment import estimate_fabric_skew_angle, align_fabric_image
from src.preprocessing.frequency_filters import suppress_weave_pattern_2d
from src.utils.io import load_image_gray, load_mask_binary, save_json
from src.utils.logging import get_logger
import csv

logger = get_logger(__name__)


def evaluate_configuration(
    pipeline: DualScalePipeline,
    train_samples: List[dict],
    test_samples: List[dict],
    config_name: str,
    save_artifacts: bool = False,
    val_samples: Optional[List[dict]] = None,
) -> Dict:
    t0 = time.perf_counter()
    pipeline.fit(train_samples)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    mask_dir = Path("outputs/masks") if save_artifacts else None
    overlay_dir = Path("outputs/overlays/detection") if save_artifacts else None
    eval_res = pipeline.evaluate(
        test_samples, split_name="test",
        save_artifacts=save_artifacts,
        mask_out_dir=mask_dir,
        overlay_out_dir=overlay_dir
    )
    eval_time = time.perf_counter() - t1

    op = eval_res["image_level"]
    raw = eval_res["raw_screening"]

    val_block = {}
    if val_samples is not None:
        v = pipeline.evaluate(val_samples, split_name="val")
        vl = v["image_level"]
        val_block = {
            "val_screening": {k: round(vl[k] * 100.0, 2) for k in ("accuracy", "recall", "precision", "specificity")},
            "val_confusion_matrix": {k: vl[k] for k in ("tp", "fn", "fp", "tn")},
            "val_pooled_dice_defective_pct": round(v["pooled_dice_defective"] * 100.0, 2),
            "val_roll_flags": [{"sample_id": r["sample_id"], "fabric_id": r["fabric_id"],
                                "is_defective": r["is_defective"], "is_flagged": r["is_flagged"]} for r in v["records"]],
        }

    return {
        **val_block,
        "config_name": config_name,
        "enable_alignment": pipeline.enable_alignment,
        "enable_fft_notch": pipeline.enable_fft_notch,
        "screening_accuracy_pct": round(op["accuracy"] * 100.0, 2),
        "recall_sensitivity_pct": round(op["recall"] * 100.0, 2),
        "precision_pct": round(op["precision"] * 100.0, 2),
        "specificity_pct": round(op["specificity"] * 100.0, 2),
        "pooled_dice_defective_pct": round(eval_res["pooled_dice_defective"] * 100.0, 2),
        "pixel_recall_pct": round(eval_res.get("recall", 0.0) * 100.0, 2),
        "pixel_precision_defective_pct": round(eval_res.get("precision_defective", 0.0) * 100.0, 2),
        "clean_fp_pct": round(eval_res["clean_fp_pct"], 4),
        "confusion_matrix": {
            "tp": op["tp"],
            "fn": op["fn"],
            "fp": op["fp"],
            "tn": op["tn"],
        },
        "raw_screening": {
            "accuracy_pct": round(raw["accuracy"] * 100.0, 2),
            "recall_pct": round(raw["recall"] * 100.0, 2),
            "fp": raw["fp"],
        },
        "fit_time_s": round(fit_time, 2),
        "eval_time_s": round(eval_time, 2),
        "total_time_s": round(fit_time + eval_time, 2),
        "records": eval_res["records"],
        "per_fabric": eval_res.get("per_fabric", {}),
    }


def generate_ablation_plot(benchmarks: List[Dict], out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#2c3e50",
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    names = [b["config_name"] for b in benchmarks]
    x = np.arange(len(names))
    width = 0.20

    accs = [b["screening_accuracy_pct"] for b in benchmarks]
    recs = [b["recall_sensitivity_pct"] for b in benchmarks]
    precs = [b["precision_pct"] for b in benchmarks]
    specs = [b["specificity_pct"] for b in benchmarks]

    c_acc = "#2980b9"
    c_rec = "#27ae60"
    c_prec = "#8e44ad"
    c_spec = "#f39c12"

    r1 = ax.bar(x - 1.5 * width, accs, width, label="Screening Accuracy (%)", color=c_acc, edgecolor="#1f618d", zorder=3)
    r2 = ax.bar(x - 0.5 * width, recs, width, label="Defect Recall / Sensitivity (%)", color=c_rec, edgecolor="#1e8449", zorder=3)
    r3 = ax.bar(x + 0.5 * width, precs, width, label="Precision (%)", color=c_prec, edgecolor="#6c3483", zorder=3)
    r4 = ax.bar(x + 1.5 * width, specs, width, label="Specificity (%)", color=c_spec, edgecolor="#d68910", zorder=3)

    # Add numeric labels
    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

    ax.set_ylabel("Performance (%)", fontsize=12, fontweight="bold")
    ax.set_title("Dual-Scale Pipeline Phase 3 Ablation Progression (AITEX 81 Held-Out Test Rolls)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_ylim(55, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved ablation chart: {out_path}")


def generate_confusion_matrices_plot(baseline: Dict, integrated: Dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    for ax, data, title in [
        (axes[0], baseline, f"Stage 1: baseline\nAcc: {baseline['screening_accuracy_pct']:.1f}% | Rec: {baseline['recall_sensitivity_pct']:.1f}%"),
        (axes[1], integrated, f"Stage 4: full pipeline\nAcc: {integrated['screening_accuracy_pct']:.1f}% | Rec: {integrated['recall_sensitivity_pct']:.1f}%"),
    ]:
        cm = np.array([
            [data["confusion_matrix"]["tp"], data["confusion_matrix"]["fn"]],
            [data["confusion_matrix"]["fp"], data["confusion_matrix"]["tn"]],
        ])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Flagged\n(Defect)", "Clear\n(Pass)"], fontsize=10, fontweight="bold")
        ax.set_yticklabels(["Actual Defect\n(53 rolls)", "Actual Clean\n(28 rolls)"], fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        # Annotate cells
        labels = [["True Positive\n(TP)", "False Negative\n(FN - Missed)"],
                  ["False Alarm\n(FP)", "True Negative\n(TN - Clean)"]]
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / max(1, int(np.sum(cm[i]))) * 100.0
                color = "white" if val > 20 else "#2c3e50"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    fig.suptitle("Production Operational Screening: Baseline vs. Phase 3 Integrated", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved confusion matrices: {out_path}")


def generate_recovered_gallery(records: List[Dict], recovered_sids: List[str], out_path: Path) -> None:
    """Gallery of the rolls that are False Negatives in Stage 1 and True Positives in Stage 4
    (derived from the benchmark records, never hard-coded)."""
    rec_records = [r for r in records if r["sample_id"] in recovered_sids]

    fig, axes = plt.subplots(len(rec_records), 3, figsize=(12, 2.5 * len(rec_records)), dpi=250)
    col_titles = ["Raw Line-Scan Fabric", "Ground Truth Defect Mask", "Phase 3 Notch Anomaly Overlay"]

    for row_idx, r in enumerate(rec_records):
        sid = r["sample_id"]
        img = load_image_gray(r["image_path"])
        gt = load_mask_binary(r["mask_path"]).astype(bool) if r["mask_path"] else np.zeros_like(img, bool)

        # Center on defect
        ys, xs = np.where(gt)
        if len(xs) > 0:
            cx = int(np.mean(xs))
            x0 = max(0, min(img.shape[1] - 512, cx - 256))
        else:
            x0 = 1000

        patch_img = img[:, x0:x0 + 512].astype(np.uint8)
        patch_gt = gt[:, x0:x0 + 512]

        pred_mask_p = r.get("pred_mask_path")
        if pred_mask_p and Path(pred_mask_p).exists():
            pred = load_mask_binary(pred_mask_p).astype(bool)[:, x0:x0 + 512]
        else:
            pred = np.zeros_like(patch_gt)

        # Visual overlay
        overlay = cv2.cvtColor(patch_img, cv2.COLOR_GRAY2RGB)
        overlay[patch_gt] = [0, 255, 0]  # Green GT
        if np.any(pred):
            overlay[pred] = [255, 0, 0]  # Red prediction
            overlap = pred & patch_gt
            if np.any(overlap):
                overlay[overlap] = [255, 255, 0]  # Yellow true positive overlap

        axes[row_idx, 0].imshow(patch_img, cmap="gray")
        axes[row_idx, 0].set_ylabel(f"{sid}\n(Fab 03)", fontsize=9, fontweight="bold")
        axes[row_idx, 1].imshow(patch_gt, cmap="hot")
        axes[row_idx, 2].imshow(overlay)

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Visual Verification: 5 Recovered False-Negative Fabric Rolls via 2D FFT Notch Filtering", fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved recovered gallery: {out_path}")


def generate_stage_progression_panels(records: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Representative spotlight rolls: 2 recovered rolls (0050_019_03, 0053_019_03) and 2 test demonstrations (0018_010_03, 0021_016_03)
    spotlights = ["0050_019_03", "0053_019_03", "0018_010_03", "0021_016_03"]
    rec_map = {r["sample_id"]: r for r in records}

    for sid in spotlights:
        if sid not in rec_map:
            continue
        r = rec_map[sid]
        img = load_image_gray(r["image_path"])
        mask = load_mask_binary(r["mask_path"]).astype(bool) if r.get("mask_path") else np.zeros_like(img, bool)

        # Crop 512x256 patch centered on defect
        ys, xs = np.where(mask) if np.any(mask) else (np.array([128]), np.array([2048]))
        cx = int(np.mean(xs))
        x0 = max(0, min(img.shape[1] - 512, cx - 256))

        p_raw = img[:, x0:x0 + 512]
        p_gt = mask[:, x0:x0 + 512]

        # Step 2: Radon Aligned
        res_align = estimate_fabric_skew_angle(p_raw, angle_range=(-2.5, 2.5))
        skew = res_align["skew_angle"]
        p_align = align_fabric_image(p_raw, skew) if abs(skew) >= 0.15 else p_raw.copy()

        # Step 3: FFT Notch Filtered
        p_notch, _ = suppress_weave_pattern_2d(p_align, notch_radius=12.0, order=3, max_peaks=12)

        # Step 4: Final Overlay
        pred_p = r.get("pred_mask_path")
        if pred_p and Path(pred_p).exists():
            pred_m = load_mask_binary(pred_p)[:, x0:x0 + 512]
        else:
            pred_m = np.zeros_like(p_raw, bool)

        overlay = cv2.cvtColor(p_raw, cv2.COLOR_GRAY2BGR)
        overlay[pred_m > 0] = [0, 230, 255]  # Cyan detected
        contours, _ = cv2.findContours(p_gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Green GT

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), dpi=200)
        axes[0].imshow(p_raw, cmap="gray")
        axes[0].set_title(f"Step 1: Raw Industrial Line-Scan Strip [{sid}]", fontsize=10, fontweight="bold")

        axes[1].imshow(p_align, cmap="gray")
        axes[1].set_title(f"Step 2: Radon Line-Integral Web Alignment (theta = {skew:.2f} deg)", fontsize=10, fontweight="bold")

        axes[2].imshow(p_notch, cmap="gray")
        axes[2].set_title("Step 3: 2D FFT Reciprocal Lattice Butterworth Cross-Notch Filtered", fontsize=10, fontweight="bold")

        axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Step 4: Dual-Scale Anomaly Segmentation Overlay (Green=GT, Yellow=Detected)", fontsize=10, fontweight="bold")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        out_f = out_dir / f"{sid}_stages.png"
        fig.savefig(out_f)
        plt.close(fig)
    print(f"Saved step-by-step stage progression overlays to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/splits_oneclass")
    parser.add_argument("--out-dir", default="outputs/visualizations/endsem")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    metrics_dir = Path(args.metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(args.splits_dir)
    train_samples = load_split(splits_dir, "train")
    test_samples = load_split(splits_dir, "test")
    val_samples = load_split(splits_dir, "val")

    print("=" * 110)
    print("PHASE 3 SYSTEMATIC INTEGRATION BENCHMARK (4-CONFIGURATION ABLATION STUDY)")
    print("=" * 110)

    configs = [
        ("1. Baseline (dual-scale morphology)", False, False),
        ("2. + Web alignment", True, False),
        ("3. + 2D FFT notch", False, True),
        ("4. Full pipeline", True, True),
    ]

    benchmarks: List[Dict] = []
    for cfg_name, en_align, en_notch in configs:
        print(f"\nEvaluating Stage: {cfg_name} (Alignment={en_align}, FFT Notch={en_notch})...")
        pipe = DualScalePipeline(
            mode="adaptive",
            enable_alignment=en_align,
            enable_fft_notch=en_notch,
        )
        save_art = (cfg_name == "4. Full pipeline")
        res = evaluate_configuration(pipe, train_samples, test_samples, cfg_name, save_artifacts=save_art, val_samples=val_samples)
        benchmarks.append(res)
        print(f"  -> Acc: {res['screening_accuracy_pct']:.2f}% | Rec: {res['recall_sensitivity_pct']:.2f}% | Prec: {res['precision_pct']:.2f}% | Spec: {res['specificity_pct']:.2f}% | TP={res['confusion_matrix']['tp']}, FN={res['confusion_matrix']['fn']}, FP={res['confusion_matrix']['fp']}, TN={res['confusion_matrix']['tn']} ({res['total_time_s']}s)")

    # Dynamically extract recovered rolls (FN in Stage 1 -> TP in Stage 4)
    base_fn = {r["sample_id"] for r in benchmarks[0]["records"] if r["is_defective"] and not r["is_flagged"]}
    final_tp = {r["sample_id"]: r for r in benchmarks[3]["records"] if r["is_defective"] and r["is_flagged"]}
    recovered_sids = sorted(list(base_fn.intersection(final_tp.keys())))
    recovered_rolls = []
    for sid in recovered_sids:
        r = final_tp[sid]
        recovered_rolls.append({
            "sample_id": sid,
            "fabric_id": r["fabric_id"],
            "defect_type": r.get("defect_type", sid.split("_")[1]),
            "dice": round(r["dice"] * 100.0, 2),
        })

    # Compute macro-average fabric Dice for all stages
    for b in benchmarks:
        if "per_fabric" in b and b["per_fabric"]:
            dices = [st["dice"] * 100.0 for st in b["per_fabric"].values() if st["dice"] is not None]
            b["macro_avg_dice_pct"] = round(float(np.mean(dices)), 2) if dices else 0.0
        else:
            b["macro_avg_dice_pct"] = 0.0

    # Save quantitative ablation metrics
    for b in benchmarks:
        b["test_roll_flags"] = [{"sample_id": r["sample_id"], "fabric_id": r["fabric_id"],
                                 "is_defective": r["is_defective"], "is_flagged": r["is_flagged"]} for r in b["records"]]
    save_benchmarks = [{k: v for k, v in b.items() if k != "records"} for b in benchmarks]
    metrics_path = metrics_dir / "phase3_integration_benchmark.json"
    save_json(metrics_path, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_summary": save_benchmarks,
        "recovered_rolls": recovered_rolls,
        "baseline_to_integrated_gain": {
            "screening_accuracy_gain_pct": round(benchmarks[3]["screening_accuracy_pct"] - benchmarks[0]["screening_accuracy_pct"], 2),
            "recall_gain_pct": round(benchmarks[3]["recall_sensitivity_pct"] - benchmarks[0]["recall_sensitivity_pct"], 2),
            "missed_rolls_reduced_from_to": [benchmarks[0]["confusion_matrix"]["fn"], benchmarks[3]["confusion_matrix"]["fn"]],
            "clean_false_alarm_rate_pct": benchmarks[3]["clean_fp_pct"],
            "recovered_rolls_count": len(recovered_rolls),
        }
    })
    print(f"\nSaved metrics: {metrics_path}")

    # Roll-screening threshold sensitivity (Stage 4). The headline results use ONE uniform
    # threshold (15 px). The legacy hand-set per-fabric map is reported only as a labelled variant.
    LEGACY_PER_FABRIC = {"00": 10, "01": 5, "02": 10, "03": 8}
    sens_pipe = DualScalePipeline(mode="adaptive", enable_alignment=True, enable_fft_notch=True)
    sens_pipe.fit(train_samples)
    sens_rows = []
    for label, min_px, per_fabric in [
        ("uniform 1 px", 1, None), ("uniform 5 px", 5, None), ("uniform 15 px (HEADLINE)", 15, None),
        ("uniform 30 px", 30, None), ("uniform 50 px", 50, None),
        ("legacy per-fabric map {00:10,01:5,02:10,03:8, others:15} (hand-set, untuned provenance)", 15, LEGACY_PER_FABRIC),
    ]:
        sens_pipe.screening_min_px = min_px
        sens_pipe.screening_thresholds = per_fabric
        r = sens_pipe.evaluate(test_samples, split_name="test")["image_level"]
        sens_rows.append([label, round(r["accuracy"] * 100, 2), round(r["recall"] * 100, 2),
                          round(r["precision"] * 100, 2), round(r["specificity"] * 100, 2),
                          r["tp"], r["fn"], r["fp"], r["tn"]])
    tables_dir0 = Path("outputs/tables")
    tables_dir0.mkdir(parents=True, exist_ok=True)
    with open(tables_dir0 / "screening_threshold_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Roll-screening rule (Stage 4, test split)", "Accuracy (%)", "Recall (%)", "Precision (%)", "Specificity (%)", "TP", "FN", "FP", "TN"])
        w.writerows(sens_rows)

    # Generate visual artifacts
    generate_ablation_plot(benchmarks, out_dir / "fig_phase3_ablation_comparison.png")
    generate_confusion_matrices_plot(benchmarks[0], benchmarks[3], out_dir / "fig_phase3_confusion_matrices.png")
    generate_recovered_gallery(benchmarks[3]["records"], recovered_sids, out_dir / "fig_recovered_rolls_gallery.png")

    # Generate step-by-step stage progression panels
    stages_dir = Path("outputs/overlays/stages")
    generate_stage_progression_panels(benchmarks[3]["records"], stages_dir)

    # Export clean CSV tables
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "ablation_study_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Web Alignment", "2D FFT Notch", "Accuracy (%)", "Recall (%)", "Precision (%)", "Specificity (%)", "Pooled Dice (%)", "Macro Dice (%)", "TP", "FN", "FP", "TN", "Runtime (s)"])
        for b in benchmarks:
            cm = b["confusion_matrix"]
            writer.writerow([
                b["config_name"], b["enable_alignment"], b["enable_fft_notch"],
                b["screening_accuracy_pct"], b["recall_sensitivity_pct"], b["precision_pct"],
                b["specificity_pct"], b["pooled_dice_defective_pct"], b["macro_avg_dice_pct"],
                cm["tp"], cm["fn"], cm["fp"], cm["tn"], b["total_time_s"]
            ])
    print(f"Saved ablation table: {tables_dir / 'ablation_study_table.csv'}")

    if "per_fabric" in benchmarks[3] and benchmarks[3]["per_fabric"]:
        with open(tables_dir / "fabric_wise_breakdown.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Fabric ID", "Ground Truth Px", "True Positives Px", "Dice (%)", "Recall (%)", "Precision (%)", "Clean False Alarm (%)"])
            fabric_dices = []
            tot_gt = 0
            tot_tp = 0
            for fid, stats in sorted(benchmarks[3]["per_fabric"].items()):
                dice_val = f"{stats['dice']*100:.2f}" if stats['dice'] is not None else "N/A"
                if stats['dice'] is not None:
                    fabric_dices.append(stats['dice'] * 100.0)
                rec_val = f"{stats['recall']*100:.2f}" if stats['recall'] is not None else "N/A"
                prec_val = f"{stats['precision']*100:.2f}" if stats['precision'] is not None else "N/A"
                tot_gt += stats["gt_px"]
                tot_tp += stats["tp"]
                writer.writerow([fid, stats["gt_px"], stats["tp"], dice_val, rec_val, prec_val, f"{stats['clean_fp_pct']:.4f}"])
            macro_dice = np.mean(fabric_dices) if fabric_dices else 0.0
            writer.writerow(["Macro-Average", "-", "-", f"{macro_dice:.2f}", "-", "-", "-"])
            pooled_dice = benchmarks[3]["pooled_dice_defective_pct"]
            pooled_rec = benchmarks[3]["pixel_recall_pct"]
            pooled_prec = benchmarks[3]["pixel_precision_defective_pct"]
            pooled_clean_fp = f"{benchmarks[3]['clean_fp_pct']:.4f}"
            writer.writerow(["Pooled Total", tot_gt, tot_tp, f"{pooled_dice:.2f}", f"{pooled_rec:.2f}", f"{pooled_prec:.2f}", pooled_clean_fp])
        print(f"Saved fabric breakdown table: {tables_dir / 'fabric_wise_breakdown.csv'}")

    # Validation-split replication of the ablation + per-fabric roll-level breakdown
    with open(tables_dir / "ablation_val_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Configuration", "Val Accuracy (%)", "Val Recall (%)", "Val Precision (%)", "Val Specificity (%)", "TP", "FN", "FP", "TN"])
        for b in benchmarks:
            vs, vc = b["val_screening"], b["val_confusion_matrix"]
            w.writerow([b["config_name"], vs["accuracy"], vs["recall"], vs["precision"], vs["specificity"], vc["tp"], vc["fn"], vc["fp"], vc["tn"]])
    with open(tables_dir / "ablation_by_fabric.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Split", "Fabric", "Configuration", "Defective rolls flagged", "Defective rolls", "Clean rolls flagged", "Clean rolls"])
        for split, key in (("val", "val_roll_flags"), ("test", "test_roll_flags")):
            for b in benchmarks:
                fabs = sorted({r["fabric_id"] for r in b[key]})
                for fid in fabs:
                    rr = [r for r in b[key] if r["fabric_id"] == fid]
                    d = [r for r in rr if r["is_defective"]]
                    c = [r for r in rr if not r["is_defective"]]
                    w.writerow([split, fid, b["config_name"], sum(r["is_flagged"] for r in d), len(d),
                                sum(r["is_flagged"] for r in c), len(c)])
    print(f"Saved validation + per-fabric ablation tables to {tables_dir}")

    print("\n" + "=" * 110)
    print("PHASE 3 INTEGRATION COMPLETE: ALL VISUALIZATIONS AND METRICS EXPORTED.")
    print("=" * 110)


if __name__ == "__main__":
    main()
