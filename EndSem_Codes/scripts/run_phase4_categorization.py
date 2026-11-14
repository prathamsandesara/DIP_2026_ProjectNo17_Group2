#!/usr/bin/env python
"""
Phase 4 Benchmark: Multi-Class Defect Categorization via Invariant Moments & Physical Taxonomy.

ECE501 Digital Image Processing (Module 7: Representation & Description, Week 13)
---------------------------------------------------------------------------------
Evaluates deterministic physical defect categorization on the 53 defective fabric rolls:
1. Hu's 7 Invariant Moments (phi_1 through phi_7)
2. Morphometric & Photometric Descriptors:
   - Aspect ratio AR = max(W, H) / min(W, H)
   - Compactness C = 4*pi*Area / Perimeter^2
   - Eccentricity epsilon from inertia tensor
   - Principal orientation angle theta
   - Radiometric contrast Delta_I = mu_defect - mu_background
3. Deterministic Taxonomy Decision Tree (NO machine learning):
   - Category 0: Longitudinal / Warp Flaw (Vertical line/streak)
   - Category 1: Transverse / Weft Flaw (Horizontal line/bar)
   - Category 2: Localized Micro-Puncture / Hole (Puncture/thread break)
   - Category 3: Photometric Contaminant (dark diffuse region)

Outputs:
- outputs/metrics/phase4_categorization_benchmark.json
- outputs/visualizations/endsem/fig_phase4_taxonomy_confusion_matrix.png
- outputs/visualizations/endsem/fig_phase4_hu_moments_scatter.png
- outputs/visualizations/endsem/fig_phase4_categorization_gallery.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.localization.moment_classifier import (
    extract_defect_descriptors,
    classify_defect_deterministic,
    draw_taxonomy_overlay,
    TAXONOMY_CATEGORIES,
    TAXONOMY_COLORS,
)
from src.utils.io import load_image_gray, load_mask_binary, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

CODE_TO_GROUND_TRUTH_CLASS = {
    # 0: Longitudinal / Warp Flaws (Vertical)
    "002": 0, "023": 0, "006": 0, "027": 0,
    # 1: Transverse / Weft Flaws (Horizontal)
    "010": 1, "022": 1, "016": 1, "036": 1,
    # 2: Localized Micro-Punctures / Holes
    "019": 2, "025": 2,
    # 3: Photometric Contaminants / Stains
    "030": 3,
}


def evaluate_taxonomy(samples: List[dict]) -> Tuple[List[Dict], np.ndarray, float]:
    records: List[Dict] = []
    conf_mat = np.zeros((4, 4), dtype=int)
    correct = 0
    total = 0
    print(f"   [Taxonomy] Extracting Hu invariant moments & physical descriptors for {len(samples)} rolls...")

    for s in samples:
        if not s["is_defective"] or not s.get("mask_path"):
            continue

        sid = s["sample_id"]
        code = sid.split("_")[1]
        if code not in CODE_TO_GROUND_TRUTH_CLASS:
            continue
        gt_class = CODE_TO_GROUND_TRUTH_CLASS[code]

        img = load_image_gray(s["image_path"])
        mask = load_mask_binary(s["mask_path"]).astype(bool)
        if not np.any(mask):
            continue

        feats = extract_defect_descriptors(img, mask)
        if not feats:
            continue

        clf = classify_defect_deterministic(feats)
        pred_class = clf["category_id"]

        is_correct = (pred_class == gt_class)
        if is_correct:
            correct += 1
        total += 1
        conf_mat[gt_class, pred_class] += 1

        rec = {
            "sample_id": sid,
            "fabric_id": s["fabric_id"],
            "defect_code": code,
            "ground_truth_class_id": gt_class,
            "ground_truth_class_name": TAXONOMY_CATEGORIES[gt_class],
            "predicted_class_id": pred_class,
            "predicted_class_name": clf["category_name"],
            "rule_triggered": clf["rule_triggered"],
            "is_correct": is_correct,
            "descriptors": feats,
            "image_path": s["image_path"],
            "mask_path": s["mask_path"],
        }
        records.append(rec)

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    return records, conf_mat, acc


def generate_confusion_matrix_heatmap(conf_mat: np.ndarray, accuracy: float, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
    })

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)
    im = ax.imshow(conf_mat, cmap="Blues", interpolation="nearest")

    classes = [TAXONOMY_CATEGORIES[i] for i in range(4)]
    short_classes = ["Warp Flaw\n(Vertical)", "Weft Flaw\n(Horizontal)", "Micro-Puncture\n(Hole / Break)", "Photometric\n(Dark Diffuse)"]

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(short_classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(short_classes, fontsize=10, fontweight="bold")

    ax.set_xlabel("Predicted Physical Category", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground-Truth Defect Category", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title(f"Phase 4 Deterministic Defect Taxonomy Confusion Matrix\nOverall Physical Classification Accuracy: {accuracy:.2f}% (No Machine Learning)", fontsize=12, fontweight="bold", pad=15)

    # Annotate numbers and row-percentages
    row_sums = conf_mat.sum(axis=1)
    for i in range(4):
        for j in range(4):
            val = conf_mat[i, j]
            row_tot = row_sums[i]
            pct = (val / row_tot * 100.0) if row_tot > 0 else 0.0
            color = "white" if val > 8 else "#1b4f72"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of Defect Instances", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved confusion matrix heatmap: {out_path}")


def generate_moments_scatter(records: List[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

    category_markers = {0: "^", 1: "s", 2: "o", 3: "D"}
    category_colors = {0: "#e67e22", 1: "#2980b9", 2: "#c0392b", 3: "#8e44ad"}

    for cat_id in range(4):
        cat_recs = [r for r in records if r["ground_truth_class_id"] == cat_id]
        if not cat_recs:
            continue
        xs = [r["descriptors"]["aspect_ratio"] for r in cat_recs]
        ys = [r["descriptors"]["hu1"] for r in cat_recs]
        ax.scatter(
            xs, ys,
            s=80,
            marker=category_markers[cat_id],
            c=category_colors[cat_id],
            edgecolors="#2c3e50",
            linewidth=1.0,
            label=f"[{cat_id}] {TAXONOMY_CATEGORIES[cat_id]} (N={len(cat_recs)})",
            alpha=0.88,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_ylim(-2.0, 1.15)
    ax.set_xlabel("Aspect Ratio (log scale: max(W,H) / min(W,H))", fontsize=11, fontweight="bold")
    ax.set_ylabel("Hu Invariant Moment $\\psi_1 = -\\mathrm{sign}(\\phi_1)\\log_{10}|\\phi_1|$", fontsize=11, fontweight="bold")
    ax.set_title("Defect Morphometry in Invariant Feature Space (N=53)\n(Warp/Weft separated by principal orientation $\\theta$, not Hu $\\psi_1$)", fontsize=11.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    ax.text(0.04, 0.06, "Warp & Weft overlap in (AR, ψ1) space; tree separates them by angle θ (|θ|≤30° vs |θ|≥60°)",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved moments scatter plot: {out_path}")


def generate_categorization_gallery(records: List[Dict], out_path: Path) -> None:
    # Select 2 representative samples per category
    samples_per_cat = {0: [], 1: [], 2: [], 3: []}
    for r in records:
        cid = r["ground_truth_class_id"]
        if r["is_correct"] and len(samples_per_cat[cid]) < 2:
            samples_per_cat[cid].append(r)

    selected = []
    for cid in range(4):
        selected.extend(samples_per_cat[cid])

    fig, axes = plt.subplots(len(selected), 3, figsize=(13, 2.3 * len(selected)), dpi=250)
    col_titles = ["Raw Line-Scan Fabric", "Ground Truth Defect Mask", "Physical Taxonomy Classification Overlay"]

    for row_idx, r in enumerate(selected):
        img = load_image_gray(r["image_path"])
        mask = load_mask_binary(r["mask_path"]).astype(bool)
        ys, xs = np.where(mask)
        cx = int(np.mean(xs))
        x0 = max(0, min(img.shape[1] - 512, cx - 256))

        p_img = img[:, x0:x0 + 512]
        p_mask = mask[:, x0:x0 + 512]

        desc = r["descriptors"]
        clf = {"category_id": r["predicted_class_id"], "category_name": r["predicted_class_name"]}
        overlay = draw_taxonomy_overlay(p_img, p_mask.astype(np.uint8), desc, clf)

        axes[row_idx, 0].imshow(p_img, cmap="gray")
        axes[row_idx, 0].set_ylabel(f"{r['sample_id']}\n({r['defect_code']})", fontsize=8, fontweight="bold")
        axes[row_idx, 1].imshow(p_mask, cmap="hot")
        axes[row_idx, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Visual Demonstration: Classical Invariant Moment Defect Categorization Across All 4 Physical Classes", fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved categorization gallery: {out_path}")


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
    test_defective = [s for s in test_samples if s["is_defective"]]
    val_defective = [s for s in val_samples if s["is_defective"] and s.get("mask_path")]

    print("=" * 110)
    print("PHASE 4 BENCHMARK: MULTI-CLASS DEFECT CATEGORIZATION VIA INVARIANT MOMENTS & PHYSICAL TAXONOMY")
    print("=" * 110)

    # 1. Oracle Diagnostic Evaluation (Test Split Ground-Truth Masks)
    t0 = time.perf_counter()
    records, conf_mat, accuracy = evaluate_taxonomy(test_defective)
    elapsed = time.perf_counter() - t0

    # 2. Unseen Validation Generalization Check
    _, val_conf_mat, val_accuracy = evaluate_taxonomy(val_defective)

    # 3. True end-to-end evaluation: the classifier sees only what the detector produced.
    #    Roll flagged -> take the LARGEST connected component of the predicted mask (no ground truth used)
    #    -> describe it (Hu moments etc.) -> classify. A roll counts as correct only if flagged AND the class matches.
    from src.pipeline import DualScalePipeline
    pipe = DualScalePipeline(enable_alignment=True, enable_fft_notch=True)
    pipe.fit(train_samples)
    mask_dir = Path("outputs/masks")
    eval_pipe = pipe.evaluate(test_samples, split_name="test", save_artifacts=True,
                              mask_out_dir=mask_dir, overlay_out_dir=Path("outputs/overlays/detection"))
    pipe_records = {r["sample_id"]: r for r in eval_pipe["records"]}

    e2e_desc = {}          # sample_id -> descriptors of the largest predicted component (or None)
    for rec in records:
        sid = rec["sample_id"]
        pr = pipe_records.get(sid, {})
        e2e_desc[sid] = None
        if not pr.get("is_flagged", False):
            continue
        pm = cv2.imread(str(mask_dir / f"{sid}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if pm is None or not np.any(pm > 127):
            continue
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats((pm > 127).astype(np.uint8), connectivity=8)
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        e2e_desc[sid] = extract_defect_descriptors(load_image_gray(rec["image_path"]), lab == big)

    detector_caught = sum(1 for r in records if pipe_records.get(r["sample_id"], {}).get("is_flagged", False))
    total_test_def = len(test_defective)
    flag_and_oracle_correct = sum(1 for r in records
                                  if pipe_records.get(r["sample_id"], {}).get("is_flagged", False) and r["is_correct"])
    e2e_correct = sum(1 for r in records
                      if e2e_desc[r["sample_id"]] is not None
                      and classify_defect_deterministic(e2e_desc[r["sample_id"]])["category_id"] == r["ground_truth_class_id"])
    cascaded_acc = (e2e_correct / total_test_def) * 100.0 if total_test_def > 0 else 0.0
    cascaded_correct = e2e_correct
    flag_oracle_acc = flag_and_oracle_correct / total_test_def * 100.0 if total_test_def > 0 else 0.0

    # Compute Per-Class Performance
    per_class = {}
    for i, name in TAXONOMY_CATEGORIES.items():
        tp = int(conf_mat[i, i])
        fn = int(conf_mat[i, :].sum() - tp)
        fp = int(conf_mat[:, i].sum() - tp)
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "instances": int(conf_mat[i, :].sum()),
            "true_positives": tp,
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "f1_score": round(f1, 2),
        }

    print(f"\n1. Oracle Diagnostic Evaluation (Given Defect Segment): {accuracy:.2f}% ({elapsed/len(records)*1000.0:.1f}ms/defect)")
    print(f"2. Validation split (frozen rules):              {val_accuracy:.2f}%")
    print(f"3. True end-to-end (largest predicted component classified): {cascaded_acc:.2f}% ({cascaded_correct}/{total_test_def})")
    print(f"   (flagged AND oracle-mask class correct, for reference:   {flag_oracle_acc:.2f}% ({flag_and_oracle_correct}/{total_test_def}))\n")

    print(f"{'Category Name':<42} | {'Count':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score'}")
    print("-" * 85)
    for name, stats in per_class.items():
        print(f"{name:<42} | {stats['instances']:<6} | {stats['precision_pct']:>8.2f}% | {stats['recall_pct']:>8.2f}% | {stats['f1_score']:>8.2f}")
    print("=" * 85)

    clean_records = []
    for r in records:
        clean_records.append({
            "sample_id": r["sample_id"],
            "fabric_id": r["fabric_id"],
            "defect_code": r["defect_code"],
            "ground_truth_class_id": r["ground_truth_class_id"],
            "ground_truth_class_name": r["ground_truth_class_name"],
            "predicted_class_id": r["predicted_class_id"],
            "predicted_class_name": r["predicted_class_name"],
            "is_correct": r["is_correct"],
            "aspect_ratio": r["descriptors"]["aspect_ratio"],
            "hu1": r["descriptors"]["hu1"],
            "hu2": r["descriptors"]["hu2"],
        })

    # Sensitivity: scale EVERY threshold of the real classifier simultaneously (no copy of the rules).
    sensitivity_results = []
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        corr_oracle = sum(1 for r in records
                          if classify_defect_deterministic(r["descriptors"], scale)["category_id"] == r["ground_truth_class_id"])
        corr_e2e = sum(1 for r in records
                       if e2e_desc[r["sample_id"]] is not None
                       and classify_defect_deterministic(e2e_desc[r["sample_id"]], scale)["category_id"] == r["ground_truth_class_id"])
        sensitivity_results.append({
            "perturbation_scale": scale,
            "perturbation_label": f"{int(round((scale - 1.0) * 100)):+d}%",
            "oracle_accuracy_pct": round(corr_oracle / len(records) * 100.0, 2),
            "oracle_correct": f"{corr_oracle}/{len(records)}",
            "cascaded_accuracy_pct": round(corr_e2e / total_test_def * 100.0, 2),
            "cascaded_correct": f"{corr_e2e}/{total_test_def}",
        })

    # Save metrics JSON
    metrics_path = metrics_dir / "phase4_categorization_benchmark.json"
    save_json(metrics_path, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_instances": len(records),
        "mean_runtime_ms": round(elapsed / max(1, len(records)) * 1000.0, 2),
        "overall_accuracy_pct": round(accuracy, 2),
        "oracle_diagnostic_accuracy_pct": round(accuracy, 2),
        "unseen_validation_accuracy_pct": round(val_accuracy, 2),
        "cascaded_end_to_end_accuracy_pct": round(cascaded_acc, 2),
        "flagged_and_oracle_class_accuracy_pct": round(flag_oracle_acc, 2),
        "detector_recall_pct": round(detector_caught / total_test_def * 100.0, 2),
        "sensitivity_analysis": sensitivity_results,
        "per_class_performance": per_class,
        "confusion_matrix": conf_mat.tolist(),
        "taxonomy_categories": TAXONOMY_CATEGORIES,
        "records": clean_records,
    })
    print(f"\nSaved taxonomy metrics: {metrics_path}")

    # Generate visual artifacts
    generate_confusion_matrix_heatmap(conf_mat, accuracy, out_dir / "fig_phase4_taxonomy_confusion_matrix.png")
    generate_moments_scatter(records, out_dir / "fig_phase4_hu_moments_scatter.png")
    generate_categorization_gallery(records, out_dir / "fig_phase4_categorization_gallery.png")

    # Save individual taxonomy overlays for all defect instances
    tax_dir = Path("outputs/overlays/taxonomy")
    tax_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        img = load_image_gray(r["image_path"])
        mask = load_mask_binary(r["mask_path"]).astype(bool)
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx = int(np.mean(xs))
            x0 = max(0, min(img.shape[1] - 512, cx - 256))
            p_img = img[:, x0:x0 + 512]
            p_mask = mask[:, x0:x0 + 512]
            desc = r["descriptors"]
            clf = {"category_id": r["predicted_class_id"], "category_name": r["predicted_class_name"]}
            overlay = draw_taxonomy_overlay(p_img, p_mask.astype(np.uint8), desc, clf)
            cv2.imwrite(str(tax_dir / f"{r['sample_id']}_{r['defect_code']}_taxonomy.png"), overlay)
    print(f"Saved {len(records)} defect taxonomy overlays to: {tax_dir}")

    # Export clean CSV tables
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(tables_dir / "multiclass_taxonomy_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class ID", "Defect Category", "Support (N)", "True Positives (TP)", "False Positives (FP)", "False Negatives (FN)", "Precision (%)", "Recall (%)", "F1-Score (%)"])
        for cid, (name, stats) in enumerate(per_class.items()):
            tp = stats["true_positives"]
            fn = int(conf_mat[cid, :].sum() - tp)
            fp = int(conf_mat[:, cid].sum() - tp)
            writer.writerow([cid, name, stats["instances"], tp, fp, fn, stats["precision_pct"], stats["recall_pct"], stats["f1_score"]])
    print(f"Saved multiclass taxonomy table: {tables_dir / 'multiclass_taxonomy_table.csv'}")

    with open(tables_dir / "multiclass_confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ground Truth \\ Predicted"] + [TAXONOMY_CATEGORIES[i] for i in range(4)])
        for i in range(4):
            writer.writerow([TAXONOMY_CATEGORIES[i]] + conf_mat[i].tolist())
    print(f"Saved multiclass confusion matrix: {tables_dir / 'multiclass_confusion_matrix.csv'}")

    with open(tables_dir / "taxonomy_sensitivity_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Perturbation Label", "Threshold Scale", "Oracle Accuracy (%)", "Oracle Correct", "End-to-end Accuracy (%)", "End-to-end Correct"])
        for sr in sensitivity_results:
            writer.writerow([sr["perturbation_label"], sr["perturbation_scale"], sr["oracle_accuracy_pct"], sr["oracle_correct"], sr["cascaded_accuracy_pct"], sr["cascaded_correct"]])
    print(f"Saved taxonomy sensitivity table: {tables_dir / 'taxonomy_sensitivity_table.csv'}")

    print("\n" + "=" * 110)
    print("PHASE 4 COMPLETE: ALL TAXONOMY METRICS AND FIGURES EXPORTED.")
    print("=" * 110)


if __name__ == "__main__":
    main()
