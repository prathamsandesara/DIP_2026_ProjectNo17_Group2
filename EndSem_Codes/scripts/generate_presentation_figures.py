"""
Regenerate the slide figures in outputs/visualizations/presentation/ from the benchmark
outputs (JSON / CSV tables, predicted masks, overlays). No number is typed by hand.

Run AFTER run_phase3_benchmark.py and run_phase4_categorization.py:
    PYTHONPATH=code python code/scripts/generate_presentation_figures.py

Figures:
  fig_midsem_vs_endsem.png          Stage-1 vs Stage-4 confusion matrices + metric bars
  fig_fabric_wise_dice.png          per-fabric Dice with pooled / macro reference lines
  fig_detection_gallery_defective.png  first defective test roll of every fabric (GT vs prediction)
  fig_clean_roll_gallery.png        first clean test roll of every fabric (prediction)
  fig_heatmap_gallery.png           anomaly-ratio heat maps of the same defective rolls
  fig_taxonomy_showcase.png         first test defect of each taxonomy class (moment overlay)

Selection rule for galleries: the first roll of each fabric in sample_id order (no cherry-picking).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.splitter import load_split
from src.utils.io import load_image_gray, load_mask_binary

OUT = Path("outputs/visualizations/presentation")
CODE_TO_CLASS = {"002": 0, "023": 0, "006": 0, "027": 0, "010": 1, "022": 1, "016": 1, "036": 1,
                 "019": 2, "025": 2, "030": 3}
CLASS_NAMES = ["Warp flaw (vertical)", "Weft flaw (horizontal)", "Micro-puncture / hole",
               "Photometric contaminant (dark diffuse)"]
SCREEN_PX = 15


def _pred_mask(sid: str) -> np.ndarray:
    m = cv2.imread(f"outputs/masks/{sid}_mask.png", cv2.IMREAD_GRAYSCALE)
    return (m > 127) if m is not None else None


def _crop(img, gt, x_pad=256):
    """Window of 512 px width centred on the GT defect (or the roll centre if clean)."""
    w = img.shape[1]
    cx = int(np.mean(np.where(gt)[1])) if gt is not None and gt.any() else w // 2
    x0 = int(np.clip(cx - x_pad, 0, w - 2 * x_pad))
    return slice(x0, x0 + 2 * x_pad)


def _outline(ax, mask, color, sl):
    cs, _ = cv2.findContours(mask[:, sl].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        c = c[:, 0, :]
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=color, lw=1.6)


def fig_midsem_vs_endsem(p3):
    s1, s4 = p3["benchmark_summary"][0], p3["benchmark_summary"][3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, b, title in ((axes[0], s1, "Stage 1: dual-scale morphology baseline"),
                         (axes[1], s4, "Stage 4: + alignment + FFT notch")):
        c = b["confusion_matrix"]
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / cm[i].sum():.1f}%)", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Pred clean", "Pred defective"])
        ax.set_yticks([0, 1], ["GT clean", "GT defective"])
        ax.set_title(f"{title}\nacc {b['screening_accuracy_pct']:.2f}%", fontsize=10)
    names = ["Accuracy", "Recall", "Precision", "Specificity", "Pooled Dice"]
    keys = ["screening_accuracy_pct", "recall_sensitivity_pct", "precision_pct", "specificity_pct",
            "pooled_dice_defective_pct"]
    x = np.arange(len(names))
    axes[2].bar(x - 0.2, [s1[k] for k in keys], 0.4, label="Stage 1")
    axes[2].bar(x + 0.2, [s4[k] for k in keys], 0.4, label="Stage 4")
    axes[2].set_xticks(x, names, rotation=20)
    axes[2].set_ylim(0, 100); axes[2].set_ylabel("%"); axes[2].legend()
    axes[2].set_title("Held-out test split (81 rolls)")
    fig.tight_layout(); fig.savefig(OUT / "fig_midsem_vs_endsem.png", dpi=160); plt.close(fig)


def fig_fabric_dice():
    rows = list(csv.DictReader(open("outputs/tables/fabric_wise_breakdown.csv")))
    fab = [r for r in rows if r["Fabric ID"].isdigit()]
    macro = next(r for r in rows if r["Fabric ID"] == "Macro-Average")["Dice (%)"]
    pooled = next(r for r in rows if r["Fabric ID"] == "Pooled Total")["Dice (%)"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    d = [float(r["Dice (%)"]) for r in fab]
    ax.bar([r["Fabric ID"] for r in fab], d, color="#3b82f6")
    for i, v in enumerate(d):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.axhline(float(pooled), color="#ef4444", ls="--", label=f"pooled {pooled}%")
    ax.axhline(float(macro), color="#f59e0b", ls="--", label=f"macro {macro}%")
    ax.set_xlabel("Fabric ID"); ax.set_ylabel("Pixel Dice (%)"); ax.set_ylim(0, 100); ax.legend()
    ax.set_title("Per-fabric Dice, Stage 4 (test split)")
    fig.tight_layout(); fig.savefig(OUT / "fig_fabric_wise_dice.png", dpi=160); plt.close(fig)


def _first_per_fabric(samples, defective):
    seen, out = set(), []
    for s in sorted(samples, key=lambda s: s["sample_id"]):
        if bool(s["is_defective"]) == defective and s["fabric_id"] not in seen:
            seen.add(s["fabric_id"]); out.append(s)
    return out


def fig_gallery(samples, defective, name):
    sel = _first_per_fabric(samples, defective)
    fig, axes = plt.subplots(len(sel), 2, figsize=(14, 2.3 * len(sel)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 1.2]})
    for r, s in enumerate(sel):
        img = load_image_gray(s["image_path"])
        gt = load_mask_binary(s["mask_path"]).astype(bool) if defective else np.zeros(img.shape, bool)
        pr = _pred_mask(s["sample_id"])
        flagged = int(pr.sum()) >= SCREEN_PX
        axes[r, 0].imshow(img, cmap="gray", aspect="auto")
        if gt.any():
            _outline(axes[r, 0], gt, "lime", slice(0, img.shape[1]))
        if pr.any():
            _outline(axes[r, 0], pr, "yellow", slice(0, img.shape[1]))
        axes[r, 0].set_title(f"{s['sample_id']}  fabric {s['fabric_id']}  -> "
                             f"{'FLAGGED' if flagged else 'not flagged'} ({int(pr.sum())} px)", fontsize=9, loc="left")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        sl = _crop(img, gt)
        axes[r, 1].imshow(img[:, sl], cmap="gray", aspect="auto")
        if gt.any():
            _outline(axes[r, 1], gt, "lime", sl)
        if pr.any():
            _outline(axes[r, 1], pr, "yellow", sl)
        axes[r, 1].set_title("zoom (green = ground truth, yellow = prediction)", fontsize=8, loc="left")
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig)


def fig_heatmaps(samples):
    sel = _first_per_fabric(samples, True)
    fig, axes = plt.subplots(len(sel), 1, figsize=(14, 1.9 * len(sel)), squeeze=False)
    for r, s in enumerate(sel):
        im = cv2.imread(f"outputs/overlays/deviations/{s['sample_id']}_deviation.png")[..., ::-1]
        gt = load_mask_binary(s["mask_path"]).astype(bool)
        axes[r, 0].imshow(im, aspect="auto")
        _outline(axes[r, 0], gt, "white", slice(0, im.shape[1]))
        axes[r, 0].set_title(f"{s['sample_id']}  anomaly-ratio heat map (white = ground truth)", fontsize=9, loc="left")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "fig_heatmap_gallery.png", dpi=130); plt.close(fig)


def fig_taxonomy(samples):
    firsts = {}
    for s in sorted(samples, key=lambda s: s["sample_id"]):
        if s["is_defective"]:
            c = CODE_TO_CLASS.get(s["sample_id"].split("_")[1])
            if c is not None and c not in firsts:
                firsts[c] = s["sample_id"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for c in range(4):
        ax = axes[c]; ax.axis("off")
        sid = firsts.get(c)
        files = sorted(Path("outputs/overlays/taxonomy").glob(f"{sid}_*_taxonomy.png")) if sid else []
        if files:
            ax.imshow(cv2.imread(str(files[0]))[..., ::-1])
        ax.set_title(f"[{c}] {CLASS_NAMES[c]}\n{sid}", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_taxonomy_showcase.png", dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p3 = json.load(open("outputs/metrics/phase3_integration_benchmark.json"))
    test = load_split(Path("data/splits_oneclass"), "test")
    fig_midsem_vs_endsem(p3)
    fig_fabric_dice()
    fig_gallery(test, True, "fig_detection_gallery_defective.png")
    fig_gallery(test, False, "fig_clean_roll_gallery.png")
    fig_heatmaps(test)
    fig_taxonomy(test)
    print("presentation figures written to", OUT)


if __name__ == "__main__":
    main()
