"""
Classical Texture Descriptor Ablation: LBP vs. GLCM vs. Gabor
ECE501 Digital Image Processing - End-Semester Project

Empirically benchmarks the classical texture descriptor families on AITEX fabric patches:
1. All Three Fused: LBP (1/3) + GLCM (1/3) + Gabor (1/3)
2. LBP Only (w_lbp = 1.0, others = 0.0)
3. GLCM Only (w_glcm = 1.0, others = 0.0)
4. Gabor Only (w_gabor = 1.0, others = 0.0)

Measures:
- Defect Contrast SNR: (mean_defect - mean_bg) / std_bg
- Patch-Level Segmentation: Recall (%), Precision (%), Mean Dice (%)
- Feature Extraction Latency per tile (ms)
- Calculated Full Roll Estimate (tile ms x 16 tiles)

Exports:
- outputs/metrics/descriptor_ablation_benchmark.json
- outputs/tables/descriptor_ablation_table.csv
"""

import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List
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
from src.texture.lbp import compute_lbp, local_lbp_histograms, lbp_map_to_pixelwise_feature
from src.texture.glcm import local_glcm_feature_maps
from src.texture.gabor import build_gabor_bank, apply_gabor_bank
from src.detection.deviation_map import normalize_deviation


def benchmark_descriptor_ablation(num_patches: int = 20):
    splits_dir = Path("data/splits_oneclass")
    test_samples = load_split(splits_dir, "test")
    defective_samples = [s for s in test_samples if s["is_defective"] and s.get("mask_path")][:num_patches]

    print(f"Benchmarking LBP, GLCM, and Gabor on {len(defective_samples)} representative defective test patches...")

    lbp_times, glcm_times, gabor_times = [], [], []
    lbp_snr_list, glcm_snr_list, gabor_snr_list, fused_snr_list = [], [], [], []
    lbp_dices, glcm_dices, gabor_dices, fused_dices = [], [], [], []
    lbp_recs, glcm_recs, gabor_recs, fused_recs = [], [], [], []
    lbp_precs, glcm_precs, gabor_precs, fused_precs = [], [], [], []

    for idx, s in enumerate(defective_samples):
        img = load_image_gray(s["image_path"])
        mask = load_mask_binary(s["mask_path"]).astype(bool)

        # Extract 256 x 256 patch centered on defect
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        x0 = max(0, min(img.shape[1] - 256, cx - 128))
        y0 = max(0, min(img.shape[0] - 256, cy - 128))

        p_img = img[y0:y0 + 256, x0:x0 + 256]
        p_mask = mask[y0:y0 + 256, x0:x0 + 256]

        if not np.any(p_mask) or np.all(p_mask):
            continue

        p_u8 = np.clip(p_img, 0, 255).astype(np.uint8)

        # 1. LBP Extraction
        t0 = time.perf_counter()
        lbp_map = compute_lbp(p_u8, radius=1, n_points=8, method="uniform")
        hists, centers = local_lbp_histograms(lbp_map, window_size=16, stride=8, n_points=8)
        lbp_feat = lbp_map_to_pixelwise_feature(hists, centers, p_u8.shape, 16)
        lbp_energy = np.mean(np.abs(lbp_feat - np.mean(lbp_feat, axis=(0, 1), keepdims=True)), axis=-1)
        t_lbp = (time.perf_counter() - t0) * 1000.0
        lbp_times.append(t_lbp)

        # 2. GLCM Extraction
        t0 = time.perf_counter()
        glcm_maps, _ = local_glcm_feature_maps(p_u8, window_size=17, distances=(1,), angles=(0, np.pi/2), n_levels=16)
        glcm_contrast = glcm_maps.get("contrast", np.zeros_like(p_u8, dtype=np.float32))
        t_glcm = (time.perf_counter() - t0) * 1000.0
        glcm_times.append(t_glcm)

        # 3. Gabor Filter Bank Extraction
        t0 = time.perf_counter()
        bank = build_gabor_bank(frequencies=[0.1, 0.2], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4])
        gabor_res = apply_gabor_bank(p_u8, bank)
        gabor_mag = gabor_res["energy"]
        t_gabor = (time.perf_counter() - t0) * 1000.0
        gabor_times.append(t_gabor)

        # Normalize features
        n_lbp = normalize_deviation(lbp_energy)
        n_glcm = normalize_deviation(glcm_contrast)
        n_gabor = normalize_deviation(gabor_mag)

        # Fused combination (1/3 each)
        n_fused = (n_lbp + n_glcm + n_gabor) / 3.0

        # Contrast SNR = (mean_defect - mean_background) / std_background
        def calc_snr(feat_map, m):
            bg = ~m
            if not np.any(m) or not np.any(bg):
                return 0.0
            sig_bg = np.std(feat_map[bg])
            if sig_bg < 1e-6:
                return 0.0
            return float((np.mean(feat_map[m]) - np.mean(feat_map[bg])) / sig_bg)

        # Patch segmentation metrics via adaptive thresholding at median + 1.5 * std
        def calc_seg(feat_map, m):
            th = np.median(feat_map) + 1.5 * np.std(feat_map)
            pred = (feat_map > th)
            inter = np.sum(pred & m)
            pr_s = np.sum(pred)
            gt_s = np.sum(m)
            dice = (2.0 * inter) / (pr_s + gt_s) if (pr_s + gt_s) > 0 else 0.0
            rec = inter / gt_s if gt_s > 0 else 0.0
            prec = inter / pr_s if pr_s > 0 else 0.0
            return dice * 100.0, rec * 100.0, prec * 100.0

        lbp_snr_list.append(calc_snr(n_lbp, p_mask))
        glcm_snr_list.append(calc_snr(n_glcm, p_mask))
        gabor_snr_list.append(calc_snr(n_gabor, p_mask))
        fused_snr_list.append(calc_snr(n_fused, p_mask))

        d, r, p = calc_seg(n_lbp, p_mask)
        lbp_dices.append(d); lbp_recs.append(r); lbp_precs.append(p)
        d, r, p = calc_seg(n_glcm, p_mask)
        glcm_dices.append(d); glcm_recs.append(r); glcm_precs.append(p)
        d, r, p = calc_seg(n_gabor, p_mask)
        gabor_dices.append(d); gabor_recs.append(r); gabor_precs.append(p)
        d, r, p = calc_seg(n_fused, p_mask)
        fused_dices.append(d); fused_recs.append(r); fused_precs.append(p)

    results = [
        {
            "descriptor": "1. All Three Fused (1/3 LBP, 1/3 GLCM, 1/3 Gabor)",
            "mean_snr": round(float(np.mean(fused_snr_list)), 2),
            "patch_dice_pct": round(float(np.mean(fused_dices)), 2),
            "patch_recall_pct": round(float(np.mean(fused_recs)), 2),
            "patch_precision_pct": round(float(np.mean(fused_precs)), 2),
            "extraction_time_ms_per_tile": round(float(np.mean(lbp_times) + np.mean(glcm_times) + np.mean(gabor_times)), 1),
            "calculated_full_roll_time_s": round(float((np.mean(lbp_times) + np.mean(glcm_times) + np.mean(gabor_times)) * 16 / 1000.0), 2),
            "primary_defect_sensitivity": "Balanced across micro & macro weave flaws",
            "course_dips_principles": "Complete statistical + frequency + structural synthesis"
        },
        {
            "descriptor": "2. LBP Only (Uniform P=8, R=1)",
            "mean_snr": round(float(np.mean(lbp_snr_list)), 2),
            "patch_dice_pct": round(float(np.mean(lbp_dices)), 2),
            "patch_recall_pct": round(float(np.mean(lbp_recs)), 2),
            "patch_precision_pct": round(float(np.mean(lbp_precs)), 2),
            "extraction_time_ms_per_tile": round(float(np.mean(lbp_times)), 1),
            "calculated_full_roll_time_s": round(float(np.mean(lbp_times) * 16 / 1000.0), 2),
            "primary_defect_sensitivity": "High sensitivity to sharp yarn breaks & thread holes",
            "course_dips_principles": "Micro-texture invariant to monotonic gray-level shifts"
        },
        {
            "descriptor": "3. GLCM Only (Co-occurrence Contrast)",
            "mean_snr": round(float(np.mean(glcm_snr_list)), 2),
            "patch_dice_pct": round(float(np.mean(glcm_dices)), 2),
            "patch_recall_pct": round(float(np.mean(glcm_recs)), 2),
            "patch_precision_pct": round(float(np.mean(glcm_precs)), 2),
            "extraction_time_ms_per_tile": round(float(np.mean(glcm_times)), 1),
            "calculated_full_roll_time_s": round(float(np.mean(glcm_times) * 16 / 1000.0), 2),
            "primary_defect_sensitivity": "Sensitive to local intensity transitions & weft cracks",
            "course_dips_principles": "Spatial spatial gray-level dependence matrix P(i,j; d, theta)"
        },
        {
            "descriptor": "4. Gabor Only (Multi-scale multi-orientation filter bank)",
            "mean_snr": round(float(np.mean(gabor_snr_list)), 2),
            "patch_dice_pct": round(float(np.mean(gabor_dices)), 2),
            "patch_recall_pct": round(float(np.mean(gabor_recs)), 2),
            "patch_precision_pct": round(float(np.mean(gabor_precs)), 2),
            "extraction_time_ms_per_tile": round(float(np.mean(gabor_times)), 1),
            "calculated_full_roll_time_s": round(float(np.mean(gabor_times) * 16 / 1000.0), 2),
            "primary_defect_sensitivity": "Directional selectivity to warp streaks & horizontal bars",
            "course_dips_principles": "Joint spatial / spatial-frequency optimal localization"
        },
    ]

    print("\n" + "=" * 105)
    print("CLASSICAL TEXTURE DESCRIPTOR ABLATION RESULTS (256x256 Patches):")
    for r in results:
        print(f"  {r['descriptor']:<55}: SNR={r['mean_snr']:>5.2f}, Dice={r['patch_dice_pct']:>5.2f}%, Rec={r['patch_recall_pct']:>5.2f}%, Prec={r['patch_precision_pct']:>5.2f}%, Latency={r['extraction_time_ms_per_tile']:>5.1f} ms/tile")
    print("=" * 105)

    # Save metrics JSON
    out_json = Path("outputs/metrics/descriptor_ablation_benchmark.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_json, {
        "num_patches_evaluated": len(lbp_snr_list),
        "patch_size": "256x256",
        "descriptors": results
    })
    print(f"Saved descriptor ablation JSON to: {out_json}")

    # Save CSV table
    out_csv = Path("outputs/tables/descriptor_ablation_table.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Descriptor Configuration", "Mean Contrast SNR", "Patch Dice (%)", "Patch Recall (%)", "Patch Precision (%)", "Latency (ms/tile)", "Calculated Full Roll Time (Tile ms x 16) (s)", "Primary Flaw Sensitivity", "DIP Foundations"])
        for r in results:
            writer.writerow([r["descriptor"], r["mean_snr"], r["patch_dice_pct"], r["patch_recall_pct"], r["patch_precision_pct"], r["extraction_time_ms_per_tile"], r["calculated_full_roll_time_s"], r["primary_defect_sensitivity"], r["course_dips_principles"]])
    print(f"Saved descriptor ablation CSV to: {out_csv}")


if __name__ == "__main__":
    benchmark_descriptor_ablation()
