"""
Full-roll evaluation of the classical texture-descriptor detector
(LBP, GLCM, Gabor and their fusion) on the same 81 held-out test rolls and
with the same screening rule as the dual-scale morphological pipeline.

Outputs:
  - outputs/metrics/texture_pipeline_benchmark.json
  - outputs/tables/texture_pipeline_table.csv

Usage:  PYTHONPATH=code python code/scripts/run_texture_descriptor_pipeline.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from src.data.splitter import load_split
from src.detection.texture_pipeline import CONFIGS, DEFAULT_SCALE, TextureDescriptorPipeline

LABELS = {
    "lbp": "LBP only (uniform, P=16, R=2)",
    "glcm": "GLCM only (Haralick features, 17x17 window)",
    "gabor": "Gabor only (4 orientations, f=0.15)",
    "fused": "LBP + GLCM + Gabor fused (equal weights)",
}


def main() -> None:
    splits = Path("data/splits_oneclass")
    train, test = load_split(splits, "train"), load_split(splits, "test")
    pipe = TextureDescriptorPipeline()
    t0 = time.perf_counter()
    pipe.fit(train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = pipe.evaluate(test)
    eval_s = time.perf_counter() - t0
    out_dir = Path("outputs")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "fit on 85 clean train rolls only; one uniform screening_min_px=15; "
                    "threshold = 99.99th percentile of per-fabric train deviation",
        "scale": DEFAULT_SCALE.__dict__, "fit_time_s": round(fit_s, 1),
        "eval_time_s": round(eval_s, 1), "configs": res,
    }
    (out_dir / "metrics" / "texture_pipeline_benchmark.json").write_text(json.dumps(payload, indent=2, default=float))
    with open(out_dir / "tables" / "texture_pipeline_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Texture-descriptor detector", "Accuracy (%)", "Recall (%)", "Precision (%)",
                    "Specificity (%)", "Pooled Dice (%)", "Macro Dice (%)", "TP", "FN", "FP", "TN"])
        for name in CONFIGS:
            r = res[name]
            w.writerow([LABELS[name], f"{r['accuracy_pct']:.2f}", f"{r['recall_pct']:.2f}",
                        f"{r['precision_pct']:.2f}", f"{r['specificity_pct']:.2f}",
                        f"{r['pooled_dice_pct']:.2f}", f"{r['macro_dice_pct']:.2f}",
                        r["tp"], r["fn"], r["fp"], r["tn"]])
            print(f"{LABELS[name]:<48} acc {r['accuracy_pct']:.1f}  rec {r['recall_pct']:.1f}  "
                  f"spec {r['specificity_pct']:.1f}  dice {r['pooled_dice_pct']:.2f}  "
                  f"({r['tp']}/{r['fn']}/{r['fp']}/{r['tn']})")
    print(f"fit {fit_s:.0f}s, eval {eval_s:.0f}s")


if __name__ == "__main__":
    main()
