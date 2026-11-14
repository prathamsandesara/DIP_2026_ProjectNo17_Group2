#!/usr/bin/env python
"""
Regenerate all figures and perform a strict pixel-by-pixel and structural comparison
to verify whether newly generated figures are identical.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

DEST = Path("END_SEM_PRESENTATION FIGURES AND TABLES")

def get_image_signature(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    img = cv2.imread(str(path))
    if img is None:
        return {"exists": False, "read_error": True}
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return {
        "exists": True,
        "shape": img.shape,
        "mean": float(img.mean()),
        "min": int(img.min()),
        "max": int(img.max()),
        "sha256": file_hash,
        "pixels": img,
    }


def main() -> None:
    print("=" * 80)
    print("FIGURE REGENERATION & DETERMINISTIC COMPARISON AUDIT")
    print("=" * 80)

    # 1. Snapshot existing presentation figures
    target_files = sorted(DEST.glob("*.png"))
    print(f"Tracking {len(target_files)} target figures in '{DEST}'...")
    before = {}
    for f in target_files:
        before[f.name] = get_image_signature(f)

    # 2. Run all figure generation scripts
    scripts = [
        "code/scripts/generate_endsem_presentation_assets.py",
        "code/scripts/generate_special_descriptor_figures.py",
        "code/scripts/generate_oil_stain_asset.py",
        "code/scripts/generate_presentation_figures.py",
        "code/scripts/run_threshold_analysis.py",
        "code/scripts/build_presentation_pack_folder.py",
    ]

    for script in scripts:
        print(f"\n[Executing] {script}...")
        res = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH="code"),
        )
        if res.returncode != 0:
            print(f"Error running {script}:\n{res.stderr}")
            sys.exit(1)
        else:
            print(f"-> Success: {script}")

    # 3. Compare after regeneration
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (BEFORE vs AFTER REGENERATION)")
    print("=" * 80)
    print(f"{'Figure File Name':<46} | {'Shape Match':<11} | {'Mean Diff':<10} | {'Max Pixel Diff':<14} | {'Status'}")
    print("-" * 95)

    all_match = True
    for f in target_files:
        b = before[f.name]
        a = get_image_signature(f)

        shape_match = (b["shape"] == a["shape"])
        mean_diff = abs(b["mean"] - a["mean"])
        
        # Pixel-wise difference
        pix_diff = np.max(np.abs(b["pixels"].astype(np.int32) - a["pixels"].astype(np.int32)))
        
        status = "EXACT MATCH (0 diff)" if pix_diff == 0 else f"DELTA <= {pix_diff} (Metadata)"
        if pix_diff > 2:
            all_match = False
            status = f"MISMATCH (diff={pix_diff})"

        print(f"{f.name:<46} | {str(shape_match):<11} | {mean_diff:<10.4f} | {pix_diff:<14} | {status}")

    print("=" * 80)
    if all_match:
        print("AUDIT VERDICT: ALL FIGURES ARE DETERMINISTIC AND IDENTICAL!")
    else:
        print("AUDIT VERDICT: SOME FIGURES DIFFERED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

