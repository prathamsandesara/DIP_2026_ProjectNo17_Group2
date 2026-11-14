"""
Verify the frozen one-class split in data/splits_oneclass/ and write outputs/tables/split_summary.csv.

The split files are the canonical split used by every benchmark. Protocol:
  * TRAIN  : defect-free rolls only (no defective roll is ever used to fit or calibrate anything).
  * VAL/TEST: the remaining defect-free rolls plus ALL defective rolls, divided roughly half/half and
              stratified by (fabric, defect code).
  * The defect-free assignment is reproducible: it equals `stratified_split(samples, seed=42)`.
    (The script that divided the defective rolls between VAL and TEST is not part of the repository, so
    that half of the split is a frozen artifact; this script checks its invariants instead.)

Usage: PYTHONPATH=code python code/scripts/verify_splits.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.data.parser import AitexParser
from src.data.splitter import stratified_split

SPLIT_DIR = Path("data/splits_oneclass")


def main() -> None:
    splits = {n: json.load(open(SPLIT_DIR / f"{n}.json"))["samples"] for n in ("train", "val", "test")}
    ids = {n: {s["sample_id"] for s in v} for n, v in splits.items()}
    checks = {
        "train/val disjoint": not (ids["train"] & ids["val"]),
        "train/test disjoint": not (ids["train"] & ids["test"]),
        "val/test disjoint": not (ids["val"] & ids["test"]),
        "train has no defective roll": not any(s["is_defective"] for s in splits["train"]),
        "every defective TEST roll has a mask": all(s["mask_path"] for s in splits["test"] if s["is_defective"]),
        # AITEX ships one defective image of an extra fabric ("08") without a mask; it is only in VAL and the
        # evaluation excludes fabric 08 (see `primary_results` in src/pipeline.py).
        "only mask-less defective roll is fabric 08": {s["sample_id"] for n in ("val", "test") for s in splits[n]
                                                        if s["is_defective"] and not s["mask_path"]} <= {"0100_025_08"},
    }
    raw = Path("data/raw")
    if (raw / "Defect_images").exists():
        samples = AitexParser(raw).parse()
        ref = stratified_split(samples, random_seed=42)
        clean = {s.sample_id for s in samples if not s.is_defective}
        for n in ("train", "val", "test"):
            checks[f"defect-free {n} == stratified_split(seed=42)"] = (
                {s.sample_id for s in ref[n] if not s.is_defective} == (ids[n] & clean))
        all_def = {s.sample_id for s in samples if s.is_defective}
        checks["all 106 defective rolls are in val or test"] = all_def == ((ids["val"] | ids["test"]) & all_def) and len(all_def) == 106
    rows = []
    for n, v in splits.items():
        for fid in sorted({s["fabric_id"] for s in v}):
            c = Counter(s["is_defective"] for s in v if s["fabric_id"] == fid)
            rows.append([n, fid, c[False], c[True]])
    out = Path("outputs/tables"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "split_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "fabric", "clean_rolls", "defective_rolls"]); w.writerows(rows)
    for k, v in checks.items():
        print(("PASS  " if v else "FAIL  ") + k)
    for n, v in splits.items():
        print(f"{n}: {len(v)} rolls ({sum(s['is_defective'] for s in v)} defective)")
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
