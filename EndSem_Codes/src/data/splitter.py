"""
Leakage-safe train/validation/test splitting.

Design goals (see METHODOLOGY.md / DATASET.md for the full rationale):

1. The TEST set's ground-truth masks must never influence any learned
   parameter (adaptive scale selection, normal-texture statistics,
   threshold selection, feature normalization). Those are all fitted on
   TRAIN (and validated on VAL) only.
2. Splitting is stratified by fabric type so every fabric type is
   represented in train/val/test where possible.
3. Splitting is stratified by defective/defect-free status (and, within
   the defective set, by defect type) so rare defect types are not
   accidentally excluded from evaluation.
4. The TRAIN split used for "normal texture" modeling and adaptive
   parameter estimation is restricted to DEFECT-FREE images only --
   defective images (even in the train split) are never used to estimate
   what "normal" texture looks like.

The split is deterministic given a fixed `random_seed`, and is written to
`data/splits/*.json` so every script re-loads the same split rather than
re-randomizing it.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from src.data.parser import Sample
from src.utils.io import save_json


def _stratum_key(s: Sample) -> str:
    if not s.is_defective:
        return f"{s.fabric_id}__defect_free"
    return f"{s.fabric_id}__{s.defect_type}"


def stratified_split(
    samples: List[Sample],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, List[Sample]]:
    """Stratify by (fabric_id, defect_status/defect_type) and split each
    stratum independently so every group is proportionally represented.

    A stratum with fewer than 3 samples is assigned entirely to TRAIN if
    defect-free (so normal-texture modeling has more data), or entirely to
    TEST if defective (so rare defect types are still evaluated at least
    once), since it cannot be meaningfully split three ways.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1"

    rng = random.Random(random_seed)
    strata: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        strata[_stratum_key(s)].append(s)

    split: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}

    for key, group in strata.items():
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            if group and group[0].is_defective:
                split["test"].extend(group)
            else:
                split["train"].extend(group)
            continue

        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = min(n_train, n - 2)  # leave at least 1 for val and 1 for test
        n_val = min(n_val, n - n_train - 1)

        split["train"].extend(group[:n_train])
        split["val"].extend(group[n_train:n_train + n_val])
        split["test"].extend(group[n_train + n_val:])

    return split


def save_split(split: Dict[str, List[Sample]], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    for name, group in split.items():
        records = [
            {
                "image_path": str(s.image_path),
                "mask_path": str(s.mask_path) if s.mask_path else None,
                "fabric_id": s.fabric_id,
                "defect_type": s.defect_type,
                "is_defective": s.is_defective,
                "sample_id": s.sample_id,
            }
            for s in group
        ]
        save_json(out_dir / f"{name}.json", {"count": len(records), "samples": records})


def load_split(split_dir: Path, name: str) -> List[Dict]:
    from src.utils.io import load_json

    data = load_json(Path(split_dir) / f"{name}.json")
    return data["samples"]


def normal_texture_training_samples(split: Dict[str, List[Sample]]) -> List[Sample]:
    """Defect-free samples from TRAIN ONLY -- the exact set that may be
    used to estimate normal-texture statistics and adaptive parameters.
    Never includes val/test samples, preventing leakage by construction.
    """
    return [s for s in split["train"] if not s.is_defective]
