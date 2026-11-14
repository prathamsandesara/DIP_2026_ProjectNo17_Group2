"""
Dataset integrity validation: used by `scripts/inspect_dataset.py` and
`scripts/prepare_dataset.py` to detect problems before any processing runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.parser import Sample
from src.utils.io import load_image_gray


@dataclass
class ValidationReport:
    total_samples: int = 0
    defect_free_count: int = 0
    defective_count: int = 0
    missing_images: List[str] = field(default_factory=list)
    missing_masks: List[str] = field(default_factory=list)
    unreadable_files: List[str] = field(default_factory=list)
    image_dimensions: Dict[str, int] = field(default_factory=dict)  # "HxW" -> count
    fabric_type_counts: Dict[str, int] = field(default_factory=dict)
    defect_type_counts: Dict[str, int] = field(default_factory=dict)

    def is_clean(self) -> bool:
        return not (self.missing_images or self.unreadable_files)

    def summary(self) -> str:
        lines = [
            f"Total samples            : {self.total_samples}",
            f"Defect-free images       : {self.defect_free_count}",
            f"Defective images         : {self.defective_count}",
            f"Missing image files      : {len(self.missing_images)}",
            f"Missing mask files (def.): {len(self.missing_masks)}",
            f"Unreadable files         : {len(self.unreadable_files)}",
            f"Distinct fabric types    : {len(self.fabric_type_counts)}",
            f"Distinct defect types    : {len(self.defect_type_counts)}",
            f"Image dimensions seen    : {self.image_dimensions}",
        ]
        return "\n".join(lines)


def validate_samples(samples: List[Sample], check_readable: bool = True) -> ValidationReport:
    report = ValidationReport(total_samples=len(samples))

    for s in samples:
        if s.is_defective:
            report.defective_count += 1
            report.defect_type_counts[s.defect_type] = report.defect_type_counts.get(s.defect_type, 0) + 1
        else:
            report.defect_free_count += 1

        report.fabric_type_counts[s.fabric_id] = report.fabric_type_counts.get(s.fabric_id, 0) + 1

        if not Path(s.image_path).exists():
            report.missing_images.append(str(s.image_path))
            continue

        if s.is_defective and (s.mask_path is None or not Path(s.mask_path).exists()):
            report.missing_masks.append(str(s.image_path))

        if check_readable:
            try:
                img = load_image_gray(s.image_path)
                dims_key = f"{img.shape[0]}x{img.shape[1]}"
                report.image_dimensions[dims_key] = report.image_dimensions.get(dims_key, 0) + 1
                if not np.isfinite(img).all():
                    report.unreadable_files.append(str(s.image_path))
            except Exception:
                report.unreadable_files.append(str(s.image_path))

    return report
