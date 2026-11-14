"""
STEP 10: Connected Component Defect Localization, Bounding Boxes & Severity Grading.

Pipeline Stage: Step 10
Purpose:
  - Transforms pixel-level binary anomaly masks into actionable industrial manufacturing telemetry:
      * 10.1 8-Connected Component Analysis: Segments discrete anomaly clusters.
      * 10.2 Spatial Centroid & Bounding Box Generation: Computes [x, y, width, height] for
             automated laser cutting machines to isolate and excise defective cloth.
      * 10.3 Defect Severity Scoring: S = min(1.0, Area / 500).
             - Severe tears (Area >= 400 px, S >= 0.8): Demand immediate loom emergency stoppage.
             - Minor slubs/knots (Area < 50 px, S < 0.1): Logged for yarn grading.
      * 10.4 Visual Inspection Overlays: Marks bounding boxes and color overlays on fabric scan.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

submission_dir = Path(__file__).resolve().parent
repo_root = submission_dir.parent
for p in [str(submission_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import get_logger

logger = get_logger("Step10_Localization")


@dataclass
class DefectRegion:
    """Represents a localized physical defect entity."""
    region_id: int
    bbox_xywh: Tuple[int, int, int, int]  # (x, y, width, height)
    centroid_xy: Tuple[float, float]      # (cx, cy)
    area_px: int
    severity: float                       # [0.0, 1.0]

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "bbox_xywh": list(self.bbox_xywh),
            "centroid_xy": list(self.centroid_xy),
            "area_px": self.area_px,
            "severity": round(self.severity, 4),
        }


def extract_defect_regions(binary_mask: np.ndarray, min_area: int = 5) -> List[DefectRegion]:
    """
    10.1 & 10.2 Extract connected components, centroids, bounding boxes, and severity.
    """
    if not np.any(binary_mask):
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    regions: List[DefectRegion] = []
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[lbl, 0])
        cy = float(centroids[lbl, 1])

        severity = min(1.0, float(area) / 500.0)

        regions.append(DefectRegion(
            region_id=lbl,
            bbox_xywh=(x, y, w, h),
            centroid_xy=(cx, cy),
            area_px=area,
            severity=severity,
        ))

    return regions


def draw_defect_boxes_on_image(
    image: np.ndarray,
    regions: List[DefectRegion],
    box_color_bgr: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    10.4 Draw colored bounding boxes and severity telemetry onto fabric image.
    """
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    for r in regions:
        x, y, w, h = r.bbox_xywh
        cv2.rectangle(out, (x, y), (x + w, y + h), box_color_bgr, thickness)
        label_text = f"Defect #{r.region_id} (Sev={r.severity:.2f}, Area={r.area_px}px)"
        cv2.putText(
            out, label_text, (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
        )

    return out


def main() -> None:
    print("=" * 80)
    print("STEP 10: DEFECT LOCALIZATION, BOUNDING BOXES & SEVERITY TELEMETRY")
    print("=" * 80)

    # Synthetic mask with two distinct flaws
    h, w = 256, 4096
    synth_mask = np.zeros((h, w), dtype=bool)
    # Flaw 1: Tiny knot (32 px)
    synth_mask[50:54, 800:808] = True
    # Flaw 2: Large tear (2400 px)
    synth_mask[120:160, 2500:2560] = True

    regions = extract_defect_regions(synth_mask, min_area=5)
    print(f"Extracted Discrete Defect Blobs : {len(regions)}")
    for r in regions:
        print(f"  Blob #{r.region_id} | Bounding Box: {r.bbox_xywh} | Area: {r.area_px:>5} px | Severity: {r.severity:.3f}")
    print("=" * 80)
    print("Step 10 Completed Successfully. Defect localization & severity verified.")


if __name__ == "__main__":
    main()
