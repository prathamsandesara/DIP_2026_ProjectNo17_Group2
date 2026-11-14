#!/usr/bin/env python
"""
Active Fabric Region Detection & Validation.

Algorithm:
1. Detect boundary-connected saturated white scanner margins (I >= 240).
2. Intersect with physical loom selvedge guides for line-scan cameras.
3. Compute support-valid masks eroded by filter footprints to avoid edge artifacts.
"""
from __future__ import annotations

from typing import Tuple, Dict, Optional
import cv2
import numpy as np

LOOM_SELVEDGE_BOUNDS: Dict[str, Tuple[int, int]] = {
    "00": (1150, 4096),
    "01": (450, 4096),
    "02": (450, 3900),
    "03": (1050, 4096),
    "04": (540, 3250),
    "05": (250, 4096),
    "06": (450, 3500),
}


def detect_active_fabric_mask(img: np.ndarray, sat_thresh: int = 240,
                               fabric_id: Optional[str] = None) -> np.ndarray:
    """
    Detect the active fabric region by segmenting boundary-connected scanner margins
    and constraining to the known physical loom selvedge guides.
    Returns: boolean mask of shape (H, W) where True indicates active fabric.
    """
    h, w = img.shape
    is_sat = (img >= sat_thresh).astype(np.uint8)

    # Connected components on saturated pixels with 8-connectivity
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(is_sat, connectivity=8)

    border_mask = np.zeros((h, w), dtype=bool)

    if num_labels > 1:
        # Check which components touch left (x=0) or right (x=w-1) borders
        left_labels = np.unique(labels[:, 0])
        right_labels = np.unique(labels[:, w - 1])
        touching_labels = set(left_labels).union(set(right_labels))
        touching_labels.discard(0)  # 0 is background (non-saturated)

        for lbl in touching_labels:
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= 500:
                border_mask |= (labels == lbl)

    # Clean border mask: fill small holes along vertical columns
    col_has_border = np.mean(border_mask, axis=0) > 0.7
    if np.any(col_has_border):
        solid_border = np.tile(col_has_border, (h, 1))
        border_mask |= solid_border

    active_mask = ~border_mask

    # Intersect with physical loom selvedge bounds if fabric_id is known
    if fabric_id is not None and fabric_id in LOOM_SELVEDGE_BOUNDS:
        x_min, x_max = LOOM_SELVEDGE_BOUNDS[fabric_id]
        active_mask[:, :x_min] = False
        active_mask[:, x_max:] = False

    return active_mask


def get_support_valid_mask(active_mask: np.ndarray, ksize: Tuple[int, int],
                           extra_margin: int = 0) -> np.ndarray:
    """
    Erode active fabric mask by filter kernel size to avoid filter border overlap artifacts.
    """
    wx = ksize[0] + 2 * extra_margin
    wy = ksize[1] + 2 * extra_margin
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (wx, wy))
    valid_mask = cv2.erode(
        active_mask.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype(bool)
    return valid_mask
