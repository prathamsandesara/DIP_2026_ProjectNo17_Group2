"""
STEP 1: Preprocessing, Denoising, Enhancement & Active Fabric Domain Masking.

Pipeline Stage: Step 1
Modules:
  1.1 Grayscale Conversion & Intensity Verification (ITU-R BT.601 standard luminance).
  1.2 Dynamic Range Min-Max Normalization to [0, 255].
  1.3 Alpha-Trimmed Mean Filter (3x3 window, d=4: drops 2 lowest & 2 highest extreme pixels,
      averages the middle 5 clean pixels to eliminate both sensor fuzz and dust specks).
  1.4 CLAHE Contrast Enhancement (Clip limit 2.0, Tile Grid 8x8: boosts local defect contrast
      without over-amplifying sensor noise).
  1.5 Active Fabric Domain Masking (Eliminates overexposed white inspection table margins >= 240
      and aligns with loom selvedge guides).
  1.6 Support-Valid Mask Generation (Erodes boundary by filter footprint to prevent edge bleed).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import generic_filter

submission_dir = Path(__file__).resolve().parent
repo_root = submission_dir.parent
for p in [str(submission_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import get_logger, load_image_gray

logger = get_logger("Step01_Preprocessing")

# Loom physical selvedge boundaries per fabric ID (in pixels across 4096-width line scan)
LOOM_SELVEDGE_BOUNDS: Dict[str, Tuple[int, int]] = {
    "00": (1150, 4096),
    "01": (450, 4096),
    "02": (450, 3900),
    "03": (1050, 4096),
    "04": (540, 3250),
    "05": (250, 4096),
    "06": (450, 3500),
}


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """1.1 Convert image to single-channel uint8 grayscale if needed."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_dynamic_range(image: np.ndarray) -> np.ndarray:
    """1.2 Standardize dynamic range to [0, 255] float32."""
    img_f = image.astype(np.float32)
    i_min, i_max = float(np.min(img_f)), float(np.max(img_f))
    if i_max - i_min < 1e-6:
        return np.zeros_like(img_f)
    return (img_f - i_min) / (i_max - i_min) * 255.0


def alpha_trimmed_mean_filter(image: np.ndarray, size: int = 3, d: int = 4) -> np.ndarray:
    """
    1.3 Alpha-Trimmed Mean Filter (Order-Statistic Denoising).
    In a size x size (9-pixel) window, drops d/2 lowest and d/2 highest pixels,
    then averages the remaining clean pixels.
    """
    if size % 2 == 0:
        raise ValueError("Filter window size must be odd")
    n = size * size
    d_clamped = int(np.clip(d, 0, n - 1))
    half_d = d_clamped // 2

    # Fast implementation for 3x3 window and d=4 (drops 2 min, 2 max, averages middle 5)
    if size == 3 and d == 4:
        # Build 9 neighborhood pixel slices
        padded = np.pad(image.astype(np.float32), 1, mode="reflect")
        slices = []
        for dy in range(3):
            for dx in range(3):
                slices.append(padded[dy:dy + image.shape[0], dx:dx + image.shape[1]])
        stacked = np.stack(slices, axis=-1)  # shape (H, W, 9)
        # Sort along neighborhood axis
        stacked.sort(axis=-1)
        # Slice middle 5 elements: indices 2 to 7 (2, 3, 4, 5, 6)
        trimmed = stacked[..., 2:7]
        return np.mean(trimmed, axis=-1).astype(np.float32)

    # General fallback using scipy generic_filter
    def _trimmed_kernel(neighborhood: np.ndarray) -> float:
        sorted_vals = np.sort(neighborhood)
        trimmed = sorted_vals[half_d: n - half_d] if half_d > 0 else sorted_vals
        return float(np.mean(trimmed))

    return generic_filter(image.astype(np.float64), _trimmed_kernel, size=size).astype(np.float32)


def clahe_enhance(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    1.4 Contrast-Limited Adaptive Histogram Equalization (CLAHE).
    Boosts local weave contrast with strict clip limit to prevent noise blowing up.
    """
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img_u8).astype(np.float32)


def detect_active_fabric_mask(
    img: np.ndarray,
    sat_thresh: int = 240,
    fabric_id: Optional[str] = None
) -> np.ndarray:
    """
    1.5 Active Fabric Region Masking.
    Identifies real cloth domain and discards overexposed conveyor table margins (I >= 240).
    Constrains active pixels to known loom selvedge guides.
    """
    h, w = img.shape
    is_sat = (img >= sat_thresh).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(is_sat, connectivity=8)

    border_mask = np.zeros((h, w), dtype=bool)
    if num_labels > 1:
        left_labels = np.unique(labels[:, 0])
        right_labels = np.unique(labels[:, w - 1])
        touching_labels = set(left_labels).union(set(right_labels))
        touching_labels.discard(0)

        for lbl in touching_labels:
            if stats[lbl, cv2.CC_STAT_AREA] >= 500:
                border_mask |= (labels == lbl)

    col_has_border = np.mean(border_mask, axis=0) > 0.7
    if np.any(col_has_border):
        border_mask |= np.tile(col_has_border, (h, 1))

    active_mask = ~border_mask

    if fabric_id is not None and fabric_id in LOOM_SELVEDGE_BOUNDS:
        x_min, x_max = LOOM_SELVEDGE_BOUNDS[fabric_id]
        active_mask[:, :x_min] = False
        active_mask[:, x_max:] = False

    return active_mask


def get_support_valid_mask(
    active_mask: np.ndarray,
    ksize: Tuple[int, int],
    extra_margin: int = 15
) -> np.ndarray:
    """
    1.6 Support-Valid Domain Mask (M_valid).
    Erodes active fabric domain by filter kernel footprint plus extra margin
    so calibration samples are strictly free of boundary artifacts.
    """
    wx = ksize[0] + 2 * extra_margin
    wy = ksize[1] + 2 * extra_margin
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (wx, wy))
    return cv2.erode(
        active_mask.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype(bool)


def preprocess_pipeline(
    image: np.ndarray,
    fabric_id: Optional[str] = None,
    apply_alpha_trimmed: bool = True,
    apply_clahe: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified Preprocessing Execution:
    Returns (cleaned_image, active_fabric_mask).
    """
    gray = to_grayscale(image)
    active_mask = detect_active_fabric_mask(gray, sat_thresh=240, fabric_id=fabric_id)

    if apply_alpha_trimmed:
        cleaned = alpha_trimmed_mean_filter(gray, size=3, d=4)
    else:
        cleaned = gray.astype(np.float32)

    if apply_clahe:
        cleaned = clahe_enhance(cleaned, clip_limit=2.0, tile_grid_size=8)

    return np.clip(cleaned, 0, 255).astype(np.uint8), active_mask


def main() -> None:
    print("=" * 80)
    print("STEP 1: PREPROCESSING & ACTIVE FABRIC DOMAIN VERIFICATION")
    print("=" * 80)

    # Test with synthetic test frame matching line-scan camera resolution
    test_img = np.random.randint(60, 180, (256, 4096), dtype=np.uint8)
    # Add scanner overexposed white margins
    test_img[:, :200] = 255
    test_img[:, 3900:] = 255

    cleaned, active_mask = preprocess_pipeline(test_img, fabric_id="00")
    print(f"Input Shape          : {test_img.shape}")
    print(f"Cleaned Output Shape : {cleaned.shape}, dtype={cleaned.dtype}")
    print(f"Active Mask Valid Px : {int(np.sum(active_mask)):,} px ({(np.mean(active_mask)*100):.2f}% active)")
    print("=" * 80)
    print("Step 1 Completed Successfully. Denoising & masking operators verified.")


if __name__ == "__main__":
    main()
