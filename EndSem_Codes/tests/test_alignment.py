"""
Unit tests for fabric alignment correction (Radon line-integral de-skewing).
"""
import cv2
import numpy as np
import pytest
from pathlib import Path

from src.preprocessing.alignment import (
    estimate_fabric_skew_angle,
    align_fabric_image,
    align_mask,
    inject_synthetic_skew,
    extract_representative_patch,
    compute_projection_variance,
)
from src.utils.io import load_image_gray


def make_synthetic_woven_grid(h=256, w=512, warp_period=8, weft_period=8):
    """Generate a clean synthetic woven grid texture with vertical warp and horizontal weft."""
    x = np.arange(w)
    y = np.arange(h)
    warp = np.cos(2 * np.pi * x / warp_period)
    weft = np.cos(2 * np.pi * y / weft_period)
    grid = (warp[None, :] * 0.5 + weft[:, None] * 0.5 + 1.0) * 100.0 + 20.0
    return grid.astype(np.float32)


def test_synthetic_grid_zero_skew():
    grid = make_synthetic_woven_grid()
    res = estimate_fabric_skew_angle(grid, angle_range=(-5.0, 5.0), coarse_step=0.5, fine_step=0.05)
    assert abs(res["skew_angle"]) <= 0.1, f"Expected near 0 skew, got {res['skew_angle']}"
    assert res["confidence"] > 2.0


def test_synthetic_grid_known_skew():
    grid = make_synthetic_woven_grid(h=300, w=512)
    injected_angle = 3.5
    tilted = inject_synthetic_skew(grid, injected_angle)
    res = estimate_fabric_skew_angle(tilted, angle_range=(-8.0, 8.0), coarse_step=0.5, fine_step=0.05)
    # The estimated skew angle should match injected_angle
    err = abs(res["skew_angle"] - injected_angle)
    assert err <= 0.15, f"Expected {injected_angle}, got {res['skew_angle']} (err={err})"


def test_align_fabric_image_shape_and_dtype():
    img = np.random.randint(50, 200, (128, 256), dtype=np.uint8).astype(np.float32)
    aligned = align_fabric_image(img, 4.2)
    assert aligned.shape == img.shape
    assert aligned.dtype == img.dtype


def test_align_mask_boolean_preservation():
    mask = np.zeros((128, 256), dtype=bool)
    mask[40:60, 100:150] = True
    aligned = align_mask(mask, 5.0)
    assert aligned.shape == mask.shape
    assert aligned.dtype == bool
    # Region should still be present
    assert np.sum(aligned) > 0


def test_real_aitex_image_alignment():
    sample_path = Path("data/raw/NODefect_images/2311694-2040n7u/0001_000_00.png")
    if not sample_path.exists():
        pytest.skip("AITEX dataset not found")

    img = load_image_gray(sample_path)
    # Native unskewed image should be very close to 0 deg
    res_native = estimate_fabric_skew_angle(img, angle_range=(-6.0, 6.0), coarse_step=0.5, fine_step=0.05)
    assert abs(res_native["skew_angle"]) <= 0.2, f"Native skew angle unexpectedly large: {res_native['skew_angle']}"

    # Injected test on real scan: +4.0 deg
    tilted = inject_synthetic_skew(img, 4.0)
    res_tilted = estimate_fabric_skew_angle(tilted, angle_range=(-8.0, 8.0), coarse_step=0.5, fine_step=0.05)
    err = abs(res_tilted["skew_angle"] - 4.0)
    assert err <= 0.15, f"Failed on real AITEX image: expected 4.0, got {res_tilted['skew_angle']} (err={err})"
