"""
Unit tests for automated 2D FFT harmonic peak detection and Butterworth cross-notch filtering.
"""
import numpy as np
import pytest
from pathlib import Path

from src.preprocessing.frequency_filters import (
    detect_harmonic_peaks_2d,
    synthesize_butterworth_notch_filter,
    suppress_weave_pattern_2d,
    butterworth_band_reject,
)
from src.utils.io import load_image_gray


def make_synthetic_periodic_weave(h=256, w=256, period_x=16, period_y=16):
    """Create a synthetic 2D periodic texture with known weave harmonics."""
    x = np.arange(w)
    y = np.arange(h)
    grid = (
        np.cos(2 * np.pi * x / period_x)[None, :]
        + np.cos(2 * np.pi * y / period_y)[:, None]
    )
    # Add a localized non-periodic defect blob in the center
    defect = np.zeros((h, w), dtype=np.float64)
    Y, X = np.ogrid[:h, :w]
    defect[(Y - h // 2) ** 2 + (X - w // 2) ** 2 < 100] = 5.0
    return (grid * 30.0 + defect + 128.0).astype(np.float32)


def test_detect_harmonic_peaks_synthetic():
    img = make_synthetic_periodic_weave(256, 256, 16, 16)
    peaks, diag = detect_harmonic_peaks_2d(img, dc_radius=15, std_multiplier=1.5)
    assert len(peaks) >= 2, f"Expected at least 2 harmonic peaks, found {len(peaks)}"
    assert diag["total_candidates_found"] >= 2


def test_synthesize_butterworth_notch_filter_properties():
    shape = (128, 128)
    cy, cx = 64, 64
    test_peaks = [(cy - 20, cx + 15), (cy - 30, cx)]
    H = synthesize_butterworth_notch_filter(shape, test_peaks, radius=8.0, order=3)

    # 1. Output shape & range
    assert H.shape == shape
    assert 0.0 <= np.min(H) <= 0.1
    assert 0.95 <= np.max(H) <= 1.0

    # 2. DC component must be preserved (H(DC) == 1.0)
    assert abs(H[cy, cx] - 1.0) < 0.02, f"DC component was attenuated: H(DC)={H[cy, cx]}"

    # 3. Harmonic coordinates must be attenuated
    for yp, xp in test_peaks:
        assert H[yp, xp] < 0.15, f"Peak at ({yp}, {xp}) not attenuated: H={H[yp, xp]}"


def test_suppress_weave_pattern_2d_suppression_ratio():
    img = make_synthetic_periodic_weave(256, 256, 16, 16)
    residual, diag = suppress_weave_pattern_2d(img, notch_radius=10.0, order=3)

    assert diag["filter_applied"] is True
    assert diag["harmonic_energy_suppression_pct"] >= 80.0, (
        f"Expected >= 80% suppression, got {diag['harmonic_energy_suppression_pct']}%"
    )
    # Background texture variance should be significantly reduced
    assert diag["residual_std"] < diag["raw_std"]


def test_real_aitex_image_frequency_filtering():
    sample_path = Path("data/raw/Defect_images/0018_010_03.png")
    if not sample_path.exists():
        pytest.skip("AITEX sample 0018_010_03 not found")

    img = load_image_gray(sample_path)
    patch = img[:, 1500:2012]
    residual, diag = suppress_weave_pattern_2d(patch, notch_radius=12.0, order=3)

    assert diag["filter_applied"] is True
    assert diag["peaks_detected"] >= 4
    assert diag["harmonic_energy_suppression_pct"] >= 75.0
