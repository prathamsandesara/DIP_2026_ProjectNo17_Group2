"""Tests for grayscale conversion, filtering, normalization, and
enhancement, exercised on REAL AITEX fabric image patches (no synthetic
arrays)."""
import numpy as np

from src.preprocessing.enhancement import clahe_enhance, histogram_equalize
from src.preprocessing.filtering import denoise, gaussian_filter, median_filter
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.normalization import min_max_normalize, zscore_normalize
from tests.conftest import requires_dataset


@requires_dataset
def test_grayscale_passthrough_for_real_image(real_defect_free_patch):
    out = to_grayscale(real_defect_free_patch)
    assert out.shape == real_defect_free_patch.shape
    assert out.dtype == np.float32


@requires_dataset
def test_gaussian_filter_reduces_high_frequency_variation(real_defect_free_patch):
    smoothed = gaussian_filter(real_defect_free_patch, sigma=2.0)
    # Local pixel-to-pixel variation (approximated by the std of the
    # first-difference along rows) should decrease after smoothing a
    # real, noisy fabric scan.
    orig_diff_std = np.std(np.diff(real_defect_free_patch, axis=1))
    smoothed_diff_std = np.std(np.diff(smoothed, axis=1))
    assert smoothed_diff_std <= orig_diff_std


@requires_dataset
def test_median_filter_output_shape(real_defect_free_patch):
    filtered = median_filter(real_defect_free_patch, size=3)
    assert filtered.shape == real_defect_free_patch.shape


@requires_dataset
def test_denoise_dispatch_none_is_identity(real_defect_free_patch):
    out = denoise(real_defect_free_patch, method="none")
    assert np.allclose(out, real_defect_free_patch.astype(np.float32))


@requires_dataset
def test_denoise_both_runs(real_defect_free_patch):
    out = denoise(real_defect_free_patch, method="both", gaussian_sigma=1.0, median_size=3)
    assert out.shape == real_defect_free_patch.shape


@requires_dataset
def test_min_max_normalize_range(real_defect_free_patch):
    out = min_max_normalize(real_defect_free_patch, out_range=(0, 255))
    assert out.min() >= 0
    assert out.max() <= 255
    assert np.isclose(out.max(), 255) or np.isclose(out.min(), 0)


@requires_dataset
def test_zscore_normalize_mean_zero(real_defect_free_patch):
    out = zscore_normalize(real_defect_free_patch)
    assert abs(np.mean(out)) < 1e-4
    assert abs(np.std(out) - 1.0) < 1e-4


@requires_dataset
def test_clahe_enhance_output_shape(real_defect_free_patch):
    normalized = min_max_normalize(real_defect_free_patch, out_range=(0, 255))
    out = clahe_enhance(normalized)
    assert out.shape == real_defect_free_patch.shape


@requires_dataset
def test_histogram_equalize_output_shape(real_defect_free_patch):
    normalized = min_max_normalize(real_defect_free_patch, out_range=(0, 255))
    out = histogram_equalize(normalized)
    assert out.shape == real_defect_free_patch.shape
