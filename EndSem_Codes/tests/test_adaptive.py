"""Tests for adaptive scale estimation, parameter selection, and threshold
selection, exercised on REAL AITEX defect-free fabric images (no
synthetic textures)."""
import numpy as np

from src.adaptive.parameter_selection import select_adaptive_scales
from src.adaptive.scale_estimation import estimate_texture_characteristics
from src.adaptive.threshold_selection import (
    mean_k_std_threshold, median_mad_threshold, otsu_threshold, percentile_threshold, select_threshold,
)
from tests.conftest import requires_dataset


@requires_dataset
def test_estimate_texture_characteristics_on_real_images(real_defect_free_image):
    # Two non-overlapping real crops stand in for "two training images"
    # of the same real fabric type.
    patch1 = real_defect_free_image[:, 0:128]
    patch2 = real_defect_free_image[:, 128:256]
    tc = estimate_texture_characteristics(
        "real_fabric", [patch1, patch2],
        candidate_variance_windows=[5, 9], variance_cv_tolerance=0.5,
        candidate_gabor_frequencies=[0.1, 0.125, 0.2], gabor_sigma=3.0,
    )
    assert tc.fabric_id == "real_fabric"
    assert tc.n_images == 2
    assert tc.mean_period_x > 0
    assert tc.mean_period_y > 0
    assert np.isfinite(tc.dominant_gabor_frequency)


@requires_dataset
def test_select_adaptive_scales_ordering_on_real_texture(real_defect_free_image):
    patch1 = real_defect_free_image[:, 0:128]
    patch2 = real_defect_free_image[:, 128:256]
    tc = estimate_texture_characteristics(
        "real_fabric", [patch1, patch2],
        candidate_variance_windows=[5, 9], variance_cv_tolerance=0.5,
        candidate_gabor_frequencies=[0.1, 0.125, 0.2], gabor_sigma=3.0,
    )
    mapping_cfg = {
        "min_glcm_window": 7, "max_glcm_window": 33, "min_lbp_radius": 1, "max_lbp_radius": 8,
        "small_scale_ratio": 0.5, "large_scale_ratio": 2.0,
    }
    scales = select_adaptive_scales(tc, mapping_cfg)
    names = [s.name for s in scales]
    assert names == ["small", "medium", "large"]
    small, medium, large = scales
    assert small.glcm_window <= medium.glcm_window <= large.glcm_window


@requires_dataset
def test_threshold_methods_on_real_deviation_scores(real_defect_free_image):
    """Threshold-selection methods, run on a real (though ad hoc, since
    it is not yet a proper fused deviation map) pixel-intensity
    distribution taken directly from a real fabric image, to confirm
    each method returns a sane, finite value on real data statistics."""
    values = real_defect_free_image.ravel().astype(np.float64)
    assert np.isfinite(otsu_threshold(values))
    assert np.isfinite(mean_k_std_threshold(values, k=2.0))
    assert np.isfinite(percentile_threshold(values, percentile=95.0))
    assert np.isfinite(median_mad_threshold(values, k=3.0))


@requires_dataset
def test_select_threshold_dispatch_on_real_data(real_defect_free_image):
    values = real_defect_free_image.ravel().astype(np.float64)
    t1 = select_threshold(values, {"method": "percentile", "percentile": 99.0})
    t2 = select_threshold(values, {"method": "mean_k_std", "k": 3.0})
    assert isinstance(t1, float)
    assert isinstance(t2, float)
