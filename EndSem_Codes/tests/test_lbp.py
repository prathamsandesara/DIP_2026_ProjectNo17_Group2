"""Tests for LBP computation and histogramming on REAL AITEX fabric image
patches (no synthetic arrays)."""
import numpy as np

from src.texture.lbp import compute_lbp, lbp_histogram, local_lbp_histograms, lbp_map_to_pixelwise_feature
from tests.conftest import requires_dataset


@requires_dataset
def test_compute_lbp_output_shape(real_preprocessed_patch):
    lbp_map = compute_lbp(real_preprocessed_patch, radius=1, n_points=8)
    assert lbp_map.shape == real_preprocessed_patch.shape


@requires_dataset
def test_compute_lbp_values_within_uniform_range(real_preprocessed_patch):
    n_points = 8
    lbp_map = compute_lbp(real_preprocessed_patch, radius=1, n_points=n_points, method="uniform")
    # Uniform LBP codes are in [0, P+1] inclusive (P uniform-transition
    # bins + 1 catch-all non-uniform bin).
    assert lbp_map.min() >= 0
    assert lbp_map.max() <= n_points + 1


@requires_dataset
def test_lbp_histogram_sums_to_one(real_preprocessed_patch):
    lbp_map = compute_lbp(real_preprocessed_patch, radius=1, n_points=8, method="uniform")
    hist = lbp_histogram(lbp_map, n_points=8, method="uniform")
    assert abs(hist.sum() - 1.0) < 1e-8
    assert len(hist) == 10  # P + 2 for uniform


@requires_dataset
def test_local_lbp_histograms_grid_shape(real_preprocessed_patch):
    lbp_map = compute_lbp(real_preprocessed_patch, radius=1, n_points=8, method="uniform")
    histograms, centers = local_lbp_histograms(lbp_map, window_size=16, stride=8, n_points=8)
    assert histograms.shape[:2] == centers.shape[:2]
    assert histograms.shape[-1] == 10


@requires_dataset
def test_lbp_pixelwise_feature_matches_image_shape(real_preprocessed_patch):
    lbp_map = compute_lbp(real_preprocessed_patch, radius=1, n_points=8, method="uniform")
    histograms, centers = local_lbp_histograms(lbp_map, window_size=16, stride=8, n_points=8)
    feature = lbp_map_to_pixelwise_feature(histograms, centers, real_preprocessed_patch.shape, 16)
    assert feature.shape[:2] == real_preprocessed_patch.shape


@requires_dataset
def test_lbp_captures_texture_difference_between_fabric_and_defect(
    real_preprocessed_patch, real_defective_image, real_defect_mask
):
    """LBP histograms of a defect region should differ measurably from
    LBP histograms of the surrounding normal fabric on a REAL defective
    image (sanity check that the descriptor is texture-sensitive, not a
    guarantee of any specific separation margin)."""
    if real_defect_mask.sum() < 25:
        import pytest
        pytest.skip("Matched real defective sample has too small a defect region for this check")

    lbp_map = compute_lbp(real_defective_image, radius=1, n_points=8, method="uniform")
    defect_hist = lbp_histogram(lbp_map[real_defect_mask.astype(bool)], n_points=8, method="uniform")
    normal_hist = lbp_histogram(lbp_map[~real_defect_mask.astype(bool)], n_points=8, method="uniform")
    assert not np.allclose(defect_hist, normal_hist)
