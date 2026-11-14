"""Tests for GLCM/Haralick feature computation on REAL AITEX fabric image
patches (no synthetic arrays)."""
import numpy as np

from src.texture.glcm import compute_glcm_features, local_glcm_feature_maps, quantize
from tests.conftest import requires_dataset


@requires_dataset
def test_quantize_range(real_defect_free_patch):
    q = quantize(real_defect_free_patch, n_levels=16)
    assert q.min() >= 0
    assert q.max() <= 15


@requires_dataset
def test_glcm_features_on_real_patch_are_finite(real_preprocessed_patch):
    feats = compute_glcm_features(real_preprocessed_patch, n_levels=16)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "ASM", "correlation"):
        assert prop in feats
        assert np.isfinite(feats[prop])


@requires_dataset
def test_glcm_homogeneity_and_energy_bounded(real_preprocessed_patch):
    feats = compute_glcm_features(real_preprocessed_patch, n_levels=16)
    assert 0.0 <= feats["homogeneity"] <= 1.0
    assert 0.0 <= feats["energy"] <= 1.0
    assert 0.0 <= feats["ASM"] <= 1.0


@requires_dataset
def test_local_glcm_feature_maps_shape(real_preprocessed_patch):
    maps, centers = local_glcm_feature_maps(real_preprocessed_patch, window_size=8, stride=8, n_levels=8)
    assert "contrast" in maps
    assert maps["contrast"].shape == real_preprocessed_patch.shape


@requires_dataset
def test_glcm_distinguishes_defect_from_normal_texture(real_defective_image, real_defect_mask):
    """Sanity check on a REAL defective image: GLCM contrast inside the
    defect region should differ from GLCM contrast in the surrounding
    normal fabric (no fixed margin asserted, only that they differ)."""
    ys, xs = np.where(real_defect_mask)
    if len(ys) < 64:
        import pytest
        pytest.skip("Matched real defective sample has too small a defect region for this check")

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    defect_patch = real_defective_image[y0:y1 + 1, x0:x1 + 1]
    normal_patch = real_defective_image[max(0, y0 - 40):y0, x0:x1 + 1] if y0 >= 40 else \
        real_defective_image[y1 + 1:min(real_defective_image.shape[0], y1 + 41), x0:x1 + 1]

    if defect_patch.shape[0] < 2 or defect_patch.shape[1] < 2 or normal_patch.size == 0:
        import pytest
        pytest.skip("Insufficient surrounding normal fabric for comparison")

    defect_feats = compute_glcm_features(defect_patch, n_levels=16)
    normal_feats = compute_glcm_features(normal_patch, n_levels=16)
    assert defect_feats["contrast"] != normal_feats["contrast"]
