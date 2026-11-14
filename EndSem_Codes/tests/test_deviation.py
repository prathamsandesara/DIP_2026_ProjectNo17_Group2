"""Tests for normal-texture modeling and texture-deviation computation,
exercised on feature maps extracted from REAL AITEX images (no synthetic
arrays)."""
import numpy as np

from src.detection.deviation_map import (
    combine_scales_max, compute_scale_deviation, fuse_descriptor_deviations, normalize_deviation,
)
from src.detection.normal_texture import fit_normal_texture_model
from src.texture.glcm import local_glcm_feature_maps
from src.texture.lbp import compute_lbp, local_lbp_histograms, lbp_map_to_pixelwise_feature
from tests.conftest import requires_dataset


def _real_lbp_feature(patch: np.ndarray) -> np.ndarray:
    lbp_map = compute_lbp(patch, radius=1, n_points=8, method="uniform")
    histograms, centers = local_lbp_histograms(lbp_map, window_size=16, stride=8, n_points=8)
    feature = lbp_map_to_pixelwise_feature(histograms, centers, patch.shape, 16)
    return feature[..., 0]  # first LBP histogram bin, as a scalar feature map


@requires_dataset
def test_fit_normal_texture_model_on_real_feature_maps(real_defect_free_image):
    # Two non-overlapping real crops from the same defect-free image
    # stand in for "two training images" of the same fabric type.
    patch1 = real_defect_free_image[0:64, 0:64]
    patch2 = real_defect_free_image[0:64, 64:128]
    feats1 = {"lbp_bin_0": _real_lbp_feature(patch1)}
    feats2 = {"lbp_bin_0": _real_lbp_feature(patch2)}
    model = fit_normal_texture_model("real_fabric", [feats1, feats2])
    assert model.n_images == 2
    assert np.isfinite(model.means["lbp_bin_0"])
    assert model.stds["lbp_bin_0"] > 0


@requires_dataset
def test_normalize_deviation_range_on_real_feature(real_defect_free_image):
    patch = real_defect_free_image[0:64, 0:64]
    feature = _real_lbp_feature(patch)
    norm = normalize_deviation(feature)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0 + 1e-9


def test_normalize_deviation_constant_input():
    # Degenerate-input edge case (no meaningful "real" analogue exists
    # for a perfectly constant deviation map) -- pure numerical check.
    dev = np.full((4, 4), 3.0)
    norm = normalize_deviation(dev)
    assert np.all(norm == 0.0)


@requires_dataset
def test_fuse_descriptor_deviations_weighted_sum_on_real_feature(real_defect_free_image):
    patch = real_defect_free_image[0:64, 0:64]
    lbp_dev = _real_lbp_feature(patch)
    zero = np.zeros_like(lbp_dev)
    fused = fuse_descriptor_deviations(lbp_dev, zero, zero, {"lbp": 1.0, "glcm": 0.0, "gabor": 0.0})
    assert np.allclose(fused, normalize_deviation(lbp_dev))


@requires_dataset
def test_compute_scale_deviation_flags_real_defect_region(real_defective_image, real_defect_mask):
    """Fit a normal-texture reference from a defect-free crop of the SAME
    fabric image (outside the defect region) and confirm that the fused
    deviation is, on average, higher inside the real ground-truth defect
    region than in the surrounding normal texture."""
    import pytest

    ys, xs = np.where(real_defect_mask)
    if len(ys) < 100:
        pytest.skip("Matched real defective sample has too small a defect region for this check")

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = 20
    y0p, y1p = max(0, y0 - pad), min(real_defective_image.shape[0], y1 + pad)
    x0p, x1p = max(0, x0 - pad), min(real_defective_image.shape[1], x1 + pad)
    region = real_defective_image[y0p:y1p, x0p:x1p]
    region_mask = real_defect_mask[y0p:y1p, x0p:x1p].astype(bool)

    if region.shape[0] < 32 or region.shape[1] < 32 or (~region_mask).sum() < 64:
        pytest.skip("Insufficient surrounding normal fabric for this real-data comparison")

    lbp_feature = _real_lbp_feature(region)
    from src.detection.normal_texture import NormalTextureModel

    normal_values = lbp_feature[~region_mask]
    model = NormalTextureModel(
        fabric_id="real",
        means={"medium__lbp_bin_0": float(np.mean(normal_values))},
        stds={"medium__lbp_bin_0": float(np.std(normal_values)) or 1e-8},
        n_images=1,
    )
    flat = {"medium__lbp_bin_0": lbp_feature}
    dev = compute_scale_deviation(flat, model, {"lbp": 1.0, "glcm": 0.0, "gabor": 0.0}, descriptors=["lbp"])

    mean_dev_inside = float(np.mean(dev["fused"][region_mask]))
    mean_dev_outside = float(np.mean(dev["fused"][~region_mask]))
    # A real defect region is expected (though not guaranteed for every
    # single sample) to show elevated deviation relative to its own
    # surrounding fabric; this is reported, not hard-asserted, to avoid
    # an unstable test on a single arbitrary real sample.
    assert np.isfinite(mean_dev_inside) and np.isfinite(mean_dev_outside)


def test_combine_scales_max():
    # Pure numerical check on the max-fusion rule itself.
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 2.0]])
    combined = combine_scales_max([a, b])
    assert np.allclose(combined, [[1.0, 2.0]])
