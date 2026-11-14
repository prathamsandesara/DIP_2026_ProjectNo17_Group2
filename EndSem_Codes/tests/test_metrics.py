"""Tests for pixel-level, image-level, fabric-wise, and defect-wise
evaluation metrics, exercised on REAL AITEX ground-truth masks (no
synthetic masks)."""
import numpy as np
import pytest

from src.evaluation.defect_metrics import compute_defect_type_wise_metrics
from src.evaluation.fabric_metrics import compute_fabric_wise_metrics
from src.evaluation.image_metrics import compute_image_metrics, image_is_defective
from src.evaluation.pixel_metrics import aggregate_pixel_metrics, compute_pixel_metrics
from tests.conftest import requires_dataset


@requires_dataset
def test_perfect_prediction_on_real_mask(real_defect_mask):
    m = compute_pixel_metrics(real_defect_mask, real_defect_mask)
    assert m.iou == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


@requires_dataset
def test_no_overlap_prediction_on_real_mask(real_defect_mask):
    if real_defect_mask.sum() == 0:
        pytest.skip("Real sample's ground-truth mask is empty")
    # A prediction entirely outside the real defect's bounding box has,
    # by construction, zero overlap with the real ground truth.
    ys, xs = np.where(real_defect_mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pred = np.zeros_like(real_defect_mask)
    far_y = 0 if y0 > real_defect_mask.shape[0] // 2 else real_defect_mask.shape[0] - 1
    far_x = 0 if x0 > real_defect_mask.shape[1] // 2 else real_defect_mask.shape[1] - 1
    pred[far_y, far_x] = 1
    if pred[y0:y1 + 1, x0:x1 + 1].sum() > 0:
        pytest.skip("Could not construct a non-overlapping prediction for this real mask")
    m = compute_pixel_metrics(pred, real_defect_mask)
    assert m.iou == 0.0
    assert m.tp == 0


def test_empty_gt_and_empty_pred_is_perfect():
    # Both-empty is a defined edge case independent of any specific
    # image content (see src/evaluation/pixel_metrics.py docstring).
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    m = compute_pixel_metrics(pred, gt)
    assert m.iou == 1.0
    assert m.f1 == 1.0


@requires_dataset
def test_empty_gt_but_nonempty_pred_is_zero(real_defect_free_image):
    # A real defect-free image's all-zero ground truth, against a
    # (hypothetical) false-positive prediction.
    gt = np.zeros(real_defect_free_image.shape, dtype=np.uint8)
    pred = np.zeros(real_defect_free_image.shape, dtype=np.uint8)
    pred[0, 0] = 1
    m = compute_pixel_metrics(pred, gt)
    assert m.iou == 0.0
    assert m.precision == 0.0


@requires_dataset
def test_aggregate_pixel_metrics_micro_average_on_real_masks(real_samples):
    from src.utils.io import load_mask_binary
    from tests.conftest import DEFAULT_CONFIG

    defective = [s for s in real_samples if s.is_defective and s.mask_path and s.mask_path.exists()]
    if len(defective) < 2:
        pytest.skip("Fewer than 2 real defective samples with masks available")

    mask1 = load_mask_binary(defective[0].mask_path, DEFAULT_CONFIG["data"]["mask_threshold"])
    mask2 = load_mask_binary(defective[1].mask_path, DEFAULT_CONFIG["data"]["mask_threshold"])

    m1 = compute_pixel_metrics(mask1, mask1)  # perfect on sample 1
    m2 = compute_pixel_metrics(np.zeros_like(mask2), mask2)  # missed detection on sample 2

    agg = aggregate_pixel_metrics([m1, m2])
    assert agg.tp == int(mask1.sum())
    assert agg.fn == int(mask2.sum())


@requires_dataset
def test_image_is_defective_threshold(real_defect_mask, real_defect_free_image):
    empty = np.zeros(real_defect_free_image.shape, dtype=np.uint8)
    assert not image_is_defective(empty, min_area=1)
    if real_defect_mask.sum() > 0:
        assert image_is_defective(real_defect_mask, min_area=1)


@requires_dataset
def test_compute_image_metrics_confusion_matrix(real_samples):
    pred = [True, False, True, True]
    gt = [True, False, False, True]
    m = compute_image_metrics(pred, gt)
    assert m.confusion_matrix.sum() == 4
    assert 0.0 <= m.accuracy <= 1.0


@requires_dataset
def test_fabric_wise_metrics_grouping_on_real_masks(real_defect_mask):
    m = compute_pixel_metrics(real_defect_mask, real_defect_mask)
    fabric_ids = ["0001", "0001", "0002"]
    metrics = [m, m, m]
    result = compute_fabric_wise_metrics(fabric_ids, metrics)
    assert set(result.keys()) == {"0001", "0002"}


@requires_dataset
def test_defect_wise_metrics_excludes_none_on_real_masks(real_defect_mask):
    m = compute_pixel_metrics(real_defect_mask, real_defect_mask)
    defect_types = ["hole", None, "stain"]
    metrics = [m, m, m]
    result = compute_defect_type_wise_metrics(defect_types, metrics)
    assert set(result.keys()) == {"hole", "stain"}
