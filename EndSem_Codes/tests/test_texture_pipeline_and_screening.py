"""Tests for the texture-descriptor detector, the classifier `scale` hook and the screening-threshold API."""
import numpy as np

from src.detection.normal_texture import NormalTextureModel
from src.detection.texture_pipeline import CONFIGS, _deviation_maps
from src.localization.moment_classifier import classify_defect_deterministic, extract_defect_descriptors
from src.pipeline import DualScalePipeline


def _flat(shape=(32, 32), value=1.0):
    keys = ["medium__lbp_bin_0", "medium__glcm_contrast", "medium__gabor_energy"]
    return {k: np.full(shape, value, dtype=np.float64) for k in keys}


def test_deviation_is_zero_on_normal_texture_and_positive_on_anomaly():
    model = NormalTextureModel("00", means={k: 1.0 for k in _flat()}, stds={k: 0.5 for k in _flat()})
    normal = _deviation_maps(_flat(), model)
    assert set(normal) == set(CONFIGS)
    assert all(np.allclose(v, 0.0) for v in normal.values())
    flat = _flat()
    for k in flat:
        flat[k][10:14, 10:14] = 3.0
    anomalous = _deviation_maps(flat, model)
    for name, dev in anomalous.items():
        assert dev[12, 12] > 3.0 and dev[0, 0] == 0.0, name


def test_classifier_scale_one_is_default_and_scale_changes_decision():
    img = np.full((100, 300), 120, np.uint8)
    mask = np.zeros_like(img, bool)
    mask[40:46, 50:150] = True           # long horizontal bar -> weft flaw
    d = extract_defect_descriptors(img, mask)
    assert classify_defect_deterministic(d)["category_id"] == classify_defect_deterministic(d, scale=1.0)["category_id"] == 1
    assert classify_defect_deterministic(d, scale=20.0)["category_id"] == 2   # thresholds scaled out of reach -> default compact class


def test_pipeline_screening_threshold_api():
    p = DualScalePipeline(screening_thresholds={"03": 8})
    assert p.screening_min_px == 15 and p.screening_thresholds == {"03": 8}
    assert DualScalePipeline().screening_thresholds is None
