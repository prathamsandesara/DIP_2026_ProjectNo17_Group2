"""
Unit tests for Moment-Invariant Region Classification and Deterministic Defect Taxonomy.

ECE501 Digital Image Processing (Module 7: Representation & Description, Week 13)
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.localization.moment_classifier import (
    compute_hu_moments_from_scratch,
    extract_defect_descriptors,
    classify_defect_deterministic,
    draw_taxonomy_overlay,
    TAXONOMY_CATEGORIES,
)


def test_hu_moments_opencv_agreement():
    """Verify from-scratch Hu moments match OpenCV's cv2.HuMoments to numerical tolerance."""
    # Create an asymmetric synthetic test shape
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 30), (70, 50), 255, -1)
    cv2.circle(mask, (35, 30), 12, 255, -1)

    raw_scratch, log_scratch = compute_hu_moments_from_scratch(mask)
    cv_moments = cv2.moments((mask > 0).astype(np.uint8))
    raw_cv = cv2.HuMoments(cv_moments).flatten()

    # Compare first 4 moments (higher moments can be very small numbers ~1e-12)
    for i in range(4):
        assert np.isclose(raw_scratch[i], raw_cv[i], rtol=1e-3, atol=1e-8), (
            f"Moment phi_{i+1} mismatch: scratch={raw_scratch[i]}, cv={raw_cv[i]}"
        )


def test_hu_moments_translation_invariance():
    """Verify Hu moments are invariant under 2D translation."""
    shape = np.zeros((150, 150), dtype=np.uint8)
    cv2.ellipse(shape, (40, 40), (25, 10), 30, 0, 360, 255, -1)

    # Shift shape by (50, 60)
    shifted = np.zeros((150, 150), dtype=np.uint8)
    cv2.ellipse(shifted, (90, 100), (25, 10), 30, 0, 360, 255, -1)

    raw_1, _ = compute_hu_moments_from_scratch(shape)
    raw_2, _ = compute_hu_moments_from_scratch(shifted)

    for i in range(4):
        assert np.isclose(raw_1[i], raw_2[i], rtol=1e-3, atol=1e-7), (
            f"Translation invariance failed for phi_{i+1}: {raw_1[i]} vs {raw_2[i]}"
        )


def test_hu_moments_scale_invariance():
    """Verify Hu moments are invariant under uniform scaling."""
    base = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(base, (30, 30), (60, 50), 255, -1)

    scaled = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(scaled, (30, 30), (90, 70), 255, -1)  # 2x scaled rectangle

    _, log_1 = compute_hu_moments_from_scratch(base)
    _, log_2 = compute_hu_moments_from_scratch(scaled)

    # First two moments phi_1 and phi_2 should match closely
    assert np.isclose(log_1[0], log_2[0], atol=0.05), f"Scale phi_1 mismatch: {log_1[0]} vs {log_2[0]}"
    assert np.isclose(log_1[1], log_2[1], atol=0.10), f"Scale phi_2 mismatch: {log_1[1]} vs {log_2[1]}"


def test_extract_defect_descriptors():
    """Verify physical and radiometric descriptors on synthetic image."""
    img = np.full((100, 100), 120.0, dtype=np.float32)
    mask = np.zeros((100, 100), dtype=bool)

    # Add dark horizontal defect
    img[45:55, 20:80] = 50.0
    mask[45:55, 20:80] = True

    desc = extract_defect_descriptors(img, mask)
    assert desc is not None
    assert desc["width"] == 60.0
    assert desc["height"] == 10.0
    assert desc["aspect_ratio"] == 6.0
    assert desc["delta_i"] < -50.0  # Strongly negative contrast
    assert desc["eccentricity"] > 0.90  # Highly elongated
    assert len(desc["hu_log"]) == 7


def test_classify_defect_deterministic_rules():
    """Verify deterministic taxonomy rules on standard defect geometries."""
    # 1. Vertical flaw
    desc_vert = {"width": 8.0, "height": 180.0, "area": 1200.0, "aspect_ratio": 22.5, "delta_i": 5.0}
    res_vert = classify_defect_deterministic(desc_vert)
    assert res_vert["category_id"] == 0
    assert "Warp" in res_vert["category_name"]

    # 2. Horizontal flaw
    desc_horiz = {"width": 450.0, "height": 12.0, "area": 3500.0, "aspect_ratio": 37.5, "delta_i": -15.0}
    res_horiz = classify_defect_deterministic(desc_horiz)
    assert res_horiz["category_id"] == 1
    assert "Weft" in res_horiz["category_name"]

    # 3. Compact puncture / hole
    desc_punct = {"width": 6.0, "height": 6.0, "area": 25.0, "aspect_ratio": 1.0, "delta_i": -5.0}
    res_punct = classify_defect_deterministic(desc_punct)
    assert res_punct["category_id"] == 2
    assert "Micro-Puncture" in res_punct["category_name"]

    # 4. Photometric stain
    desc_stain = {"width": 12.0, "height": 10.0, "area": 95.0, "aspect_ratio": 1.2, "delta_i": -35.0}
    res_stain = classify_defect_deterministic(desc_stain)
    assert res_stain["category_id"] == 3
    assert "Photometric" in res_stain["category_name"]


def test_draw_taxonomy_overlay():
    """Verify diagnostic overlay generation."""
    img = np.full((120, 200), 128.0, dtype=np.float32)
    mask = np.zeros((120, 200), dtype=bool)
    mask[50:60, 80:120] = True

    desc = extract_defect_descriptors(img, mask)
    clf = classify_defect_deterministic(desc)
    overlay = draw_taxonomy_overlay(img, mask, desc, clf)

    assert overlay.shape == (120, 200, 3)
    assert overlay.dtype == np.uint8
    assert np.any(overlay > 0)
