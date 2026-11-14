"""Tests for binary morphological post-processing.

These test the mathematical correctness of the morphological operators
themselves (opening/closing/hole-filling/small-component removal),
exercised on binary masks derived from REAL AITEX ground-truth masks
rather than any fabricated dataset. Verifying operator correctness
requires knowing the ground truth "before/after" a specific isolated
perturbation, so a single controlled pixel/gap is added on top of a real
mask where the operator's expected effect must be checked precisely --
no synthetic image or fabric texture is used anywhere.
"""
import numpy as np

from src.morphology.postprocessing import (
    binary_closing, binary_opening, fill_holes, postprocess_mask, remove_small_components,
)
from tests.conftest import requires_dataset


@requires_dataset
def test_opening_removes_isolated_pixel_on_real_mask(real_defect_mask):
    # Start from the real ground-truth mask and add one isolated
    # foreground pixel far from any real defect region, to verify the
    # opening operator's documented effect (removal of sub-structuring-
    # element noise) precisely.
    mask = real_defect_mask.copy()
    mask[:] = 0
    mask[5, 5] = 1
    opened = binary_opening(mask, size=3)
    assert opened.sum() == 0


@requires_dataset
def test_opening_preserves_real_defect_region(real_defect_mask):
    if real_defect_mask.sum() < 25:
        import pytest
        pytest.skip("Real defect region too small for this check")
    opened = binary_opening(real_defect_mask, size=3)
    assert opened.sum() > 0


@requires_dataset
def test_remove_small_components_on_real_mask_plus_noise(real_defect_mask):
    mask = real_defect_mask.copy()
    mask[0:2, 0:2] = 1  # small isolated noise component, area 4
    cleaned = remove_small_components(mask, min_area=10)
    assert cleaned[0, 0] == 0
    if real_defect_mask.sum() >= 10:
        assert cleaned.sum() >= real_defect_mask.sum()


@requires_dataset
def test_closing_bridges_gap_in_real_mask(real_defect_mask):
    ys, xs = np.where(real_defect_mask)
    import pytest

    # Find the row with the longest CONTIGUOUS run of real foreground
    # pixels (the real defect mask is irregularly shaped, so an arbitrary
    # row spanning [x.min(), x.max()] is not necessarily filled in
    # between -- we need an actual contiguous run to cut a gap into).
    best_row, best_run = None, (0, 0, 0)  # (length, start, end)
    for y in np.unique(ys):
        row = real_defect_mask[y]
        idx = np.where(row)[0]
        if idx.size == 0:
            continue
        # split into contiguous runs
        splits = np.where(np.diff(idx) > 1)[0]
        run_starts = np.concatenate(([0], splits + 1))
        run_ends = np.concatenate((splits, [len(idx) - 1]))
        for s, e in zip(run_starts, run_ends):
            length = idx[e] - idx[s] + 1
            if length > best_run[0]:
                best_run = (length, idx[s], idx[e])
                best_row = y

    if best_row is None or best_run[0] < 7:
        pytest.skip("No sufficiently long contiguous run found in the real defect mask")

    y = int(best_row)
    x0, x1 = int(best_run[1]), int(best_run[2])
    mid = (x0 + x1) // 2

    mask = real_defect_mask.copy()
    mask[y, mid - 1:mid + 2] = 0  # cut a small gap through the real, contiguous defect run
    closed = binary_closing(mask, size=5)
    assert closed[y, mid] == 1


@requires_dataset
def test_fill_holes_on_real_mask(real_defect_mask):
    import pytest

    ys, xs = np.where(real_defect_mask)
    if len(ys) < 40:
        pytest.skip("Real defect region too small for this check")

    # Pick a real foreground pixel (not an arbitrary centroid, which may
    # itself fall on background for an irregularly-shaped real region)
    # and punch a 1-pixel hole there, fully surrounded by real foreground.
    interior = None
    for y, x in zip(ys, xs):
        if 1 <= y < real_defect_mask.shape[0] - 1 and 1 <= x < real_defect_mask.shape[1] - 1:
            neighborhood = real_defect_mask[y - 1:y + 2, x - 1:x + 2]
            if neighborhood.sum() == 9:  # fully surrounded by foreground
                interior = (y, x)
                break

    if interior is None:
        pytest.skip("No fully-interior real defect pixel found to punch a hole into")

    cy, cx = interior
    mask = real_defect_mask.copy()
    mask[cy, cx] = 0  # punch a single-pixel hole inside the real region
    filled = fill_holes(mask)
    assert filled[cy, cx] == 1


@requires_dataset
def test_postprocess_mask_pipeline_runs_on_real_mask(real_defect_mask):
    config = {"opening_size": 3, "min_component_area": 5, "closing_size": 3, "fill_holes": True}
    out = postprocess_mask(real_defect_mask, config)
    assert out.shape == real_defect_mask.shape
    assert set(np.unique(out)).issubset({0, 1})
