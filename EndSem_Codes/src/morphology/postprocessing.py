"""
Binary morphological post-processing of the thresholded defect mask.

Pipeline (see PIPELINE.md):

    Binary mask
        -> Opening   (erosion then dilation): removes small isolated
                       noise pixels/blobs that survive thresholding
                       without shrinking genuine larger defect regions.
        -> Remove small components: explicit area-based cleanup of any
                       residual small connected components below a
                       minimum-area threshold (belt-and-braces alongside
                       opening).
        -> Closing   (dilation then erosion): fills small holes inside a
                       defect region and bridges small gaps, so that one
                       physical defect broken into several nearby blobs
                       by noise is reported as one connected region.
        -> Hole filling: fills any fully enclosed background holes inside
                       a foreground defect region (binary_fill_holes).

Each operation is a classical binary morphology primitive defined with a
structuring element B:

    Erosion:  (A (-) B)(x) = 1 iff B fits entirely inside A when centered at x
    Dilation: (A (+) B)(x) = 1 iff B intersects A when centered at x
    Opening:  A o B = (A (-) B) (+) B
    Closing:  A . B = (A (+) B) (-) B

All structuring-element sizes and the minimum-component area are
configurable (`configs/*.yaml`: `morphology`).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage


def disk_se(radius: int) -> np.ndarray:
    """ECE501 Module 5 Part 1: circular (disk) binary structuring element.

    More isotropic than a square SE -- does not bias erosion/dilation
    along horizontal/vertical axes, which matters for detecting
    oblique defects (diagonal bar defects, skewed stains).
    A disk of radius r has area ~ pi*r^2 vs (2r+1)^2 for a square.
    """
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2).astype(bool)


def binary_opening(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Opening with disk SE of radius size//2 (Module 5 Pt1)."""
    structure = disk_se(max(1, size // 2))
    return ndimage.binary_opening(mask.astype(bool), structure=structure).astype(np.uint8)


def binary_closing(mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Closing with disk SE of radius size//2 (Module 5 Pt1)."""
    structure = disk_se(max(1, size // 2))
    return ndimage.binary_closing(mask.astype(bool), structure=structure).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int = 10) -> np.ndarray:
    labeled, n_components = ndimage.label(mask.astype(bool))
    if n_components == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
    remove_labels = [i + 1 for i, s in enumerate(sizes) if s < min_area]
    cleaned = mask.copy().astype(np.uint8)
    for lbl in remove_labels:
        cleaned[labeled == lbl] = 0
    return cleaned


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask.astype(bool)).astype(np.uint8)


def postprocess_mask(mask: np.ndarray, config: Dict) -> np.ndarray:
    """Full morphological pipeline: opening -> closing -> fill_holes -> remove_small_components."""
    out = mask
    if config.get("opening_size", 0) > 0:
        out = binary_opening(out, size=config["opening_size"])
    if config.get("closing_size", 0) > 0:
        out = binary_closing(out, size=config["closing_size"])
    if config.get("fill_holes", False):
        out = fill_holes(out)
    min_area = config.get("min_component_area", 200)
    if min_area > 0:
        out = remove_small_components(out, min_area=min_area)
    return out
