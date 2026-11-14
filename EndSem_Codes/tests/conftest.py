"""
Shared pytest fixtures for testing against the REAL AITEX dataset placed
under `data/raw/` (see data/raw/README.md). No synthetic/fabricated image
data is generated anywhere in this test suite -- every fixture here loads
actual AITEX images/masks discovered by `src.data.parser.AitexParser`.

Tests that need the dataset are skipped (not failed) if `data/raw/` is
empty or not yet populated, so `pytest` remains runnable before the
dataset is placed, but on this machine (dataset present) every test
below exercises real fabric image data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.parser import AitexParser
from src.preprocessing import preprocess_image
from src.preprocessing.grayscale import to_grayscale
from src.utils.io import load_config, load_image_gray, load_mask_binary

def _find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, p.parents[1], p.parents[2]]:
        if (parent / "data").exists():
            return parent
    return p.parents[1]

PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG = load_config(PROJECT_ROOT / "code" / "configs" / "default.yaml" if (PROJECT_ROOT / "code" / "configs" / "default.yaml").exists() else PROJECT_ROOT / "configs" / "default.yaml")
DATASET_ROOT = PROJECT_ROOT / DEFAULT_CONFIG["paths"]["dataset_root"]


def _parse_real_samples():
    if not DATASET_ROOT.exists():
        return []
    parser = AitexParser(DATASET_ROOT, DEFAULT_CONFIG["data"]["raw"])
    return parser.parse()


_REAL_SAMPLES = _parse_real_samples()

requires_dataset = pytest.mark.skipif(
    len(_REAL_SAMPLES) == 0,
    reason="Real AITEX dataset not found under data/raw/ -- see data/raw/README.md",
)


@pytest.fixture(scope="session")
def real_samples():
    """All parsed real AITEX samples (Sample objects)."""
    if not _REAL_SAMPLES:
        pytest.skip("Real AITEX dataset not found under data/raw/")
    return _REAL_SAMPLES


@pytest.fixture(scope="session")
def real_defect_free_image(real_samples) -> np.ndarray:
    """One real, defect-free AITEX image (grayscale float32 array)."""
    sample = next(s for s in real_samples if not s.is_defective)
    return load_image_gray(sample.image_path)


@pytest.fixture(scope="session")
def real_defective_sample(real_samples):
    """One real defective AITEX sample that has a matching ground-truth mask."""
    sample = next(s for s in real_samples if s.is_defective and s.mask_path is not None and s.mask_path.exists())
    return sample


@pytest.fixture(scope="session")
def real_defective_image(real_defective_sample) -> np.ndarray:
    return load_image_gray(real_defective_sample.image_path)


@pytest.fixture(scope="session")
def real_defect_mask(real_defective_sample, real_defective_image) -> np.ndarray:
    mask = load_mask_binary(real_defective_sample.mask_path, DEFAULT_CONFIG["data"]["mask_threshold"])
    return mask


@pytest.fixture(scope="session")
def real_defect_free_patch(real_defect_free_image) -> np.ndarray:
    """A small (64x64) crop of a real defect-free image, for fast
    per-function unit tests that don't need the full 4096x256 frame."""
    img = real_defect_free_image
    h, w = img.shape
    y0 = h // 2 - 32 if h >= 64 else 0
    x0 = w // 2 - 32 if w >= 64 else 0
    return img[y0:y0 + min(64, h), x0:x0 + min(64, w)]


@pytest.fixture(scope="session")
def real_preprocessed_patch(real_defect_free_patch) -> np.ndarray:
    enhanced, _ = preprocess_image(real_defect_free_patch, DEFAULT_CONFIG["preprocessing"])
    return enhanced
