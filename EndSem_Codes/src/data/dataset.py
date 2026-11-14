"""
High-level dataset object wrapping `parser.AitexParser` output.

`FabricDataset` loads (lazily) images/masks as numpy arrays and exposes
convenient filtering by fabric type, defect type, and defect-free status,
used throughout preprocessing, adaptive-parameter estimation, and
evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data.parser import AitexParser, Sample
from src.utils.io import load_image_gray, load_mask_binary


@dataclass
class LoadedSample:
    sample: Sample
    image: np.ndarray
    mask: Optional[np.ndarray]  # None if no ground-truth mask available


class FabricDataset:
    """In-memory-lazy view over the AITEX dataset.

    Parameters
    ----------
    root : Path
        Dataset root (see data/raw/README.md).
    raw_config : dict, optional
        The `data.raw` section of the YAML config (glob patterns etc.).
    mask_threshold : float
        Intensity threshold used to binarize ground-truth masks.
    """

    def __init__(self, root: Path, raw_config: Optional[Dict] = None, mask_threshold: float = 127.0):
        self.root = Path(root)
        self.parser = AitexParser(self.root, raw_config)
        self.samples: List[Sample] = self.parser.parse()
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.samples)

    def fabric_types(self) -> List[str]:
        return self.parser.fabric_types(self.samples)

    def defect_types(self) -> List[str]:
        return self.parser.defect_types(self.samples)

    def defect_free_samples(self) -> List[Sample]:
        return [s for s in self.samples if not s.is_defective]

    def defective_samples(self) -> List[Sample]:
        return [s for s in self.samples if s.is_defective]

    def filter(
        self,
        fabric_id: Optional[str] = None,
        defect_type: Optional[str] = None,
        is_defective: Optional[bool] = None,
    ) -> List[Sample]:
        result = self.samples
        if fabric_id is not None:
            result = [s for s in result if s.fabric_id == fabric_id]
        if defect_type is not None:
            result = [s for s in result if s.defect_type == defect_type]
        if is_defective is not None:
            result = [s for s in result if s.is_defective == is_defective]
        return result

    def load(self, sample: Sample) -> LoadedSample:
        """Load the image (and mask, if present) for one sample as arrays."""
        image = load_image_gray(sample.image_path)
        mask = None
        if sample.mask_path is not None and sample.mask_path.exists():
            mask = load_mask_binary(sample.mask_path, threshold=self.mask_threshold)
        elif not sample.is_defective:
            # A defect-free image with no mask file is defined as an
            # all-zero (no-defect) ground truth.
            mask = np.zeros(image.shape, dtype=np.uint8)
        return LoadedSample(sample=sample, image=image, mask=mask)

    def iter_loaded(self, samples: Optional[List[Sample]] = None):
        for s in (samples if samples is not None else self.samples):
            yield self.load(s)
