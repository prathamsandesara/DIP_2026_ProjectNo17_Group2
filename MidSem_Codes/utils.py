"""
Submission Helper Utilities for Fabric Defect Inspection Pipeline.
Provides image I/O, mask overlays, logging, and CPU parallelization helpers.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def get_logger(name: str = "fabric_inspection") -> logging.Logger:
    """Configure and return a standard console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("Submission.utils")


def load_image_gray(image_path: str | Path) -> np.ndarray:
    """Load an image in single-channel uint8 grayscale."""
    p = str(Path(image_path).resolve())
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at: {p}")
    return img


def load_mask_binary(mask_path: Optional[str | Path]) -> np.ndarray:
    """Load a ground-truth defect mask as a boolean array."""
    if mask_path is None or not Path(mask_path).exists():
        return np.zeros((256, 4096), dtype=bool)
    p = str(Path(mask_path).resolve())
    mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((256, 4096), dtype=bool)
    return mask > 127


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a binary defect mask onto a grayscale image in vivid color.
    Returns: BGR uint8 image suitable for inspection/saving.
    """
    if image.ndim == 2:
        base_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = image.copy()

    overlay = base_bgr.copy()
    overlay[mask > 0] = color_bgr
    return cv2.addWeighted(overlay, alpha, base_bgr, 1.0 - alpha, 0)


def resolve_n_workers(requested: int = 12) -> int:
    """Safely clamp thread/process worker count to physical CPU capacity."""
    available = os.cpu_count() or 1
    return max(1, min(int(requested), available))


def limit_thread_oversubscription() -> None:
    """Cap internal BLAS/OpenCV threadpools inside child processes."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
