"""
I/O helpers: config loading, image/mask reading & writing, path resolution.

Keeping all filesystem and YAML access here means the rest of the codebase
never hard-codes paths or duplicates config-parsing logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

PathLike = Union[str, Path]


def load_config(config_path: PathLike) -> Dict[str, Any]:
    """Load a YAML configuration file into a plain dict.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` on top of `base` and return a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_image_gray(path: PathLike) -> np.ndarray:
    """Load an image from disk as a single-channel float32 array in [0, 255].

    AITEX fabric images are single-channel greyscale 8-bit images already,
    but this loader also handles RGB inputs defensively by converting them.
    """
    path = Path(str(path).replace("\\", "/"))
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)
    else:  # pragma: no cover - fallback path
        from PIL import Image

        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float32)


def load_mask_binary(path: PathLike, threshold: float = 127.0) -> np.ndarray:
    """Load a ground-truth mask and binarize it to {0, 1} uint8.

    AITEX masks encode defect pixels as non-zero intensity; anything above
    `threshold` (default 127 on an 8-bit scale) is treated as defect.
    """
    mask = load_image_gray(path)
    return (mask > threshold).astype(np.uint8)


def save_image(path: PathLike, image: np.ndarray) -> None:
    """Save a float or uint8 image array to disk, creating parent dirs."""
    path = Path(str(path).replace("\\", "/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image
    if img.dtype != np.uint8:
        img_min, img_max = float(np.min(img)), float(np.max(img))
        if img_max - img_min < 1e-12:
            img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    if cv2 is not None:
        cv2.imwrite(str(path), img)
    else:  # pragma: no cover
        from PIL import Image

        Image.fromarray(img).save(path)


def save_json(path: PathLike, data: Dict[str, Any]) -> None:
    path = Path(str(path).replace("\\", "/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_json(path: PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: PathLike) -> Path:
    path = Path(str(path).replace("\\", "/"))
    path.mkdir(parents=True, exist_ok=True)
    return path
