"""
Visualization helpers used by scripts/generate_visualizations.py and by the
notebooks. All functions save figures to disk (no interactive display
required) so they run headlessly from the command line.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PathLike = Union[str, Path]


def save_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    out_path: PathLike,
    ncols: int = 4,
    cmap: str = "gray",
    suptitle: Optional[str] = None,
) -> None:
    """Lay out a list of images (grayscale or RGB, or heatmaps) in a grid.

    Used to render the full 16-panel DIP-pipeline figure required by the
    project spec (original -> ... -> localization overlay).
    """
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            im = ax.imshow(img, cmap=cmap)
        else:
            im = ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bar_comparison(
    metrics_by_method: Dict[str, Dict[str, float]],
    metric_names: Sequence[str],
    out_path: PathLike,
    title: str = "Metric comparison",
) -> None:
    """Grouped bar chart comparing several methods across several metrics.

    `metrics_by_method` = {"LBP": {"iou": .., "precision": ..}, "GLCM": {...}, ...}
    """
    methods = list(metrics_by_method.keys())
    n_methods = len(methods)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(1.5 * n_metrics + 2, 5))
    for i, method in enumerate(methods):
        values = [metrics_by_method[method].get(m, np.nan) for m in metric_names]
        ax.bar(x + i * width, values, width, label=method)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: PathLike,
    title: str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_distribution(
    values: np.ndarray,
    out_path: PathLike,
    title: str = "Distribution",
    xlabel: str = "Value",
    bins: int = 50,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values.ravel(), bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bar(
    labels: Sequence[str],
    values: Sequence[float],
    out_path: PathLike,
    title: str = "",
    ylabel: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 4))
    ax.bar(labels, values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary mask onto a grayscale image as a colored overlay."""
    if image.dtype != np.uint8:
        img_min, img_max = float(np.min(image)), float(np.max(image))
        denom = (img_max - img_min) if img_max > img_min else 1.0
        image = ((image - img_min) / denom * 255.0).astype(np.uint8)
    rgb = np.stack([image] * 3, axis=-1).astype(np.float32)
    color_layer = np.zeros_like(rgb)
    color_layer[..., 0] = color[0]
    color_layer[..., 1] = color[1]
    color_layer[..., 2] = color[2]
    mask_bool = mask.astype(bool)
    rgb[mask_bool] = (1 - alpha) * rgb[mask_bool] + alpha * color_layer[mask_bool]
    return rgb.astype(np.uint8)


def draw_bounding_boxes(image_rgb: np.ndarray, boxes: List[Dict]) -> np.ndarray:
    """Draw bounding boxes (from localization.connected_components) on an RGB image."""
    import cv2

    out = image_rgb.copy()
    for box in boxes:
        x, y, w, h = box["bbox"]
        conf = box.get("confidence", None)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if conf is not None:
            cv2.putText(
                out, f"{conf:.2f}", (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
    return out
