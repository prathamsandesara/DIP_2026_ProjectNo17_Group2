"""
Boundary Acutance and Edge Penumbra Profiling for Defect Etiology Discrimination.

DIP Theoretical Foundations:
---------------------------
Ref: Gonzalez & Woods, Digital Image Processing, Chapter 10 (Edge Detection & Boundary Characteristics).

Physics of Textile Anomalies:
- Physical Punctures / Thread Severances (Holes):
  Produce a sharp step-edge discontinuity at the boundary between the intact fabric
  and the empty void. The normal gradient magnitude along the boundary contour is steep,
  and the penumbra transition width is narrow (1-2 pixels).
- Photometric Contaminants (Stains / Oil Smudges):
  Liquid capillary action causes oil/dye to wick into adjacent fibers, producing a soft,
  diffuse boundary profile with low acutance and a wide transition penumbra.

Key Descriptor:
- Normalized Boundary Acutance (NBA):
  Mean gradient magnitude along the immediate 1-pixel morphological boundary of the defect,
  normalized by the radiometric intensity drop |Delta_I|:
      NBA = Mean_Gradient(Boundary) / (|Delta_I| + 1.0)
  High NBA indicates a crisp puncture edge (Hole).
  Low NBA indicates capillary wicking / diffuse penumbra (Photometric Stain).
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def compute_boundary_acutance(
    img: np.ndarray,
    mask: np.ndarray,
    delta_i: float = 0.0,
) -> Dict[str, float]:
    """
    Compute edge sharpness (acutance) and boundary transition metrics.

    Args:
        img: Grayscale image (uint8).
        mask: Defect binary mask (uint8 or bool).
        delta_i: Pre-computed radiometric contrast (mu_defect - mu_background).

    Returns:
        Dict containing:
            - 'boundary_gradient_mean': Mean Sobel gradient magnitude on defect perimeter
            - 'normalized_acutance': Gradient normalized by radiometric contrast
            - 'penumbra_diffuseness': Inverse acutance representing edge blur
    """
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return {
            "boundary_gradient_mean": 0.0,
            "normalized_acutance": 0.0,
            "penumbra_diffuseness": 0.0,
        }

    # Extract 1-pixel outer boundary using morphological gradient (dilation - original)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel)
    boundary = (dilated > 0) & (binary == 0)

    if not np.any(boundary):
        return {
            "boundary_gradient_mean": 0.0,
            "normalized_acutance": 0.0,
            "penumbra_diffuseness": 0.0,
        }

    # Sobel gradient magnitude
    img_f32 = img.astype(np.float32)
    gx = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    mean_grad = float(np.mean(grad_mag[boundary]))
    abs_delta = abs(delta_i) if abs(delta_i) > 1.0 else 1.0
    norm_acutance = mean_grad / abs_delta
    diffuseness = 1.0 / (norm_acutance + 1e-4)

    return {
        "boundary_gradient_mean": round(mean_grad, 2),
        "normalized_acutance": round(norm_acutance, 3),
        "penumbra_diffuseness": round(diffuseness, 3),
    }

