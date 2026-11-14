"""
Laws' Texture Energy Measures (TEM) for Fabric Defect Characterization.

DIP Theoretical Foundations:
---------------------------
Formulated by Kenneth I. Laws (1980), "Rapid Texture Identification", SPIE.
Ref: Gonzalez & Woods, Digital Image Processing, Chapter 10 & 11 (Texture Representation).

Uses combinations of 1D micro-texture kernels:
- Level (L5)  = [ 1,  4,  6,  4,  1]  (Center-weighted local average)
- Edge  (E5)  = [-1, -2,  0,  2,  1]  (First-order gradient)
- Spot  (S5)  = [-1,  0,  2,  0, -1]  (Second-order point/spot detector)
- Wave  (W5)  = [-1,  2,  0, -2,  1]  (Periodic wave detector)
- Ripple(R5)  = [ 1, -4,  6, -4,  1]  (High-frequency ripple detector)

By forming 2D outer products (e.g., E5L5, L5E5, S5S5, E5E5), these separable filters
capture localized micro-texture structures. Local texture energy is computed as the moving
absolute deviation.

Key Metrics for Defect Diagnosis:
1. Texture Intactness Ratio (TIR):
   Ratio of spot/edge energy inside the candidate ROI vs. the surrounding background.
   - Intact weave with photometric dye absorption (Stain): TIR remains high (~0.65 - 0.75).
   - Ruptured weave / void (Hole): yarn structure is demolished (TIR collapses).
2. Directional Energy Asymmetry (DEA):
   Energy contrast between horizontal (L5E5) and vertical (E5L5) filter responses:
   - Positive DEA indicates horizontal pick disruption (Weft flaw).
   - Negative DEA indicates longitudinal thread disruption (Warp flaw).
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


# 1D Laws Kernels (length 5)
L5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32)
E5 = np.array([-1.0, -2.0, 0.0, 2.0, 1.0], dtype=np.float32)
S5 = np.array([-1.0, 0.0, 2.0, 0.0, -1.0], dtype=np.float32)
W5 = np.array([-1.0, 2.0, 0.0, -2.0, 1.0], dtype=np.float32)
R5 = np.array([1.0, -4.0, 6.0, -4.0, 1.0], dtype=np.float32)

# Standard 2D Laws Energy Kernels
KERNEL_E5L5 = np.outer(E5, L5)  # Vertical edge / longitudinal response
KERNEL_L5E5 = np.outer(L5, E5)  # Horizontal edge / transverse response
KERNEL_S5S5 = np.outer(S5, S5)  # Spot / localized point anomaly
KERNEL_E5E5 = np.outer(E5, E5)  # Diagonal / corner energy
KERNEL_R5R5 = np.outer(R5, R5)  # Micro-ripple roughness


def compute_laws_energy_map(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Compute local Laws' Texture Energy by 2D convolution followed by absolute response.
    Uses OpenCV's optimized separable/2D filter2D.
    """
    img_f32 = img.astype(np.float32)
    filtered = cv2.filter2D(img_f32, cv2.CV_32F, kernel)
    return np.abs(filtered)


def extract_laws_defect_descriptors(
    img: np.ndarray,
    mask: np.ndarray,
    bg_ring_dilation: int = 15,
) -> Dict[str, float]:
    """
    Extract defect-specific Laws' Texture Energy features for a segmented defect.

    Args:
        img: Grayscale image (uint8).
        mask: Binary mask of the defect (uint8 or bool).
        bg_ring_dilation: Radius of morphological dilation to construct the local clean ring.

    Returns:
        Dict of classical Laws' texture descriptors:
            - 'laws_spot_intactness': Ratio of S5S5 spot energy (inside vs background)
            - 'laws_edge_intactness': Ratio of E5E5 edge energy (inside vs background)
            - 'laws_directional_asymmetry': Normalized difference between L5E5 (horizontal) and E5L5 (vertical)
            - 'laws_spot_inside': Mean S5S5 energy inside defect
            - 'laws_spot_bg': Mean S5S5 energy in surrounding background
    """
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return {
            "laws_spot_intactness": 1.0,
            "laws_edge_intactness": 1.0,
            "laws_directional_asymmetry": 0.0,
            "laws_spot_inside": 0.0,
            "laws_spot_bg": 0.0,
        }

    # Construct clean background ring around defect
    kernel_dil = np.ones((bg_ring_dilation, bg_ring_dilation), np.uint8)
    dilated = cv2.dilate(binary, kernel_dil).astype(bool)
    bg_ring = dilated & (~(binary.astype(bool)))

    # Compute energy maps for Spot, Horizontal Edge, Vertical Edge
    e_s5s5 = compute_laws_energy_map(img, KERNEL_S5S5)
    e_e5l5 = compute_laws_energy_map(img, KERNEL_E5L5)
    e_l5e5 = compute_laws_energy_map(img, KERNEL_L5E5)
    e_e5e5 = compute_laws_energy_map(img, KERNEL_E5E5)

    def_mask = binary.astype(bool)

    # Spot Energy
    spot_in = float(np.mean(e_s5s5[def_mask]))
    spot_bg = float(np.mean(e_s5s5[bg_ring])) if np.any(bg_ring) else (spot_in + 1e-5)
    spot_intactness = spot_in / max(1e-5, spot_bg)

    # Edge Energy
    edge_in = float(np.mean(e_e5e5[def_mask]))
    edge_bg = float(np.mean(e_e5e5[bg_ring])) if np.any(bg_ring) else (edge_in + 1e-5)
    edge_intactness = edge_in / max(1e-5, edge_bg)

    # Directional Asymmetry: (Horizontal - Vertical) / (Horizontal + Vertical)
    horiz_in = float(np.mean(e_l5e5[def_mask]))
    vert_in = float(np.mean(e_e5l5[def_mask]))
    denom = horiz_in + vert_in + 1e-5
    directional_asymmetry = (horiz_in - vert_in) / denom

    return {
        "laws_spot_intactness": round(spot_intactness, 3),
        "laws_edge_intactness": round(edge_intactness, 3),
        "laws_directional_asymmetry": round(directional_asymmetry, 3),
        "laws_spot_inside": round(spot_in, 2),
        "laws_spot_bg": round(spot_bg, 2),
    }

