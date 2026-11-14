"""
Moment-Invariant Region Classification and Deterministic Defect Taxonomy.

ECE501 Digital Image Processing (Module 7: Representation & Description, Week 13)
---------------------------------------------------------------------------------
Implements:
1. Spatial, Central, and Scale-Normalized Moments (m_pq, mu_pq, eta_pq).
2. Hu's 7 Invariant Moments (phi_1 through phi_7) invariant to translation,
   scaling, and 2D planar rotation.
3. Morphometric Descriptors:
   - Compactness (isoperimetric quotient) C = 4*pi*Area / Perimeter^2
   - Bounding-box aspect ratio AR = max(W, H) / min(W, H)
   - Extent (fill ratio) = Area / (W * H)
   - Eccentricity epsilon from inertia tensor eigenvalues
   - Principal orientation angle theta via second-order central moments
   - Radiometric contrast Delta_I = mu_defect - mu_background
4. Deterministic Physical Taxonomy Decision Tree (NO black-box machine learning):
   - Category 0: Longitudinal / Warp Flaws (Broken warp threads, vertical streaks)
   - Category 1: Transverse / Weft Flaws (Broken picks, missing wefts, horizontal bars)
   - Category 2: Localized Micro-Punctures & Holes (Punctures, tears, thread breaks)
   - Category 3: Photometric Contaminants (Dark diffuse regions, negative contrast Delta_I <= -10.0)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.texture.laws_energy import extract_laws_defect_descriptors


TAXONOMY_CATEGORIES = {
    0: "Longitudinal / Warp Flaw (Vertical)",
    1: "Transverse / Weft Flaw (Horizontal)",
    2: "Localized Micro-Puncture / Hole",
    3: "Photometric Contaminant (dark diffuse region)",
}

TAXONOMY_COLORS = {
    0: (0, 165, 255),    # Orange for vertical warp flaws
    1: (255, 100, 0),    # Blue for horizontal weft flaws
    2: (0, 0, 255),      # Red for physical punctures / holes
    3: (180, 0, 180),    # Purple for photometric stains
}


def compute_hu_moments_from_scratch(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hu's 7 Moment Invariants from first principles.

    Returns:
        raw_hu: 7-element array of raw Hu invariants [phi_1 .. phi_7]
        log_hu: 7-element array of sign-preserved log-transformed invariants
                psi_i = -sign(phi_i) * log10(|phi_i| + 1e-30)
    """
    binary = (mask > 0).astype(np.float64)
    y_coords, x_coords = np.where(binary > 0)
    if len(x_coords) == 0:
        return np.zeros(7, dtype=np.float64), np.zeros(7, dtype=np.float64)

    # 1. Spatial Moments m_pq
    m00 = float(len(x_coords))
    m10 = float(np.sum(x_coords))
    m01 = float(np.sum(y_coords))

    # Centroid
    x_bar = m10 / m00
    y_bar = m01 / m00

    # 2. Central Moments mu_pq
    x_diff = x_coords - x_bar
    y_diff = y_coords - y_bar

    mu20 = float(np.sum(x_diff ** 2))
    mu02 = float(np.sum(y_diff ** 2))
    mu11 = float(np.sum(x_diff * y_diff))

    mu30 = float(np.sum(x_diff ** 3))
    mu03 = float(np.sum(y_diff ** 3))
    mu21 = float(np.sum((x_diff ** 2) * y_diff))
    mu12 = float(np.sum(x_diff * (y_diff ** 2)))

    # 3. Normalized Central Moments eta_pq = mu_pq / (m00 ** ((p + q) / 2 + 1))
    def norm_mu(mu: float, p: int, q: int) -> float:
        gamma = (p + q) / 2.0 + 1.0
        return mu / (m00 ** gamma)

    eta20 = norm_mu(mu20, 2, 0)
    eta02 = norm_mu(mu02, 0, 2)
    eta11 = norm_mu(mu11, 1, 1)

    eta30 = norm_mu(mu30, 3, 0)
    eta03 = norm_mu(mu03, 0, 3)
    eta21 = norm_mu(mu21, 2, 1)
    eta12 = norm_mu(mu12, 1, 2)

    # 4. Hu's 7 Invariant Moments (Hu, 1962; Gonzalez & Woods Eq. 10.3)
    # phi_1: Spread / Inertia
    phi1 = eta20 + eta02

    # phi_2: Difference of principal inertias
    phi2 = (eta20 - eta02) ** 2 + 4.0 * (eta11 ** 2)

    # phi_3: Third-order asymmetry
    phi3 = (eta30 - 3.0 * eta12) ** 2 + (3.0 * eta21 - eta03) ** 2

    # phi_4: Third-order variance
    phi4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2

    # phi_5: Planar skewness
    term1 = (eta30 - 3.0 * eta12) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3.0 * ((eta21 + eta03) ** 2))
    term2 = (3.0 * eta21 - eta03) * (eta21 + eta03) * (3.0 * ((eta30 + eta12) ** 2) - (eta21 + eta03) ** 2)
    phi5 = term1 + term2

    # phi_6: Second and third order interaction
    phi6 = (eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) + 4.0 * eta11 * (eta30 + eta12) * (eta21 + eta03)

    # phi_7: Skew invariant (distinguishes mirror reflections)
    term3 = (3.0 * eta21 - eta03) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3.0 * ((eta21 + eta03) ** 2))
    term4 = (eta30 - 3.0 * eta12) * (eta21 + eta03) * (3.0 * ((eta30 + eta12) ** 2) - (eta21 + eta03) ** 2)
    phi7 = term3 - term4

    raw_hu = np.array([phi1, phi2, phi3, phi4, phi5, phi6, phi7], dtype=np.float64)

    # Sign-preserving log transform for numerical interpretation
    log_hu = np.zeros(7, dtype=np.float64)
    for i in range(7):
        v = raw_hu[i]
        if abs(v) > 1e-30:
            log_hu[i] = -np.sign(v) * np.log10(abs(v))
        else:
            log_hu[i] = 0.0

    return raw_hu, log_hu


def extract_defect_descriptors(
    img: np.ndarray,
    mask: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Extract complete physical, geometric, and moment descriptors for a segmented defect.
    """
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None

    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    area = float(len(xs))
    aspect_ratio = max(w, h) / max(1.0, min(w, h))
    extent = area / max(1.0, w * h)

    # Contours & Compactness
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    compactness = (4.0 * np.pi * area) / max(1.0, perimeter ** 2) if perimeter > 0 else 0.0

    # Moments & Hu Invariants
    raw_hu, log_hu = compute_hu_moments_from_scratch(binary)

    # Orientation & Eccentricity from central moments
    moments = cv2.moments(binary)
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]
    diff_mu = mu20 - mu02
    discriminant = np.sqrt(diff_mu ** 2 + 4.0 * (mu11 ** 2))
    lambda1 = (mu20 + mu02 + discriminant) / 2.0
    lambda2 = max(0.0, (mu20 + mu02 - discriminant) / 2.0)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - lambda2 / max(1e-6, lambda1))))
    theta_deg = float(np.degrees(0.5 * np.arctan2(2.0 * mu11, diff_mu)))

    # Radiometric Contrast Delta I (Defect intensity vs. local background ring)
    dilated = cv2.dilate(binary, np.ones((15, 15), np.uint8)).astype(bool)
    bg_ring = dilated & (~(binary.astype(bool)))
    mean_def = float(np.mean(img[binary.astype(bool)]))
    mean_bg = float(np.mean(img[bg_ring])) if np.any(bg_ring) else 128.0
    delta_i = mean_def - mean_bg

    # Boundary gradient sharpness
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    boundary = dilated ^ (binary.astype(bool))
    edge_sharpness = float(np.mean(grad_mag[boundary])) if np.any(boundary) else 0.0

    # Classical Laws' Texture Energy Measures (Kenneth Laws, 1980)
    laws_desc = extract_laws_defect_descriptors(img, binary)

    return {
        "area": area,
        "width": w,
        "height": h,
        "aspect_ratio": round(aspect_ratio, 2),
        "extent": round(extent, 3),
        "compactness": round(compactness, 3),
        "eccentricity": round(eccentricity, 3),
        "theta_deg": round(theta_deg, 1),
        "delta_i": round(delta_i, 1),
        "edge_sharpness": round(edge_sharpness, 1),
        "hu_raw": [float(x) for x in raw_hu],
        "hu_log": [round(float(x), 3) for x in log_hu],
        "hu1": round(float(log_hu[0]), 3),
        "hu2": round(float(log_hu[1]), 3),
        "laws_spot_intactness": laws_desc["laws_spot_intactness"],
        "laws_edge_intactness": laws_desc["laws_edge_intactness"],
        "laws_directional_asymmetry": laws_desc["laws_directional_asymmetry"],
        "laws_spot_inside": laws_desc["laws_spot_inside"],
        "laws_spot_bg": laws_desc["laws_spot_bg"],
    }


def classify_defect_deterministic(descriptors: Dict[str, Any], scale: float = 1.0) -> Dict[str, Any]:
    """
    Deterministic Decision Tree using classical DIP Invariants (No Machine Learning).

    All thresholds are hand-set constants (not fitted). `scale` multiplies EVERY numeric
    threshold (sizes, ratios, contrast, edge sharpness and the two angle bounds) simultaneously
    and exists only for the sensitivity analysis; the model itself uses scale = 1.0.

    Categories:
    - Category 0: Longitudinal / Warp Flaw (Vertical streak, broken warp thread)
    - Category 1: Transverse / Weft Flaw (Horizontal line flaw, missing weft pick)
    - Category 2: Localized Micro-Puncture / Hole (Thread breakage, tear, knot)
    - Category 3: Photometric Contaminant (Dark diffuse region, Delta_I <= -10.0)
    """
    w = float(descriptors.get("width", 1.0))
    h = float(descriptors.get("height", 1.0))
    area = float(descriptors.get("area", w * h))
    ar = float(descriptors.get("aspect_ratio", max(w, h) / max(1.0, min(w, h))))
    delta_i = float(descriptors.get("delta_i", 0.0))
    theta = abs(float(descriptors.get("theta_deg", 0.0)))
    edge = float(descriptors.get("edge_sharpness", 0.0))
    extent = float(descriptors.get("extent", 0.5))
    spot_tir = float(descriptors.get("laws_spot_intactness", 0.70))
    spot_bg = float(descriptors.get("laws_spot_bg", 0.0))
    k = float(scale)
    th_h = min(30.0 * k, 90.0)   # horizontal-axis bound
    th_v = min(60.0 * k, 90.0)   # vertical-axis bound

    # Rule 1: Transverse / Weft Flaws (Horizontal Orientation Dominance)
    # Width distinctly exceeds height, or horizontal inertial axis (theta <= 30 deg)
    if (w >= 30.0 * k and w >= 1.5 * k * h) or (w >= 100.0 * k and w > h) or (w >= 20.0 * k and ar >= 2.0 * k and theta <= th_h):
        return {
            "category_id": 1,
            "category_name": TAXONOMY_CATEGORIES[1],
            "rule_triggered": f"Transverse Weft Orientation (W={w:.0f}px, H={h:.0f}px, W/H={w/max(1.0,h):.1f}, Theta={theta:.1f}deg)",
            "confidence": 0.95 if w >= 100.0 else 0.88,
        }

    # Rule 2: Photometric Contaminant (dark diffuse region): strongly negative radiometric contrast
    # with Laws' texture intactness (liquid capillary wicking preserves underlying weave ribs).
    # A: Broad dark absorption with extent >= 0.30
    if (delta_i <= -10.0 * k and area >= 75.0 * k and ar <= 1.8 * k and extent >= 0.30 * k and spot_tir <= 0.78) or \
       (delta_i <= -18.0 * k and area >= 70.0 * k and ar <= 2.2 * k and extent >= 0.65 * k and spot_tir <= 0.78) or \
       (delta_i <= -10.0 * k and h >= 70.0 * k and w <= 25.0 * k and edge <= 130.0 * k):
        return {
            "category_id": 3,
            "category_name": TAXONOMY_CATEGORIES[3],
            "rule_triggered": f"Broad Photometric Absorption (Delta_I={delta_i:.1f}, Area={area:.0f}px, AR={ar:.2f}, Spot_TIR={spot_tir:.2f})",
            "confidence": 0.92,
        }

    # B: Localized dark dye smudge identified via Laws texture background spot energy & weave absorption
    if (delta_i <= -20.0 * k and spot_bg >= 70.0 * k and edge >= 140.0 * k and ar <= 1.25 * k and area >= 30.0 * k and 0.60 <= spot_tir <= 0.78):
        return {
            "category_id": 3,
            "category_name": TAXONOMY_CATEGORIES[3],
            "rule_triggered": f"Laws Localized Stain Disruption (Delta_I={delta_i:.1f}, Spot_BG={spot_bg:.1f}, Spot_TIR={spot_tir:.2f})",
            "confidence": 0.90,
        }

    # Rule 3: Longitudinal / Warp Flaws (Vertical Orientation Dominance)
    # Height distinctly exceeds width, or vertical inertial axis (theta >= 60 deg)
    if (h >= 40.0 * k and h >= 1.4 * k * w) or (h >= 100.0 * k and h > w) or (h >= 25.0 * k and ar >= 2.5 * k and theta >= th_v):
        return {
            "category_id": 0,
            "category_name": TAXONOMY_CATEGORIES[0],
            "rule_triggered": f"Longitudinal Warp Orientation (H={h:.0f}px, W={w:.0f}px, H/W={h/max(1.0,w):.1f}, Theta={theta:.1f}deg)",
            "confidence": 0.95 if h >= 100.0 else 0.88,
        }

    # Rule 4: Localized Micro-Puncture / Hole / Thread Breakage (Default compact flaw)
    # Physical Manifestation: Compact mechanical flaw confined to 1-2 yarn crossings (Area < 80 px)
    return {
        "category_id": 2,
        "category_name": TAXONOMY_CATEGORIES[2],
        "rule_triggered": f"Compact Physical Puncture (W={w:.0f}px, H={h:.0f}px, Area={area:.0f}px, AR={ar:.2f})",
        "confidence": 0.85,
    }


def draw_taxonomy_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    descriptors: Dict[str, Any],
    classification: Dict[str, Any],
) -> np.ndarray:
    """
    Render visual diagnostic overlay displaying defect bounding box,
    principal orientation axis, invariant moment stats, and taxonomy label.
    """
    h_img, w_img = img.shape[:2]
    overlay = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cat_id = classification["category_id"]
    color = TAXONOMY_COLORS.get(cat_id, (0, 255, 0))

    # Colorize defect mask
    overlay[mask > 0] = color

    # Bounding Box
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        cv2.rectangle(overlay, (max(0, x0 - 3), max(0, y0 - 3)),
                      (min(w_img - 1, x1 + 3), min(h_img - 1, y1 + 3)), color, 2)

        # Centroid & Principal Axis
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        theta = np.radians(descriptors["theta_deg"])
        axis_len = max(20, int(max(descriptors["width"], descriptors["height"]) / 2.0))
        dx = int(axis_len * np.cos(theta))
        dy = int(axis_len * np.sin(theta))
        cv2.line(overlay, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 255, 255), 1, cv2.LINE_AA)

        # Annotation badge
        cat_name = classification["category_name"]
        text_top = f"[{cat_id}] {cat_name}"
        text_sub = f"AR={descriptors['aspect_ratio']:.1f} | Hu1={descriptors['hu1']:.2f} | dI={descriptors['delta_i']:.1f}"

        # Draw text background banner
        by0 = max(0, y0 - 36)
        cv2.rectangle(overlay, (max(0, x0 - 4), by0), (min(w_img - 1, x0 + 320), max(0, y0 - 4)), (30, 30, 30), -1)
        cv2.putText(overlay, text_top, (x0, max(12, y0 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, text_sub, (x0, max(24, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay
