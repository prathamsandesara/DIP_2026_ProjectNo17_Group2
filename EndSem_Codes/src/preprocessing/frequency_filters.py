"""
Frequency-Domain Filtering for Fabric Defect Detection & Weave Pattern Suppression.

ECE501 Module 2 (Part 2) & Module 3 (Week 6): Fourier Transform & Frequency Filtering
-------------------------------------------------------------------------------------
Fabric has a dominant periodic weave structure (yarn crossing at known spatial frequencies).
In the 2D Fourier spectrum (np.fft.fft2), this periodic grid manifests as sharp, high-energy
harmonic impulses at discrete coordinates (u_k, v_k).

Defects (oil stains, holes, missing picks, knots), in contrast, are aperiodic and transient,
meaning their energy is distributed smoothly across a broad continuum of frequencies.

This module implements:
1. Automated 2D FFT harmonic peak detection: identifies the fundamental warp and weft
   harmonic coordinates (u_k, v_k) from the 2D power spectrum without manual tuning.
2. Smooth 2D Butterworth cross-notch band-reject filtering: attenuates the periodic weave
   peaks and their symmetric conjugates without introducing Gibbs ringing artifacts.
3. 2D Butterworth band-reject filtering for isotropic radial suppression.
4. Comprehensive energy suppression diagnostics and residual reconstruction.

Mathematical formulation (Gonzalez & Woods, 4th ed., Sec. 4.8 - 4.10):
    F(u, v) = FFT2{ f(x, y) }
    G(u, v) = F(u, v) * H_notch(u, v)
    g(x, y) = Re{ IFFT2{ G(u, v) } }
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import maximum_filter


# ---------------------------------------------------------------------------
# 1. Automated 2D FFT Harmonic Peak Detection
# ---------------------------------------------------------------------------

def detect_harmonic_peaks_2d(
    image: np.ndarray,
    dc_radius: int = 15,
    neighborhood_size: int = 11,
    std_multiplier: float = 2.0,
    max_peaks: int = 16,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Automatically identify periodic weave harmonic coordinates in the 2D DFT spectrum.

    Parameters:
        image: 2D array of pixel intensities.
        dc_radius: Radius around DC (center) to exclude from peak searching.
        neighborhood_size: Footprint for local non-maximum suppression.
        std_multiplier: Threshold multiplier above spectral mean (mean + k*std).
        max_peaks: Maximum number of dominant harmonic pairs to retain.

    Returns:
        peaks: List of (row, col) coordinates in the fftshifted spectrum (upper half-plane).
        diagnostics: Dict containing centered log-power spectrum and DC coordinates.
    """
    img = image.astype(np.float64)
    h, w = img.shape
    cy, cx = h // 2, w // 2

    # 2D FFT with DC centered
    F_shift = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(F_shift)
    log_power = np.log1p(power)

    # Exclude DC region
    Y, X = np.ogrid[:h, :w]
    dist_dc = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    power_masked = power.copy()
    power_masked[dist_dc < dc_radius] = 0.0

    # Local maximum filter for isolated peak finding
    local_max = maximum_filter(power_masked, size=neighborhood_size)
    is_peak = (power_masked == local_max) & (power_masked > 0.0)

    # Exclude lower half-plane and left half of DC line (conjugate symmetry of real signals: F(-u, -v) = F*(u, v))
    is_peak[cy + 1 :, :] = False
    is_peak[cy, : cx + 1] = False

    # Statistical significance threshold
    bg_vals = power_masked[dist_dc >= dc_radius]
    thresh = float(np.mean(bg_vals) + std_multiplier * np.std(bg_vals))
    candidate_indices = np.argwhere(is_peak & (power_masked > thresh))

    # Sort candidates by descending spectral energy
    sorted_peaks = sorted(
        candidate_indices,
        key=lambda pt: float(power_masked[pt[0], pt[1]]),
        reverse=True,
    )
    selected_peaks = [(int(pt[0]), int(pt[1])) for pt in sorted_peaks[:max_peaks]]

    diagnostics = {
        "dc_center": (cy, cx),
        "log_power": log_power,
        "power_threshold": thresh,
        "total_candidates_found": len(candidate_indices),
        "selected_peaks": selected_peaks,
    }
    return selected_peaks, diagnostics


# ---------------------------------------------------------------------------
# 2. 2D Butterworth Cross-Notch Filter Synthesis
# ---------------------------------------------------------------------------

def synthesize_butterworth_notch_filter(
    shape: Tuple[int, int],
    peaks: List[Tuple[int, int]],
    radius: float = 12.0,
    order: int = 3,
) -> np.ndarray:
    """
    Build a smooth, continuous 2D Butterworth band-reject cross-notch filter mask H(u, v).
    Each peak (u_k, v_k) and its complex conjugate (-u_k, -v_k) are smoothly attenuated.

    H_k(u, v) = 1 / ( 1 + [ radius^2 / (D_k * D_{-k}) ]^order )
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]

    H = np.ones((h, w), dtype=np.float64)
    eps = 1e-8

    for y_p, x_p in peaks:
        # Distance to positive harmonic peak
        d_pos = np.sqrt((Y - y_p) ** 2 + (X - x_p) ** 2)

        # Distance to conjugate peak (symmetric reflection through DC center)
        y_neg = 2 * cy - y_p
        x_neg = 2 * cx - x_p
        d_neg = np.sqrt((Y - y_neg) ** 2 + (X - x_neg) ** 2)

        d_prod = d_pos * d_neg
        H_k = 1.0 / (1.0 + (radius ** 2 / (d_prod + eps)) ** order)
        H *= H_k

    return H


# ---------------------------------------------------------------------------
# 3. End-to-End Weave Pattern Suppression
# ---------------------------------------------------------------------------

def suppress_weave_pattern_2d(
    image: np.ndarray,
    notch_radius: float = 12.0,
    order: int = 3,
    max_peaks: int = 16,
    std_multiplier: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    End-to-end frequency-domain periodic weave suppression.
    Automatically detects weave harmonics in the 2D DFT spectrum, applies a smooth
    Butterworth cross-notch filter, and returns the aperiodic defect residual.

    Returns:
        residual: Spatial domain image with weave grid attenuated.
        diagnostics: Full quantitative metrics including harmonic energy suppression ratio.
    """
    img = image.astype(np.float64)
    h, w = img.shape
    cy, cx = h // 2, w // 2

    # Step 1: Forward 2D FFT
    F_shift = np.fft.fftshift(np.fft.fft2(img))
    power_before = np.abs(F_shift) ** 2

    # Step 2: Automated Harmonic Peak Finding
    peaks, diag = detect_harmonic_peaks_2d(
        image,
        max_peaks=max_peaks,
        std_multiplier=std_multiplier,
    )

    if len(peaks) == 0:
        # No significant periodic peaks detected (already smooth or uniform)
        return image.copy(), {
            "suppression_ratio": 0.0,
            "peaks_detected": 0,
            "filter_applied": False,
        }

    # Step 3: Synthesize Butterworth Cross-Notch Filter
    H_notch = synthesize_butterworth_notch_filter(
        (h, w), peaks, radius=notch_radius, order=order
    )

    # Step 4: Multiply in Frequency Domain & Inverse FFT
    G_shift = F_shift * H_notch
    residual = np.fft.ifft2(np.fft.ifftshift(G_shift)).real

    # Step 5: Compute Quantitative Harmonic Energy Suppression Ratio
    Y, X = np.ogrid[:h, :w]
    harmonic_mask = np.zeros((h, w), dtype=bool)
    for y_p, x_p in peaks:
        y_neg, x_neg = 2 * cy - y_p, 2 * cx - x_p
        harmonic_mask |= (np.sqrt((Y - y_p) ** 2 + (X - x_p) ** 2) < notch_radius)
        harmonic_mask |= (np.sqrt((Y - y_neg) ** 2 + (X - x_neg) ** 2) < notch_radius)

    energy_before = float(np.sum(power_before[harmonic_mask]))
    energy_after = float(np.sum(np.abs(G_shift[harmonic_mask]) ** 2))
    suppression_ratio = float(
        (1.0 - energy_after / (energy_before + 1e-8)) * 100.0
    ) if energy_before > 0 else 0.0

    diag_out = {
        "peaks_detected": len(peaks),
        "peak_coordinates": peaks,
        "harmonic_energy_suppression_pct": round(suppression_ratio, 2),
        "raw_std": round(float(np.std(image)), 3),
        "residual_std": round(float(np.std(residual)), 3),
        "filter_applied": True,
        "log_power_before": diag["log_power"],
        "log_power_after": np.log1p(np.abs(G_shift)),
        "filter_mask": H_notch,
    }

    return residual.astype(image.dtype), diag_out


# ---------------------------------------------------------------------------
# 4. Butterworth Radial Band-Reject Filter (Module 2 Pt 2)
# ---------------------------------------------------------------------------

def _butterworth_band_reject_mask(
    shape: Tuple[int, int],
    center_freq: float,
    bandwidth: float,
    order: int = 2,
) -> np.ndarray:
    """
    Build a 2-D Butterworth band-reject filter mask H(u,v) of size `shape`.
    """
    rows, cols = shape
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U ** 2 + V ** 2)

    D0 = center_freq
    BW = bandwidth
    eps = 1e-8
    denom = D ** 2 - D0 ** 2
    ratio = (D * BW) / (np.abs(denom) + eps)
    H = 1.0 / (1.0 + ratio ** (2 * order))
    return H


def butterworth_band_reject(
    image: np.ndarray,
    center_freq: float,
    bandwidth: float = 0.03,
    order: int = 2,
) -> np.ndarray:
    """
    Apply a Butterworth band-reject filter to suppress the periodic weave
    component at `center_freq` (cycles/pixel).
    """
    img = image.astype(np.float64)
    F = np.fft.fft2(img)
    H = _butterworth_band_reject_mask(img.shape, center_freq, bandwidth, order)
    G = F * H
    result = np.fft.ifft2(G).real
    return result.astype(image.dtype)


def notch_reject(
    image: np.ndarray,
    notch_centers: List[Tuple[float, float]],
    radius: float = 0.02,
) -> np.ndarray:
    """
    Notch-reject filter for specific normalized frequency components.
    """
    img = image.astype(np.float64)
    rows, cols = img.shape
    F = np.fft.fft2(img)

    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    V, U = np.meshgrid(v, u)

    H = np.ones((rows, cols), dtype=np.float64)
    for (u0, v0) in notch_centers:
        dist_pos = np.sqrt((U - u0) ** 2 + (V - v0) ** 2)
        dist_neg = np.sqrt((U + u0) ** 2 + (V + v0) ** 2)
        H[dist_pos < radius] = 0.0
        H[dist_neg < radius] = 0.0

    G = F * H
    result = np.fft.ifft2(G).real
    return result.astype(image.dtype)


def apply_band_reject_to_deviation(
    deviation_map: np.ndarray,
    dominant_freq: float,
    bandwidth: float = 0.03,
    order: int = 2,
) -> np.ndarray:
    """
    Convenience wrapper: apply Butterworth band-reject to a texture-deviation map.
    """
    if dominant_freq <= 0:
        return deviation_map
    filtered = butterworth_band_reject(deviation_map, dominant_freq, bandwidth, order)
    return np.clip(filtered, 0, None).astype(deviation_map.dtype)
