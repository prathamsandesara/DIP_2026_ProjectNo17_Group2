"""
Local (windowed) first-order statistical image analysis: mean, variance,
gradient magnitude, and autocorrelation-based periodicity estimation.

These classical statistics are simpler than LBP/GLCM/Gabor but are the
basis for (a) adaptive scale estimation (src/adaptive/scale_estimation.py)
and (b) a lightweight, fast texture-uniformity signal fused into the
deviation map.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, sobel


def local_mean(image: np.ndarray, window_size: int = 9) -> np.ndarray:
    return uniform_filter(image.astype(np.float64), size=window_size)


def local_variance(image: np.ndarray, window_size: int = 9) -> np.ndarray:
    """Var(I) = E[I^2] - E[I]^2, computed with local box filters."""
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=window_size)
    mean_sq = uniform_filter(img ** 2, size=window_size)
    var = mean_sq - mean ** 2
    return np.clip(var, 0, None)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude: sqrt(Gx^2 + Gy^2)."""
    img = image.astype(np.float64)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return np.hypot(gx, gy)


def autocorrelation_2d(image: np.ndarray) -> np.ndarray:
    """Normalized 2-D autocorrelation via the Wiener-Khinchin theorem:

        R = IFFT( FFT(I) * conj(FFT(I)) )

    computed on the mean-subtracted image and normalized so R(0,0) = 1.
    Used to estimate the dominant weave repetition period (see
    src/adaptive/scale_estimation.py) -- periodic fabric texture produces
    strong secondary autocorrelation peaks at the weave pitch.
    """
    img = image.astype(np.float64) - np.mean(image)
    fft = np.fft.fft2(img)
    power_spectrum = fft * np.conj(fft)
    autocorr = np.fft.ifft2(power_spectrum).real
    autocorr = np.fft.fftshift(autocorr)
    peak = autocorr.max()
    if peak > 1e-12:
        autocorr = autocorr / peak
    return autocorr


def dominant_period(image: np.ndarray, axis: int = 1, min_lag: int = 2) -> float:
    """Estimate the dominant repetition period (in pixels) along `axis`
    from the 1-D marginal autocorrelation profile through the image
    center, by locating the first strong secondary peak beyond `min_lag`.
    """
    autocorr = autocorrelation_2d(image)
    center = autocorr.shape[axis] // 2
    profile = np.take(autocorr, indices=range(autocorr.shape[axis]), axis=axis)
    if axis == 1:
        profile = profile[autocorr.shape[0] // 2, :]
    else:
        profile = profile[:, autocorr.shape[1] // 2]

    half = profile[center + min_lag:]
    if len(half) < 3:
        return float(min_lag)
    # first local maximum
    for i in range(1, len(half) - 1):
        if half[i] > half[i - 1] and half[i] >= half[i + 1]:
            return float(i + min_lag)
    return float(np.argmax(half) + min_lag)
