"""
Adaptive texture-scale estimation from DEFECT-FREE training images.

This is the central "adaptive" mechanism of the project: instead of
hand-picking texture-analysis parameters (LBP radius, GLCM window,
Gabor frequency, ...), those parameters are DERIVED, per fabric type,
from measurable statistics of that fabric type's own normal texture.

Exact procedure (see METHODOLOGY.md for the full write-up)
-------------------------------------------------------------
Given the set of defect-free TRAIN images for one fabric type:

1. Dominant spatial period P (pixels): estimated per image via the 2-D
   autocorrelation function (src/texture/local_statistics.dominant_period)
   along both axes, then averaged over all training images of that
   fabric type. P approximates the weave repetition pitch.

2. Local variance scale: the window size at which local variance
   (src/texture/local_statistics.local_variance) stabilizes (its
   coefficient of variation across the image drops below a configurable
   tolerance) gives a natural neighborhood size at which the texture
   looks statistically homogeneous.

3. Gradient statistics: the median Sobel gradient magnitude gives a
   sense of how fine/coarse the texture edges are, used as a secondary
   signal for LBP radius (finer edges -> smaller radius resolves them).

4. Gabor energy peak frequency: a small bank of candidate Gabor
   frequencies (configs/adaptive.yaml: `candidate_gabor_frequencies`) is
   applied to the training images; the frequency whose mean response
   energy is highest is the fabric's dominant spatial frequency
   f_dominant = 1 / P_gabor.

These measurements are then converted into concrete parameters by
`parameter_selection.py`. Both this module's outputs (raw measurements)
and the derived parameters are logged/saved so the mapping is fully
reproducible and auditable (per REPORT_GUIDE.md's requirement to explain
the adaptive methodology mathematically).

Leakage safety: this module must only ever be called with images drawn
from `splitter.normal_texture_training_samples` (TRAIN, defect-free).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.texture.gabor import apply_gabor_bank, build_gabor_bank
from src.texture.local_statistics import dominant_period, gradient_magnitude, local_variance
from src.utils.parallel import limit_thread_oversubscription


@dataclass
class TextureCharacteristics:
    """Measured normal-texture characteristics for one fabric type,
    averaged over its defect-free training images."""

    fabric_id: str
    n_images: int
    mean_period_x: float
    mean_period_y: float
    mean_local_variance: float
    variance_stabilization_window: int
    median_gradient_magnitude: float
    dominant_gabor_frequency: float
    gabor_energy_at_dominant: float


def _variance_stabilization_window(image: np.ndarray, candidate_windows: List[int], cv_tolerance: float) -> int:
    """Return the smallest candidate window size whose local-variance map
    has a coefficient of variation (std/mean) below `cv_tolerance`,
    i.e. the texture "looks statistically homogeneous" at that scale.
    Falls back to the largest candidate if none satisfy the tolerance.
    """
    for w in sorted(candidate_windows):
        var_map = local_variance(image, window_size=w)
        mean_var = float(np.mean(var_map))
        std_var = float(np.std(var_map))
        cv = std_var / mean_var if mean_var > 1e-8 else np.inf
        if cv <= cv_tolerance:
            return w
    return max(candidate_windows)


def _dominant_gabor_frequency(
    image: np.ndarray,
    candidate_frequencies: List[float],
    sigma: float,
    mean_period: float = 0.0,
) -> tuple:
    """Return (frequency, mean_energy) for the fabric's dominant spatial
    frequency.

    Uses the autocorrelation-derived mean_period (already computed before
    this function is called) to set the Gabor frequency as f = 1/period.
    This is the most reliable approach for the 256x4096 AITEX images:
      - The 2-D FFT radial-averaging approach fails because max_r =
        min(128, 2048) = 128, so the max representable frequency is only
        127/4096 = 0.031 cyc/px, which is below all candidate frequencies.
      - The autocorrelation period is already robustly estimated from
        multiple training images, giving 5-13 px periods.

    The result is clamped to [min(candidates), max(candidates)].
    """
    if mean_period > 0:
        freq_from_period = 1.0 / mean_period
        best_freq = float(np.clip(
            freq_from_period,
            min(candidate_frequencies),
            max(candidate_frequencies),
        ))
    else:
        # Fallback: use middle candidate
        best_freq = float(candidate_frequencies[len(candidate_frequencies) // 2])

    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    bank = build_gabor_bank(
        frequencies=[best_freq], orientations=orientations,
        sigma_x=sigma, sigma_y=sigma,
    )
    responses = apply_gabor_bank(image, bank)
    mean_energy = float(np.mean(responses["energy"]))
    return best_freq, mean_energy


def _measure_one_image(
    image: np.ndarray,
    candidate_variance_windows: List[int],
    variance_cv_tolerance: float,
    candidate_gabor_frequencies: List[float],
    gabor_sigma: float,
) -> Tuple[float, float, float, int, float, float, float]:
    """Top-level (picklable) worker: compute every per-image measurement
    (period, variance, stabilization window, gradient, Gabor frequency)
    for ONE image. Kept at module level so it can be dispatched to a
    `ProcessPoolExecutor` -- each training image's measurement is fully
    independent of every other image's, making this parallelizable
    across CPU cores with no change to the result (only to how long it
    takes)."""
    period_x = dominant_period(image, axis=1)
    period_y = dominant_period(image, axis=0)
    # Use the mean of x/y periods to drive Gabor frequency selection.
    # Prefer the shorter period (higher frequency) if the two differ a lot.
    mean_period = float(np.mean([period_x, period_y]))
    variance = float(np.mean(local_variance(image, window_size=9)))
    window = _variance_stabilization_window(image, list(candidate_variance_windows), variance_cv_tolerance)
    gradient = float(np.median(gradient_magnitude(image)))
    freq, energy = _dominant_gabor_frequency(
        image, list(candidate_gabor_frequencies), gabor_sigma,
        mean_period=mean_period,
    )
    return period_x, period_y, variance, window, gradient, freq, energy


def _measure_one_image_star(args):
    return _measure_one_image(*args)


def estimate_texture_characteristics(
    fabric_id: str,
    images: List[np.ndarray],
    candidate_variance_windows: List[int] = (5, 9, 13, 17, 21),
    variance_cv_tolerance: float = 0.35,
    candidate_gabor_frequencies: List[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
    gabor_sigma: float = 3.0,
    n_workers: int = 1,
) -> TextureCharacteristics:
    """Estimate normal-texture characteristics for one fabric type from a
    list of its defect-free (grayscale, preprocessed) training images.

    `n_workers` (default 1, fully serial -- identical behavior to before
    parallel support was added) controls how many CPU processes are used
    to measure images in parallel; see `src.pipeline.resolve_n_workers`
    for the config-driven way this is normally set from
    `configs/default.yaml`'s `compute.n_workers`.
    """
    if not images:
        raise ValueError(f"No defect-free training images provided for fabric {fabric_id}")

    if n_workers <= 1 or len(images) <= 1:
        per_image = [
            _measure_one_image(img, candidate_variance_windows, variance_cv_tolerance,
                                candidate_gabor_frequencies, gabor_sigma)
            for img in images
        ]
    else:
        tasks = [
            (img, candidate_variance_windows, variance_cv_tolerance, candidate_gabor_frequencies, gabor_sigma)
            for img in images
        ]
        with ProcessPoolExecutor(max_workers=n_workers, initializer=limit_thread_oversubscription) as executor:
            per_image = list(executor.map(_measure_one_image_star, tasks))

    periods_x = [r[0] for r in per_image]
    periods_y = [r[1] for r in per_image]
    variances = [r[2] for r in per_image]
    windows = [r[3] for r in per_image]
    gradients = [r[4] for r in per_image]
    freqs = [r[5] for r in per_image]
    energies = [r[6] for r in per_image]

    return TextureCharacteristics(
        fabric_id=fabric_id,
        n_images=len(images),
        mean_period_x=float(np.mean(periods_x)),
        mean_period_y=float(np.mean(periods_y)),
        mean_local_variance=float(np.mean(variances)),
        variance_stabilization_window=int(round(np.median(windows))),
        median_gradient_magnitude=float(np.mean(gradients)),
        dominant_gabor_frequency=float(np.median(freqs)),
        gabor_energy_at_dominant=float(np.mean(energies)),
    )


def characteristics_to_dict(tc: TextureCharacteristics) -> Dict:
    return {
        "fabric_id": tc.fabric_id,
        "n_images": tc.n_images,
        "mean_period_x": tc.mean_period_x,
        "mean_period_y": tc.mean_period_y,
        "mean_local_variance": tc.mean_local_variance,
        "variance_stabilization_window": tc.variance_stabilization_window,
        "median_gradient_magnitude": tc.median_gradient_magnitude,
        "dominant_gabor_frequency": tc.dominant_gabor_frequency,
        "gabor_energy_at_dominant": tc.gabor_energy_at_dominant,
    }
