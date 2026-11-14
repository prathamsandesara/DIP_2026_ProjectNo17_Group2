"""
Gabor filtering: joint spatial/frequency texture analysis.

The 2-D Gabor function
------------------------
A Gabor filter is a sinusoidal plane wave modulated by a Gaussian envelope:

    g(x, y; lambda, theta, psi, sigma, gamma) =
        exp( -(x'^2 + gamma^2 y'^2) / (2 sigma^2) ) * cos(2*pi*x'/lambda + psi)

    x' =  x cos(theta) + y sin(theta)
    y' = -x sin(theta) + y cos(theta)

where lambda is the wavelength (inverse spatial frequency) of the
sinusoid, theta is its orientation, psi is the phase offset, sigma is the
Gaussian envelope's spatial extent (bandwidth/scale), and gamma is the
spatial aspect ratio. Convolving an image with g gives a response that is
large where the image locally matches that specific spatial frequency and
orientation, and small otherwise.

Why Gabor for fabric texture
-------------------------------
Woven fabric is close to a locally periodic texture: threads repeat at a
fairly stable spatial frequency (the weave pitch) and a small number of
dominant orientations (warp/weft). Gabor filters are the classical
frequency-domain complement to LBP/GLCM because they directly probe *is
there periodic structure at frequency f and orientation theta here*,
which is precisely what a defect (float, broken pick, hole) disrupts --
it locally breaks or shifts the periodicity. Because Gabor filters are
also spatially localized (via the Gaussian envelope), the response is a
per-pixel map, not just a global frequency spectrum, giving spatial
localization capability that a global Fourier transform lacks.

Multi-scale / multi-orientation Gabor bank
---------------------------------------------
A single (lambda, theta) captures one frequency/orientation combination.
A *bank* of filters spanning several wavelengths (scales) and orientations
is used so that: (a) different fabric weaves (different pitch) are
matched by an appropriate wavelength, and (b) both principal weave
directions are captured. The energy of each response (or its statistics)
forms the Gabor feature set used in normal-texture modeling and deviation
computation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve


def build_gabor_bank(
    frequencies: List[float] = (0.1, 0.2, 0.3),
    orientations: List[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    sigma_x: float = 3.0,
    sigma_y: float = 3.0,
) -> List[Tuple[float, float, np.ndarray]]:
    """Construct a bank of Gabor kernels spanning `frequencies` (spatial
    frequency = 1/lambda) x `orientations`. Returns a list of
    (frequency, orientation, complex kernel) tuples."""
    bank = []
    for freq in frequencies:
        for theta in orientations:
            kernel = gabor_kernel(frequency=freq, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y)
            bank.append((freq, theta, kernel))
    return bank


def apply_gabor_bank(image: np.ndarray, bank: List[Tuple[float, float, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Convolve `image` with every kernel in the bank; return the magnitude
    response map for each (frequency, orientation) pair, keyed by a
    readable string, plus an aggregate "energy" map (sum of squared
    responses across the whole bank -- a scale/orientation-invariant
    texture-richness map)."""
    responses: Dict[str, np.ndarray] = {}
    energy = np.zeros_like(image, dtype=np.float64)

    for freq, theta, kernel in bank:
        real = convolve(image.astype(np.float64), np.real(kernel), mode="reflect")
        imag = convolve(image.astype(np.float64), np.imag(kernel), mode="reflect")
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        key = f"f{freq:.3f}_t{np.degrees(theta):.0f}"
        responses[key] = magnitude
        energy += magnitude ** 2

    responses["energy"] = energy
    return responses


def local_gabor_energy(response_map: np.ndarray, window_size: int = 16) -> np.ndarray:
    """Local (windowed) mean energy of a Gabor response map, via uniform
    filtering -- a smoothed, spatially-local summary suitable for
    comparison against the normal-texture reference."""
    from scipy.ndimage import uniform_filter

    return uniform_filter(response_map, size=window_size)
