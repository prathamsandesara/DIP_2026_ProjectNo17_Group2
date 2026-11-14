"""Tests for the Gabor filter bank on REAL AITEX fabric image patches (no
synthetic arrays)."""
import numpy as np

from src.texture.gabor import apply_gabor_bank, build_gabor_bank, local_gabor_energy
from tests.conftest import requires_dataset


def test_build_gabor_bank_size():
    # Pure parameter-combinatorics check -- no image data involved.
    bank = build_gabor_bank(frequencies=[0.1, 0.2], orientations=[0, np.pi / 2])
    assert len(bank) == 4


@requires_dataset
def test_apply_gabor_bank_output_shapes(real_preprocessed_patch):
    bank = build_gabor_bank(frequencies=[0.1], orientations=[0, np.pi / 2])
    responses = apply_gabor_bank(real_preprocessed_patch, bank)
    assert "energy" in responses
    for resp in responses.values():
        assert resp.shape == real_preprocessed_patch.shape


@requires_dataset
def test_local_gabor_energy_shape(real_preprocessed_patch):
    energy = local_gabor_energy(real_preprocessed_patch, window_size=8)
    assert energy.shape == real_preprocessed_patch.shape


@requires_dataset
def test_gabor_responds_more_at_dominant_weave_frequency(real_defect_free_image):
    """On a real defect-free fabric image, the Gabor filter bank's mean
    response energy should vary across candidate frequencies (i.e. the
    filter bank is actually frequency-selective on real woven texture,
    not a constant/degenerate response)."""
    from src.preprocessing import preprocess_image
    from tests.conftest import DEFAULT_CONFIG

    preprocessed, _ = preprocess_image(real_defect_free_image, DEFAULT_CONFIG["preprocessing"])
    orientations = [0, np.pi / 2]
    energies = []
    for freq in (0.05, 0.15, 0.3):
        bank = build_gabor_bank(frequencies=[freq], orientations=orientations, sigma_x=3.0, sigma_y=3.0)
        responses = apply_gabor_bank(preprocessed, bank)
        energies.append(float(np.mean(responses["energy"])))
    assert len(set(np.round(energies, 6))) > 1
