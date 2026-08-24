import numpy as np

from app.ml.drift import drift_level, population_stability_index, standardized_mean_shift


def test_identical_samples_are_stable() -> None:
    reference = np.linspace(0, 1, 100)
    psi = population_stability_index(reference, reference.copy())
    assert psi < 1e-9
    assert drift_level(psi) == "stable"


def test_shifted_distribution_has_positive_psi() -> None:
    reference = np.linspace(0, 1, 200)
    current = np.linspace(0.5, 1.5, 200)
    assert population_stability_index(reference, current) > 0


def test_standardized_mean_shift_detects_change() -> None:
    reference = np.array([0.0, 1.0, 2.0, 3.0])
    current = reference + 2.0
    assert standardized_mean_shift(reference, current) > 1.0
