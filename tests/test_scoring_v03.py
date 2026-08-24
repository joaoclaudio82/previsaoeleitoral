import numpy as np

from app.ml.scoring import (
    binary_log_loss,
    brier_score,
    expected_calibration_error,
    interval_coverage,
)


def test_perfect_forecast_has_zero_brier() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, y) == 0.0


def test_better_probabilities_have_lower_log_loss() -> None:
    y = np.array([0.0, 1.0])
    good = binary_log_loss(y, np.array([0.1, 0.9]))
    bad = binary_log_loss(y, np.array([0.4, 0.6]))
    assert good < bad


def test_interval_coverage() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    lower = np.array([9.0, 21.0, 25.0])
    upper = np.array([11.0, 23.0, 35.0])
    assert interval_coverage(actual, lower, upper) == 2 / 3


def test_calibration_error_is_small_for_matching_bins() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, bins=2) == 0.0
