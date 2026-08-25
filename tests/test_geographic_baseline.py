import numpy as np
import pandas as pd

from app.ml.geographic_baseline import apply_national_swing_state_lean


def test_geographic_baseline_preserves_simplex_and_applies_lean() -> None:
    national = np.tile(np.array([[0.45, 0.40, 0.15]]), (1000, 1))
    priors = pd.DataFrame([
        {"uf": "CE", "candidate_id": "a", "candidate_name": "A", "prior_share": 60.0, "national_prior_share": 45.0, "prior_concentration": 5000.0},
        {"uf": "CE", "candidate_id": "b", "candidate_name": "B", "prior_share": 30.0, "national_prior_share": 40.0, "prior_concentration": 5000.0},
        {"uf": "CE", "candidate_id": "c", "candidate_name": "C", "prior_share": 10.0, "national_prior_share": 15.0, "prior_concentration": 5000.0},
    ])

    forecast = apply_national_swing_state_lean(
        national,
        ["a", "b", "c"],
        ["A", "B", "C"],
        priors,
        seed=12,
    )

    assert forecast.state_ids == ["CE"]
    assert forecast.state_draws.shape == (1000, 1, 3)
    assert np.allclose(forecast.state_draws.sum(axis=2), 1.0)
    assert forecast.state_draws[:, 0, 0].mean() > national[:, 0].mean()
    assert forecast.state_draws[:, 0, 1].mean() < national[:, 1].mean()


def test_no_lean_is_close_to_national_distribution() -> None:
    national = np.tile(np.array([[0.50, 0.35, 0.15]]), (2000, 1))
    priors = pd.DataFrame([
        {"uf": "SP", "candidate_id": "a", "candidate_name": "A", "prior_share": 50.0, "national_prior_share": 50.0, "prior_concentration": 10000.0},
        {"uf": "SP", "candidate_id": "b", "candidate_name": "B", "prior_share": 35.0, "national_prior_share": 35.0, "prior_concentration": 10000.0},
        {"uf": "SP", "candidate_id": "c", "candidate_name": "C", "prior_share": 15.0, "national_prior_share": 15.0, "prior_concentration": 10000.0},
    ])

    forecast = apply_national_swing_state_lean(national, ["a", "b", "c"], ["A", "B", "C"], priors, seed=33)
    assert np.allclose(forecast.state_draws[:, 0, :].mean(axis=0), national.mean(axis=0), atol=0.01)
