from datetime import date
import numpy as np
import pandas as pd

from app.services.hierarchical_polls import fit_hierarchical_poll_model


def _polls() -> pd.DataFrame:
    rows = []
    for poll_id, institute, mode, uf, shares in [
        ("N1", "Alpha", "telephone", "BR", [55, 35]),
        ("N2", "Beta", "online", "BR", [51, 39]),
        ("SP1", "Alpha", "telephone", "SP", [45, 45]),
    ]:
        for candidate_id, share in zip(["A", "B"], shares):
            rows.append({
                "poll_id": poll_id, "field_date": "2026-07-20", "institute": institute,
                "collection_mode": mode, "target_population": "registered_voters",
                "candidate_id": candidate_id, "candidate_name": candidate_id,
                "share": share, "undecided_share": 10, "sample_size": 1200,
                "margin_error": 2.5, "scope": "national" if uf == "BR" else "state", "uf": uf,
            })
    return pd.DataFrame(rows)


def test_hierarchical_posterior_is_normalized_and_regional():
    priors = pd.DataFrame([
        {"uf": "SP", "candidate_id": "A", "candidate_name": "A", "prior_share": 48, "prior_strength": 3, "registered_voters": 1000},
        {"uf": "SP", "candidate_id": "B", "candidate_name": "B", "prior_share": 52, "prior_strength": 3, "registered_voters": 1000},
    ])
    posterior = fit_hierarchical_poll_model(_polls(), date(2026, 7, 21), state_priors=priors, n_draws=600, seed=7)
    assert np.allclose(posterior.national_draws.sum(axis=1), 100.0)
    assert posterior.state_ids == ["SP"]
    assert set(posterior.institute_reliability["institute"]) == {"Alpha", "Beta"}
    assert posterior.diagnostics["uses_external_institute_quality"] is False
    assert posterior.diagnostics["correlated_candidate_error"] is True
