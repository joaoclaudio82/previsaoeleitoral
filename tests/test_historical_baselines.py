from datetime import date

import numpy as np
import pandas as pd

from app.ml.historical_baselines import fit_polling_baseline


def _polls() -> pd.DataFrame:
    rows = []
    for poll_id, field_date, sample, a, b in [
        ("p1", "2022-09-01", 1000, 45.0, 55.0),
        ("p2", "2022-09-20", 2000, 48.0, 52.0),
    ]:
        rows.extend(
            [
                {"poll_id": poll_id, "field_date": field_date, "sample_size": sample, "candidate_id": "a", "candidate_name": "A", "share": a},
                {"poll_id": poll_id, "field_date": field_date, "sample_size": sample, "candidate_id": "b", "candidate_name": "B", "share": b},
            ]
        )
    return pd.DataFrame(rows)


def test_baselines_return_simplex_draws() -> None:
    for method in ("latest_poll", "simple_mean", "recency_weighted", "sample_recency_weighted"):
        result = fit_polling_baseline(_polls(), date(2022, 9, 25), method=method, n_draws=200, seed=3)
        assert result.draws.shape == (200, 2)
        assert np.allclose(result.draws.sum(axis=1), 1.0)
        assert np.all(result.draws > 0)


def test_latest_poll_tracks_latest_field_date() -> None:
    result = fit_polling_baseline(_polls(), date(2022, 9, 25), method="latest_poll", n_draws=1000, seed=4)
    assert abs(result.draws[:, 0].mean() - 0.48) < 0.03
