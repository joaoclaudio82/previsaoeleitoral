from datetime import date

import numpy as np
import pandas as pd

from app.ml.dynamic_polling import fit_dynamic_polling_baseline


def _poll_rows(poll_id: str, field_date: str, shares: tuple[float, float, float]) -> list[dict[str, object]]:
    rows = []
    for candidate_id, candidate_name, share in zip(("a", "b", "c"), ("A", "B", "C"), shares):
        rows.append({
            "poll_id": poll_id,
            "field_date": field_date,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "share": share,
            "margin_error": 2.0,
            "sample_size": 1000,
        })
    return rows


def test_dynamic_forecast_is_normalized_and_leakage_safe() -> None:
    rows = []
    rows += _poll_rows("p1", "2022-09-10", (42.0, 38.0, 20.0))
    rows += _poll_rows("p2", "2022-09-20", (44.0, 37.0, 19.0))
    base = pd.DataFrame(rows)
    with_future = pd.concat([
        base,
        pd.DataFrame(_poll_rows("future", "2022-09-30", (60.0, 25.0, 15.0))),
    ], ignore_index=True)

    cutoff = date(2022, 9, 20)
    target = date(2022, 10, 2)
    first = fit_dynamic_polling_baseline(base, cutoff, forecast_date=target, n_draws=500, seed=77)
    second = fit_dynamic_polling_baseline(with_future, cutoff, forecast_date=target, n_draws=500, seed=77)

    assert np.allclose(first.draws, second.draws)
    assert np.allclose(first.draws.sum(axis=1), 1.0)
    assert np.all(first.draws > 0)


def test_dynamic_uncertainty_grows_with_forecast_horizon() -> None:
    rows = []
    rows += _poll_rows("p1", "2022-09-10", (42.0, 38.0, 20.0))
    rows += _poll_rows("p2", "2022-09-20", (44.0, 37.0, 19.0))
    polls = pd.DataFrame(rows)
    cutoff = date(2022, 9, 20)

    now = fit_dynamic_polling_baseline(polls, cutoff, forecast_date=cutoff, n_draws=2000, seed=91)
    later = fit_dynamic_polling_baseline(polls, cutoff, forecast_date=date(2022, 10, 5), n_draws=2000, seed=91)

    assert later.draws[:, 0].std() > now.draws[:, 0].std()
