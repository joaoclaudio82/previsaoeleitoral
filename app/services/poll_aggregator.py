from __future__ import annotations

from datetime import date
import pandas as pd

from app.services.hierarchical_polls import fit_hierarchical_poll_model


def aggregate_polls(polls: pd.DataFrame, as_of_date: date, half_life_days: float = 24.0) -> pd.DataFrame:
    """Compatibility wrapper returning the national hierarchical posterior summary."""
    posterior = fit_hierarchical_poll_model(
        polls,
        as_of_date,
        n_draws=2_000,
        half_life_days=half_life_days,
    )
    return posterior.national_summary
