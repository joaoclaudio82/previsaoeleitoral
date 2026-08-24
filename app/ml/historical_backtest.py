from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.data.candidate_identity import canonical_candidate
from app.data.geography import BRAZIL_UF_SET
from app.data.historical_snapshots import build_snapshots
from app.data.historical_state_priors import build_state_priors
from app.services.hierarchical_polls import PollPosterior, fit_hierarchical_poll_model


@dataclass(frozen=True, slots=True)
class HistoricalBacktestResult:
    forecasts: pd.DataFrame
    snapshot_summary: pd.DataFrame
    state_summary: pd.DataFrame


def _candidate_frame(polls: pd.DataFrame) -> pd.DataFrame:
    return polls[["candidate_id", "candidate_name"]].drop_duplicates().reset_index(drop=True)


def _actual_selected(results: pd.DataFrame, candidate_names: list[str], round_number: int = 1) -> pd.DataFrame:
    frame = results[pd.to_numeric(results["round"], errors="coerce") == round_number].copy()
    frame["uf"] = frame["uf"].astype(str).str.upper().str.strip()
    frame = frame[frame["uf"].isin(BRAZIL_UF_SET)].copy()
    frame["canonical"] = frame["candidate_name"].map(canonical_candidate)
    wanted = {canonical_candidate(name) for name in candidate_names}
    frame = frame[frame["canonical"].isin(wanted)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["uf", "canonical", "actual_share"])
    grouped = frame.groupby(["uf", "canonical"], as_index=False)["votes"].sum()
    state_totals = grouped.groupby("uf")["votes"].transform("sum")
    grouped["actual_share"] = grouped["votes"] / state_totals.where(state_totals > 0)
    national = grouped.groupby("canonical", as_index=False)["votes"].sum()
    national["uf"] = "BR"
    national["actual_share"] = national["votes"] / national["votes"].sum()
    return pd.concat([grouped, national], ignore_index=True)


def _posterior_records(
    posterior: PollPosterior,
    actual: pd.DataFrame,
    *,
    election_year: int,
    snapshot_date: date,
    days_before_election: int,
    scorable: bool,
) -> list[dict[str, object]]:
    candidate_canonical = [canonical_candidate(name) for name in posterior.candidate_names]
    actual_lookup = actual.set_index(["uf", "canonical"])["actual_share"].to_dict() if not actual.empty else {}
    records: list[dict[str, object]] = []

    def append_geo(uf: str, draws: np.ndarray, level: str) -> None:
        winners = np.argmax(draws, axis=1)
        actual_values = np.array([actual_lookup.get((uf, key), np.nan) for key in candidate_canonical], dtype=float)
        valid_actual = np.isfinite(actual_values)
        actual_winner = int(np.nanargmax(actual_values)) if valid_actual.any() and np.nansum(actual_values) > 0 else -1
        for idx, (candidate_id, candidate_name, canonical) in enumerate(
            zip(posterior.candidate_ids, posterior.candidate_names, candidate_canonical)
        ):
            values = draws[:, idx] / 100.0
            records.append(
                {
                    "election_year": election_year,
                    "snapshot_date": snapshot_date.isoformat(),
                    "days_before_election": days_before_election,
                    "scorable": bool(scorable),
                    "uf": uf,
                    "level": level,
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "canonical_candidate": canonical,
                    "win_probability": float(np.mean(winners == idx)),
                    "predicted_share": float(values.mean()),
                    "lower": float(np.quantile(values, 0.05)),
                    "upper": float(np.quantile(values, 0.95)),
                    "actual_share": float(actual_values[idx]) if np.isfinite(actual_values[idx]) else np.nan,
                    "outcome": int(idx == actual_winner) if scorable and actual_winner >= 0 else np.nan,
                }
            )

    append_geo("BR", posterior.national_draws, "national")
    for state_index, uf in enumerate(posterior.state_ids):
        if uf not in BRAZIL_UF_SET:
            continue
        append_geo(uf, posterior.state_draws[:, state_index, :], "state")
    return records


def _scored(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["scorable"] & frame["outcome"].notna() & frame["actual_share"].notna()].copy()


def _snapshot_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _scored(frame)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (year, days, level), group in frame.groupby(["election_year", "days_before_election", "level"]):
        probability = np.clip(group["win_probability"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
        outcome = group["outcome"].to_numpy(dtype=float)
        rows.append(
            {
                "election_year": int(year),
                "days_before_election": int(days),
                "level": str(level),
                "observations": len(group),
                "brier": float(np.mean(np.square(probability - outcome))),
                "log_loss": float(-np.mean(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability))),
                "vote_share_mae": float(np.mean(np.abs(group["predicted_share"] - group["actual_share"]))),
                "interval_coverage": float(np.mean((group["actual_share"] >= group["lower"]) & (group["actual_share"] <= group["upper"]))),
            }
        )
    return pd.DataFrame(rows).sort_values(["election_year", "days_before_election", "level"]).reset_index(drop=True)


def _state_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    states = _scored(frame)
    states = states[(states["level"] == "state") & states["uf"].isin(BRAZIL_UF_SET)].copy()
    if states.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (year, uf), group in states.groupby(["election_year", "uf"]):
        rows.append(
            {
                "election_year": int(year),
                "uf": str(uf),
                "observations": len(group),
                "vote_share_mae": float(np.mean(np.abs(group["predicted_share"] - group["actual_share"]))),
                "interval_coverage": float(np.mean((group["actual_share"] >= group["lower"]) & (group["actual_share"] <= group["upper"]))),
                "winner_brier": float(np.mean(np.square(group["win_probability"] - group["outcome"]))),
            }
        )
    return pd.DataFrame(rows).sort_values(["election_year", "uf"]).reset_index(drop=True)


def run_historical_backtest(
    polls: pd.DataFrame,
    official_results: pd.DataFrame,
    election_date: date,
    *,
    election_year: int,
    previous_results: pd.DataFrame | None = None,
    scoring_start_date: date | None = None,
    offsets: tuple[int, ...] = (180, 120, 90, 60, 30, 15, 7, 3, 1),
    posterior_draws: int = 4000,
    seed: int = 42,
) -> HistoricalBacktestResult:
    snapshots = build_snapshots(polls, election_date, offsets=offsets)
    if not snapshots:
        raise ValueError(f"No historical snapshots available for {election_year}")

    records: list[dict[str, object]] = []
    for index, (days, snapshot) in enumerate(sorted(snapshots.items(), reverse=True)):
        snapshot_date = election_date - timedelta(days=days)
        snapshot_candidates = _candidate_frame(snapshot)
        state_priors = None
        if previous_results is not None and not previous_results.empty:
            previous_first_round = previous_results[pd.to_numeric(previous_results["round"], errors="coerce") == 1]
            state_priors = build_state_priors(previous_first_round, snapshot_candidates)
        actual = _actual_selected(official_results, snapshot_candidates["candidate_name"].tolist(), round_number=1)
        scorable = scoring_start_date is None or snapshot_date >= scoring_start_date
        posterior = fit_hierarchical_poll_model(
            snapshot,
            snapshot_date,
            state_priors=state_priors,
            calibration=None,
            n_draws=posterior_draws,
            seed=seed + index,
        )
        records.extend(
            _posterior_records(
                posterior,
                actual,
                election_year=election_year,
                snapshot_date=snapshot_date,
                days_before_election=days,
                scorable=scorable,
            )
        )
    forecasts = pd.DataFrame(records)
    return HistoricalBacktestResult(
        forecasts=forecasts,
        snapshot_summary=_snapshot_metrics(forecasts),
        state_summary=_state_metrics(forecasts),
    )
