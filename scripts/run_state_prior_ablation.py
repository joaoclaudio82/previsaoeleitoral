from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.candidate_identity import canonical_candidate
from app.data.historical_manifest import get_election
from app.data.historical_snapshots import build_snapshots
from app.data.historical_state_priors import build_state_priors
from app.ml.posterior_predictive import add_correlated_predictive_error
from app.services.hierarchical_polls import fit_hierarchical_poll_model


def _neutral_priors(previous_results: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    ufs = sorted(set(previous_results["uf"].dropna().astype(str)) - {"BR", "ZZ"})
    k = max(len(candidates), 1)
    rows = []
    for uf in ufs:
        for candidate in candidates.itertuples(index=False):
            rows.append({
                "uf": uf,
                "candidate_id": str(candidate.candidate_id),
                "candidate_name": str(candidate.candidate_name),
                "prior_share": 100.0 / k,
                "national_prior_share": 100.0 / k,
                "prior_strength": 2.5,
                "prior_concentration": 50.0,
                "prior_source": "neutral_ablation",
            })
    return pd.DataFrame(rows)


def _actual_state(results: pd.DataFrame, uf: str, candidate_names: list[str]) -> dict[str, float]:
    frame = results[pd.to_numeric(results["round"], errors="coerce") == 1].copy()
    frame = frame[frame["uf"].astype(str).str.upper() == uf].copy()
    frame["canonical"] = frame["candidate_name"].map(canonical_candidate)
    wanted = {canonical_candidate(name) for name in candidate_names}
    frame = frame[frame["canonical"].isin(wanted)]
    totals = frame.groupby("canonical")["votes"].sum()
    total = float(totals.sum())
    return {str(key): float(value / total) for key, value in totals.items()} if total > 0 else {}


def _score(draws: np.ndarray, candidate_names: list[str], actual: dict[str, float]) -> tuple[float, float, float]:
    actual_values = np.array([actual.get(canonical_candidate(name), np.nan) for name in candidate_names])
    valid = np.isfinite(actual_values)
    if not valid.any():
        return np.nan, np.nan, np.nan
    predictions = draws.mean(axis=0) / 100.0
    lower = np.quantile(draws, 0.05, axis=0) / 100.0
    upper = np.quantile(draws, 0.95, axis=0) / 100.0
    mae = float(np.nanmean(np.abs(predictions[valid] - actual_values[valid])))
    coverage = float(np.nanmean((actual_values[valid] >= lower[valid]) & (actual_values[valid] <= upper[valid])))
    actual_winner = int(np.nanargmax(actual_values))
    winner_prob = np.bincount(np.argmax(draws, axis=1), minlength=draws.shape[1]) / len(draws)
    outcome = np.arange(draws.shape[1]) == actual_winner
    brier = float(np.mean(np.square(winner_prob - outcome.astype(float))))
    return mae, coverage, brier


def main() -> int:
    parser = argparse.ArgumentParser(description="Symmetric ablation of historical state leans")
    parser.add_argument("--years", nargs="+", type=int, default=[2014, 2018, 2022])
    parser.add_argument("--offsets", nargs="+", type=int, default=[15, 7, 3, 1])
    parser.add_argument("--data-root", type=Path, default=Path("data/historical"))
    parser.add_argument("--output", type=Path, default=Path("reports/state_prior_ablation"))
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=9090)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for year in args.years:
        election = get_election(year)
        processed = args.data_root / str(year) / "processed"
        polls = pd.read_csv(processed / "polls_model_schema.csv")
        results = pd.read_csv(processed / "presidential_results.csv")
        previous_year = max((candidate for candidate in (2010, 2014, 2018) if candidate < year), default=None)
        if previous_year is None:
            continue
        previous = pd.read_csv(args.data_root / str(previous_year) / "processed" / "presidential_results.csv")
        previous = previous[pd.to_numeric(previous["round"], errors="coerce") == 1].copy()
        snapshots = build_snapshots(polls, election.first_round_date, offsets=tuple(args.offsets))

        for snap_index, (days, snapshot) in enumerate(sorted(snapshots.items(), reverse=True)):
            cutoff = election.first_round_date - timedelta(days=int(days))
            columns = ["candidate_id", "candidate_name"] + (["party"] if "party" in snapshot.columns else [])
            candidates = snapshot[columns].drop_duplicates().sort_values("candidate_id")
            priors = {
                "historical_state_lean": build_state_priors(previous, candidates),
                "neutral_state_prior": _neutral_priors(previous, candidates),
            }
            for model_index, (model, prior) in enumerate(priors.items()):
                posterior = fit_hierarchical_poll_model(
                    snapshot,
                    cutoff,
                    state_priors=prior,
                    calibration=None,
                    forecast_date=election.first_round_date,
                    n_draws=args.draws,
                    seed=args.seed + year + snap_index * 20 + model_index,
                )
                _, state_draws = add_correlated_predictive_error(
                    posterior.national_draws,
                    posterior.state_draws,
                    posterior.residual_covariance,
                    seed=args.seed + year * 100 + snap_index * 20 + model_index,
                )
                for state_index, uf in enumerate(posterior.state_ids):
                    if uf in {"BR", "ZZ"}:
                        continue
                    actual = _actual_state(results, uf, posterior.candidate_names)
                    mae, coverage, brier = _score(state_draws[:, state_index, :], posterior.candidate_names, actual)
                    rows.append({
                        "model": model,
                        "election_year": year,
                        "days_before_election": int(days),
                        "uf": uf,
                        "vote_share_mae": mae,
                        "interval_coverage": coverage,
                        "winner_brier": brier,
                    })

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "state_prior_ablation_by_state.csv", index=False)
    aggregate = frame.groupby("model", as_index=False).agg(
        vote_share_mae=("vote_share_mae", "mean"),
        interval_coverage=("interval_coverage", "mean"),
        winner_brier=("winner_brier", "mean"),
        observations=("vote_share_mae", "size"),
    )
    aggregate.to_csv(args.output / "state_prior_ablation_aggregate.csv", index=False)
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
