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
from app.ml.historical_backtest import run_historical_backtest
from app.ml.historical_baselines import fit_polling_baseline
from app.ml.scoring import binary_log_loss, brier_score, expected_calibration_error, interval_coverage, mean_absolute_error
from app.services.hierarchical_polls import fit_hierarchical_poll_model


BASELINES = ("latest_poll", "simple_mean", "recency_weighted", "sample_recency_weighted")


def _load_year(root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = root / str(year) / "processed"
    return pd.read_csv(processed / "polls_model_schema.csv"), pd.read_csv(processed / "presidential_results.csv")


def _actual_lookup(results: pd.DataFrame, candidate_names: list[str]) -> dict[str, float]:
    frame = results[pd.to_numeric(results["round"], errors="coerce") == 1].copy()
    frame["canonical"] = frame["candidate_name"].map(canonical_candidate)
    wanted = {canonical_candidate(name) for name in candidate_names}
    frame = frame[frame["canonical"].isin(wanted)]
    national = frame.groupby("canonical", as_index=False)["votes"].sum()
    total = national["votes"].sum()
    if total <= 0:
        return {}
    return dict(zip(national["canonical"], national["votes"] / total))


def _records_from_draws(draws: np.ndarray, candidate_ids: list[str], candidate_names: list[str], actual: dict[str, float], *, model: str, year: int, days: int, snapshot_date: str, scorable: bool) -> list[dict[str, object]]:
    winners = np.argmax(draws, axis=1)
    actual_values = np.array([actual.get(canonical_candidate(name), np.nan) for name in candidate_names])
    actual_winner = int(np.nanargmax(actual_values)) if np.isfinite(actual_values).any() else -1
    rows: list[dict[str, object]] = []
    for idx, (candidate_id, candidate_name) in enumerate(zip(candidate_ids, candidate_names)):
        values = draws[:, idx]
        rows.append({"model": model, "election_year": year, "days_before_election": days, "snapshot_date": snapshot_date, "level": "national", "uf": "BR", "candidate_id": candidate_id, "candidate_name": candidate_name, "win_probability": float(np.mean(winners == idx)), "predicted_share": float(values.mean()), "lower": float(np.quantile(values, 0.05)), "upper": float(np.quantile(values, 0.95)), "actual_share": float(actual_values[idx]) if np.isfinite(actual_values[idx]) else np.nan, "outcome": int(idx == actual_winner) if scorable and actual_winner >= 0 else np.nan, "scorable": bool(scorable)})
    return rows


def _neutral_state_priors(previous_results: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    ufs = sorted(set(previous_results["uf"].dropna().astype(str)) - {"BR", "ZZ"})
    k = max(len(candidates), 1)
    rows = []
    for uf in ufs:
        for row in candidates.itertuples(index=False):
            rows.append({"uf": uf, "candidate_id": str(row.candidate_id), "candidate_name": str(row.candidate_name), "prior_share": 100.0 / k, "prior_strength": 0.75, "prior_source": "neutral_ablation"})
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame[frame["scorable"] & frame["outcome"].notna() & frame["actual_share"].notna()].copy()
    rows = []
    for keys, group in scored.groupby(["model", "election_year", "days_before_election", "level"], dropna=False):
        model, year, days, level = keys
        y = group["outcome"].to_numpy(dtype=float)
        p = np.clip(group["win_probability"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
        rows.append({"model": model, "election_year": int(year), "days_before_election": int(days), "level": level, "observations": len(group), "brier": brier_score(y, p), "log_loss": binary_log_loss(y, p), "ece": expected_calibration_error(y, p, bins=min(10, max(2, len(group) // 3))), "vote_share_mae": mean_absolute_error(group["actual_share"].to_numpy(), group["predicted_share"].to_numpy()), "interval_coverage": interval_coverage(group["actual_share"].to_numpy(), group["lower"].to_numpy(), group["upper"].to_numpy())})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ElectionAI with polling baselines and state-prior ablations")
    parser.add_argument("--years", nargs="+", type=int, default=[2014, 2018, 2022])
    parser.add_argument("--data-root", type=Path, default=Path("data/historical"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for year in args.years:
        election = get_election(year)
        polls, results = _load_year(args.data_root, year)
        snapshots = build_snapshots(polls, election.first_round_date)
        previous_year = max((candidate for candidate in (2010, 2014, 2018) if candidate < year), default=None)
        previous_results = None
        if previous_year is not None:
            path = args.data_root / str(previous_year) / "processed" / "presidential_results.csv"
            if path.exists():
                previous_results = pd.read_csv(path)
        for snap_index, (days, snapshot) in enumerate(sorted(snapshots.items(), reverse=True)):
            snapshot_date = election.first_round_date - timedelta(days=days)
            candidates = snapshot[["candidate_id", "candidate_name"]].drop_duplicates().sort_values("candidate_id")
            actual = _actual_lookup(results, candidates["candidate_name"].tolist())
            scorable = election.scoring_start_date is None or snapshot_date >= election.scoring_start_date
            for method_index, method in enumerate(BASELINES):
                fitted = fit_polling_baseline(snapshot, snapshot_date, method=method, n_draws=args.draws, seed=args.seed + year + snap_index * 10 + method_index)
                rows.extend(_records_from_draws(fitted.draws, fitted.candidate_ids, fitted.candidate_names, actual, model=method, year=year, days=days, snapshot_date=snapshot_date.isoformat(), scorable=scorable))
            national = fit_hierarchical_poll_model(snapshot, snapshot_date, state_priors=None, calibration=None, n_draws=args.draws, seed=args.seed + year + snap_index * 10 + 7)
            rows.extend(_records_from_draws(national.national_draws / 100.0, national.candidate_ids, national.candidate_names, actual, model="hierarchical_national", year=year, days=days, snapshot_date=snapshot_date.isoformat(), scorable=scorable))
        if previous_results is not None:
            full = run_historical_backtest(polls, results, election.first_round_date, election_year=year, previous_results=previous_results, scoring_start_date=election.scoring_start_date, posterior_draws=args.draws, seed=args.seed + year + 100).forecasts
            full["model"] = "electionai_full"
            rows.extend(full.to_dict("records"))
            for snap_index, (days, snapshot) in enumerate(sorted(snapshots.items(), reverse=True)):
                snapshot_date = election.first_round_date - timedelta(days=days)
                candidates = snapshot[["candidate_id", "candidate_name"]].drop_duplicates().sort_values("candidate_id")
                neutral = _neutral_state_priors(previous_results, candidates)
                posterior = fit_hierarchical_poll_model(snapshot, snapshot_date, state_priors=neutral, calibration=None, n_draws=args.draws, seed=args.seed + year + snap_index * 10 + 8)
                actual_state = results[pd.to_numeric(results["round"], errors="coerce") == 1].copy()
                actual_state["canonical"] = actual_state["candidate_name"].map(canonical_candidate)
                wanted = {canonical_candidate(name) for name in posterior.candidate_names}
                actual_state = actual_state[actual_state["canonical"].isin(wanted)]
                for state_index, uf in enumerate(posterior.state_ids):
                    state = actual_state[actual_state["uf"].astype(str) == uf]
                    totals = state.groupby("canonical")["votes"].sum()
                    total = totals.sum()
                    actual = {key: value / total for key, value in totals.items()} if total > 0 else {}
                    scorable = election.scoring_start_date is None or snapshot_date >= election.scoring_start_date
                    state_rows = _records_from_draws(posterior.state_draws[:, state_index, :] / 100.0, posterior.candidate_ids, posterior.candidate_names, actual, model="hierarchical_neutral_state_prior", year=year, days=days, snapshot_date=snapshot_date.isoformat(), scorable=scorable)
                    for row in state_rows:
                        row["level"] = "state"
                        row["uf"] = uf
                    rows.extend(state_rows)
    forecasts = pd.DataFrame(rows)
    forecasts.to_csv(args.output / "comparison_forecasts.csv", index=False)
    metrics = _metrics(forecasts)
    metrics.to_csv(args.output / "comparison_metrics.csv", index=False)
    aggregate = metrics.groupby(["model", "level"], as_index=False).agg(brier=("brier", "mean"), log_loss=("log_loss", "mean"), ece=("ece", "mean"), vote_share_mae=("vote_share_mae", "mean"), interval_coverage=("interval_coverage", "mean")).sort_values(["level", "vote_share_mae", "brier"])
    aggregate.to_csv(args.output / "aggregate_metrics.csv", index=False)
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
