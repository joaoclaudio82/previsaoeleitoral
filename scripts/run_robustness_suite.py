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
from app.ml.dynamic_polling import fit_dynamic_polling_baseline
from app.ml.geographic_baseline import apply_national_swing_state_lean
from app.ml.historical_baselines import fit_polling_baseline
from app.ml.posterior_predictive import add_correlated_predictive_error
from app.services.hierarchical_polls import fit_hierarchical_poll_model


PRIOR_STRENGTHS = (0.5, 1.0, 2.5, 5.0, 10.0)


def _candidate_frame(snapshot: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate_id", "candidate_name"] + (["party"] if "party" in snapshot.columns else [])
    return snapshot[cols].drop_duplicates().sort_values("candidate_id").reset_index(drop=True)


def _actual(results: pd.DataFrame, uf: str, candidate_names: list[str]) -> np.ndarray:
    frame = results[pd.to_numeric(results["round"], errors="coerce") == 1].copy()
    frame["uf"] = frame["uf"].astype(str).str.upper().str.strip()
    if uf != "BR":
        frame = frame[frame["uf"] == uf].copy()
    frame["canonical"] = frame["candidate_name"].map(canonical_candidate)
    wanted = [canonical_candidate(name) for name in candidate_names]
    frame = frame[frame["canonical"].isin(set(wanted))]
    totals = frame.groupby("canonical")["votes"].sum()
    values = np.array([float(totals.get(key, 0.0)) for key in wanted], dtype=float)
    if values.sum() <= 0:
        return np.full(len(wanted), np.nan)
    return values / values.sum()


def _score(draws: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    values = np.asarray(draws, dtype=float)
    if values.max() > 1.5:
        values = values / 100.0
    values = np.clip(values, 1e-9, None)
    values = values / values.sum(axis=1, keepdims=True)
    valid = np.isfinite(actual)
    if not valid.any():
        return {"vote_share_mae": np.nan, "coverage": np.nan, "winner_brier": np.nan, "interval_width": np.nan}
    center = values.mean(axis=0)
    lower = np.quantile(values, 0.05, axis=0)
    upper = np.quantile(values, 0.95, axis=0)
    actual_winner = int(np.nanargmax(actual))
    winner_probability = np.bincount(np.argmax(values, axis=1), minlength=values.shape[1]) / len(values)
    outcome = np.arange(values.shape[1]) == actual_winner
    return {
        "vote_share_mae": float(np.mean(np.abs(center[valid] - actual[valid]))),
        "coverage": float(np.mean((actual[valid] >= lower[valid]) & (actual[valid] <= upper[valid]))),
        "winner_brier": float(np.mean(np.square(winner_probability - outcome.astype(float)))),
        "interval_width": float(np.mean((upper - lower)[valid])),
    }


def _neutral_priors(previous: pd.DataFrame, candidates: pd.DataFrame, *, strength: float = 2.5) -> pd.DataFrame:
    ufs = sorted(set(previous["uf"].dropna().astype(str)) - {"BR", "ZZ"})
    k = max(len(candidates), 1)
    rows: list[dict[str, object]] = []
    for uf in ufs:
        for candidate in candidates.itertuples(index=False):
            rows.append({
                "uf": uf,
                "candidate_id": str(candidate.candidate_id),
                "candidate_name": str(candidate.candidate_name),
                "prior_share": 100.0 / k,
                "national_prior_share": 100.0 / k,
                "prior_strength": float(strength),
                "prior_concentration": max(6.0, float(strength) * 20.0),
                "prior_source": "neutral_robustness",
            })
    return pd.DataFrame(rows)


def _record(rows: list[dict[str, object]], *, model: str, year: int, days: int, uf: str, level: str, metrics: dict[str, float], prior_strength: float | None = None) -> None:
    rows.append({
        "model": model,
        "election_year": int(year),
        "days_before_election": int(days),
        "uf": uf,
        "level": level,
        "prior_strength": prior_strength,
        **metrics,
    })


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["model", "level"], as_index=False)
        .agg(
            vote_share_mae=("vote_share_mae", "mean"),
            coverage=("coverage", "mean"),
            winner_brier=("winner_brier", "mean"),
            interval_width=("interval_width", "mean"),
            observations=("vote_share_mae", "size"),
        )
        .sort_values(["level", "vote_share_mae"])
    )


def _leave_one_election_out(frame: pd.DataFrame) -> pd.DataFrame:
    state = frame[(frame["level"] == "state") & frame["model"].isin(["electionai_full", "national_swing_state_lean", "neutral_state_prior"])].copy()
    years = sorted(state["election_year"].unique())
    rows: list[dict[str, object]] = []
    for omitted in years:
        kept = state[state["election_year"] != omitted]
        for model, group in kept.groupby("model"):
            rows.append({
                "omitted_election": int(omitted),
                "model": model,
                "vote_share_mae": float(group["vote_share_mae"].mean()),
                "coverage": float(group["coverage"].mean()),
                "winner_brier": float(group["winner_brier"].mean()),
                "observations": len(group),
            })
    return pd.DataFrame(rows)


def _loeo_prior_tuning(sensitivity: pd.DataFrame) -> pd.DataFrame:
    years = sorted(sensitivity["election_year"].unique())
    rows: list[dict[str, object]] = []
    for test_year in years:
        training = sensitivity[sensitivity["election_year"] != test_year]
        train_scores = (
            training.groupby("prior_strength", as_index=False)
            .agg(vote_share_mae=("vote_share_mae", "mean"), winner_brier=("winner_brier", "mean"))
        )
        train_scores["objective"] = train_scores["vote_share_mae"] + 0.25 * train_scores["winner_brier"]
        selected = float(train_scores.sort_values(["objective", "prior_strength"]).iloc[0]["prior_strength"])
        test = sensitivity[(sensitivity["election_year"] == test_year) & (sensitivity["prior_strength"] == selected)]
        rows.append({
            "test_election": int(test_year),
            "selected_prior_strength": selected,
            "test_vote_share_mae": float(test["vote_share_mae"].mean()),
            "test_coverage": float(test["coverage"].mean()),
            "test_winner_brier": float(test["winner_brier"].mean()),
            "test_observations": len(test),
        })
    return pd.DataFrame(rows)


def _cluster_bootstrap(frame: pd.DataFrame, *, left: str, right: str, seed: int, replicates: int = 20000) -> dict[str, float | str]:
    state = frame[(frame["level"] == "state") & frame["model"].isin([left, right])].copy()
    election_means = state.groupby(["election_year", "model"])["vote_share_mae"].mean().unstack("model").dropna()
    deltas = (election_means[right] - election_means[left]).to_numpy(dtype=float)
    if len(deltas) == 0:
        return {"comparison": f"{right} minus {left}", "mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "probability_positive": np.nan}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(replicates, len(deltas)), replace=True).mean(axis=1)
    return {
        "comparison": f"{right} minus {left}",
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "probability_positive": float(np.mean(sampled > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-election robustness suite for ElectionAI")
    parser.add_argument("--years", nargs="+", type=int, default=[2014, 2018, 2022])
    parser.add_argument("--offsets", nargs="+", type=int, default=[15, 7, 3, 1])
    parser.add_argument("--data-root", type=Path, default=Path("data/historical"))
    parser.add_argument("--output", type=Path, default=Path("reports/robustness"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

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
            candidates = _candidate_frame(snapshot)

            # National temporal robustness baseline.
            dynamic = fit_dynamic_polling_baseline(
                snapshot,
                cutoff,
                forecast_date=election.first_round_date,
                n_draws=args.draws,
                seed=args.seed + year + snap_index,
            )
            _record(
                rows,
                model="dynamic_random_walk",
                year=year,
                days=days,
                uf="BR",
                level="national",
                metrics=_score(dynamic.draws, _actual(results, "BR", dynamic.candidate_names)),
            )

            recency = fit_polling_baseline(
                snapshot,
                cutoff,
                method="recency_weighted",
                n_draws=args.draws,
                seed=args.seed + year + snap_index + 100,
            )
            _record(
                rows,
                model="recency_weighted",
                year=year,
                days=days,
                uf="BR",
                level="national",
                metrics=_score(recency.draws, _actual(results, "BR", recency.candidate_names)),
            )

            historical_prior = build_state_priors(previous, candidates, prior_strength=2.5, fallback_strength=0.75)
            neutral_prior = _neutral_priors(previous, candidates, strength=2.5)

            full = fit_hierarchical_poll_model(
                snapshot,
                cutoff,
                state_priors=historical_prior,
                calibration=None,
                forecast_date=election.first_round_date,
                n_draws=args.draws,
                seed=args.seed + year * 10 + snap_index,
            )
            full_national, full_states = add_correlated_predictive_error(
                full.national_draws,
                full.state_draws,
                full.residual_covariance,
                seed=args.seed + year * 100 + snap_index,
            )
            _record(rows, model="electionai_full", year=year, days=days, uf="BR", level="national", metrics=_score(full_national, _actual(results, "BR", full.candidate_names)))

            neutral = fit_hierarchical_poll_model(
                snapshot,
                cutoff,
                state_priors=neutral_prior,
                calibration=None,
                forecast_date=election.first_round_date,
                n_draws=args.draws,
                seed=args.seed + year * 10 + snap_index + 1,
            )
            _, neutral_states = add_correlated_predictive_error(
                neutral.national_draws,
                neutral.state_draws,
                neutral.residual_covariance,
                seed=args.seed + year * 100 + snap_index + 1,
            )

            geographic = apply_national_swing_state_lean(
                recency.draws,
                recency.candidate_ids,
                recency.candidate_names,
                historical_prior,
                seed=args.seed + year * 100 + snap_index + 2,
            )

            for state_index, uf in enumerate(full.state_ids):
                actual = _actual(results, uf, full.candidate_names)
                _record(rows, model="electionai_full", year=year, days=days, uf=uf, level="state", metrics=_score(full_states[:, state_index, :], actual))
                if uf in neutral.state_ids:
                    neutral_index = neutral.state_ids.index(uf)
                    _record(rows, model="neutral_state_prior", year=year, days=days, uf=uf, level="state", metrics=_score(neutral_states[:, neutral_index, :], actual))
                if uf in geographic.state_ids:
                    geo_index = geographic.state_ids.index(uf)
                    geo_actual = _actual(results, uf, geographic.candidate_names)
                    _record(rows, model="national_swing_state_lean", year=year, days=days, uf=uf, level="state", metrics=_score(geographic.state_draws[:, geo_index, :], geo_actual))

            # Sensitivity of the hierarchical state prior strength.
            for strength_index, strength in enumerate(PRIOR_STRENGTHS):
                prior = build_state_priors(
                    previous,
                    candidates,
                    prior_strength=float(strength),
                    fallback_strength=max(0.25, float(strength) * 0.3),
                )
                posterior = fit_hierarchical_poll_model(
                    snapshot,
                    cutoff,
                    state_priors=prior,
                    calibration=None,
                    forecast_date=election.first_round_date,
                    n_draws=args.draws,
                    seed=args.seed + year * 1000 + snap_index * 20 + strength_index,
                )
                _, state_draws = add_correlated_predictive_error(
                    posterior.national_draws,
                    posterior.state_draws,
                    posterior.residual_covariance,
                    seed=args.seed + year * 2000 + snap_index * 20 + strength_index,
                )
                for state_index, uf in enumerate(posterior.state_ids):
                    metrics = _score(state_draws[:, state_index, :], _actual(results, uf, posterior.candidate_names))
                    sensitivity_rows.append({
                        "election_year": year,
                        "days_before_election": int(days),
                        "uf": uf,
                        "prior_strength": float(strength),
                        **metrics,
                    })

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "robustness_records.csv", index=False)
    aggregate = _aggregate(frame)
    aggregate.to_csv(args.output / "robustness_aggregate.csv", index=False)

    per_election = (
        frame.groupby(["model", "level", "election_year"], as_index=False)
        .agg(vote_share_mae=("vote_share_mae", "mean"), coverage=("coverage", "mean"), winner_brier=("winner_brier", "mean"), interval_width=("interval_width", "mean"))
    )
    per_election.to_csv(args.output / "robustness_by_election.csv", index=False)

    influence = _leave_one_election_out(frame)
    influence.to_csv(args.output / "leave_one_election_out_influence.csv", index=False)

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(args.output / "prior_strength_records.csv", index=False)
    sensitivity_aggregate = (
        sensitivity.groupby("prior_strength", as_index=False)
        .agg(vote_share_mae=("vote_share_mae", "mean"), coverage=("coverage", "mean"), winner_brier=("winner_brier", "mean"), interval_width=("interval_width", "mean"))
    )
    sensitivity_aggregate.to_csv(args.output / "prior_strength_sensitivity.csv", index=False)

    loeo_tuned = _loeo_prior_tuning(sensitivity)
    loeo_tuned.to_csv(args.output / "leave_one_election_out_prior_tuning.csv", index=False)

    bootstrap = pd.DataFrame([
        _cluster_bootstrap(frame, left="electionai_full", right="neutral_state_prior", seed=args.seed + 1),
        _cluster_bootstrap(frame, left="electionai_full", right="national_swing_state_lean", seed=args.seed + 2),
    ])
    bootstrap.to_csv(args.output / "election_cluster_bootstrap.csv", index=False)

    print("\nAggregate robustness metrics")
    print(aggregate.to_string(index=False))
    print("\nPrior-strength sensitivity")
    print(sensitivity_aggregate.to_string(index=False))
    print("\nLeave-one-election-out prior tuning")
    print(loeo_tuned.to_string(index=False))
    print("\nElection-cluster bootstrap (diagnostic; only three independent cycles)")
    print(bootstrap.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
