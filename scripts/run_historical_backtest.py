from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.data.historical_manifest import get_election
from app.ml.calibration_report import calibration_by_group, calibration_slope_intercept, reliability_bins
from app.ml.historical_backtest import run_historical_backtest


def _load_year(root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = root / str(year) / "processed"
    polls_path = processed / "polls_model_schema.csv"
    results_path = processed / "presidential_results.csv"
    if not polls_path.exists():
        raise FileNotFoundError(f"Missing historical polls: {polls_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing official results: {results_path}")
    return pd.read_csv(polls_path), pd.read_csv(results_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ElectionAI temporal backtests on historical Brazilian elections")
    parser.add_argument("--years", nargs="+", type=int, default=[2014, 2018, 2022])
    parser.add_argument("--data-root", type=Path, default=Path("data/historical"))
    parser.add_argument("--output", type=Path, default=Path("reports/historical_backtest"))
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_forecasts: list[pd.DataFrame] = []
    all_snapshots: list[pd.DataFrame] = []
    all_states: list[pd.DataFrame] = []
    for year in args.years:
        election = get_election(year)
        polls, results = _load_year(args.data_root, year)
        previous_results = None
        previous_year = max((candidate for candidate in (2010, 2014, 2018) if candidate < year), default=None)
        if previous_year is not None:
            previous_path = args.data_root / str(previous_year) / "processed" / "presidential_results.csv"
            if previous_path.exists():
                previous_results = pd.read_csv(previous_path)
        result = run_historical_backtest(
            polls,
            results,
            election.first_round_date,
            election_year=year,
            previous_results=previous_results,
            posterior_draws=args.draws,
            seed=args.seed + year,
        )
        all_forecasts.append(result.forecasts)
        all_snapshots.append(result.snapshot_summary)
        if not result.state_summary.empty:
            all_states.append(result.state_summary)

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    snapshots = pd.concat(all_snapshots, ignore_index=True)
    states = pd.concat(all_states, ignore_index=True) if all_states else pd.DataFrame()
    by_election = calibration_by_group(forecasts, ["election_year", "level"])
    by_uf = calibration_by_group(forecasts[forecasts["level"] == "state"], ["election_year", "uf"]) if (forecasts["level"] == "state").any() else pd.DataFrame()
    bins = reliability_bins(forecasts)
    slope = calibration_slope_intercept(forecasts)

    forecasts.to_csv(args.output / "forecasts.csv", index=False)
    snapshots.to_csv(args.output / "metrics_by_snapshot.csv", index=False)
    states.to_csv(args.output / "metrics_by_state.csv", index=False)
    by_election.to_csv(args.output / "calibration_by_election.csv", index=False)
    by_uf.to_csv(args.output / "calibration_by_uf.csv", index=False)
    bins.to_csv(args.output / "reliability_bins.csv", index=False)
    summary = {
        "years": args.years,
        "forecast_rows": len(forecasts),
        "posterior_draws": args.draws,
        "calibration": slope,
        "note": "Backtests use only information available on or before each snapshot date.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
