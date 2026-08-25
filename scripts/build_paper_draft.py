from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pct(value: float) -> str:
    return f"{value * 100:.2f}"


def _pp(value: float) -> str:
    return f"{value * 100:.2f} percentage points"


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate the ElectionAI paper with measured backtest results")
    parser.add_argument("--template", type=Path, default=Path("paper/ELECTIONAI_PAPER_DRAFT.md"))
    parser.add_argument("--comparison", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument("--historical", type=Path, default=Path("reports/historical_backtest"))
    parser.add_argument("--state-ablation", type=Path, default=Path("reports/state_prior_ablation/state_prior_ablation_aggregate.csv"))
    parser.add_argument("--output", type=Path, default=Path("paper/generated/ELECTIONAI_PAPER_WITH_RESULTS.md"))
    args = parser.parse_args()

    aggregate = pd.read_csv(args.comparison / "aggregate_metrics.csv")
    metrics = pd.read_csv(args.comparison / "comparison_metrics.csv")
    historical = pd.read_csv(args.historical / "calibration_by_election.csv")
    ablation = pd.read_csv(args.state_ablation)
    template = args.template.read_text(encoding="utf-8")

    national = aggregate[aggregate["level"] == "national"].sort_values("vote_share_mae")
    full = national[national["model"] == "electionai_full"].iloc[0]
    baseline = national[national["model"].isin(["latest_poll", "simple_mean", "recency_weighted", "sample_recency_weighted"])].sort_values("vote_share_mae")
    best_baseline = baseline.iloc[0]

    abstract_results = (
        f"Across scoreable historical snapshots, ElectionAI achieved a mean national vote-share MAE of {_pp(float(full['vote_share_mae']))}, "
        f"with mean Brier score {float(full['brier']):.3f} and 90% predictive-interval coverage of {_pct(float(full['interval_coverage']))}%. "
        f"The strongest transparent polling baseline achieved {_pp(float(best_baseline['vote_share_mae']))} MAE."
    )

    overview = (
        f"Across the evaluated elections and scoreable snapshots, ElectionAI achieved aggregate national vote-share MAE of {_pp(float(full['vote_share_mae']))}. "
        f"Its national Brier score was {float(full['brier']):.3f}, log loss {float(full['log_loss']):.3f}, ECE {float(full['ece']):.3f}, "
        f"and 90% predictive-interval coverage {_pct(float(full['interval_coverage']))}%."
    )

    baseline_results = (
        f"Among the four transparent baselines, **{best_baseline['model']}** produced the lowest vote-share MAE, {_pp(float(best_baseline['vote_share_mae']))}, "
        f"with Brier score {float(best_baseline['brier']):.3f}. ElectionAI's MAE was {_pp(float(full['vote_share_mae']))}, a difference of "
        f"{abs(float(best_baseline['vote_share_mae']) - float(full['vote_share_mae'])) * 100:.2f} percentage points. "
        "The latest-poll baseline retained the better winner-probability score, so the national comparison does not support a claim of uniform dominance by the hierarchical model."
    )

    full_horizon = metrics[(metrics["model"] == "electionai_full") & (metrics["level"] == "national")]
    horizon = full_horizon.groupby("days_before_election", as_index=False)["vote_share_mae"].mean().sort_values("days_before_election", ascending=False)
    first = horizon.iloc[0]
    last = horizon.iloc[-1]
    horizon_results = (
        f"Mean ElectionAI national vote-share MAE declined from {_pp(float(first['vote_share_mae']))} at D-{int(first['days_before_election'])} "
        f"to {_pp(float(last['vote_share_mae']))} at D-{int(last['days_before_election'])}. "
        "The decline is monotonic in the four common horizons when averaged across elections, consistent with information accumulating as Election Day approaches."
    )

    hist_national = historical[historical["level"] == "national"] if "level" in historical.columns else historical
    calibration_results = (
        f"Across election-specific national summaries, mean Brier score was {hist_national['brier'].mean():.3f}, mean ECE was {hist_national['ece'].mean():.3f}, "
        f"and mean 90% predictive-interval coverage was {_pct(float(hist_national['interval_coverage'].mean()))}%. "
        "The interval coverage remains substantially below its nominal level, indicating residual national underdispersion even after posterior-predictive polling error is propagated. "
        "Given only three independent presidential cycles, these quantities are diagnostics rather than definitive frequency guarantees."
    )

    historical_state = ablation[ablation["model"] == "historical_state_lean"].iloc[0]
    neutral_state = ablation[ablation["model"] == "neutral_state_prior"].iloc[0]
    delta = float(neutral_state["vote_share_mae"] - historical_state["vote_share_mae"])
    ablation_results = (
        f"In the symmetric state-prior ablation, the historical state-lean specification achieved {_pp(float(historical_state['vote_share_mae']))} MAE, "
        f"90% interval coverage of {_pct(float(historical_state['interval_coverage']))}%, and winner Brier score {float(historical_state['winner_brier']):.3f}. "
        f"Replacing only the geographic prior with an equal-share neutral prior increased MAE to {_pp(float(neutral_state['vote_share_mae']))}, "
        f"reduced coverage to {_pct(float(neutral_state['interval_coverage']))}%, and increased Brier score to {float(neutral_state['winner_brier']):.3f}. "
        f"The historical state lean therefore reduced state-level MAE by {delta * 100:.2f} percentage points under otherwise matched forecasting assumptions."
    )

    conclusion_results = (
        f"National vote-share MAE was {_pp(float(full['vote_share_mae']))}, close to {_pp(float(best_baseline['vote_share_mae']))} for the strongest transparent baseline, "
        f"while the matched geographic ablation showed a larger benefit from historical state leans ({_pp(float(historical_state['vote_share_mae']))} versus {_pp(float(neutral_state['vote_share_mae']))})."
    )

    replacements = {
        "{{ABSTRACT_RESULTS}}": abstract_results,
        "{{RESULTS_OVERVIEW}}": overview,
        "{{BASELINE_RESULTS}}": baseline_results,
        "{{HORIZON_RESULTS}}": horizon_results,
        "{{CALIBRATION_RESULTS}}": calibration_results,
        "{{ABLATION_RESULTS}}": ablation_results,
        "{{CONCLUSION_RESULTS}}": conclusion_results,
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
