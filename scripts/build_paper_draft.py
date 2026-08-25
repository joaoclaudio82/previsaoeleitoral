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
    parser.add_argument("--output", type=Path, default=Path("paper/generated/ELECTIONAI_PAPER_WITH_RESULTS.md"))
    args = parser.parse_args()

    aggregate = pd.read_csv(args.comparison / "aggregate_metrics.csv")
    metrics = pd.read_csv(args.comparison / "comparison_metrics.csv")
    historical = pd.read_csv(args.historical / "calibration_by_election.csv")
    template = args.template.read_text(encoding="utf-8")

    national = aggregate[aggregate["level"] == "national"].sort_values("vote_share_mae")
    full_national = national[national["model"] == "electionai_full"]
    if full_national.empty:
        full_national = national[national["model"] == "hierarchical_national"]
    full = full_national.iloc[0]
    best = national.iloc[0]
    baseline = national[national["model"].isin(["latest_poll", "simple_mean", "recency_weighted", "sample_recency_weighted"])].sort_values("vote_share_mae")
    best_baseline = baseline.iloc[0]

    state = aggregate[aggregate["level"] == "state"].copy()
    full_state = state[state["model"] == "electionai_full"]
    neutral_state = state[state["model"] == "hierarchical_neutral_state_prior"]

    abstract_results = (
        f"Across scoreable historical snapshots, the selected ElectionAI national specification achieved a mean vote-share MAE of {_pp(float(full['vote_share_mae']))}, "
        f"with mean Brier score {float(full['brier']):.3f} and mean 90% interval coverage {_pct(float(full['interval_coverage']))}%. "
        f"The strongest transparent polling baseline achieved {_pp(float(best_baseline['vote_share_mae']))} MAE."
    )

    overview = (
        f"Across the evaluated elections and scoreable snapshots, **{best['model']}** obtained the lowest aggregate national vote-share MAE ({_pp(float(best['vote_share_mae']))}). "
        f"The ElectionAI hierarchical specification recorded a Brier score of {float(full['brier']):.3f}, log loss of {float(full['log_loss']):.3f}, "
        f"ECE of {float(full['ece']):.3f}, and interval coverage of {_pct(float(full['interval_coverage']))}%."
    )

    baseline_results = (
        f"Among the four transparent baselines, **{best_baseline['model']}** performed best in aggregate, with vote-share MAE of {_pp(float(best_baseline['vote_share_mae']))} "
        f"and Brier score {float(best_baseline['brier']):.3f}. The corresponding hierarchical specification produced {_pp(float(full['vote_share_mae']))} MAE. "
        "The complete model-by-election and model-by-horizon values are provided in the generated replication tables."
    )

    full_horizon = metrics[(metrics["model"].isin(["electionai_full", "hierarchical_national"])) & (metrics["level"] == "national")]
    if not full_horizon.empty:
        horizon = full_horizon.groupby("days_before_election", as_index=False)["vote_share_mae"].mean().sort_values("days_before_election", ascending=False)
        first = horizon.iloc[0]
        last = horizon.iloc[-1]
        horizon_results = (
            f"Mean national vote-share MAE changed from {_pp(float(first['vote_share_mae']))} at D-{int(first['days_before_election'])} "
            f"to {_pp(float(last['vote_share_mae']))} at D-{int(last['days_before_election'])}. "
            "Figure 1 reports the full error trajectory for all compared models."
        )
    else:
        horizon_results = "No scoreable hierarchical horizon metrics were available."

    hist_national = historical[historical["level"] == "national"] if "level" in historical.columns else historical
    calibration_results = (
        f"Across election-specific national summaries, mean Brier score was {hist_national['brier'].mean():.3f}, mean ECE was {hist_national['ece'].mean():.3f}, "
        f"and mean posterior interval coverage was {_pct(float(hist_national['interval_coverage'].mean()))}%. "
        "Given the small number of independent presidential cycles, these calibration estimates are treated as diagnostics rather than definitive frequency guarantees."
    )

    if not full_state.empty and not neutral_state.empty:
        f = full_state.iloc[0]
        n = neutral_state.iloc[0]
        delta = float(n["vote_share_mae"] - f["vote_share_mae"])
        direction = "reduced" if delta > 0 else "increased"
        ablation_results = (
            f"Replacing historical state priors with weak neutral priors changed aggregate state-level MAE from {_pp(float(f['vote_share_mae']))} to {_pp(float(n['vote_share_mae']))}. "
            f"Thus, historical priors {direction} state-level MAE by {abs(delta) * 100:.2f} percentage points in this retrospective sample."
        )
    else:
        ablation_results = "The state-prior ablation did not produce enough scoreable observations for aggregate comparison."

    conclusion_results = (
        f"In the retrospective sample, the hierarchical national specification achieved {_pp(float(full['vote_share_mae']))} aggregate vote-share MAE, "
        f"compared with {_pp(float(best_baseline['vote_share_mae']))} for the strongest transparent baseline."
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
