from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_error_horizon(metrics: pd.DataFrame, output: Path) -> None:
    national = metrics[metrics["level"] == "national"].copy()
    if national.empty:
        return
    summary = national.groupby(["model", "days_before_election"], as_index=False)["vote_share_mae"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in summary.groupby("model"):
        group = group.sort_values("days_before_election", ascending=False)
        ax.plot(group["days_before_election"], group["vote_share_mae"] * 100.0, marker="o", label=model)
    ax.invert_xaxis()
    ax.set_xlabel("Days before election")
    ax.set_ylabel("Mean absolute vote-share error (percentage points)")
    ax.set_title("Forecast error across electoral horizons")
    ax.legend(fontsize=8)
    _save(fig, output / "fig1_error_by_horizon.png")


def figure_model_comparison(aggregate: pd.DataFrame, output: Path) -> None:
    national = aggregate[aggregate["level"] == "national"].sort_values("vote_share_mae")
    if national.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(national["model"], national["vote_share_mae"] * 100.0)
    ax.set_xlabel("Mean absolute vote-share error (percentage points)")
    ax.set_title("National model comparison")
    ax.invert_yaxis()
    _save(fig, output / "fig2_model_comparison.png")


def figure_reliability(forecasts: pd.DataFrame, output: Path) -> None:
    scored = forecasts[forecasts["scorable"] & forecasts["outcome"].notna()].copy()
    scored = scored[scored["model"] == "electionai_full"]
    if scored.empty:
        return
    edges = np.linspace(0.0, 1.0, 11)
    x, y, n = [], [], []
    for idx in range(10):
        mask = (scored["win_probability"] >= edges[idx]) & (
            (scored["win_probability"] < edges[idx + 1]) if idx < 9 else (scored["win_probability"] <= edges[idx + 1])
        )
        group = scored[mask]
        if group.empty:
            continue
        x.append(group["win_probability"].mean())
        y.append(group["outcome"].mean())
        n.append(len(group))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if x:
        ax.scatter(x, y, s=np.maximum(np.asarray(n) * 8, 25), label="ElectionAI")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    _save(fig, output / "fig3_reliability.png")


def figure_state_ablation(aggregate: pd.DataFrame, output: Path) -> None:
    states = aggregate[(aggregate["level"] == "state") & aggregate["model"].isin(["electionai_full", "hierarchical_neutral_state_prior"])].copy()
    if states.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(states["model"], states["vote_share_mae"] * 100.0)
    ax.set_ylabel("Mean absolute vote-share error (percentage points)")
    ax.set_title("Ablation: historical state priors")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, output / "fig4_state_prior_ablation.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ElectionAI paper figures")
    parser.add_argument("--comparison", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.comparison / "comparison_metrics.csv")
    aggregate = pd.read_csv(args.comparison / "aggregate_metrics.csv")
    forecasts = pd.read_csv(args.comparison / "comparison_forecasts.csv")
    figure_error_horizon(metrics, args.output)
    figure_model_comparison(aggregate, args.output)
    figure_reliability(forecasts, args.output)
    figure_state_ablation(aggregate, args.output)
    print(f"Figures written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
