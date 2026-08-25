from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pp(value: float) -> str:
    return f"{100.0 * value:.2f} percentage points"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Insert robustness results into the generated ElectionAI paper")
    parser.add_argument("--paper", type=Path, default=Path("paper/generated/ELECTIONAI_PAPER_WITH_RESULTS.md"))
    parser.add_argument("--robustness", type=Path, default=Path("reports/robustness"))
    args = parser.parse_args()

    text = args.paper.read_text(encoding="utf-8")
    aggregate = pd.read_csv(args.robustness / "robustness_aggregate.csv")
    loeo = pd.read_csv(args.robustness / "leave_one_election_out_prior_tuning.csv")
    sensitivity = pd.read_csv(args.robustness / "prior_strength_sensitivity.csv")
    bootstrap = pd.read_csv(args.robustness / "election_cluster_bootstrap.csv")

    state = aggregate[aggregate["level"] == "state"].set_index("model")
    full = state.loc["electionai_full"]
    simple = state.loc["national_swing_state_lean"]
    neutral = state.loc["neutral_state_prior"]

    best_sensitivity = sensitivity.sort_values(["vote_share_mae", "prior_strength"]).iloc[0]
    loeo_mean = float(loeo["test_vote_share_mae"].mean())
    loeo_coverage = float(loeo["test_coverage"].mean())

    boot_neutral = bootstrap[bootstrap["comparison"].str.contains("neutral_state_prior")].iloc[0]
    boot_simple = bootstrap[bootstrap["comparison"].str.contains("national_swing_state_lean")].iloc[0]

    section = f"""
### 5.5 Cross-election robustness and sensitivity

The geographic result was subjected to additional tests designed to reduce dependence on a particular election or prior specification. First, ElectionAI was compared with an explicit **national-swing + historical-state-lean** benchmark that applies the previous-election state-vs-national pattern directly to a current national polling distribution without the full hierarchical pollster and survey-design structure. ElectionAI achieved state-level MAE of {_pp(float(full['vote_share_mae']))}, compared with {_pp(float(simple['vote_share_mae']))} for this simpler geographic benchmark and {_pp(float(neutral['vote_share_mae']))} for the neutral-state-prior specification. This comparison separates the value of geographic persistence itself from the additional structure of the full hierarchical model.

Second, prior strength was varied over a prespecified grid. The lowest pooled state-level MAE on that grid occurred at prior strength {float(best_sensitivity['prior_strength']):.2f}, with MAE {_pp(float(best_sensitivity['vote_share_mae']))} and 90% interval coverage {_pct(float(best_sensitivity['coverage']))}. Because selecting a hyperparameter on the same elections used for evaluation would be optimistic, we also conducted leave-one-election-out tuning: for each held-out presidential election, prior strength was selected using only the other two cycles. Across the three held-out evaluations, mean state-level MAE was {_pp(loeo_mean)} and mean 90% interval coverage was {_pct(loeo_coverage)}.

Third, uncertainty in the geographic comparison was assessed by resampling at the **election level**, rather than treating candidate-by-state-by-snapshot observations as independent. For historical state leans versus neutral priors, the election-cluster bootstrap estimated a mean MAE advantage of {_pp(float(boot_neutral['mean_delta']))} for ElectionAI, with a 95% cluster-bootstrap interval from {_pp(float(boot_neutral['ci_low']))} to {_pp(float(boot_neutral['ci_high']))}. Against the simpler national-swing + state-lean benchmark, the corresponding mean difference was {_pp(float(boot_simple['mean_delta']))}, with a 95% interval from {_pp(float(boot_simple['ci_low']))} to {_pp(float(boot_simple['ci_high']))}. These intervals remain coarse because only three independent presidential cycles are available; they are therefore reported as robustness diagnostics rather than conventional large-sample inferential guarantees.
"""

    marker = "\n## 6. Discussion\n"
    if marker not in text:
        raise ValueError("Discussion section marker not found in generated paper")
    if "### 5.5 Cross-election robustness and sensitivity" in text:
        before, rest = text.split("### 5.5 Cross-election robustness and sensitivity", 1)
        _, after = rest.split(marker, 1)
        text = before.rstrip() + "\n\n" + section.strip() + marker + after
    else:
        text = text.replace(marker, "\n" + section.strip() + "\n" + marker, 1)
    args.paper.write_text(text, encoding="utf-8")
    print(args.paper)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
