from __future__ import annotations

import pandas as pd


def compute_state_weights(
    frame: pd.DataFrame,
    electorate_column: str = "eligible_voters",
    turnout_column: str = "expected_turnout",
) -> pd.DataFrame:
    required = {"uf", electorate_column, turnout_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing state-weight columns: {sorted(missing)}")
    output = frame.copy()
    electorate = pd.to_numeric(output[electorate_column], errors="raise")
    turnout = pd.to_numeric(output[turnout_column], errors="raise")
    if (electorate < 0).any():
        raise ValueError("electorate cannot be negative")
    if ((turnout < 0) | (turnout > 1)).any():
        raise ValueError("expected turnout must be between 0 and 1")
    output["expected_votes"] = electorate * turnout
    total = float(output["expected_votes"].sum())
    if total <= 0:
        raise ValueError("expected national votes must be positive")
    output["national_weight"] = output["expected_votes"] / total
    return output


def aggregate_state_share(
    state_shares: pd.DataFrame,
    state_weights: pd.DataFrame,
    share_column: str = "share",
) -> float:
    merged = state_shares.merge(state_weights[["uf", "national_weight"]], on="uf", how="inner", validate="many_to_one")
    if merged.empty:
        raise ValueError("no overlapping UFs between shares and weights")
    return float((merged[share_column] * merged["national_weight"]).sum() / merged["national_weight"].sum())
