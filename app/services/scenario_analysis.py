from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    turnout_multiplier: float = 1.0
    undecided_shift_points: float = 0.0
    poll_shift_points: float = 0.0


def apply_poll_shift(frame: pd.DataFrame, candidate_id: str, points: float) -> pd.DataFrame:
    if "candidate_id" not in frame.columns or "share" not in frame.columns:
        raise ValueError("candidate_id and share are required")
    output = frame.copy()
    mask = output["candidate_id"].astype(str) == str(candidate_id)
    output.loc[mask, "share"] = (pd.to_numeric(output.loc[mask, "share"], errors="raise") + points).clip(0, 100)
    return output


def apply_turnout_multiplier(frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    if multiplier <= 0:
        raise ValueError("turnout multiplier must be positive")
    if "expected_turnout" not in frame.columns:
        raise ValueError("expected_turnout is required")
    output = frame.copy()
    output["expected_turnout"] = (pd.to_numeric(output["expected_turnout"], errors="raise") * multiplier).clip(0, 1)
    return output


def standard_scenarios() -> list[Scenario]:
    return [
        Scenario("baseline"),
        Scenario("lower_turnout", turnout_multiplier=0.97),
        Scenario("higher_turnout", turnout_multiplier=1.03),
        Scenario("poll_error_minus_2", poll_shift_points=-2.0),
        Scenario("poll_error_plus_2", poll_shift_points=2.0),
    ]
