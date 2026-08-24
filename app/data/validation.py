from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def validate_columns(frame: pd.DataFrame, required: set[str]) -> ValidationReport:
    report = ValidationReport()
    missing = sorted(required.difference(frame.columns))
    if missing:
        report.errors.append(f"Missing columns: {', '.join(missing)}")
    return report


def validate_probability_column(frame: pd.DataFrame, column: str) -> ValidationReport:
    report = ValidationReport()
    if column not in frame.columns:
        report.errors.append(f"Missing probability column: {column}")
        return report
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        report.errors.append(f"{column} contains non-numeric or null values")
    if ((values < 0) | (values > 1)).any():
        report.errors.append(f"{column} must be between 0 and 1")
    return report


def validate_poll_frame(frame: pd.DataFrame) -> ValidationReport:
    required = {
        "poll_id",
        "institute",
        "fieldwork_end",
        "sample_size",
        "candidate_id",
        "share",
    }
    report = validate_columns(frame, required)
    if report.errors:
        return report
    sample = pd.to_numeric(frame["sample_size"], errors="coerce")
    share = pd.to_numeric(frame["share"], errors="coerce")
    if sample.isna().any() or (sample <= 0).any():
        report.errors.append("sample_size must contain positive numbers")
    if share.isna().any() or ((share < 0) | (share > 100)).any():
        report.errors.append("share must be numeric and between 0 and 100")
    duplicated = frame.duplicated(["poll_id", "candidate_id"], keep=False)
    if duplicated.any():
        report.errors.append("duplicate poll_id/candidate_id observations detected")
    return report
