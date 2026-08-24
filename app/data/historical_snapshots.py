from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from datetime import date, timedelta

import pandas as pd


DEFAULT_OFFSETS = (180, 120, 90, 60, 30, 15, 7, 3, 1)


def candidate_id(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _poll_id(year: int, pollster: str, poll_date: str, source_table: object) -> str:
    raw = f"{year}|{pollster}|{poll_date}|{source_table}".encode("utf-8")
    return f"hist_{year}_{hashlib.sha1(raw).hexdigest()[:12]}"


def to_model_poll_schema(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "poll_date", "pollster", "candidate_name", "share"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Historical polls missing columns: {sorted(missing)}")
    frame = raw.copy()
    frame["poll_date"] = pd.to_datetime(frame["poll_date"], errors="coerce").dt.date
    frame = frame[frame["poll_date"].notna()].copy()
    frame["candidate_id"] = frame["candidate_name"].map(candidate_id)
    sample_source = frame["sample_size"] if "sample_size" in frame.columns else pd.Series(1000, index=frame.index)
    frame["sample_size"] = pd.to_numeric(sample_source, errors="coerce").fillna(1000).clip(lower=100)
    frame["margin_error"] = 98.0 / frame["sample_size"].map(math.sqrt)
    source_tables = frame["source_table"] if "source_table" in frame.columns else pd.Series(0, index=frame.index)
    frame["poll_id"] = [
        _poll_id(int(y), str(p), d.isoformat(), t)
        for y, p, d, t in zip(frame["year"], frame["pollster"], frame["poll_date"], source_tables)
    ]
    source_urls = frame["source_url"] if "source_url" in frame.columns else pd.Series("", index=frame.index)
    result = pd.DataFrame(
        {
            "poll_id": frame["poll_id"],
            "field_date": frame["poll_date"].map(date.isoformat),
            "institute": frame["pollster"].astype(str),
            "candidate_id": frame["candidate_id"],
            "candidate_name": frame["candidate_name"].astype(str),
            "share": pd.to_numeric(frame["share"], errors="coerce") * 100.0,
            "sample_size": frame["sample_size"].astype(int),
            "margin_error": frame["margin_error"],
            "collection_mode": "unknown",
            "target_population": "registered_voters",
            "undecided_share": 0.0,
            "scope": "national",
            "uf": "BR",
            "source_url": source_urls,
        }
    )
    return result.dropna(subset=["share"]).drop_duplicates(["poll_id", "candidate_id"]).reset_index(drop=True)


def snapshot_dates(election_date: date, offsets: tuple[int, ...] = DEFAULT_OFFSETS) -> list[date]:
    return [election_date - timedelta(days=days) for days in offsets]


def _candidate_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    meta = frame.groupby("poll_id", as_index=False).agg(field_date=("field_date", "max"))
    signatures = (
        frame.groupby("poll_id")["candidate_id"]
        .apply(lambda values: tuple(sorted(set(map(str, values)))))
        .rename("signature")
        .reset_index()
    )
    return meta.merge(signatures, on="poll_id", how="inner")


def build_snapshots(
    polls: pd.DataFrame,
    election_date: date,
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
    max_age_days: int = 90,
) -> dict[int, pd.DataFrame]:
    frame = polls.copy()
    frame["field_date"] = pd.to_datetime(frame["field_date"]).dt.date
    snapshots: dict[int, pd.DataFrame] = {}
    for days in offsets:
        cutoff = election_date - timedelta(days=days)
        earliest = cutoff - timedelta(days=max_age_days)
        eligible = frame[(frame["field_date"] <= cutoff) & (frame["field_date"] >= earliest)].copy()
        if eligible.empty:
            continue
        signatures = _candidate_signatures(eligible).sort_values(["field_date", "poll_id"], ascending=[False, False])
        latest_signature = signatures.iloc[0]["signature"]
        compatible_ids = signatures.loc[signatures["signature"] == latest_signature, "poll_id"]
        eligible = eligible[eligible["poll_id"].isin(compatible_ids)].copy()
        if eligible["candidate_id"].nunique() < 2:
            continue
        eligible["snapshot_date"] = cutoff.isoformat()
        eligible["days_before_election"] = days
        eligible["candidate_slate_hash"] = hashlib.sha1("|".join(latest_signature).encode("utf-8")).hexdigest()[:12]
        snapshots[days] = eligible.reset_index(drop=True)
    return snapshots
