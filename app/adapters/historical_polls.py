from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Iterable

import pandas as pd

from app.data.historical_manifest import get_election


NON_CANDIDATE_TOKENS = {
    "polling firm", "publisher/pollster", "pollster", "date", "date(s) administered",
    "polling period", "sample size", "sample", "lead", "others", "other",
    "blank/null/undec.", "blank/null/undec", "undecided", "abst.", "abstention",
    "results", "result", "source", "margin of error",
    "instituto", "data", "periodo", "periodo da pesquisa", "amostra", "margem de erro",
    "branco", "nulo", "indeciso", "nao sabe", "nenhum", "outros", "resultado",
}

PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    if isinstance(copy.columns, pd.MultiIndex):
        names: list[str] = []
        for column in copy.columns:
            parts = [str(part) for part in column if str(part) != "nan"]
            names.append(" | ".join(dict.fromkeys(parts)))
        copy.columns = names
    else:
        copy.columns = [str(c) for c in copy.columns]
    return copy


def _parse_percent(value: object) -> float | None:
    text = str(value).strip().replace(",", ".")
    if text in {"", "nan", "N/A", "–", "—", "-"}:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%?", text)
    if not match:
        return None
    number = float(match.group(1))
    return number / 100.0 if number > 1.0 else number


def _poll_end_date(value: object, year: int) -> date | None:
    text = str(value).replace("–", "-").replace("—", "-").strip()
    if not text or text.lower() == "nan":
        return None
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        return parsed.date()
    folded = _fold(text)
    pt_match = re.search(r"(?:\d{1,2}\s*(?:-|a)\s*)?(\d{1,2})\s+de\s+([a-z]+)(?:\s+de\s+(\d{4}))?", folded)
    if pt_match and pt_match.group(2) in PT_MONTHS:
        return date(int(pt_match.group(3) or year), PT_MONTHS[pt_match.group(2)], int(pt_match.group(1)))
    month_match = re.search(
        r"(?:\d{1,2}\s*[-/]\s*)?(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"(?:\s+(\d{4}))?",
        text,
        flags=re.IGNORECASE,
    )
    if month_match:
        day = int(month_match.group(1))
        month = pd.to_datetime(month_match.group(2), format="%B").month
        return date(int(month_match.group(3) or year), month, day)
    return None


def _candidate_columns(frame: pd.DataFrame) -> list[str]:
    result: list[str] = []
    for column in frame.columns:
        folded = _fold(column)
        if any(token in folded for token in NON_CANDIDATE_TOKENS):
            continue
        values = frame[column].map(_parse_percent)
        if values.notna().mean() >= 0.20:
            result.append(column)
    return result


def _find_column(columns: Iterable[str], *tokens: str) -> str | None:
    for column in columns:
        folded = _fold(column)
        if any(token in folded for token in tokens):
            return column
    return None


def extract_poll_tables(url: str, year: int, round_number: int = 1) -> pd.DataFrame:
    tables = pd.read_html(url, flavor="lxml")
    rows: list[dict[str, object]] = []
    for table_index, raw in enumerate(tables):
        frame = _flatten_columns(raw)
        pollster_col = _find_column(frame.columns, "polling firm", "publisher/pollster", "pollster", "instituto")
        date_col = _find_column(frame.columns, "date(s) administered", "polling period", "periodo da pesquisa", "periodo", "data")
        if pollster_col is None or date_col is None:
            continue
        sample_col = _find_column(frame.columns, "sample size", "sample", "amostra")
        candidates = _candidate_columns(frame)
        if len(candidates) < 2:
            continue
        for _, source_row in frame.iterrows():
            poll_date = _poll_end_date(source_row[date_col], year)
            if poll_date is None:
                continue
            pollster = str(source_row[pollster_col]).strip()
            if not pollster or _fold(pollster) in {"nan", "results", "resultado", "eleicoes"}:
                continue
            sample_size: int | None = None
            if sample_col is not None:
                match = re.search(r"[\d,\.]+", str(source_row[sample_col]))
                if match:
                    digits = re.sub(r"[^0-9]", "", match.group(0))
                    sample_size = int(digits) if digits else None
            for candidate_col in candidates:
                share = _parse_percent(source_row[candidate_col])
                if share is None:
                    continue
                candidate_name = str(candidate_col).split(" | ")[-1].strip()
                rows.append(
                    {
                        "year": year,
                        "round": round_number,
                        "poll_date": poll_date.isoformat(),
                        "pollster": pollster,
                        "sample_size": sample_size,
                        "candidate_name": candidate_name,
                        "share": share,
                        "scope": "BR",
                        "source_url": url,
                        "source_table": table_index,
                        "source_kind": "published_poll_table",
                    }
                )
    if not rows:
        raise ValueError(f"No polling tables could be parsed from {url}")
    return pd.DataFrame(rows).drop_duplicates().sort_values(["poll_date", "pollster", "candidate_name"]).reset_index(drop=True)


def load_historical_polls(year: int, round_number: int = 1) -> pd.DataFrame:
    election = get_election(year)
    if election.polling_page is None:
        raise ValueError(f"No structured polling source configured for {year}")
    return extract_poll_tables(election.polling_page, year=year, round_number=round_number)
