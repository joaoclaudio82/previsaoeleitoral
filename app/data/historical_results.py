from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


def _norm_column(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


def read_tse_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("latin-1", "utf-8-sig", "utf-8"):
        for sep in (";", ","):
            try:
                frame = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
                if len(frame.columns) > 3:
                    frame.columns = [_norm_column(col) for col in frame.columns]
                    return frame
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_error = exc
    raise ValueError(f"Could not parse TSE file {path}: {last_error}")


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing expected TSE columns; tried {names}")


def normalize_presidential_results(frame: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    frame = frame.copy()
    cargo_col = next((c for c in ("DS_CARGO", "DS_CARGO_PERGUNTA") if c in frame.columns), None)
    if cargo_col is not None:
        frame = frame[frame[cargo_col].astype(str).str.upper().str.contains("PRESIDENT", na=False)]
    if year is not None and "ANO_ELEICAO" in frame.columns:
        frame = frame[pd.to_numeric(frame["ANO_ELEICAO"], errors="coerce") == year]

    uf_col = _first_existing(frame, ("SG_UF", "UF"))
    turn_col = _first_existing(frame, ("NR_TURNO", "NUM_TURNO"))
    candidate_col = _first_existing(frame, ("NM_VOTAVEL", "NM_CANDIDATO", "DS_CARGO_PERGUNTA"))
    votes_col = _first_existing(frame, ("QT_VOTOS", "QT_VOTOS_NOMINAIS", "QT_VOTOS_NOMINAIS_VALIDOS"))
    party_col = next((c for c in ("SG_PARTIDO", "NR_PARTIDO", "NM_PARTIDO") if c in frame.columns), None)

    clean = pd.DataFrame(
        {
            "year": year if year is not None else pd.to_numeric(frame.get("ANO_ELEICAO"), errors="coerce"),
            "round": pd.to_numeric(frame[turn_col], errors="coerce"),
            "uf": frame[uf_col].astype(str).str.upper().str.strip(),
            "candidate_name": frame[candidate_col].astype(str).str.strip(),
            "party": frame[party_col].astype(str).str.upper().str.strip() if party_col else "UNKNOWN",
            "votes": pd.to_numeric(frame[votes_col], errors="coerce").fillna(0).astype("int64"),
        }
    )
    clean = clean[~clean["candidate_name"].str.upper().isin({"BRANCO", "NULO", "#NULO#", "NULO TÉCNICO"})]
    grouped = clean.groupby(["year", "round", "uf", "candidate_name", "party"], as_index=False, dropna=False)["votes"].sum()
    valid_totals = grouped.groupby(["year", "round", "uf"])["votes"].transform("sum")
    grouped["vote_share"] = grouped["votes"] / valid_totals.where(valid_totals > 0)
    return grouped.sort_values(["year", "round", "uf", "votes"], ascending=[True, True, True, False]).reset_index(drop=True)


def normalize_turnout(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    frame = frame.copy()
    uf_col = _first_existing(frame, ("SG_UF", "UF"))
    eligible_col = _first_existing(frame, ("QT_APTOS", "QT_ELEITORES", "QT_ELEITOR_APTO"))
    present_col = _first_existing(frame, ("QT_COMPARECIMENTO", "QT_COMPARECIMENTO_TOT", "QT_ELEITORES_COMPARECIMENTO"))
    turn_col = next((c for c in ("NR_TURNO", "NUM_TURNO") if c in frame.columns), None)
    result = pd.DataFrame(
        {
            "year": year,
            "round": pd.to_numeric(frame[turn_col], errors="coerce") if turn_col else 1,
            "uf": frame[uf_col].astype(str).str.upper().str.strip(),
            "eligible": pd.to_numeric(frame[eligible_col], errors="coerce").fillna(0),
            "present": pd.to_numeric(frame[present_col], errors="coerce").fillna(0),
        }
    )
    result = result.groupby(["year", "round", "uf"], as_index=False)[["eligible", "present"]].sum()
    result["turnout_rate"] = result["present"] / result["eligible"].where(result["eligible"] > 0)
    return result
