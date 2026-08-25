from __future__ import annotations

import io
import re
import unicodedata

import httpx
import pandas as pd


STATE_TO_UF = {"acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM", "bahia": "BA", "ceara": "CE", "distrito federal": "DF", "espirito santo": "ES", "goias": "GO", "maranhao": "MA", "mato grosso": "MT", "mato grosso do sul": "MS", "minas gerais": "MG", "para": "PA", "paraiba": "PB", "parana": "PR", "pernambuco": "PE", "piaui": "PI", "rio de janeiro": "RJ", "rio grande do norte": "RN", "rio grande do sul": "RS", "rondonia": "RO", "roraima": "RR", "santa catarina": "SC", "sao paulo": "SP", "sergipe": "SE", "tocantins": "TO"}
RESULT_CONFIG = {
    2010: {"url": "https://pt.wikipedia.org/wiki/Resultados_da_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_2010", "candidates": [("Dilma", "Dilma Rousseff", "PT"), ("Serra", "José Serra", "PSDB"), ("Marina", "Marina Silva", "PV")]},
    2014: {"url": "https://pt.wikipedia.org/wiki/Resultados_da_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_2014", "candidates": [("Dilma", "Dilma Rousseff", "PT"), ("Aecio", "Aécio Neves", "PSDB"), ("Marina", "Marina Silva", "PSB")]},
    2018: {"url": "https://pt.wikipedia.org/wiki/Resultados_da_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_2018", "candidates": [("Bolsonaro", "Jair Bolsonaro", "PSL"), ("Haddad", "Fernando Haddad", "PT"), ("Ciro", "Ciro Gomes", "PDT"), ("Alckmin", "Geraldo Alckmin", "PSDB"), ("Amoedo", "João Amoêdo", "NOVO")]},
    2022: {"url": "https://pt.wikipedia.org/wiki/Resultados_da_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_2022", "candidates": [("Lula", "Lula", "PT"), ("Bolsonaro", "Jair Bolsonaro", "PL"), ("Tebet", "Simone Tebet", "MDB"), ("Ciro", "Ciro Gomes", "PDT")]},
}


def _fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+", " ", text).strip()


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    if isinstance(copy.columns, pd.MultiIndex):
        copy.columns = [" | ".join(str(part) for part in column if str(part) != "nan") for column in copy.columns]
    else:
        copy.columns = [str(column) for column in copy.columns]
    return copy


def _numeric(series: pd.Series) -> pd.Series:
    def parse(value: object) -> float:
        text = str(value).replace("\xa0", " ").strip()
        text = re.sub(r"[^0-9,.-]", "", text)
        if not text:
            return float("nan")
        if "," in text and "." not in text:
            text = text.replace(".", "").replace(",", ".")
        elif text.count(".") > 1:
            text = text.replace(".", "")
        try:
            return float(text)
        except ValueError:
            return float("nan")
    return series.map(parse)


def _state_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        folded = _fold(column)
        if "estado" in folded or "unidade federativa" in folded:
            return column
    for column in frame.columns[:3]:
        states = frame[column].map(_fold).map(lambda value: value in STATE_TO_UF)
        if states.sum() >= 20:
            return column
    return None


def _candidate_vote_column(frame: pd.DataFrame, token: str) -> str | None:
    matches = [column for column in frame.columns if _fold(token) in _fold(column)]
    if not matches:
        return None
    scored = []
    for column in matches:
        values = _numeric(frame[column])
        large = values[values > 100]
        score = (len(large), float(large.median()) if not large.empty else 0.0)
        scored.append((score, column))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0][0] > 0 else None


def _read_tables(url: str) -> list[pd.DataFrame]:
    response = httpx.get(url, follow_redirects=True, timeout=60.0, headers={"User-Agent": "ElectionAI-research/0.3 (academic historical election forecasting; contact via GitHub)"})
    response.raise_for_status()
    return pd.read_html(io.StringIO(response.text), flavor="lxml")


def load_secondary_state_results(year: int) -> pd.DataFrame:
    if year not in RESULT_CONFIG:
        raise ValueError(f"No secondary result source configured for {year}")
    config = RESULT_CONFIG[year]
    for raw in _read_tables(config["url"]):
        frame = _flatten(raw)
        state_col = _state_column(frame)
        if state_col is None:
            continue
        candidate_columns = []
        for token, candidate_name, party in config["candidates"]:
            column = _candidate_vote_column(frame, token)
            if column is not None:
                candidate_columns.append((candidate_name, party, column))
        if len(candidate_columns) < 2:
            continue
        rows = []
        for _, source in frame.iterrows():
            uf = STATE_TO_UF.get(_fold(source[state_col]))
            if uf is None:
                continue
            for candidate_name, party, column in candidate_columns:
                value = _numeric(pd.Series([source[column]])).iloc[0]
                if pd.isna(value) or value <= 0:
                    continue
                rows.append({"year": year, "round": 1, "uf": uf, "candidate_name": candidate_name, "party": party, "votes": int(round(value))})
        if len({row["uf"] for row in rows}) >= 20:
            result = pd.DataFrame(rows)
            result["vote_share"] = result["votes"] / result.groupby(["year", "round", "uf"])["votes"].transform("sum")
            return result
    raise RuntimeError(f"Could not parse secondary state results for {year}")
