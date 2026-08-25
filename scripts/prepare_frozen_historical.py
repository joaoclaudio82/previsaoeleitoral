from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.historical_snapshots import to_model_poll_schema  # noqa: E402


FROZEN_ROOT = ROOT / "data" / "historical_frozen"
OUTPUT_ROOT = ROOT / "data" / "historical"

RESULT_SCHEMA = {
    2010: {
        "a": ("Dilma Rousseff", "PT"),
        "b": ("José Serra", "PSDB"),
        "c": ("Marina Silva", "PV"),
        "other": ("Others", "OTHER"),
    },
    2014: {
        "a": ("Dilma Rousseff", "PT"),
        "b": ("Aécio Neves", "PSDB"),
        "c": ("Marina Silva", "PSB"),
        "other": ("Others", "OTHER"),
    },
    2018: {
        "a": ("Jair Bolsonaro", "PSL"),
        "b": ("Fernando Haddad", "PT"),
        "c": ("Ciro Gomes", "PDT"),
        "other": ("Others", "OTHER"),
    },
    2022: {
        "a": ("Lula", "PT"),
        "b": ("Jair Bolsonaro", "PL"),
        "c": ("Simone Tebet", "MDB"),
        "other": ("Others", "OTHER"),
    },
}

POLL_SCHEMA = {
    2014: {
        "dilma": ("Dilma Rousseff", "PT"),
        "aecio": ("Aécio Neves", "PSDB"),
        "marina": ("Marina Silva", "PSB"),
        "others": ("Others", "OTHER"),
    },
    2018: {
        "bolsonaro": ("Jair Bolsonaro", "PSL"),
        "haddad": ("Fernando Haddad", "PT"),
        "ciro": ("Ciro Gomes", "PDT"),
        "others": ("Others", "OTHER"),
    },
    2022: {
        "lula": ("Lula", "PT"),
        "bolsonaro": ("Jair Bolsonaro", "PL"),
        "tebet": ("Simone Tebet", "MDB"),
        "others": ("Others", "OTHER"),
    },
}

EXPECTED_UFS = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_results() -> dict[int, pd.DataFrame]:
    source = FROZEN_ROOT / "results_state.csv"
    wide = pd.read_csv(source)
    outputs: dict[int, pd.DataFrame] = {}
    for year, mapping in RESULT_SCHEMA.items():
        year_frame = wide[wide["year"] == year].copy()
        ufs = set(year_frame["uf"].astype(str))
        if ufs != EXPECTED_UFS:
            missing = sorted(EXPECTED_UFS - ufs)
            extra = sorted(ufs - EXPECTED_UFS)
            raise ValueError(f"Frozen results for {year} have invalid UF coverage; missing={missing}, extra={extra}")
        rows: list[dict[str, object]] = []
        for row in year_frame.itertuples(index=False):
            for key, (candidate_name, party) in mapping.items():
                votes = int(getattr(row, f"{key}_votes"))
                rows.append(
                    {
                        "year": year,
                        "round": 1,
                        "uf": str(row.uf),
                        "candidate_name": candidate_name,
                        "party": party,
                        "votes": votes,
                    }
                )
        result = pd.DataFrame(rows)
        result["vote_share"] = result["votes"] / result.groupby(["year", "round", "uf"])["votes"].transform("sum")
        outputs[year] = result.sort_values(["uf", "votes"], ascending=[True, False]).reset_index(drop=True)
    return outputs


def build_polls(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = FROZEN_ROOT / f"polls_{year}.csv"
    wide = pd.read_csv(source)
    mapping = POLL_SCHEMA[year]
    rows: list[dict[str, object]] = []
    for row in wide.itertuples(index=False):
        for column, (candidate_name, party) in mapping.items():
            share = pd.to_numeric(getattr(row, column), errors="coerce")
            if pd.isna(share):
                continue
            rows.append(
                {
                    "year": year,
                    "round": 1,
                    "poll_date": str(row.poll_date),
                    "pollster": str(row.pollster),
                    "sample_size": getattr(row, "sample_size", None),
                    "margin_error": getattr(row, "margin_error", None),
                    "candidate_name": candidate_name,
                    "party": party,
                    "share": float(share) / 100.0,
                    "undecided_share": getattr(row, "undecided_share", 0.0),
                    "scope": "BR",
                    "source_url": str(getattr(row, "source_url", "")),
                    "source_table": getattr(row, "source_table", 0),
                    "source_kind": "frozen_published_poll_snapshot",
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError(f"No frozen polling rows for {year}")
    expected_candidates = {name for name, _ in mapping.values()}
    observed = set(raw["candidate_name"].unique())
    if observed != expected_candidates:
        raise ValueError(f"Candidate mismatch for {year}: expected={expected_candidates}, observed={observed}")
    model = to_model_poll_schema(raw)
    return raw, model


def write_manifest(inputs: list[Path], outputs: list[Path]) -> None:
    payload = {
        "schema_version": 1,
        "purpose": "Immutable research inputs for ElectionAI retrospective backtesting",
        "scoreable_elections": [2014, 2018, 2022],
        "prior_only_elections": [2010],
        "common_forecast_horizons_days": [15, 7, 3, 1],
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in inputs
        ],
        "generated": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "notes": [
            "2010 is used only to construct priors for the 2014 state-level model.",
            "The scored elections use a common D-15, D-7, D-3 and D-1 horizon set.",
            "Minor first-round candidates are represented as a single Others category to preserve the vote simplex.",
            "Frozen polling inputs preserve reported undecided/blank/null shares when available.",
        ],
    }
    target = OUTPUT_ROOT / "manifest.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    inputs = [FROZEN_ROOT / "results_state.csv"] + [FROZEN_ROOT / f"polls_{year}.csv" for year in POLL_SCHEMA]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen research inputs: {missing}")

    results = build_results()
    output_paths: list[Path] = []
    for year, result in results.items():
        processed = OUTPUT_ROOT / str(year) / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        result_path = processed / "presidential_results.csv"
        result.to_csv(result_path, index=False)
        output_paths.append(result_path)
        if year in POLL_SCHEMA:
            raw, model = build_polls(year)
            raw_path = processed / "polls_raw.csv"
            model_path = processed / "polls_model_schema.csv"
            raw.to_csv(raw_path, index=False)
            model.to_csv(model_path, index=False)
            output_paths.extend([raw_path, model_path])

    write_manifest(inputs, output_paths)
    print(f"Prepared frozen historical research dataset: {len(output_paths)} generated files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
