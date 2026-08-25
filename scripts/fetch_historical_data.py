from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.historical_polls import load_historical_polls
from app.adapters.tse_historical import discover_presidential_results, discover_turnout, download_verified, extract_if_archive
from app.data.historical_manifest import get_election
from app.data.historical_results import normalize_presidential_results, normalize_turnout, read_tse_csv
from app.data.historical_snapshots import to_model_poll_schema


def _parse_candidate_files(paths: list[Path], year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for path in paths:
        if path.suffix.lower() not in {".csv", ".txt"}:
            continue
        try:
            raw = read_tse_csv(path)
            normalized = normalize_presidential_results(raw, year=year)
            if not normalized.empty:
                frames.append(normalized)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if not frames:
        raise RuntimeError(f"No presidential TSE result file could be normalized for {year}: {errors[:5]}")
    result = pd.concat(frames, ignore_index=True).drop_duplicates()
    return result.groupby(["year", "round", "uf", "candidate_name", "party"], as_index=False)["votes"].sum().assign(
        vote_share=lambda df: df["votes"] / df.groupby(["year", "round", "uf"])["votes"].transform("sum")
    )


def fetch_year(year: int, root: Path, include_polls: bool = True, include_turnout: bool = True) -> dict[str, object]:
    election = get_election(year)
    year_dir = root / str(year)
    raw_dir = year_dir / "raw"
    processed_dir = year_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"year": year, "first_round_date": election.first_round_date.isoformat(), "sources": []}

    resource = discover_presidential_results(year)
    downloaded = download_verified(resource, raw_dir)
    extracted = extract_if_archive(downloaded, raw_dir / "results_extracted")
    results = _parse_candidate_files(extracted, year)
    result_path = processed_dir / "presidential_results.csv"
    results.to_csv(result_path, index=False)
    manifest["sources"].append({"kind": "tse_results", "url": downloaded.source_url, "sha256": downloaded.sha256, "resource_name": downloaded.resource_name})

    if include_polls and election.polling_page:
        published = load_historical_polls(year, round_number=1)
        published_path = processed_dir / "published_polls_raw.csv"
        published.to_csv(published_path, index=False)
        model_polls = to_model_poll_schema(published)
        model_path = processed_dir / "polls_model_schema.csv"
        model_polls.to_csv(model_path, index=False)
        manifest["sources"].append({"kind": "published_poll_table", "url": election.polling_page, "rows": len(published)})

    if include_turnout and election.tse_turnout_package:
        try:
            turnout_resource = discover_turnout(year)
            turnout_download = download_verified(turnout_resource, raw_dir / "turnout")
            turnout_files = extract_if_archive(turnout_download, raw_dir / "turnout_extracted")
            turnout_frames: list[pd.DataFrame] = []
            for path in turnout_files:
                if path.suffix.lower() not in {".csv", ".txt"}:
                    continue
                try:
                    turnout_frames.append(normalize_turnout(read_tse_csv(path), year))
                except Exception:
                    continue
            if turnout_frames:
                turnout = pd.concat(turnout_frames, ignore_index=True)
                turnout = turnout.groupby(["year", "round", "uf"], as_index=False)[["eligible", "present"]].sum()
                turnout["turnout_rate"] = turnout["present"] / turnout["eligible"].where(turnout["eligible"] > 0)
                turnout.to_csv(processed_dir / "turnout.csv", index=False)
                manifest["sources"].append({"kind": "tse_turnout", "url": turnout_download.source_url, "sha256": turnout_download.sha256})
        except Exception as exc:
            manifest["sources"].append({"kind": "tse_turnout", "status": "unavailable", "error": type(exc).__name__})

    (year_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch auditable historical Brazilian presidential election data")
    parser.add_argument("--years", nargs="+", type=int, default=[2010, 2014, 2018, 2022])
    parser.add_argument("--output", type=Path, default=Path("data/historical"))
    parser.add_argument("--no-polls", action="store_true")
    parser.add_argument("--no-turnout", action="store_true")
    args = parser.parse_args()
    manifests = [fetch_year(year, args.output, not args.no_polls, not args.no_turnout) for year in args.years]
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
