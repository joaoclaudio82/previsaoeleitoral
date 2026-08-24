from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.data.fingerprints import dataframe_sha256
from app.governance.data_quality_gate import evaluate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an ElectionAI CSV before ingestion")
    parser.add_argument("dataset", choices=["polls", "turnout", "fundamentals"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.path)
    decision = evaluate_dataset(args.dataset, frame)
    output = {
        "dataset": args.dataset,
        "path": str(args.path),
        "rows": len(frame),
        "sha256": dataframe_sha256(frame),
        "allowed": decision.allowed,
        "errors": list(decision.errors),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if decision.allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
