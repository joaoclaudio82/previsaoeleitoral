from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.ml.backtesting import evaluate_binary_forecasts, summarize_by_group


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate historical ElectionAI forecast records")
    parser.add_argument("path", type=Path, help="CSV with outcome and win_probability columns")
    parser.add_argument("--group-by", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.path)
    metrics = evaluate_binary_forecasts(frame)
    payload: dict[str, object] = {"overall": asdict(metrics)}
    if args.group_by:
        grouped = summarize_by_group(frame, args.group_by)
        payload["groups"] = grouped.to_dict(orient="records")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
