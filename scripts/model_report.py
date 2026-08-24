from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.ml.backtesting import evaluate_binary_forecasts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a compact ElectionAI validation report")
    parser.add_argument("backtest_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("model-validation-report.md"))
    args = parser.parse_args()

    frame = pd.read_csv(args.backtest_csv)
    metrics = asdict(evaluate_binary_forecasts(frame))
    lines = [
        "# ElectionAI — Model Validation Report",
        "",
        f"Observações: **{metrics['observations']}**",
        f"Brier score: **{metrics['brier']:.4f}**",
        f"Log loss: **{metrics['log_loss']:.4f}**",
        f"Expected calibration error: **{metrics['calibration_error']:.4f}**",
    ]
    if metrics["vote_share_mae"] is not None:
        lines.append(f"MAE de participação: **{metrics['vote_share_mae']:.4f}**")
    if metrics["interval_coverage"] is not None:
        lines.append(f"Cobertura do intervalo: **{metrics['interval_coverage']:.2%}**")
    lines.extend([
        "",
        "> Este relatório mede desempenho histórico. Não transforma dados sintéticos em previsão eleitoral real.",
        "",
    ])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
