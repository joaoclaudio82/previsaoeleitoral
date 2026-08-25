from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.services.agent_scenarios import load_agent_scenario
from app.services.predictor import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara baseline Bayesiano com um cenário MiroFish experimental.")
    parser.add_argument("--scenario", type=Path, default=ROOT / settings.agent_scenario_path)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--strength", type=float, default=settings.agent_scenario_strength)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    return parser.parse_args()


def _load_inputs() -> dict:
    return {
        "polls": pd.read_csv(ROOT / settings.polls_path),
        "fundamentals": pd.read_csv(ROOT / settings.fundamentals_path),
        "state_priors": pd.read_csv(ROOT / settings.state_priors_path),
        "turnout": pd.read_csv(ROOT / settings.turnout_path),
    }


def _run(inputs: dict, args: argparse.Namespace, scenario=None):
    return predict(
        **inputs,
        as_of_date=args.as_of,
        model_path=ROOT / settings.model_path,
        pollster_calibration_path=ROOT / settings.pollster_calibration_path,
        turnout_model_path=ROOT / settings.turnout_model_path,
        transfer_model_path=ROOT / settings.transfer_model_path,
        n_simulations=args.simulations,
        posterior_draws=min(settings.posterior_draws, args.simulations),
        seed=args.seed,
        agent_scenario=scenario,
        agent_scenario_strength=args.strength,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.strength <= 1.0:
        raise SystemExit("--strength deve estar entre 0 e 1")

    inputs = _load_inputs()
    scenario = load_agent_scenario(args.scenario)
    baseline = _run(inputs, args)
    hybrid = _run(inputs, args, scenario=scenario)

    base = baseline.candidates.set_index("candidate_id")
    experimental = hybrid.candidates.set_index("candidate_id")
    comparison = pd.DataFrame(
        {
            "candidate_name": experimental["candidate_name"],
            "baseline_first_round": base["expected_first_round_share"],
            "scenario_first_round": experimental["expected_first_round_share"],
            "delta_first_round": experimental["expected_first_round_share"] - base["expected_first_round_share"],
            "baseline_win_probability": base["win_probability"],
            "scenario_win_probability": experimental["win_probability"],
            "delta_win_probability": experimental["win_probability"] - base["win_probability"],
        }
    ).reset_index()

    print("EXPERIMENTO CONTRAFACTUAL — NÃO SUBSTITUI O FORECAST BAYESIANO")
    print(f"event_id={scenario.event_id} | strength={args.strength:.2f}")
    print(comparison.to_string(index=False))
    print("\nDiagnósticos do cenário:")
    print(hybrid.diagnostics.get("agent_scenario"))


if __name__ == "__main__":
    main()
