from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.mirofish import MiroFishClient
from app.agents.event_encoder import build_simulation_requirement
from app.core.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria um projeto MiroFish para um cenário eleitoral experimental.")
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--seed", action="append", required=True, help="Arquivo PDF/MD/TXT; pode ser repetido.")
    parser.add_argument("--base-url", default=settings.mirofish_base_url)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fundamentals = pd.read_csv(ROOT / settings.fundamentals_path)
    states = pd.read_csv(ROOT / settings.state_priors_path)["uf"].dropna().astype(str).unique().tolist()
    requirement = build_simulation_requirement(
        event_id=args.event_id,
        title=args.title,
        description=args.description,
        candidates=fundamentals[["candidate_id", "candidate_name"]],
        state_ids=states,
        as_of_date=args.as_of,
    )
    seed_paths = [Path(item) for item in args.seed]
    with MiroFishClient(args.base_url) as client:
        project = client.generate_ontology(
            seed_paths,
            simulation_requirement=requirement,
            project_name=f"ElectionAI-{args.event_id}",
            additional_context=(
                "Camada experimental. Produzir efeitos condicionais e incertos; não declarar vencedor eleitoral."
            ),
        )
        graph = client.build_graph(project["project_id"])

    print("Projeto MiroFish criado.")
    print(f"project_id={project['project_id']}")
    print(f"graph_task_id={graph.get('task_id')}")
    print("Após o grafo concluir, crie/prepare a simulação no MiroFish e exporte o JSON do contrato ElectionAI.")


if __name__ == "__main__":
    main()
