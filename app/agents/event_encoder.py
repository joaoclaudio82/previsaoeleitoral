from __future__ import annotations

from datetime import date
import json

import pandas as pd


OUTPUT_SCHEMA = {
    "event_id": "string",
    "title": "string",
    "summary": "string",
    "source": "mirofish",
    "experimental": True,
    "simulation_runs": "integer >= 1",
    "candidate_shocks": [
        {
            "candidate_id": "string",
            "uf": "two-letter UF",
            "vote_shift_mean": "percentage points in [-10, 10]",
            "vote_shift_sd": "percentage points in [0, 5]",
            "confidence": "number in [0, 1]",
            "rationale": "short explanation",
        }
    ],
    "state_shocks": [
        {
            "uf": "two-letter UF",
            "turnout_shift_mean": "fraction in [-0.10, 0.10]",
            "turnout_shift_sd": "fraction in [0, 0.05]",
            "undecided_shift_mean": "percentage points in [-10, 10]",
            "undecided_shift_sd": "percentage points in [0, 5]",
            "confidence": "number in [0, 1]",
            "rationale": "short explanation",
        }
    ],
    "provenance": {"mirofish_simulation_id": "optional string"},
}


def build_simulation_requirement(
    *,
    event_id: str,
    title: str,
    description: str,
    candidates: pd.DataFrame,
    state_ids: list[str],
    as_of_date: date,
) -> str:
    """Build a MiroFish requirement that produces an auditable shock payload.

    The prompt explicitly prevents the social simulator from declaring an election
    winner. Its task is limited to estimating conditional behavioral shocks.
    """
    required = {"candidate_id", "candidate_name"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em candidates: {sorted(missing)}")

    candidate_records = candidates[["candidate_id", "candidate_name"]].drop_duplicates().to_dict("records")
    states = sorted({str(uf).upper() for uf in state_ids})
    schema_text = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    candidate_text = json.dumps(candidate_records, ensure_ascii=False, indent=2)

    return f"""Você está executando uma simulação social experimental para ElectionAI.
Data de corte: {as_of_date.isoformat()}.
Evento: {title} (id={event_id}).
Descrição: {description}

Candidatos válidos:
{candidate_text}

UFs válidas: {', '.join(states)}

Objetivo científico:
- simular propagação social e reação coletiva ao evento;
- estimar apenas deslocamentos condicionais de intenção de voto, comparecimento e indecisos;
- representar incerteza por média e desvio-padrão;
- usar confiança baixa quando a evidência emergente for fraca ou contraditória;
- não declarar vencedor e não substituir pesquisas, TSE ou o posterior Bayesiano.

Ao final, produza um bloco JSON estritamente compatível com o contrato abaixo. Não inclua UFs ou candidatos inexistentes. Use zero quando não houver efeito detectável.
{schema_text}
"""
