from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicationDecision:
    allowed: bool
    status: str
    watermark: str | None
    reason: str


def assess_publication(
    dataset_type: str,
    election_year: int,
    validation_status: str,
) -> PublicationDecision:
    if dataset_type == "synthetic":
        return PublicationDecision(
            allowed=False,
            status="BLOCKED_SYNTHETIC_DEMONSTRATION",
            watermark="DEMONSTRAÇÃO SINTÉTICA — NÃO É PREVISÃO ELEITORAL",
            reason=(
                "Resultados produzidos com dados sintéticos não podem ser apresentados como "
                f"previsão da eleição presidencial de {election_year}."
            ),
        )
    if validation_status != "independently_validated":
        return PublicationDecision(
            allowed=False,
            status="BLOCKED_PENDING_INDEPENDENT_VALIDATION",
            watermark="MODELO NÃO VALIDADO PARA PUBLICAÇÃO",
            reason="A publicação externa exige backtesting temporal e validação metodológica independente.",
        )
    return PublicationDecision(
        allowed=True,
        status="PUBLICATION_ELIGIBLE",
        watermark=None,
        reason="Dados operacionais e modelo informados como independentemente validados.",
    )
