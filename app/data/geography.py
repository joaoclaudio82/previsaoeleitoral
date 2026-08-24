from __future__ import annotations

BRAZIL_UFS: tuple[str, ...] = (
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
    "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
)

BRAZIL_UF_SET = frozenset(BRAZIL_UFS)


def is_brazilian_uf(value: object) -> bool:
    return str(value).upper().strip() in BRAZIL_UF_SET
