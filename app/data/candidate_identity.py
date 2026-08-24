from __future__ import annotations

import re
import unicodedata


ALIASES: dict[str, set[str]] = {
    "lula": {"lula", "luiz inacio lula da silva", "luiz inácio lula da silva"},
    "bolsonaro": {"jair bolsonaro", "jair messias bolsonaro", "bolsonaro"},
    "dilma": {"dilma rousseff", "dilma vana rousseff"},
    "aecio": {"aecio neves", "aécio neves", "aecio neves da cunha"},
    "haddad": {"fernando haddad", "haddad"},
    "ciro": {"ciro gomes", "ciro ferreira gomes", "ciro"},
    "marina": {"marina silva", "maria osmarina marina silva vaz de lima", "marina"},
    "alckmin": {"geraldo alckmin", "geraldo jose rodrigues alckmin filho", "alckmin"},
    "tebet": {"simone tebet", "simone nassar tebet", "tebet"},
    "amoedo": {"joao amoedo", "joão amoêdo", "joao dionisio filgueira barreto amoedo", "amoedo", "amoêdo"},
}


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


_LOOKUP = {
    normalize_name(alias): canonical
    for canonical, aliases in ALIASES.items()
    for alias in aliases
}


def canonical_candidate(value: str) -> str:
    normalized = normalize_name(value)
    if normalized in _LOOKUP:
        return _LOOKUP[normalized]
    tokens = normalized.split()
    matches: list[tuple[int, str]] = []
    for alias, canonical in _LOOKUP.items():
        alias_tokens = alias.split()
        if all(token in tokens for token in alias_tokens):
            matches.append((len(alias_tokens), canonical))
    if matches:
        matches.sort(reverse=True)
        return matches[0][1]
    return normalized.replace(" ", "_")
