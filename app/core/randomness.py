from __future__ import annotations

import hashlib
import random

import numpy as np


def child_seed(root_seed: int, namespace: str) -> int:
    """Derive a stable 32-bit seed from a root seed and namespace."""
    payload = f"{root_seed}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def rng(root_seed: int, namespace: str) -> np.random.Generator:
    return np.random.default_rng(child_seed(root_seed, namespace))


def seed_global(root_seed: int) -> None:
    random.seed(root_seed)
    np.random.seed(root_seed)
