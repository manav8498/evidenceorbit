"""Deterministic hashing embedder for the fast-path soft signal.

This is not a neural embedding. It is a stable bag-of-tokens projection so
contradiction candidates can be scored without a model and without
per-process hash salt. Swap in a real embedder via the Embedder protocol
when you want semantic similarity; the judge still has to confirm.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from evidenceorbit.facts import tokenize


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right, strict=True)))
