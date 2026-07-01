"""Centralized seeded RNG factory for reproducible simulation."""

from __future__ import annotations

from numpy.random import Generator, SeedSequence, default_rng


class RNGFactory:
    """Creates independent, reproducible random streams from a single seed.

    Usage:
        factory = RNGFactory(42)
        clinical_rng, market_rng, copula_rng = factory.spawn(3)
    """

    def __init__(self, seed: int):
        self._seed_seq = SeedSequence(seed)

    @property
    def seed(self) -> int:
        return self._seed_seq.entropy

    def spawn(self, n: int = 1) -> list[Generator]:
        return [default_rng(s) for s in self._seed_seq.spawn(n)]

    def spawn_one(self) -> Generator:
        return self.spawn(1)[0]
