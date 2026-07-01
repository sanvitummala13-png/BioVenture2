"""Portfolio-level aggregation of venture-style drug-discovery bets.

A portfolio of *n_bets* equal-sized investments, each drawing an
exit multiple from a PayoffDistribution. Optionally modulated by a
clinical success probability: each bet independently survives
(Bernoulli) before receiving its exit multiple, capturing the
"most bets fail, a few win big" dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from bioventure.valuation.payoff import PayoffDistribution


@dataclass(frozen=True)
class PortfolioResult:
    """Result of a single portfolio simulation.

    Attributes
    ----------
    exit_multiples : np.ndarray
        Shape ``(n_bets,)`` — realised gross multiple per bet
        (0 if the bet failed clinically).
    total_invested_m : float
        Total capital deployed (million USD).
    total_returned_m : float
        Total exit proceeds (million USD).
    gross_multiple : float
        Portfolio MOIC = total_returned / total_invested.
    net_profit_m : float
        total_returned - total_invested (million USD).
    success_rate : float
        Fraction of bets with multiple > 1.0.
    """

    exit_multiples: np.ndarray
    total_invested_m: float
    total_returned_m: float
    gross_multiple: float
    net_profit_m: float
    success_rate: float


class PortfolioModel:
    """Simulate a portfolio of equal-sized venture bets.

    Parameters
    ----------
    n_bets : int
        Number of investments (must be >= 1).
    bet_size_million_usd : float
        Capital per bet in million USD (must be > 0).
    payoff : PayoffDistribution
        Distribution of gross exit multiples.
    """

    def __init__(
        self,
        n_bets: int,
        bet_size_million_usd: float,
        payoff: PayoffDistribution,
    ) -> None:
        if n_bets < 1:
            raise ValueError(f"n_bets must be >= 1, got {n_bets}")
        if bet_size_million_usd <= 0:
            raise ValueError(
                f"bet_size_million_usd must be > 0, got {bet_size_million_usd}"
            )
        self._n_bets = n_bets
        self._bet_size = bet_size_million_usd
        self._payoff = payoff
        self._total_invested = n_bets * bet_size_million_usd

    @property
    def n_bets(self) -> int:
        return self._n_bets

    @property
    def bet_size_million_usd(self) -> float:
        return self._bet_size

    @property
    def total_invested_m(self) -> float:
        return self._total_invested

    def simulate(
        self,
        rng: Generator,
        success_probability: float = 1.0,
    ) -> PortfolioResult:
        """Run a single portfolio simulation.

        Parameters
        ----------
        rng : Generator
            NumPy random generator.
        success_probability : float
            Per-bet probability of clinical success in [0, 1].
            Failed bets receive a 0× multiple.

        Returns
        -------
        PortfolioResult
        """
        self._validate_prob(success_probability)

        raw_multiples = self._payoff.sample(self._n_bets, rng)

        if success_probability < 1.0:
            survived = rng.random(self._n_bets) < success_probability
            multiples = raw_multiples * survived
        else:
            multiples = raw_multiples

        total_returned = float(np.sum(multiples) * self._bet_size)

        return PortfolioResult(
            exit_multiples=multiples,
            total_invested_m=self._total_invested,
            total_returned_m=total_returned,
            gross_multiple=total_returned / self._total_invested,
            net_profit_m=total_returned - self._total_invested,
            success_rate=float(np.mean(multiples > 1.0)),
        )

    def simulate_batch(
        self,
        n: int,
        rng: Generator,
        success_probabilities: np.ndarray | float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Vectorised batch of portfolio simulations.

        Parameters
        ----------
        n : int
            Number of portfolio simulations.
        rng : Generator
            NumPy random generator.
        success_probabilities : np.ndarray or float
            Per-simulation clinical success probability.
            Scalar applies the same PoS to all simulations.
            Array of shape ``(n,)`` gives per-simulation PoS
            (e.g. from clinical model draws).

        Returns
        -------
        dict[str, np.ndarray]
            ``"exit_multiples"``  : ``(n, n_bets)``
            ``"total_returned_m"`` : ``(n,)``
            ``"gross_multiple"``   : ``(n,)``
            ``"net_profit_m"``     : ``(n,)``
            ``"success_rate"``     : ``(n,)``
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        sp = np.asarray(success_probabilities, dtype=np.float64)
        if sp.ndim == 0:
            sp = np.full(n, float(sp))
        if sp.shape != (n,):
            raise ValueError(
                f"success_probabilities must be scalar or shape ({n},), "
                f"got shape {sp.shape}"
            )
        if np.any(sp < 0) or np.any(sp > 1):
            raise ValueError("success_probabilities must be in [0, 1]")

        raw = self._payoff.sample(n * self._n_bets, rng).reshape(n, self._n_bets)

        uniform = rng.random((n, self._n_bets))
        survived = uniform < sp[:, np.newaxis]
        multiples = raw * survived

        returns_per_bet = multiples * self._bet_size
        total_returned = returns_per_bet.sum(axis=1)

        return {
            "exit_multiples": multiples,
            "total_returned_m": total_returned,
            "gross_multiple": total_returned / self._total_invested,
            "net_profit_m": total_returned - self._total_invested,
            "success_rate": (multiples > 1.0).mean(axis=1),
        }

    @staticmethod
    def _validate_prob(p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"success_probability must be in [0, 1], got {p}"
            )
        