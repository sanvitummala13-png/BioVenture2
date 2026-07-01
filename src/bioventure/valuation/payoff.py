"""Venture exit-multiple distributions for drug-discovery bets.

Models the heavy-tailed, skewed payoff structure of venture-style
investments where a few large exits dominate portfolio returns.

Supported distributions:
- **Lognormal**: exp(N(μ, σ²)) — standard for venture multiples.
- **Pareto**: power-law tail x^(-α) — heavier tail, more extreme winners.

Each draw produces a gross exit multiple (e.g. 0.0× = total loss,
1.0× = break-even, 10× = home run).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from scipy import stats


class PayoffDistribution(ABC):
    """Abstract base for exit-multiple distributions."""

    @abstractmethod
    def sample(self, n: int, rng: Generator) -> np.ndarray:
        """Draw *n* gross exit multiples.

        Returns
        -------
        np.ndarray
            Shape ``(n,)`` of non-negative multiples.
        """

    @abstractmethod
    def mean(self) -> float:
        """Theoretical mean of the distribution."""

    @abstractmethod
    def median(self) -> float:
        """Theoretical median of the distribution."""


class LognormalPayoff(PayoffDistribution):
    """Lognormal exit-multiple distribution.

    If X ~ LN(μ, σ²), then:
        mean   = exp(μ + σ²/2)
        median = exp(μ)

    Parameters
    ----------
    mu : float
        Mean of log-multiple.
    sigma : float
        Std deviation of log-multiple (must be > 0).
    """

    def __init__(self, mu: float, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self._mu = mu
        self._sigma = sigma

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self, n: int, rng: Generator) -> np.ndarray:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return rng.lognormal(self._mu, self._sigma, size=n)

    def mean(self) -> float:
        return float(np.exp(self._mu + 0.5 * self._sigma ** 2))

    def median(self) -> float:
        return float(np.exp(self._mu))


class ParetoPayoff(PayoffDistribution):
    """Pareto exit-multiple distribution (shifted to start at 0).

    Uses a Pareto Type I distribution shifted so that the minimum
    multiple is 0× (total loss). The raw Pareto has support [x_m, ∞);
    we set x_m = 1 and subtract 1, giving support [0, ∞).

    PDF: f(x) = α / (x + 1)^(α+1),  x >= 0

    Parameters
    ----------
    alpha : float
        Shape parameter controlling tail heaviness (must be > 2 for
        finite variance, > 1 for finite mean).
    """

    def __init__(self, alpha: float) -> None:
        if alpha <= 1.0:
            raise ValueError(
                f"alpha must be > 1 for finite mean, got {alpha}"
            )
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def sample(self, n: int, rng: Generator) -> np.ndarray:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        raw = (rng.pareto(self._alpha, size=n))
        return raw

    def mean(self) -> float:
        return 1.0 / (self._alpha - 1.0)

    def median(self) -> float:
        return float(2.0 ** (1.0 / self._alpha) - 1.0)


def build_payoff(distribution: str, params: dict) -> PayoffDistribution:
    """Factory: construct a PayoffDistribution from config strings.

    Parameters
    ----------
    distribution : str
        ``"lognormal"`` or ``"pareto"``.
    params : dict
        Distribution-specific parameters.

    Returns
    -------
    PayoffDistribution
    """
    dist = distribution.lower().strip()
    if dist == "lognormal":
        if "mu" not in params or "sigma" not in params:
            raise ValueError("Lognormal payoff requires 'mu' and 'sigma'")
        return LognormalPayoff(mu=params["mu"], sigma=params["sigma"])
    if dist == "pareto":
        if "alpha" not in params:
            raise ValueError("Pareto payoff requires 'alpha'")
        return ParetoPayoff(alpha=params["alpha"])
    raise ValueError(
        f"Unknown payoff distribution '{distribution}', "
        f"expected 'lognormal' or 'pareto'"
    )
