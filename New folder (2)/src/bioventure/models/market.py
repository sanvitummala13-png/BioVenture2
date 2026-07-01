"""Market-size model: stochastic total market × adoption = AI-enabled market.

The model has two layers:
1. **Total market path** — evolved by a pluggable StochasticProcess
   (GBM, jump-diffusion, or regime-switching) from a 2025 anchor.
2. **AI-enabled market** — total market × adoption fraction (from
   the Bass diffusion model).

R&D compression affects *returns* (valuation layer), not market size,
so it is intentionally absent here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from bioventure.processes.base import StochasticProcess


@dataclass(frozen=True)
class MarketResult:
    """Result of a single Monte Carlo market-path draw.

    Attributes
    ----------
    total_market : np.ndarray
        Shape ``(n_steps + 1,)`` — total market size path (billion USD).
    adoption_fraction : np.ndarray
        Shape ``(n_steps + 1,)`` — AI adoption fraction per year.
    ai_enabled_market : np.ndarray
        Shape ``(n_steps + 1,)`` — AI-enabled market segment (billion USD).
    terminal_total : float
        Total market at final year.
    terminal_ai_enabled : float
        AI-enabled market at final year.
    """

    total_market: np.ndarray
    adoption_fraction: np.ndarray
    ai_enabled_market: np.ndarray
    terminal_total: float
    terminal_ai_enabled: float


class MarketModel:
    """Stochastic market model: total size × AI adoption.

    Parameters
    ----------
    anchor_2025_billion_usd : float
        Market size anchor at end of 2025 (must be > 0).
    """

    def __init__(self, anchor_2025_billion_usd: float) -> None:
        if anchor_2025_billion_usd <= 0:
            raise ValueError(
                f"anchor must be > 0, got {anchor_2025_billion_usd}"
            )
        self._anchor = anchor_2025_billion_usd

    @property
    def anchor(self) -> float:
        return self._anchor

    def evolve(
        self,
        process: StochasticProcess,
        adoption_curve: np.ndarray,
        t: float,
        dt: float,
        rng: Generator,
    ) -> MarketResult:
        """Simulate a single market path.

        Parameters
        ----------
        process : StochasticProcess
            Configured process instance (carries its own drift/vol params).
        adoption_curve : np.ndarray
            Shape ``(n_steps + 1,)`` adoption fractions from
            ``AdoptionModel.draw()``.
        t : float
            Time horizon in years.
        dt : float
            Step size in years.
        rng : Generator
            NumPy random generator.

        Returns
        -------
        MarketResult
        """
        total = process.evolve(self._anchor, t, dt, rng)
        adoption = np.asarray(adoption_curve, dtype=np.float64)
        self._check_alignment(total, adoption)

        ai_enabled = total * adoption

        return MarketResult(
            total_market=total,
            adoption_fraction=adoption,
            ai_enabled_market=ai_enabled,
            terminal_total=float(total[-1]),
            terminal_ai_enabled=float(ai_enabled[-1]),
        )

    def evolve_batch(
        self,
        process: StochasticProcess,
        adoption_curves: np.ndarray,
        t: float,
        dt: float,
        n_paths: int,
        rng: Generator,
    ) -> dict[str, np.ndarray]:
        """Simulate multiple market paths (vectorised).

        The process generates ``n_paths`` total-market paths with
        its fixed parameters. Each path is paired with the
        corresponding row of ``adoption_curves``.

        Parameters
        ----------
        process : StochasticProcess
            Configured process instance.
        adoption_curves : np.ndarray
            Shape ``(n_paths, n_steps + 1)`` adoption fractions.
        t : float
            Time horizon.
        dt : float
            Step size.
        n_paths : int
            Number of paths.
        rng : Generator
            NumPy random generator.

        Returns
        -------
        dict[str, np.ndarray]
            ``"total_market"``      : ``(n_paths, n_steps + 1)``
            ``"adoption_fraction"`` : ``(n_paths, n_steps + 1)``
            ``"ai_enabled_market"`` : ``(n_paths, n_steps + 1)``
            ``"terminal_total"``    : ``(n_paths,)``
            ``"terminal_ai_enabled"``: ``(n_paths,)``
        """
        total = process.evolve_batch(self._anchor, t, dt, n_paths, rng)
        adoption = np.asarray(adoption_curves, dtype=np.float64)

        if adoption.ndim != 2:
            raise ValueError(
                f"adoption_curves must be 2-D, got {adoption.ndim}-D"
            )
        if adoption.shape[0] != n_paths:
            raise ValueError(
                f"adoption_curves has {adoption.shape[0]} rows, "
                f"expected {n_paths}"
            )
        if adoption.shape[1] != total.shape[1]:
            raise ValueError(
                f"adoption_curves has {adoption.shape[1]} columns, "
                f"expected {total.shape[1]} (n_steps + 1)"
            )

        ai_enabled = total * adoption

        return {
            "total_market": total,
            "adoption_fraction": adoption,
            "ai_enabled_market": ai_enabled,
            "terminal_total": total[:, -1],
            "terminal_ai_enabled": ai_enabled[:, -1],
        }

    @classmethod
    def from_config(cls, anchor_2025_billion_usd: float) -> MarketModel:
        """Build from config value.

        Parameters
        ----------
        anchor_2025_billion_usd : float
            From ``base.yaml`` → ``market.anchor_2025_billion_usd``.
        """
        return cls(anchor_2025_billion_usd)

    @staticmethod
    def _check_alignment(
        total: np.ndarray,
        adoption: np.ndarray,
    ) -> None:
        """Verify that total-market and adoption arrays are compatible."""
        if adoption.ndim != 1:
            raise ValueError(
                f"adoption_curve must be 1-D, got {adoption.ndim}-D"
            )
        if len(adoption) != len(total):
            raise ValueError(
                f"adoption_curve length ({len(adoption)}) must match "
                f"total market path length ({len(total)})"
            )
        