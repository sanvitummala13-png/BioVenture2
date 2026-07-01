"""Bass diffusion model for AI-enabled drug discovery adoption.

The Bass model describes technology adoption as:

    F(t) = (1 - exp(-(p+q)·t)) / (1 + (q/p)·exp(-(p+q)·t))

where:
    p = coefficient of innovation (external influence),
    q = coefficient of imitation  (word-of-mouth / network effects),
    F(t) ∈ [0, 1] is the cumulative adoption fraction at time t.

This implementation adds:
- A ceiling < 1 (maximum addressable adoption fraction).
- A nonzero start_fraction at t=0 (adoption already underway in 2026),
  handled by computing the implied time-offset on the Bass curve.

Stochasticity enters through the prior draws of p and q; given (p, q)
the curve itself is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


def _bass_cdf(t: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Standard Bass cumulative adoption fraction.

    Parameters
    ----------
    t : np.ndarray
        Time values (any shape, broadcastable with p and q).
    p : np.ndarray
        Innovation coefficient (broadcastable).
    q : np.ndarray
        Imitation coefficient (broadcastable).

    Returns
    -------
    np.ndarray
        F(t) in [0, 1], same shape as broadcast(t, p, q).
    """
    pq = p + q
    exp_term = np.exp(-pq * t)
    return (1.0 - exp_term) / (1.0 + (q / p) * exp_term)


def _bass_time_offset(f0: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Find t0 such that bass_cdf(t0, p, q) = f0.

    Derived by inverting the Bass CDF:
        x = (1 - f0) / (1 + f0·q/p)
        t0 = -ln(x) / (p + q)

    Parameters
    ----------
    f0 : np.ndarray
        Initial adoption fraction (relative to ceiling) in [0, 1).
    p : np.ndarray
        Innovation coefficient.
    q : np.ndarray
        Imitation coefficient.

    Returns
    -------
    np.ndarray
        Time offsets (0 when f0 = 0).
    """
    f0 = np.asarray(f0, dtype=np.float64)
    x = (1.0 - f0) / (1.0 + f0 * q / p)
    return np.where(f0 > 0, -np.log(x) / (p + q), 0.0)


class AdoptionModel:
    """Bass diffusion model for AI adoption in drug discovery.

    Parameters
    ----------
    ceiling : float
        Maximum adoption fraction in (0, 1].
    start_fraction : float
        Adoption fraction at simulation start (t=0) in [0, ceiling).
    """

    def __init__(self, ceiling: float, start_fraction: float) -> None:
        if not 0.0 < ceiling <= 1.0:
            raise ValueError(f"ceiling must be in (0, 1], got {ceiling}")
        if not 0.0 <= start_fraction < ceiling:
            raise ValueError(
                f"start_fraction must be in [0, ceiling={ceiling}), "
                f"got {start_fraction}"
            )
        self._ceiling = ceiling
        self._start_fraction = start_fraction
        self._f0 = start_fraction / ceiling

    @property
    def ceiling(self) -> float:
        return self._ceiling

    @property
    def start_fraction(self) -> float:
        return self._start_fraction

    def compute_curve(
        self,
        p: float,
        q: float,
        n_years: int,
    ) -> np.ndarray:
        """Compute the adoption fraction curve for a single (p, q) draw.

        Parameters
        ----------
        p : float
            Innovation coefficient (must be > 0).
        q : float
            Imitation coefficient (must be > 0).
        n_years : int
            Number of years to simulate (returns n_years + 1 points
            including t=0).

        Returns
        -------
        np.ndarray
            Shape ``(n_years + 1,)`` adoption fractions in [0, ceiling].
        """
        self._validate_params(p, q)
        if n_years < 1:
            raise ValueError(f"n_years must be >= 1, got {n_years}")

        t_rel = np.arange(n_years + 1, dtype=np.float64)
        t0 = _bass_time_offset(self._f0, p, q)
        raw = _bass_cdf(t0 + t_rel, p, q)
        return np.minimum(raw * self._ceiling, self._ceiling)

    def draw(self, p: float, q: float, n_years: int) -> np.ndarray:
        """Alias for ``compute_curve`` — used when p, q come from priors."""
        return self.compute_curve(p, q, n_years)

    def draw_batch(
        self,
        p_array: np.ndarray,
        q_array: np.ndarray,
        n_years: int,
    ) -> np.ndarray:
        """Vectorised batch: compute adoption curves for many (p, q) draws.

        Parameters
        ----------
        p_array : np.ndarray
            Shape ``(n,)`` innovation coefficients.
        q_array : np.ndarray
            Shape ``(n,)`` imitation coefficients.
        n_years : int
            Number of simulation years.

        Returns
        -------
        np.ndarray
            Shape ``(n, n_years + 1)`` adoption fractions.
        """
        p_arr = np.asarray(p_array, dtype=np.float64)
        q_arr = np.asarray(q_array, dtype=np.float64)

        if p_arr.ndim != 1 or q_arr.ndim != 1:
            raise ValueError("p_array and q_array must be 1-D")
        if len(p_arr) != len(q_arr):
            raise ValueError(
                f"p_array and q_array must have same length, "
                f"got {len(p_arr)} and {len(q_arr)}"
            )
        if np.any(p_arr <= 0):
            raise ValueError("All p values must be > 0")
        if np.any(q_arr <= 0):
            raise ValueError("All q values must be > 0")
        if n_years < 1:
            raise ValueError(f"n_years must be >= 1, got {n_years}")

        n = len(p_arr)
        t_rel = np.arange(n_years + 1, dtype=np.float64)

        t0 = _bass_time_offset(self._f0, p_arr, q_arr)

        t_shifted = t0[:, np.newaxis] + t_rel[np.newaxis, :]
        p_2d = p_arr[:, np.newaxis]
        q_2d = q_arr[:, np.newaxis]

        raw = _bass_cdf(t_shifted, p_2d, q_2d)
        return np.minimum(raw * self._ceiling, self._ceiling)

    @classmethod
    def from_config(cls, ceiling: float, start_fraction: float) -> AdoptionModel:
        """Build from config values.

        Parameters
        ----------
        ceiling : float
            From ``base.yaml`` → ``adoption.ceiling``.
        start_fraction : float
            From ``base.yaml`` → ``adoption.start_fraction``.
        """
        return cls(ceiling, start_fraction)

    @staticmethod
    def _validate_params(p: float, q: float) -> None:
        if p <= 0:
            raise ValueError(f"Bass p must be > 0, got {p}")
        if q <= 0:
            raise ValueError(f"Bass q must be > 0, got {q}")
        