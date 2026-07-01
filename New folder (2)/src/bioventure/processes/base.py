"""Abstract base for pluggable stochastic market processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator


class StochasticProcess(ABC):
    """Base class for all stochastic market-price processes.

    Every process evolves an initial value ``S0`` over a time grid and
    returns the full path including the initial value.

    Path shape conventions:
    - ``evolve``       → ``(n_steps + 1,)``
    - ``evolve_batch`` → ``(n_paths, n_steps + 1)``

    where ``n_steps = int(T / dt)`` and column 0 is always ``S0``.

    Parameters
    ----------
    params : dict[str, float | list]
        Process-specific parameters (drift, volatility, jump intensity, …).
        Validated by each concrete subclass.
    """

    def __init__(self, params: dict) -> None:
        self._params = dict(params)
        self._validate()

    @property
    def params(self) -> dict:
        """Return a copy of the process parameters."""
        return dict(self._params)

    @abstractmethod
    def _validate(self) -> None:
        """Check that ``self._params`` contains valid values.

        Raises
        ------
        ValueError
            If any required parameter is missing or out of range.
        """

    @abstractmethod
    def evolve(
        self,
        s0: float,
        t: float,
        dt: float,
        rng: Generator,
    ) -> np.ndarray:
        """Simulate a single price path.

        Parameters
        ----------
        s0 : float
            Initial value (must be > 0).
        t : float
            Total time horizon (years).
        dt : float
            Time-step size (years).
        rng : Generator
            NumPy random generator.

        Returns
        -------
        np.ndarray
            Shape ``(n_steps + 1,)`` path starting at ``s0``.
        """

    @abstractmethod
    def evolve_batch(
        self,
        s0: float,
        t: float,
        dt: float,
        n_paths: int,
        rng: Generator,
    ) -> np.ndarray:
        """Simulate multiple independent paths (vectorised where possible).

        Parameters
        ----------
        s0 : float
            Initial value (must be > 0).
        t : float
            Total time horizon (years).
        dt : float
            Time-step size (years).
        n_paths : int
            Number of independent paths.
        rng : Generator
            NumPy random generator.

        Returns
        -------
        np.ndarray
            Shape ``(n_paths, n_steps + 1)`` array of paths.
        """

    def _check_evolve_inputs(self, s0: float, t: float, dt: float) -> int:
        """Shared input validation for evolve / evolve_batch.

        Returns
        -------
        int
            Number of time steps.

        Raises
        ------
        ValueError
            If inputs are out of range.
        """
        if s0 <= 0:
            raise ValueError(f"s0 must be > 0, got {s0}")
        if t <= 0:
            raise ValueError(f"t must be > 0, got {t}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if dt > t:
            raise ValueError(f"dt ({dt}) must be <= t ({t})")
        n_steps = int(round(t / dt))
        if n_steps < 1:
            raise ValueError(f"t/dt must yield >= 1 step, got {n_steps}")
        return n_steps
    