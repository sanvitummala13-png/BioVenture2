"""Markov regime-switching process.

Model:
    At each time step the process occupies one of K regimes.
    Regime transitions follow a discrete-time Markov chain with
    row-stochastic transition matrix P.

    Within each regime k the price follows a GBM step:
        S(t+dt) = S(t) · exp((μ_k - σ_k²/2)·dt + σ_k·√dt·Z)

    This captures structural market shifts (e.g. hype → adoption → maturity)
    where drift and volatility differ across regimes.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from bioventure.processes.base import StochasticProcess


class RegimeSwitching(StochasticProcess):
    """Markov regime-switching GBM with K states.

    Parameters (via ``params`` dict)
    ----------
    n_regimes : int
        Number of regimes K (must be >= 2).
    regimes : list[dict]
        One dict per regime with keys ``"drift"`` and ``"volatility"``.
        Optional ``"name"`` key for labelling.
    transition_matrix : list[list[float]]
        K × K row-stochastic matrix. ``P[i][j]`` = probability of
        moving from regime *i* to regime *j* in one time step.
    initial_regime : int
        Starting regime index (0-based).
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        regimes = self._params["regimes"]
        self._n_regimes = int(self._params["n_regimes"])
        self._drifts = np.array([r["drift"] for r in regimes], dtype=np.float64)
        self._volatilities = np.array([r["volatility"] for r in regimes], dtype=np.float64)
        self._transition = np.array(self._params["transition_matrix"], dtype=np.float64)
        self._cum_transition = np.cumsum(self._transition, axis=1)
        self._initial_regime = int(self._params["initial_regime"])

    def _validate(self) -> None:
        required = ("n_regimes", "regimes", "transition_matrix", "initial_regime")
        for key in required:
            if key not in self._params:
                raise ValueError(f"RegimeSwitching requires '{key}' parameter")

        k = int(self._params["n_regimes"])
        if k < 2:
            raise ValueError(f"n_regimes must be >= 2, got {k}")

        regimes = self._params["regimes"]
        if len(regimes) != k:
            raise ValueError(f"Expected {k} regimes, got {len(regimes)}")
        for i, r in enumerate(regimes):
            if "drift" not in r or "volatility" not in r:
                raise ValueError(f"Regime {i} must have 'drift' and 'volatility'")
            if r["volatility"] <= 0:
                raise ValueError(
                    f"Regime {i} volatility must be > 0, got {r['volatility']}"
                )

        tm = np.array(self._params["transition_matrix"], dtype=np.float64)
        if tm.shape != (k, k):
            raise ValueError(
                f"transition_matrix must be ({k}, {k}), got {tm.shape}"
            )
        if np.any(tm < 0):
            raise ValueError("transition_matrix entries must be >= 0")
        row_sums = tm.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-8))[0]
            raise ValueError(
                f"transition_matrix rows must sum to 1.0, "
                f"violations at rows {bad.tolist()}: {row_sums[bad].tolist()}"
            )

        ir = int(self._params["initial_regime"])
        if not 0 <= ir < k:
            raise ValueError(f"initial_regime must be in [0, {k}), got {ir}")

    @property
    def n_regimes(self) -> int:
        return self._n_regimes

    @property
    def initial_regime(self) -> int:
        return self._initial_regime

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._transition.copy()

    def _transition_single(self, current: int, rng: Generator) -> int:
        """Transition one path to its next regime."""
        u = rng.random()
        new = int((u >= self._cum_transition[current]).sum())
        return min(new, self._n_regimes - 1)

    def _transition_batch(self, current: np.ndarray, rng: Generator) -> np.ndarray:
        """Transition a vector of paths to their next regimes."""
        cum = self._cum_transition[current]
        u = rng.random(len(current))
        new = (u[:, np.newaxis] >= cum).sum(axis=1)
        return np.minimum(new, self._n_regimes - 1).astype(np.intp)

    def evolve(
        self,
        s0: float,
        t: float,
        dt: float,
        rng: Generator,
    ) -> np.ndarray:
        """Simulate a single regime-switching path.

        Parameters
        ----------
        s0 : float
            Initial value.
        t : float
            Time horizon.
        dt : float
            Step size.
        rng : Generator
            Random generator.

        Returns
        -------
        np.ndarray
            Shape ``(n_steps + 1,)`` price path.
        """
        n_steps = self._check_evolve_inputs(s0, t, dt)
        sqrt_dt = np.sqrt(dt)

        log_path = np.empty(n_steps + 1)
        log_path[0] = np.log(s0)
        regime = self._initial_regime

        for step in range(n_steps):
            regime = self._transition_single(regime, rng)
            mu = self._drifts[regime]
            sigma = self._volatilities[regime]
            z = rng.standard_normal()
            log_path[step + 1] = (
                log_path[step]
                + (mu - 0.5 * sigma ** 2) * dt
                + sigma * sqrt_dt * z
            )

        return np.exp(log_path)

    def evolve_batch(
        self,
        s0: float,
        t: float,
        dt: float,
        n_paths: int,
        rng: Generator,
    ) -> np.ndarray:
        """Simulate multiple regime-switching paths.

        The step loop is unavoidable (regime at step *t* depends on
        step *t-1*), but within each step the GBM increments and
        regime transitions are fully vectorised across paths.

        Parameters
        ----------
        s0 : float
            Initial value.
        t : float
            Time horizon.
        dt : float
            Step size.
        n_paths : int
            Number of paths.
        rng : Generator
            Random generator.

        Returns
        -------
        np.ndarray
            Shape ``(n_paths, n_steps + 1)`` array of paths.
        """
        n_steps = self._check_evolve_inputs(s0, t, dt)
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")

        sqrt_dt = np.sqrt(dt)
        log_paths = np.empty((n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(s0)
        regimes = np.full(n_paths, self._initial_regime, dtype=np.intp)

        for step in range(n_steps):
            regimes = self._transition_batch(regimes, rng)
            mu = self._drifts[regimes]
            sigma = self._volatilities[regimes]
            z = rng.standard_normal(n_paths)
            log_paths[:, step + 1] = (
                log_paths[:, step]
                + (mu - 0.5 * sigma ** 2) * dt
                + sigma * sqrt_dt * z
            )

        return np.exp(log_paths)
    