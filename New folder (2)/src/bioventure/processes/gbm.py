"""Geometric Brownian Motion process.

Model:
    dS = μ·S·dt + σ·S·dW

Exact discrete solution (log-normal step):
    S(t+dt) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z),  Z ~ N(0,1)

Known moments (for testing):
    E[S(t)]   = S0 · exp(μ·t)
    Var[S(t)] = S0² · exp(2μ·t) · (exp(σ²·t) - 1)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from bioventure.processes.base import StochasticProcess


class GeometricBrownianMotion(StochasticProcess):
    """GBM: the baseline smooth-growth-with-noise process.

    Parameters (via ``params`` dict)
    ----------
    drift : float
        Annualised drift μ.
    volatility : float
        Annualised volatility σ (must be > 0).
    """

    def _validate(self) -> None:
        if "drift" not in self._params:
            raise ValueError("GBM requires 'drift' parameter")
        if "volatility" not in self._params:
            raise ValueError("GBM requires 'volatility' parameter")
        if self._params["volatility"] <= 0:
            raise ValueError(
                f"GBM 'volatility' must be > 0, got {self._params['volatility']}"
            )

    @property
    def drift(self) -> float:
        return float(self._params["drift"])

    @property
    def volatility(self) -> float:
        return float(self._params["volatility"])

    def evolve(
        self,
        s0: float,
        t: float,
        dt: float,
        rng: Generator,
    ) -> np.ndarray:
        """Simulate a single GBM path.

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
            Shape ``(n_steps + 1,)`` path.
        """
        n_steps = self._check_evolve_inputs(s0, t, dt)
        mu = self.drift
        sigma = self.volatility

        drift_term = (mu - 0.5 * sigma ** 2) * dt
        diffusion_scale = sigma * np.sqrt(dt)

        z = rng.standard_normal(n_steps)
        log_increments = drift_term + diffusion_scale * z

        log_path = np.empty(n_steps + 1)
        log_path[0] = np.log(s0)
        np.cumsum(log_increments, out=log_path[1:])
        log_path[1:] += log_path[0]

        return np.exp(log_path)

    def evolve_batch(
        self,
        s0: float,
        t: float,
        dt: float,
        n_paths: int,
        rng: Generator,
    ) -> np.ndarray:
        """Simulate multiple independent GBM paths (fully vectorised).

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
            Shape ``(n_paths, n_steps + 1)`` array.
        """
        n_steps = self._check_evolve_inputs(s0, t, dt)
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")

        mu = self.drift
        sigma = self.volatility

        drift_term = (mu - 0.5 * sigma ** 2) * dt
        diffusion_scale = sigma * np.sqrt(dt)

        z = rng.standard_normal((n_paths, n_steps))
        log_increments = drift_term + diffusion_scale * z

        log_paths = np.empty((n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(s0)
        np.cumsum(log_increments, axis=1, out=log_paths[:, 1:])
        log_paths[:, 1:] += log_paths[:, 0:1]

        return np.exp(log_paths)
    