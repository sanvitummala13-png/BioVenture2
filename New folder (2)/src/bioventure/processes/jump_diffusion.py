"""Merton jump-diffusion process.

Model:
    dS/S = (μ - λk)dt + σ·dW + dJ

where:
    J is a compound Poisson process with log-normal jump sizes,
    k = E[e^J - 1] = exp(jump_mean + jump_std²/2) - 1  (compensator).

Discrete step:
    S(t+dt) = S(t) · exp((μ - σ²/2 - λk)·dt + σ·√dt·Z + ΣJ_i)

    Z ~ N(0,1)                          (diffusion)
    N_t ~ Poisson(λ·dt)                 (jump count)
    J_i ~ N(jump_mean, jump_std²)       (individual jump sizes)
    ΣJ_i ~ N(N_t·jump_mean, N_t·jump_std²)   (sum of N_t jumps)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from bioventure.processes.base import StochasticProcess


class JumpDiffusion(StochasticProcess):
    """Merton jump-diffusion: GBM plus compound Poisson shocks.

    Parameters (via ``params`` dict)
    ----------
    drift : float
        Annualised drift μ.
    volatility : float
        Annualised volatility σ (must be > 0).
    jump_intensity : float
        Expected number of jumps per year λ (must be >= 0).
    jump_mean : float
        Mean of the log-jump size distribution.
    jump_std : float
        Std deviation of the log-jump size distribution (must be > 0).
    """

    def _validate(self) -> None:
        required = ("drift", "volatility", "jump_intensity", "jump_mean", "jump_std")
        for key in required:
            if key not in self._params:
                raise ValueError(f"JumpDiffusion requires '{key}' parameter")
        if self._params["volatility"] <= 0:
            raise ValueError(
                f"JumpDiffusion 'volatility' must be > 0, got {self._params['volatility']}"
            )
        if self._params["jump_intensity"] < 0:
            raise ValueError(
                f"JumpDiffusion 'jump_intensity' must be >= 0, "
                f"got {self._params['jump_intensity']}"
            )
        if self._params["jump_std"] <= 0:
            raise ValueError(
                f"JumpDiffusion 'jump_std' must be > 0, got {self._params['jump_std']}"
            )

    @property
    def drift(self) -> float:
        return float(self._params["drift"])

    @property
    def volatility(self) -> float:
        return float(self._params["volatility"])

    @property
    def jump_intensity(self) -> float:
        return float(self._params["jump_intensity"])

    @property
    def jump_mean(self) -> float:
        return float(self._params["jump_mean"])

    @property
    def jump_std(self) -> float:
        return float(self._params["jump_std"])

    @property
    def compensator(self) -> float:
        """k = E[e^J - 1], removes drift bias introduced by jumps."""
        return np.exp(self.jump_mean + 0.5 * self.jump_std ** 2) - 1.0

    def _simulate_jump_sums(
        self,
        shape: tuple[int, ...],
        dt: float,
        rng: Generator,
    ) -> np.ndarray:
        """Draw the total log-jump contribution for each element in *shape*.

        For each element:
        - N ~ Poisson(λ·dt) jumps occur.
        - The sum of N iid N(jump_mean, jump_std²) values is
          N(N·jump_mean, N·jump_std²).

        Parameters
        ----------
        shape : tuple[int, ...]
            Output shape (e.g. ``(n_steps,)`` or ``(n_paths, n_steps)``).
        dt : float
            Time-step size.
        rng : Generator
            Random generator.

        Returns
        -------
        np.ndarray
            Array of summed jump sizes matching *shape*.
        """
        lam = self.jump_intensity
        jm = self.jump_mean
        js = self.jump_std

        n_jumps = rng.poisson(lam * dt, shape)
        safe_n = np.maximum(n_jumps, 1)
        raw_sums = rng.normal(safe_n * jm, np.sqrt(safe_n) * js)
        return np.where(n_jumps > 0, raw_sums, 0.0)

    def evolve(
        self,
        s0: float,
        t: float,
        dt: float,
        rng: Generator,
    ) -> np.ndarray:
        """Simulate a single jump-diffusion path.

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
        k = self.compensator

        compensated_drift = (mu - 0.5 * sigma ** 2 - self.jump_intensity * k) * dt
        diffusion_scale = sigma * np.sqrt(dt)

        z = rng.standard_normal(n_steps)
        jump_sums = self._simulate_jump_sums((n_steps,), dt, rng)
        log_increments = compensated_drift + diffusion_scale * z + jump_sums

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
        """Simulate multiple independent jump-diffusion paths.

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
        k = self.compensator

        compensated_drift = (mu - 0.5 * sigma ** 2 - self.jump_intensity * k) * dt
        diffusion_scale = sigma * np.sqrt(dt)

        z = rng.standard_normal((n_paths, n_steps))
        jump_sums = self._simulate_jump_sums((n_paths, n_steps), dt, rng)
        log_increments = compensated_drift + diffusion_scale * z + jump_sums

        log_paths = np.empty((n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(s0)
        np.cumsum(log_increments, axis=1, out=log_paths[:, 1:])
        log_paths[:, 1:] += log_paths[:, 0:1]

        return np.exp(log_paths)
    