"""Pre-allocated result recorder for Monte Carlo iterations.

The recorder owns fixed-size NumPy arrays and fills them in-place
during the simulation loop — no appending, no reallocation. After the
loop completes, call ``to_dataframe()`` to get a tidy pandas DataFrame
or access the raw arrays directly.

Usage inside the engine loop::

    recorder = Recorder(n_iterations, n_steps, param_names)
    for i in range(n_iterations):
        ...
        recorder.record(i, params, cum_pos, compression, market, portfolio)
    df = recorder.to_dataframe()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bioventure.models.market import MarketResult
from bioventure.models.rd_compression import CompressionResult
from bioventure.valuation.portfolio import PortfolioResult


class Recorder:
    """Fixed-size array container for Monte Carlo outputs.

    Parameters
    ----------
    n_iterations : int
        Total number of MC iterations (array row count).
    n_steps : int
        Number of time steps per path (determines path-array columns).
    param_names : list[str]
        Sampler parameter names — one ``(n_iterations,)`` array per name.
    """

    def __init__(
        self,
        n_iterations: int,
        n_steps: int,
        param_names: list[str],
    ) -> None:
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        self._n = n_iterations
        self._n_steps = n_steps
        self._idx = 0
        self._param_names = sorted(param_names)

        n = n_iterations
        n_pts = n_steps + 1

        self.params: dict[str, np.ndarray] = {
            name: np.empty(n, dtype=np.float64) for name in self._param_names
        }

        self.cumulative_pos = np.empty(n, dtype=np.float64)
        self.compressed_timeline = np.empty(n, dtype=np.float64)
        self.compressed_cost = np.empty(n, dtype=np.float64)

        self.terminal_total_market = np.empty(n, dtype=np.float64)
        self.terminal_ai_enabled = np.empty(n, dtype=np.float64)

        self.gross_multiple = np.empty(n, dtype=np.float64)
        self.net_profit_m = np.empty(n, dtype=np.float64)
        self.success_rate = np.empty(n, dtype=np.float64)

        self.total_market_paths = np.empty((n, n_pts), dtype=np.float64)
        self.ai_enabled_paths = np.empty((n, n_pts), dtype=np.float64)
        self.adoption_curves = np.empty((n, n_pts), dtype=np.float64)

    @property
    def n_iterations(self) -> int:
        return self._n

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def param_names(self) -> list[str]:
        return list(self._param_names)

    @property
    def n_recorded(self) -> int:
        return self._idx

    @property
    def is_complete(self) -> bool:
        return self._idx == self._n

    def record(
        self,
        i: int,
        params: dict[str, float],
        cumulative_pos: float,
        compression: CompressionResult,
        market: MarketResult,
        portfolio: PortfolioResult,
    ) -> None:
        """Store results for iteration *i*.

        Parameters
        ----------
        i : int
            Zero-based iteration index.
        params : dict[str, float]
            Sampler parameter draws.
        cumulative_pos : float
            Product of clinical phase probabilities.
        compression : CompressionResult
            R&D compression output.
        market : MarketResult
            Market model output.
        portfolio : PortfolioResult
            Portfolio simulation output.
        """
        if not 0 <= i < self._n:
            raise IndexError(
                f"Iteration index {i} out of range [0, {self._n})"
            )

        for name in self._param_names:
            self.params[name][i] = params[name]

        self.cumulative_pos[i] = cumulative_pos
        self.compressed_timeline[i] = compression.compressed_timeline_years
        self.compressed_cost[i] = compression.compressed_cost_billion_usd

        self.terminal_total_market[i] = market.terminal_total
        self.terminal_ai_enabled[i] = market.terminal_ai_enabled
        self.total_market_paths[i] = market.total_market
        self.ai_enabled_paths[i] = market.ai_enabled_market
        self.adoption_curves[i] = market.adoption_fraction

        self.gross_multiple[i] = portfolio.gross_multiple
        self.net_profit_m[i] = portfolio.net_profit_m
        self.success_rate[i] = portfolio.success_rate

        self._idx = max(self._idx, i + 1)

    def to_dataframe(self) -> pd.DataFrame:
        """Export scalar results to a pandas DataFrame.

        Path arrays (``total_market_paths``, ``ai_enabled_paths``,
        ``adoption_curves``) are excluded — access them directly as
        attributes when needed.

        Returns
        -------
        pd.DataFrame
            One row per iteration, columns for every parameter and
            scalar outcome.
        """
        data: dict[str, np.ndarray] = {}

        for name in self._param_names:
            data[f"param_{name}"] = self.params[name].copy()

        data["cumulative_pos"] = self.cumulative_pos.copy()
        data["compressed_timeline"] = self.compressed_timeline.copy()
        data["compressed_cost"] = self.compressed_cost.copy()
        data["terminal_total_market"] = self.terminal_total_market.copy()
        data["terminal_ai_enabled"] = self.terminal_ai_enabled.copy()
        data["gross_multiple"] = self.gross_multiple.copy()
        data["net_profit_m"] = self.net_profit_m.copy()
        data["success_rate"] = self.success_rate.copy()

        return pd.DataFrame(data)
    