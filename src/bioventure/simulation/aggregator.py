"""Aggregator: reduce a completed Recorder into summary statistics.

Takes the raw per-iteration arrays from a Recorder and produces:
- Percentile distributions (p5/p25/p50/p75/p95 + mean + std) for every
  scalar outcome and every sampled parameter.
- Full financial metrics via ``compute_summary`` (MOIC, IRR, VaR, CVaR).
- Year-by-year market path statistics (median + percentile bands).
- Simple sensitivity scores: Spearman correlation of each input parameter
  with each outcome (proxy for first-order sensitivity).

All output is plain Python dicts of floats / numpy arrays so callers can
serialise however they like (JSON, Parquet, print to console, feed into
visualisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from bioventure.simulation.recorder import Recorder
from bioventure.valuation.metrics import (
    compute_summary,
    percentile_summary,
    simple_irr,
)


@dataclass
class AggregationResult:
    """Container for all aggregated simulation outputs.

    Attributes
    ----------
    n_iterations : int
        Number of MC iterations summarised.
    financial : dict[str, float]
        MOIC/IRR/VaR/CVaR/PoL metrics from ``compute_summary``.
    outcomes : dict[str, dict[str, float]]
        Percentile summaries for each scalar outcome column.
        Keys: ``"cumulative_pos"``, ``"compressed_timeline"``,
        ``"compressed_cost"``, ``"terminal_total_market"``,
        ``"terminal_ai_enabled"``, ``"gross_multiple"``,
        ``"net_profit_m"``, ``"success_rate"``.
    parameters : dict[str, dict[str, float]]
        Percentile summary for each sampled parameter.
    market_paths : dict[str, np.ndarray]
        Year-by-year statistics across all paths.
        ``"total_market_*"`` and ``"ai_enabled_*"`` with
        suffixes ``_p5``, ``_p25``, ``_p50``, ``_p75``, ``_p95``, ``_mean``.
    sensitivity : dict[str, dict[str, float]]
        Spearman ρ of each parameter against each outcome.
        ``sensitivity[outcome_name][param_name] = rho``.
    irr_summary : dict[str, float]
        Percentile summary of annualised IRR across iterations.
    """

    n_iterations: int
    financial: dict[str, float] = field(default_factory=dict)
    outcomes: dict[str, dict[str, float]] = field(default_factory=dict)
    parameters: dict[str, dict[str, float]] = field(default_factory=dict)
    market_paths: dict[str, np.ndarray] = field(default_factory=dict)
    sensitivity: dict[str, dict[str, float]] = field(default_factory=dict)
    irr_summary: dict[str, float] = field(default_factory=dict)


_OUTCOME_FIELDS = (
    "cumulative_pos",
    "compressed_timeline",
    "compressed_cost",
    "terminal_total_market",
    "terminal_ai_enabled",
    "gross_multiple",
    "net_profit_m",
    "success_rate",
)

_PERCENTILES = (5, 25, 50, 75, 95)


class Aggregator:
    """Reduce raw Recorder arrays into summary statistics.

    Parameters
    ----------
    horizon_years : float
        Investment horizon used for IRR computation.
    total_invested_m : float
        Total capital deployed in million USD (used for IRR).
    percentiles : tuple[float, ...]
        Percentile levels for distribution summaries (0–100 scale).
    sensitivity_outcomes : tuple[str, ...]
        Outcome columns to correlate against parameters in sensitivity
        analysis.  Each must be a valid attribute of ``Recorder``.
    """

    def __init__(
        self,
        horizon_years: float,
        total_invested_m: float,
        percentiles: tuple[float, ...] = _PERCENTILES,
        sensitivity_outcomes: tuple[str, ...] = (
            "gross_multiple",
            "terminal_ai_enabled",
            "cumulative_pos",
        ),
    ) -> None:
        if horizon_years <= 0:
            raise ValueError(f"horizon_years must be > 0, got {horizon_years}")
        if total_invested_m <= 0:
            raise ValueError(
                f"total_invested_m must be > 0, got {total_invested_m}"
            )
        self._horizon = horizon_years
        self._total_invested = total_invested_m
        self._percentiles = tuple(percentiles)
        self._sens_outcomes = tuple(sensitivity_outcomes)

    @property
    def horizon_years(self) -> float:
        return self._horizon

    @property
    def total_invested_m(self) -> float:
        return self._total_invested

    def summarise(self, recorder: Recorder) -> AggregationResult:
        """Aggregate a completed Recorder into an AggregationResult.

        Parameters
        ----------
        recorder : Recorder
            Must have ``is_complete == True``; warns otherwise.

        Returns
        -------
        AggregationResult
        """
        if not recorder.is_complete:
            import warnings
            warnings.warn(
                f"Recorder has {recorder.n_recorded} of "
                f"{recorder.n_iterations} iterations recorded; "
                "aggregating partial results.",
                stacklevel=2,
            )

        n = recorder.n_iterations
        result = AggregationResult(n_iterations=n)

        result.financial = compute_summary(
            gross_multiples=recorder.gross_multiple,
            total_invested=self._total_invested,
            horizon_years=self._horizon,
        )

        for col in _OUTCOME_FIELDS:
            arr = getattr(recorder, col)
            result.outcomes[col] = percentile_summary(arr, self._percentiles)

        for name in recorder.param_names:
            result.parameters[name] = percentile_summary(
                recorder.params[name], self._percentiles
            )

        result.market_paths = self._path_statistics(recorder)

        irr_values = simple_irr(
            total_returned=recorder.net_profit_m + self._total_invested,
            total_invested=self._total_invested,
            horizon_years=self._horizon,
        )
        result.irr_summary = percentile_summary(irr_values, self._percentiles)

        result.sensitivity = self._sensitivity(recorder)

        return result

    def _path_statistics(self, recorder: Recorder) -> dict[str, np.ndarray]:
        """Compute percentile bands across all market paths per time step."""
        out: dict[str, np.ndarray] = {}

        for prefix, paths in (
            ("total_market", recorder.total_market_paths),
            ("ai_enabled", recorder.ai_enabled_paths),
            ("adoption", recorder.adoption_curves),
        ):
            for p in self._percentiles:
                key = f"{prefix}_p{int(p)}"
                out[key] = np.percentile(paths, p, axis=0)
            out[f"{prefix}_mean"] = paths.mean(axis=0)
            out[f"{prefix}_std"] = paths.std(axis=0)

        return out

    def _sensitivity(
        self, recorder: Recorder
    ) -> dict[str, dict[str, float]]:
        """Spearman correlation of each parameter with each outcome.

        Returns
        -------
        dict[str, dict[str, float]]
            ``result[outcome][param] = spearman_rho``
        """
        sensitivity: dict[str, dict[str, float]] = {}

        for outcome_name in self._sens_outcomes:
            outcome_arr = getattr(recorder, outcome_name, None)
            if outcome_arr is None:
                continue
            row: dict[str, float] = {}
            for param_name in recorder.param_names:
                param_arr = recorder.params[param_name]
                rho, _ = stats.spearmanr(param_arr, outcome_arr)
                row[param_name] = float(rho) if not np.isnan(rho) else 0.0
            sensitivity[outcome_name] = row

        return sensitivity

    def to_flat_dict(self, result: AggregationResult) -> dict[str, Any]:
        """Flatten AggregationResult into a single dict of scalar values.

        Path arrays are excluded (they are not scalar). Useful for
        logging, JSON serialisation, or feeding into a DataFrame.

        Parameters
        ----------
        result : AggregationResult

        Returns
        -------
        dict[str, Any]
            Keys are ``"<section>.<field>"`` strings.
        """
        flat: dict[str, Any] = {"n_iterations": result.n_iterations}

        for key, val in result.financial.items():
            flat[f"financial.{key}"] = val

        for outcome, stats_dict in result.outcomes.items():
            for stat, val in stats_dict.items():
                flat[f"outcome.{outcome}.{stat}"] = val

        for param, stats_dict in result.parameters.items():
            clean = param.replace(".", "_")
            for stat, val in stats_dict.items():
                flat[f"param.{clean}.{stat}"] = val

        for stat, val in result.irr_summary.items():
            flat[f"irr.{stat}"] = val

        for outcome, param_rhos in result.sensitivity.items():
            for param, rho in param_rhos.items():
                clean = param.replace(".", "_")
                flat[f"sensitivity.{outcome}.{clean}"] = rho

        return flat
    