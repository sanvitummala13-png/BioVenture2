"""AI-driven R&D compression model.

Maps AI capability to fractional reductions in drug-development
timeline and cost. The compression factors are stochastic (drawn
from priors during Monte Carlo) and bounded on (0, 1).

Outputs:
- Compressed timeline (years).
- Compressed cost (billion USD).
- Time saved and cost saved (absolute).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


@dataclass(frozen=True)
class CompressionResult:
    """Result of one Monte Carlo draw of R&D compression.

    Attributes
    ----------
    ai_compression_factor : float
        Fractional timeline reduction drawn this iteration (0, 1).
    cost_reduction_factor : float
        Fractional cost reduction drawn this iteration (0, 1).
    compressed_timeline_years : float
        Baseline timeline after AI compression.
    compressed_cost_billion_usd : float
        Baseline cost after AI compression.
    time_saved_years : float
        Absolute years saved.
    cost_saved_billion_usd : float
        Absolute cost saved.
    """

    ai_compression_factor: float
    cost_reduction_factor: float
    compressed_timeline_years: float
    compressed_cost_billion_usd: float
    time_saved_years: float
    cost_saved_billion_usd: float


class RDCompressionModel:
    """Stochastic model for AI-driven R&D timeline and cost compression.

    Parameters
    ----------
    baseline_timeline_years : float
        Pre-AI average drug-development timeline (must be > 0).
    baseline_cost_billion_usd : float
        Pre-AI average drug-development cost (must be > 0).
    """

    def __init__(
        self,
        baseline_timeline_years: float,
        baseline_cost_billion_usd: float,
    ) -> None:
        if baseline_timeline_years <= 0:
            raise ValueError(
                f"baseline_timeline_years must be > 0, got {baseline_timeline_years}"
            )
        if baseline_cost_billion_usd <= 0:
            raise ValueError(
                f"baseline_cost_billion_usd must be > 0, got {baseline_cost_billion_usd}"
            )
        self._baseline_timeline = baseline_timeline_years
        self._baseline_cost = baseline_cost_billion_usd

    @property
    def baseline_timeline_years(self) -> float:
        return self._baseline_timeline

    @property
    def baseline_cost_billion_usd(self) -> float:
        return self._baseline_cost

    def compress(
        self,
        ai_compression_factor: float,
        cost_reduction_factor: float,
    ) -> CompressionResult:
        """Apply compression factors to the baseline.

        Parameters
        ----------
        ai_compression_factor : float
            Fractional timeline reduction in (0, 1).
        cost_reduction_factor : float
            Fractional cost reduction in (0, 1).

        Returns
        -------
        CompressionResult
        """
        self._validate_factor("ai_compression_factor", ai_compression_factor)
        self._validate_factor("cost_reduction_factor", cost_reduction_factor)

        time_saved = self._baseline_timeline * ai_compression_factor
        cost_saved = self._baseline_cost * cost_reduction_factor

        return CompressionResult(
            ai_compression_factor=ai_compression_factor,
            cost_reduction_factor=cost_reduction_factor,
            compressed_timeline_years=self._baseline_timeline - time_saved,
            compressed_cost_billion_usd=self._baseline_cost - cost_saved,
            time_saved_years=time_saved,
            cost_saved_billion_usd=cost_saved,
        )

    def draw(
        self,
        ai_compression_factor: float,
        cost_reduction_factor: float,
    ) -> CompressionResult:
        """Alias for ``compress`` — used when factors come from prior draws."""
        return self.compress(ai_compression_factor, cost_reduction_factor)

    def draw_batch(
        self,
        ai_compression_factors: np.ndarray,
        cost_reduction_factors: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Vectorised batch compression.

        Parameters
        ----------
        ai_compression_factors : np.ndarray
            Shape ``(n,)`` array of timeline compression draws in (0, 1).
        cost_reduction_factors : np.ndarray
            Shape ``(n,)`` array of cost compression draws in (0, 1).

        Returns
        -------
        dict[str, np.ndarray]
            Each value has shape ``(n,)``.
        """
        ai = np.asarray(ai_compression_factors, dtype=np.float64)
        cost = np.asarray(cost_reduction_factors, dtype=np.float64)

        if ai.ndim != 1 or cost.ndim != 1:
            raise ValueError("Factor arrays must be 1-D")
        if len(ai) != len(cost):
            raise ValueError(
                f"Factor arrays must have same length, "
                f"got {len(ai)} and {len(cost)}"
            )
        if np.any(ai <= 0) or np.any(ai >= 1):
            raise ValueError("ai_compression_factors must be in (0, 1)")
        if np.any(cost <= 0) or np.any(cost >= 1):
            raise ValueError("cost_reduction_factors must be in (0, 1)")

        time_saved = self._baseline_timeline * ai
        cost_saved = self._baseline_cost * cost

        return {
            "ai_compression_factor": ai,
            "cost_reduction_factor": cost,
            "compressed_timeline_years": self._baseline_timeline - time_saved,
            "compressed_cost_billion_usd": self._baseline_cost - cost_saved,
            "time_saved_years": time_saved,
            "cost_saved_billion_usd": cost_saved,
        }

    @classmethod
    def from_config(
        cls,
        baseline_timeline_years: float,
        baseline_cost_billion_usd: float,
    ) -> RDCompressionModel:
        """Build from config values.

        Parameters
        ----------
        baseline_timeline_years : float
            From ``base.yaml`` → ``rd_compression.baseline_timeline_years``.
        baseline_cost_billion_usd : float
            From ``base.yaml`` → ``rd_compression.baseline_cost_billion_usd``.
        """
        return cls(baseline_timeline_years, baseline_cost_billion_usd)

    @staticmethod
    def _validate_factor(name: str, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}")
        