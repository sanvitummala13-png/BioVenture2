"""Risk-adjusted discounting for present-value calculations.

Provides discount-factor computation, NPV of cashflow streams, and
risk-adjusted NPV (rNPV) where cashflows are weighted by their
probability of realisation.

Discount rate = risk-free rate + risk premium (build-up method).
"""

from __future__ import annotations

import numpy as np


class DiscountModel:
    """Constant-rate discounting model.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate (e.g. 0.045 for 4.5%).
    risk_premium : float
        Annual risk premium above risk-free (e.g. 0.08 for 8%).
    """

    def __init__(self, risk_free_rate: float, risk_premium: float) -> None:
        if risk_free_rate < 0:
            raise ValueError(
                f"risk_free_rate must be >= 0, got {risk_free_rate}"
            )
        if risk_premium < 0:
            raise ValueError(
                f"risk_premium must be >= 0, got {risk_premium}"
            )
        self._rf = risk_free_rate
        self._rp = risk_premium

    @property
    def risk_free_rate(self) -> float:
        return self._rf

    @property
    def risk_premium(self) -> float:
        return self._rp

    @property
    def discount_rate(self) -> float:
        return self._rf + self._rp

    def discount_factor(self, t: float) -> float:
        """Discount factor for a single time point.

        Parameters
        ----------
        t : float
            Years from present (must be >= 0).

        Returns
        -------
        float
            ``1 / (1 + r)^t``
        """
        if t < 0:
            raise ValueError(f"t must be >= 0, got {t}")
        return (1.0 + self.discount_rate) ** (-t)

    def discount_factors(self, times: np.ndarray) -> np.ndarray:
        """Vectorised discount factors for an array of time points.

        Parameters
        ----------
        times : np.ndarray
            Years from present (all >= 0).

        Returns
        -------
        np.ndarray
            Same shape as *times*.
        """
        times = np.asarray(times, dtype=np.float64)
        if np.any(times < 0):
            raise ValueError("All times must be >= 0")
        return (1.0 + self.discount_rate) ** (-times)

    def present_value(self, future_value: float, t: float) -> float:
        """Discount a single future value to present.

        Parameters
        ----------
        future_value : float
            Cash amount at time *t*.
        t : float
            Years from present.

        Returns
        -------
        float
            Present value.
        """
        return future_value * self.discount_factor(t)

    def npv(
        self,
        cashflows: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """Net present value of a cashflow stream.

        Parameters
        ----------
        cashflows : np.ndarray
            Shape ``(k,)`` — cash amounts (positive = inflow).
        times : np.ndarray
            Shape ``(k,)`` — year of each cashflow.

        Returns
        -------
        float
            Sum of discounted cashflows.
        """
        cf = np.asarray(cashflows, dtype=np.float64)
        t = np.asarray(times, dtype=np.float64)
        self._check_alignment_1d(cf, t)
        return float(np.sum(cf * self.discount_factors(t)))

    def rnpv(
        self,
        cashflows: np.ndarray,
        times: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        """Risk-adjusted NPV: cashflows weighted by success probability.

        rNPV = Σ (cashflow_i × probability_i × discount_factor_i)

        Parameters
        ----------
        cashflows : np.ndarray
            Shape ``(k,)`` — cash amounts.
        times : np.ndarray
            Shape ``(k,)`` — year of each cashflow.
        probabilities : np.ndarray
            Shape ``(k,)`` — probability each cashflow is realised,
            all in [0, 1].

        Returns
        -------
        float
        """
        cf = np.asarray(cashflows, dtype=np.float64)
        t = np.asarray(times, dtype=np.float64)
        p = np.asarray(probabilities, dtype=np.float64)
        self._check_alignment_1d(cf, t)
        if len(p) != len(cf):
            raise ValueError(
                f"probabilities length ({len(p)}) must match "
                f"cashflows length ({len(cf)})"
            )
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("probabilities must be in [0, 1]")
        return float(np.sum(cf * p * self.discount_factors(t)))

    def npv_batch(
        self,
        cashflow_matrix: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """NPV for a batch of cashflow streams.

        Parameters
        ----------
        cashflow_matrix : np.ndarray
            Shape ``(n, k)`` — each row is one simulation's cashflows.
        times : np.ndarray
            Shape ``(k,)`` — shared time grid.

        Returns
        -------
        np.ndarray
            Shape ``(n,)`` — NPV per simulation.
        """
        cf = np.asarray(cashflow_matrix, dtype=np.float64)
        t = np.asarray(times, dtype=np.float64)
        if cf.ndim != 2:
            raise ValueError(f"cashflow_matrix must be 2-D, got {cf.ndim}-D")
        if t.ndim != 1:
            raise ValueError(f"times must be 1-D, got {t.ndim}-D")
        if cf.shape[1] != len(t):
            raise ValueError(
                f"cashflow_matrix has {cf.shape[1]} columns, "
                f"expected {len(t)} (length of times)"
            )
        df = self.discount_factors(t)
        return cf @ df

    @classmethod
    def from_config(
        cls,
        risk_free_rate: float,
        risk_premium: float,
    ) -> DiscountModel:
        """Build from config values.

        Parameters
        ----------
        risk_free_rate : float
            From ``base.yaml`` → ``valuation.discount.risk_free_rate``.
        risk_premium : float
            From ``base.yaml`` → ``valuation.discount.risk_premium``.
        """
        return cls(risk_free_rate, risk_premium)

    @staticmethod
    def _check_alignment_1d(a: np.ndarray, b: np.ndarray) -> None:
        if a.ndim != 1:
            raise ValueError(f"Expected 1-D array, got {a.ndim}-D")
        if b.ndim != 1:
            raise ValueError(f"Expected 1-D array, got {b.ndim}-D")
        if len(a) != len(b):
            raise ValueError(
                f"Array lengths must match, got {len(a)} and {len(b)}"
            )
        