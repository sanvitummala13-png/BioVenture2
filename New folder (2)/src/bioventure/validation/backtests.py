"""Backtest simulated prediction intervals against historical observations.

Reliability check: a well-calibrated simulation should show empirical
coverage ≈ stated confidence level.  If the 90 % PI contains 90 % of
historical observations, the model is well-calibrated at that level.

Usage::

    result = backtest_coverage(
        observed=hist_market_values,       # shape (n_obs,)
        simulated_paths=recorder.total_market_paths,  # shape (n_sims, n_time)
        time_indices=np.array([2, 5, 9]), # which time step each obs belongs to
    )
    print(result.coverage)   # {"p50": 0.52, "p80": 0.79, "p90": 0.91, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Coverage statistics from comparing simulated PI to observed data.

    Attributes
    ----------
    coverage : dict[str, float]
        Empirical coverage per confidence level, e.g. ``{"p90": 0.88}``.
        A well-calibrated model shows coverage ≈ stated level.
    interval_widths : dict[str, np.ndarray]
        Mean PI width per confidence level (same unit as the observed values).
    lower_bounds : dict[str, np.ndarray]
        Shape ``(n_obs,)`` — lower PI bound per observation.
    upper_bounds : dict[str, np.ndarray]
        Shape ``(n_obs,)`` — upper PI bound per observation.
    n_observations : int
        Number of historical observations used.
    observed : np.ndarray
        The observed values passed in (shape ``(n_obs,)``).
    sharpness : float
        Mean width of the 90 % PI — narrower = sharper = more informative.
        Only meaningful when coverage is adequate.
    """

    coverage: dict[str, float] = field(default_factory=dict)
    interval_widths: dict[str, np.ndarray] = field(default_factory=dict)
    lower_bounds: dict[str, np.ndarray] = field(default_factory=dict)
    upper_bounds: dict[str, np.ndarray] = field(default_factory=dict)
    n_observations: int = 0
    observed: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpness: float = float("nan")


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def backtest_coverage(
    observed: np.ndarray,
    simulated_paths: np.ndarray,
    time_indices: np.ndarray | None = None,
    confidence_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> BacktestResult:
    """Compare observed values against simulated prediction intervals.

    Parameters
    ----------
    observed : np.ndarray
        Shape ``(n_obs,)`` — historical point observations.
    simulated_paths : np.ndarray
        Shape ``(n_sims, n_time)`` — distribution of simulated values.
        Each column is the cross-simulation distribution at one time step.
    time_indices : np.ndarray or None
        Shape ``(n_obs,)`` integer indices mapping each observation to a
        column of *simulated_paths*.  If ``None``, assumes ``n_obs == n_time``
        and maps ``observed[t]`` to ``simulated_paths[:, t]``.
    confidence_levels : tuple[float, ...]
        Nominal coverage levels in (0, 1), e.g. ``(0.50, 0.90)``.

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError
        If shapes are inconsistent or *confidence_levels* are out of range.
    """
    observed = np.asarray(observed, dtype=np.float64)
    simulated_paths = np.asarray(simulated_paths, dtype=np.float64)

    if observed.ndim != 1:
        raise ValueError(
            f"observed must be 1-D, got {observed.ndim}-D"
        )
    if simulated_paths.ndim != 2:
        raise ValueError(
            f"simulated_paths must be 2-D, got {simulated_paths.ndim}-D"
        )

    n_sims, n_time = simulated_paths.shape
    n_obs = len(observed)

    if time_indices is None:
        if n_obs != n_time:
            raise ValueError(
                f"observed length ({n_obs}) must equal simulated_paths "
                f"n_time ({n_time}) when time_indices is None"
            )
        time_indices = np.arange(n_obs)
    else:
        time_indices = np.asarray(time_indices, dtype=int)
        if time_indices.shape != (n_obs,):
            raise ValueError(
                f"time_indices shape {time_indices.shape} must match "
                f"observed shape ({n_obs},)"
            )
        if np.any(time_indices < 0) or np.any(time_indices >= n_time):
            raise ValueError(
                f"time_indices must be in [0, {n_time - 1}]"
            )

    for cl in confidence_levels:
        if not 0.0 < cl < 1.0:
            raise ValueError(
                f"confidence level must be in (0, 1), got {cl}"
            )

    result = BacktestResult(
        n_observations=n_obs,
        observed=observed.copy(),
    )

    for cl in confidence_levels:
        alpha = 1.0 - cl
        lo_pct = (alpha / 2.0) * 100.0
        hi_pct = (1.0 - alpha / 2.0) * 100.0

        lower = np.empty(n_obs)
        upper = np.empty(n_obs)

        for i, t in enumerate(time_indices):
            col = simulated_paths[:, t]
            lower[i] = np.percentile(col, lo_pct)
            upper[i] = np.percentile(col, hi_pct)

        inside = (observed >= lower) & (observed <= upper)
        label = f"p{int(cl * 100)}"

        result.coverage[label] = float(inside.mean())
        result.interval_widths[label] = upper - lower
        result.lower_bounds[label] = lower
        result.upper_bounds[label] = upper

    if "p90" in result.interval_widths:
        result.sharpness = float(result.interval_widths["p90"].mean())

    return result


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def coverage_error(result: BacktestResult) -> dict[str, float]:
    """Signed calibration error per level: empirical - nominal.

    A positive error means the model is over-confident (PI is too narrow).
    A negative error means the model is under-confident (PI is too wide).

    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    dict[str, float]
        Keys match ``result.coverage`` (e.g. ``"p90"``).
        Values are ``empirical_coverage - nominal_coverage``.
    """
    errors: dict[str, float] = {}
    for label, empirical in result.coverage.items():
        nominal = int(label[1:]) / 100.0
        errors[label] = empirical - nominal
    return errors


def reliability_diagram_data(
    result: BacktestResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (nominal, empirical) coverage pairs for a reliability diagram.

    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(nominal_levels, empirical_coverages)`` — both shape ``(k,)``
        sorted ascending by nominal level.
    """
    pairs = sorted(
        (int(label[1:]) / 100.0, cov)
        for label, cov in result.coverage.items()
    )
    nominal = np.array([p[0] for p in pairs])
    empirical = np.array([p[1] for p in pairs])
    return nominal, empirical
