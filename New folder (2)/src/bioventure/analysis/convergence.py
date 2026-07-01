"""Monte Carlo convergence diagnostics.

Three tools:
- ``check_convergence`` — running-statistic stability test on a single metric.
- ``convergence_batch`` — apply the above to many metrics at once.
- ``effective_sample_size`` — lag-1 autocorrelation ESS estimate.
- ``gelman_rubin`` — R-hat diagnostic for multi-chain MCMC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConvergenceResult:
    """Running-statistic convergence report for one metric.

    Attributes
    ----------
    metric_name : str
    windows : np.ndarray
        Iteration counts at which statistics were evaluated.
    running_means : np.ndarray
        Cumulative mean at each window.
    running_stds : np.ndarray
        Cumulative std at each window.
    running_p5 : np.ndarray
    running_p50 : np.ndarray
    running_p95 : np.ndarray
    converged : bool
        ``True`` if the relative change in mean dropped below *tolerance*
        at any consecutive window pair.
    convergence_iteration : int or None
        First window index at which the criterion was satisfied; ``None``
        if convergence was never reached.
    relative_change : float
        Relative change in mean between the final two windows.
    """

    metric_name: str
    windows: np.ndarray
    running_means: np.ndarray
    running_stds: np.ndarray
    running_p5: np.ndarray
    running_p50: np.ndarray
    running_p95: np.ndarray
    converged: bool
    convergence_iteration: int | None
    relative_change: float


def check_convergence(
    values: np.ndarray,
    windows: np.ndarray | None = None,
    tolerance: float = 0.01,
    metric_name: str = "metric",
) -> ConvergenceResult:
    """Check convergence of a Monte Carlo estimator via running statistics.

    Evaluates the cumulative mean, std, p5, p50, and p95 at successive
    window sizes. Declares convergence when the relative change in mean
    between consecutive windows falls below *tolerance*.

    Parameters
    ----------
    values : np.ndarray
        Shape ``(n,)`` — simulation outcomes in draw order.
    windows : np.ndarray, optional
        Integer iteration counts at which to evaluate.  Default: up to 20
        evenly-spaced checkpoints from ``max(10, n // 20)`` to ``n``.
    tolerance : float
        Relative-change threshold (default 0.01 = 1 %).
    metric_name : str
        Label stored in the result.

    Returns
    -------
    ConvergenceResult
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"values must be 1-D, got {v.ndim}-D")
    n = len(v)
    if n < 2:
        raise ValueError(f"Need at least 2 values, got {n}")
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")

    if windows is None:
        start = max(10, n // 20)
        n_pts = min(20, n - start + 1)
        windows = np.unique(
            np.linspace(start, n, num=n_pts).astype(int)
        )
    else:
        windows = np.asarray(windows, dtype=int)
        if np.any(windows < 1) or np.any(windows > n):
            raise ValueError(
                f"All windows must be in [1, {n}], "
                f"got range [{windows.min()}, {windows.max()}]"
            )

    nw = len(windows)
    running_means = np.empty(nw)
    running_stds = np.empty(nw)
    running_p5 = np.empty(nw)
    running_p50 = np.empty(nw)
    running_p95 = np.empty(nw)

    for k, w in enumerate(windows):
        sub = v[:w]
        running_means[k] = sub.mean()
        running_stds[k] = sub.std()
        running_p5[k] = np.percentile(sub, 5)
        running_p50[k] = np.percentile(sub, 50)
        running_p95[k] = np.percentile(sub, 95)

    convergence_iteration: int | None = None
    converged = False
    for k in range(1, nw):
        prev = running_means[k - 1]
        curr = running_means[k]
        denom = abs(prev) if abs(prev) > 1e-12 else 1.0
        if abs(curr - prev) / denom < tolerance:
            if not converged:
                convergence_iteration = int(windows[k])
                converged = True

    if nw >= 2:
        prev = running_means[-2]
        curr = running_means[-1]
        denom = abs(prev) if abs(prev) > 1e-12 else 1.0
        final_rel_change = abs(curr - prev) / denom
    else:
        final_rel_change = float("nan")

    return ConvergenceResult(
        metric_name=metric_name,
        windows=windows,
        running_means=running_means,
        running_stds=running_stds,
        running_p5=running_p5,
        running_p50=running_p50,
        running_p95=running_p95,
        converged=converged,
        convergence_iteration=convergence_iteration,
        relative_change=final_rel_change,
    )


def convergence_batch(
    values_dict: dict[str, np.ndarray],
    windows: np.ndarray | None = None,
    tolerance: float = 0.01,
) -> dict[str, ConvergenceResult]:
    """Run ``check_convergence`` on multiple metrics simultaneously.

    Parameters
    ----------
    values_dict : dict[str, np.ndarray]
        Mapping from metric name to shape-``(n,)`` simulation outcomes.
    windows : np.ndarray, optional
        Shared window schedule applied to every metric.  ``None`` lets each
        metric auto-select its own schedule.
    tolerance : float
        Relative-change threshold forwarded to ``check_convergence``.

    Returns
    -------
    dict[str, ConvergenceResult]
        One result per metric, keyed by metric name.
    """
    return {
        name: check_convergence(
            v,
            windows=windows,
            tolerance=tolerance,
            metric_name=name,
        )
        for name, v in values_dict.items()
    }


def effective_sample_size(values: np.ndarray) -> float:
    """Estimate ESS from lag-1 autocorrelation.

    Uses the approximation ESS ≈ n / (1 + 2ρ₁) where ρ₁ is the lag-1
    autocorrelation coefficient.  Returns *n* for near-independent draws.

    Parameters
    ----------
    values : np.ndarray
        Shape ``(n,)`` — simulation outcomes.

    Returns
    -------
    float
        Effective sample size in the interval ``(0, n]``.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"values must be 1-D, got {v.ndim}-D")
    n = len(v)
    if n < 2:
        raise ValueError(f"Need at least 2 values, got {n}")

    centered = v - v.mean()
    variance = float(np.dot(centered, centered)) / n

    if variance < 1e-14:
        return float(n)

    lag1 = float(np.dot(centered[:-1], centered[1:])) / (n * variance)
    lag1 = np.clip(lag1, -1.0 + 1e-9, 1.0 - 1e-9)

    denom = 1.0 + 2.0 * lag1
    if denom <= 0.0:
        return float(n)

    return min(float(n), n / denom)


def gelman_rubin(chains: np.ndarray) -> float:
    """Gelman-Rubin R-hat diagnostic for MCMC convergence.

    Values near 1.0 indicate convergence; the common threshold is R-hat < 1.01.

    Parameters
    ----------
    chains : np.ndarray
        Shape ``(n_chains, n_samples)`` — independent parallel chains.

    Returns
    -------
    float
        R-hat statistic.
    """
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        raise ValueError(
            f"chains must be 2-D (n_chains, n_samples), got {chains.ndim}-D"
        )
    n_chains, n = chains.shape
    if n_chains < 2:
        raise ValueError(f"Need at least 2 chains, got {n_chains}")
    if n < 2:
        raise ValueError(f"Need at least 2 samples per chain, got {n}")

    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean()
    B = n * float(np.var(chain_means, ddof=1))
    W = float(np.var(chains, axis=1, ddof=1).mean())

    if W < 1e-14:
        return 1.0

    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))
