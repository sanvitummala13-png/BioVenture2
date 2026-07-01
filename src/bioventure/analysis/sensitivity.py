"""Spearman rank-correlation sensitivity analysis.

Quantifies how strongly each input parameter drives each output metric.
Spearman's ρ is model-free and monotone-relationship-safe — appropriate
for the highly skewed, non-Gaussian distributions produced by Monte Carlo
drug-discovery simulations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SensitivityResult:
    """Output of a Spearman sensitivity analysis run.

    Attributes
    ----------
    param_names : list[str]
        Ordered input parameter labels.
    metric_names : list[str]
        Ordered output metric labels.
    rho : np.ndarray
        Shape ``(n_params, n_metrics)`` — Spearman ρ for every
        (parameter, metric) pair.
    pvalues : np.ndarray
        Shape ``(n_params, n_metrics)`` — two-sided p-values for
        the hypothesis H₀: ρ = 0.
    """

    param_names: list[str]
    metric_names: list[str]
    rho: np.ndarray
    pvalues: np.ndarray


def spearman_sensitivity(
    param_draws: dict[str, np.ndarray] | np.ndarray,
    outcome_draws: dict[str, np.ndarray] | np.ndarray,
    param_names: list[str] | None = None,
    metric_names: list[str] | None = None,
) -> SensitivityResult:
    """Compute Spearman ρ between every (parameter, outcome) pair.

    Parameters
    ----------
    param_draws : dict[str, ndarray] or ndarray
        If dict: keys are parameter names, values shape ``(n_sims,)``.
        If ndarray: shape ``(n_sims, n_params)``; *param_names* required.
    outcome_draws : dict[str, ndarray] or ndarray
        If dict: keys are metric names, values shape ``(n_sims,)``.
        If ndarray: shape ``(n_sims, n_metrics)``; *metric_names* required.
    param_names : list[str], optional
        Required when *param_draws* is ndarray.
    metric_names : list[str], optional
        Required when *outcome_draws* is ndarray.

    Returns
    -------
    SensitivityResult
    """
    if isinstance(param_draws, dict):
        param_names = list(param_draws.keys())
        param_matrix = np.column_stack(
            [np.asarray(param_draws[k], dtype=np.float64) for k in param_names]
        )
    else:
        param_matrix = np.asarray(param_draws, dtype=np.float64)
        if param_names is None:
            raise ValueError(
                "param_names is required when param_draws is an ndarray"
            )

    if isinstance(outcome_draws, dict):
        metric_names = list(outcome_draws.keys())
        outcome_matrix = np.column_stack(
            [np.asarray(outcome_draws[k], dtype=np.float64) for k in metric_names]
        )
    else:
        outcome_matrix = np.asarray(outcome_draws, dtype=np.float64)
        if metric_names is None:
            raise ValueError(
                "metric_names is required when outcome_draws is an ndarray"
            )

    if param_matrix.ndim == 1:
        param_matrix = param_matrix.reshape(-1, 1)
    if outcome_matrix.ndim == 1:
        outcome_matrix = outcome_matrix.reshape(-1, 1)

    n_sims_p = param_matrix.shape[0]
    n_sims_o = outcome_matrix.shape[0]
    if n_sims_p != n_sims_o:
        raise ValueError(
            f"param_draws and outcome_draws must have the same number of "
            f"simulations; got {n_sims_p} and {n_sims_o}"
        )
    if n_sims_p < 3:
        raise ValueError(
            f"Need at least 3 simulations for Spearman correlation, got {n_sims_p}"
        )

    n_params = param_matrix.shape[1]
    n_metrics = outcome_matrix.shape[1]

    rho = np.empty((n_params, n_metrics))
    pvalues = np.empty((n_params, n_metrics))

    for i in range(n_params):
        for j in range(n_metrics):
            r, p = stats.spearmanr(param_matrix[:, i], outcome_matrix[:, j])
            rho[i, j] = float(r)
            pvalues[i, j] = float(p)

    return SensitivityResult(
        param_names=list(param_names),
        metric_names=list(metric_names),
        rho=rho,
        pvalues=pvalues,
    )


def top_k_params(
    result: SensitivityResult,
    metric: str,
    k: int | None = None,
    absolute: bool = True,
) -> list[tuple[str, float]]:
    """Return parameters ranked by influence on *metric*.

    Parameters
    ----------
    result : SensitivityResult
    metric : str
        Name of the output metric to rank against.
    k : int, optional
        How many top parameters to return. ``None`` returns all.
    absolute : bool
        If ``True`` (default), rank by |ρ|; otherwise by signed ρ descending.

    Returns
    -------
    list[tuple[str, float]]
        ``(param_name, rho)`` pairs sorted by descending importance.
    """
    if metric not in result.metric_names:
        raise ValueError(
            f"Metric '{metric}' not found. "
            f"Available metrics: {result.metric_names}"
        )
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    j = result.metric_names.index(metric)
    rho_col = result.rho[:, j]

    key_fn = (lambda r: abs(r)) if absolute else (lambda r: r)
    pairs = sorted(
        zip(result.param_names, rho_col),
        key=lambda x: key_fn(x[1]),
        reverse=True,
    )

    if k is not None:
        pairs = pairs[:k]

    return [(name, float(rho)) for name, rho in pairs]


def tornado_data(
    result: SensitivityResult,
    metric: str,
) -> tuple[list[str], np.ndarray]:
    """Extract (names, rho_values) sorted for a tornado chart.

    Parameters and their signed ρ values, ordered so the most
    influential (highest |ρ|) appears first.

    Parameters
    ----------
    result : SensitivityResult
    metric : str
        Name of the output metric.

    Returns
    -------
    tuple[list[str], np.ndarray]
        ``(param_names_sorted, rho_values_sorted)`` in descending |ρ| order.
    """
    pairs = top_k_params(result, metric, absolute=True)
    names = [p for p, _ in pairs]
    rhos = np.array([r for _, r in pairs])
    return names, rhos
