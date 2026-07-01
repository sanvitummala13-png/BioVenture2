"""K-fold cross-validation for Bayesian calibration.

Evaluates out-of-sample predictive performance by repeatedly holding out a
fold of the calibration data, fitting MAP on the training portion, then
scoring the held-out fold.  A model with good out-of-sample log-likelihood
is not over-fitting its calibration targets.

Usage::

    result = cross_validate(
        prior_sampler=prior_sampler,
        target=clinical_target,
        n_folds=5,
    )
    print(f"Mean OOS log-likelihood: {result.mean_test_ll:.3f}")
    print(f"Std:                     {result.std_test_ll:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bioventure.calibration.bayesian import BayesianCalibrator, CalibrationTarget
from bioventure.calibration.likelihoods import log_likelihood as _dispatch_ll
from bioventure.distributions.priors import PriorSampler


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Calibration and scoring outcome for one CV fold.

    Attributes
    ----------
    fold : int
        Zero-based fold index.
    train_log_likelihood : float
        Log-likelihood on training observations at MAP params.
    test_log_likelihood : float
        Out-of-sample log-likelihood on held-out observations.
    n_train : int
        Number of training observations in this fold.
    n_test : int
        Number of held-out observations in this fold.
    params : dict[str, float]
        MAP parameter estimate from training data.
    converged : bool
        Whether the MAP optimiser converged.
    """

    fold: int
    train_log_likelihood: float
    test_log_likelihood: float
    n_train: int
    n_test: int
    params: dict[str, float]
    converged: bool


@dataclass
class CrossValidationResult:
    """Aggregated k-fold cross-validation report.

    Attributes
    ----------
    fold_results : list[FoldResult]
        Per-fold detail.
    n_folds : int
        Number of folds requested.
    n_completed : int
        Folds that converged successfully (may be < n_folds on failure).
    mean_test_ll : float
        Mean out-of-sample log-likelihood across converged folds.
    std_test_ll : float
        Standard deviation of fold test log-likelihoods.
    mean_train_ll : float
        Mean in-sample log-likelihood (for diagnosing over-fit).
    generalization_gap : float
        ``mean_train_ll - mean_test_ll``; positive means the model
        fits training data better than held-out data.
    """

    fold_results: list[FoldResult] = field(default_factory=list)
    n_folds: int = 0
    n_completed: int = 0
    mean_test_ll: float = float("nan")
    std_test_ll: float = float("nan")
    mean_train_ll: float = float("nan")
    generalization_gap: float = float("nan")

    def summary(self) -> str:
        """Return a compact text summary of CV results."""
        lines = [
            f"K-fold CV: {self.n_completed}/{self.n_folds} folds converged",
            f"  Mean test  log-lik: {self.mean_test_ll:+.4f}",
            f"  Std  test  log-lik: {self.std_test_ll:.4f}",
            f"  Mean train log-lik: {self.mean_train_ll:+.4f}",
            f"  Generalisation gap: {self.generalization_gap:+.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def cross_validate(
    prior_sampler: PriorSampler,
    target: CalibrationTarget,
    n_folds: int = 5,
    seed: int = 42,
    map_options: dict[str, Any] | None = None,
    shuffle: bool = True,
) -> CrossValidationResult:
    """Run k-fold cross-validation on a single CalibrationTarget.

    Parameters
    ----------
    prior_sampler : PriorSampler
        Provides prior distributions for MAP calibration.
    target : CalibrationTarget
        Calibration target whose observations will be split into folds.
        The ``model_fn`` must return a prediction vector of the same
        length as ``target.observed``.
    n_folds : int
        Number of folds (must be >= 2 and <= n_observations).
    seed : int
        Random seed for fold assignment (ignored when ``shuffle=False``).
    map_options : dict or None
        Passed to ``BayesianCalibrator.fit_map(options=...)``.
    shuffle : bool
        Shuffle observation indices before splitting into folds.

    Returns
    -------
    CrossValidationResult

    Raises
    ------
    ValueError
        If ``n_folds < 2`` or ``n_folds > n_observations``.
    """
    n_obs = len(target.observed)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_folds > n_obs:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed n_observations ({n_obs})"
        )

    indices = np.arange(n_obs)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    fold_index_sets = _split_indices(indices, n_folds)

    fold_results: list[FoldResult] = []

    for fold_i, test_idx in enumerate(fold_index_sets):
        train_idx = np.concatenate(
            [fold_index_sets[j] for j in range(n_folds) if j != fold_i]
        )

        train_target = _subset_target(target, train_idx)
        test_target = _subset_target(target, test_idx)

        calibrator = BayesianCalibrator(prior_sampler, [train_target])
        cal_result = calibrator.fit_map(options=map_options)

        train_ll = _score_target(train_target, cal_result.params)
        test_ll = _score_target(test_target, cal_result.params)

        fold_results.append(FoldResult(
            fold=fold_i,
            train_log_likelihood=train_ll,
            test_log_likelihood=test_ll,
            n_train=len(train_idx),
            n_test=len(test_idx),
            params=cal_result.params,
            converged=cal_result.success,
        ))

    return _aggregate(fold_results, n_folds)


# ---------------------------------------------------------------------------
# Leave-one-out convenience wrapper
# ---------------------------------------------------------------------------

def leave_one_out(
    prior_sampler: PriorSampler,
    target: CalibrationTarget,
    seed: int = 42,
    map_options: dict[str, Any] | None = None,
) -> CrossValidationResult:
    """Leave-one-out cross-validation (LOO-CV).

    Equivalent to ``cross_validate(..., n_folds=n_observations)``.
    Only practical for small datasets (each fold calibrates on n-1 points).

    Parameters
    ----------
    prior_sampler : PriorSampler
    target : CalibrationTarget
    seed : int
        Seed used if shuffle=True; for LOO the order does not affect results.
    map_options : dict or None

    Returns
    -------
    CrossValidationResult
    """
    n_obs = len(target.observed)
    return cross_validate(
        prior_sampler=prior_sampler,
        target=target,
        n_folds=n_obs,
        seed=seed,
        map_options=map_options,
        shuffle=False,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_indices(
    indices: np.ndarray,
    n_folds: int,
) -> list[np.ndarray]:
    """Split indices into n_folds roughly-equal-sized arrays."""
    return [arr for arr in np.array_split(indices, n_folds) if len(arr) > 0]


def _subset_target(
    target: CalibrationTarget,
    indices: np.ndarray,
) -> CalibrationTarget:
    """Return a CalibrationTarget restricted to a subset of observations."""
    indices = np.asarray(indices, dtype=int)

    sigma_sub = (
        target.sigma[indices]
        if isinstance(target.sigma, np.ndarray)
        else target.sigma
    )
    n_trials_sub = (
        target.n_trials[indices]
        if target.n_trials is not None
        else None
    )

    def _subset_model_fn(params, _idx=indices):
        full_pred = target.model_fn(params)
        return np.asarray(full_pred)[_idx]

    return CalibrationTarget(
        name=f"{target.name}_fold",
        likelihood_type=target.likelihood_type,
        observed=target.observed[indices],
        model_fn=_subset_model_fn,
        sigma=sigma_sub,
        n_trials=n_trials_sub,
    )


def _score_target(
    target: CalibrationTarget,
    params: dict[str, float],
) -> float:
    """Evaluate the log-likelihood of a target at fixed params."""
    predicted = target.model_fn(params)
    return _dispatch_ll(
        target.likelihood_type,
        target.observed,
        np.asarray(predicted, dtype=np.float64),
        sigma=target.sigma,
        n_trials=target.n_trials,
    )


def _aggregate(
    fold_results: list[FoldResult],
    n_folds: int,
) -> CrossValidationResult:
    """Compute summary statistics from per-fold results."""
    converged = [r for r in fold_results if r.converged]
    n_completed = len(converged)

    if n_completed == 0:
        return CrossValidationResult(
            fold_results=fold_results,
            n_folds=n_folds,
            n_completed=0,
        )

    test_lls = np.array([r.test_log_likelihood for r in converged])
    train_lls = np.array([r.train_log_likelihood for r in converged])

    mean_test = float(test_lls.mean())
    std_test = float(test_lls.std()) if n_completed > 1 else 0.0
    mean_train = float(train_lls.mean())

    return CrossValidationResult(
        fold_results=fold_results,
        n_folds=n_folds,
        n_completed=n_completed,
        mean_test_ll=mean_test,
        std_test_ll=std_test,
        mean_train_ll=mean_train,
        generalization_gap=mean_train - mean_test,
    )
