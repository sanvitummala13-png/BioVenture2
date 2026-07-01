"""Log-likelihood functions for Bayesian calibration.

Three likelihood families matching the calibration targets in calibration.yaml:

- **Binomial**: clinical phase-transition counts (k successes / n trials).
- **Normal**: market-size observations with Gaussian noise.
- **Least-squares**: adoption-curve anchors (normal with unit noise, up to constant).

All functions return *log*-likelihoods (negative = worse fit) so that
MAP estimation minimises their negation and MCMC samples from their
exponentiation.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Binomial log-likelihood
# ---------------------------------------------------------------------------

def binomial_log_likelihood(
    k: np.ndarray | int,
    n: np.ndarray | int,
    p: np.ndarray | float,
    eps: float = 1e-12,
) -> float:
    """Log-likelihood for independent binomial observations.

    Drops the combinatorial constant (which is irrelevant for optimisation).

    Parameters
    ----------
    k : array-like of int
        Observed success counts per trial group (shape ``(m,)``).
    n : array-like of int
        Number of trials per group (shape ``(m,)``).
    p : array-like of float
        Predicted success probabilities per group (shape ``(m,)``),
        clipped to ``[eps, 1 - eps]`` to avoid log(0).
    eps : float
        Numerical floor / ceiling for ``p``.

    Returns
    -------
    float
        Σ_i [ k_i · log(p_i) + (n_i − k_i) · log(1 − p_i) ]
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)

    if np.any(n < 0):
        raise ValueError("n (number of trials) must be >= 0")
    if np.any(k < 0) or np.any(k > n):
        raise ValueError("k must satisfy 0 <= k <= n")

    return float(np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# Normal (Gaussian) log-likelihood
# ---------------------------------------------------------------------------

def normal_log_likelihood(
    observed: np.ndarray | float,
    predicted: np.ndarray | float,
    sigma: np.ndarray | float,
) -> float:
    """Gaussian log-likelihood for continuous observations.

    Parameters
    ----------
    observed : array-like
        Observed values (shape ``(m,)``).
    predicted : array-like
        Model-predicted values (shape ``(m,)``).
    sigma : array-like or float
        Observation standard deviation. Must be > 0. Can be scalar
        (shared noise) or array (heteroscedastic).

    Returns
    -------
    float
        Σ_i [ −0.5 · ((obs_i − pred_i) / σ_i)² − log(σ_i · √(2π)) ]
    """
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)

    if np.any(sig <= 0):
        raise ValueError("sigma must be > 0")

    residuals = obs - pred
    return float(
        np.sum(
            -0.5 * (residuals / sig) ** 2
            - np.log(sig * np.sqrt(2.0 * np.pi))
        )
    )


# ---------------------------------------------------------------------------
# Least-squares log-likelihood
# ---------------------------------------------------------------------------

def least_squares_log_likelihood(
    observed: np.ndarray | float,
    predicted: np.ndarray | float,
) -> float:
    """Least-squares log-likelihood (normal with unit sigma, up to a constant).

    Equivalent to ``normal_log_likelihood(observed, predicted, sigma=1)``,
    but drops the constant term so that the value is comparable only within
    the same target (not across targets with different sigma assumptions).

    Parameters
    ----------
    observed : array-like
        Observed values.
    predicted : array-like
        Model-predicted values.

    Returns
    -------
    float
        ``−0.5 · Σ_i (obs_i − pred_i)²``
    """
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    return float(-0.5 * np.sum((obs - pred) ** 2))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def log_likelihood(
    likelihood_type: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    sigma: np.ndarray | float | None = None,
    n_trials: np.ndarray | None = None,
) -> float:
    """Dispatch to the appropriate log-likelihood function.

    Parameters
    ----------
    likelihood_type : str
        One of ``"binomial"``, ``"normal"``, ``"least_squares"``.
    observed : np.ndarray
        Observed data.
    predicted : np.ndarray
        Model predictions.
    sigma : array-like or float, optional
        Required for ``"normal"``.
    n_trials : array-like, optional
        Required for ``"binomial"`` — number of trials per group.

    Returns
    -------
    float
        Log-likelihood value.
    """
    lt = likelihood_type.lower().strip()

    if lt == "binomial":
        if n_trials is None:
            raise ValueError(
                "'n_trials' is required for binomial likelihood"
            )
        return binomial_log_likelihood(observed, n_trials, predicted)

    if lt == "normal":
        if sigma is None:
            raise ValueError("'sigma' is required for normal likelihood")
        return normal_log_likelihood(observed, predicted, sigma)

    if lt == "least_squares":
        return least_squares_log_likelihood(observed, predicted)

    raise ValueError(
        f"Unknown likelihood_type '{likelihood_type}'. "
        f"Supported: 'binomial', 'normal', 'least_squares'"
    )
