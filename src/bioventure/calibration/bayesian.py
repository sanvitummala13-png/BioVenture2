"""Bayesian calibration: MAP estimation and optional MCMC via emcee.

Calibration fits model parameters to observed data by maximising the
log-posterior:

    log p(θ | data) = log p(data | θ)  +  log p(θ)
                    = Σ_targets log_likelihood  +  Σ_params log_prior

MAP uses L-BFGS-B (scipy) and is always available.
MCMC uses emcee's EnsembleSampler and requires ``pip install emcee``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import optimize

from bioventure.distributions.priors import PriorSampler, PriorSpec
from bioventure.calibration.likelihoods import log_likelihood as _dispatch_ll

try:
    import emcee as emcee  # type: ignore[import-not-found]
    _HAS_EMCEE = True
except ImportError:
    _HAS_EMCEE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_bounds(spec: PriorSpec) -> tuple[float, float]:
    """Return (lower, upper) optimisation bounds implied by a prior family."""
    family = spec.family
    if family == "beta":
        return (1e-9, 1.0 - 1e-9)
    if family == "lognormal":
        return (1e-9, np.inf)
    if family == "bernoulli":
        return (0.0, 1.0)
    if family == "uniform":
        return (float(spec.params["low"]), float(spec.params["high"]))
    if family == "triangular":
        return (float(spec.params["left"]), float(spec.params["right"]))
    if family == "poisson":
        return (0.0, np.inf)
    return (-np.inf, np.inf)


def _log_pdf(spec: PriorSpec, value: float) -> float:
    """Log-density of a prior at *value*, handling both continuous/discrete."""
    dist = spec.dist
    try:
        return float(dist.logpdf(value))
    except AttributeError:
        return float(dist.logpmf(int(round(value))))


def _clip_to_bounds(
    vec: np.ndarray,
    bounds: list[tuple[float, float]],
    fallback: float = 1e6,
) -> np.ndarray:
    """Clip a parameter vector to finite versions of its bounds."""
    out = vec.copy()
    for i, (lo, hi) in enumerate(bounds):
        lo_c = lo if np.isfinite(lo) else -fallback
        hi_c = hi if np.isfinite(hi) else fallback
        out[i] = np.clip(out[i], lo_c, hi_c)
    return out


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationTarget:
    """One calibration target: observed data + model function + likelihood.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"clinical"``, ``"market"``).
    likelihood_type : str
        One of ``"binomial"``, ``"normal"``, ``"least_squares"``.
    observed : np.ndarray
        Observed data array.
    model_fn : callable
        Maps ``params: dict[str, float]`` → predicted values ``np.ndarray``.
        Should be deterministic (no RNG).
    sigma : float or np.ndarray or None
        Observation noise standard deviation. Required for ``"normal"``.
    n_trials : np.ndarray or None
        Trial counts per group. Required for ``"binomial"``.
    """

    name: str
    likelihood_type: str
    observed: np.ndarray
    model_fn: Callable[[dict[str, float]], np.ndarray]
    sigma: np.ndarray | float | None = None
    n_trials: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.observed = np.asarray(self.observed, dtype=np.float64)
        valid = {"binomial", "normal", "least_squares"}
        if self.likelihood_type not in valid:
            raise ValueError(
                f"likelihood_type '{self.likelihood_type}' not in {valid}"
            )
        if self.likelihood_type == "normal" and self.sigma is None:
            raise ValueError(
                f"CalibrationTarget '{self.name}': sigma required for 'normal'"
            )
        if self.likelihood_type == "binomial" and self.n_trials is None:
            raise ValueError(
                f"CalibrationTarget '{self.name}': n_trials required for 'binomial'"
            )


@dataclass
class CalibrationResult:
    """Output of a calibration run.

    Attributes
    ----------
    method : str
        ``"map"`` or ``"mcmc"``.
    params : dict[str, float]
        Best-fit parameter vector (MAP point, or highest-posterior MCMC draw).
    log_posterior_value : float
        Log-posterior at *params*.
    success : bool
        Whether the optimiser converged (MAP) or chain ran to completion (MCMC).
    message : str
        Optimiser / sampler status message.
    mcmc_chain : np.ndarray or None
        Shape ``(n_samples, n_params)`` — flat chain after burn-in and thinning.
        ``None`` for MAP results.
    diagnostics : dict[str, Any]
        Extra diagnostics: n_iter, acceptance_fraction, autocorr_time, etc.
    """

    method: str
    params: dict[str, float]
    log_posterior_value: float
    success: bool
    message: str = ""
    mcmc_chain: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class BayesianCalibrator:
    """Fit model parameters to observed data via MAP or MCMC.

    Parameters
    ----------
    prior_sampler : PriorSampler
        Contains all uncertain parameter priors (used for log-prior).
    targets : list[CalibrationTarget]
        One or more calibration targets.
    """

    def __init__(
        self,
        prior_sampler: PriorSampler,
        targets: list[CalibrationTarget],
    ) -> None:
        if not targets:
            raise ValueError("At least one CalibrationTarget is required")

        self._priors = prior_sampler
        self._targets = list(targets)
        self._param_names: list[str] = sorted(prior_sampler.names)
        self._n_params: int = len(self._param_names)
        self._bounds: list[tuple[float, float]] = [
            _get_bounds(prior_sampler[name]) for name in self._param_names
        ]

    @property
    def param_names(self) -> list[str]:
        return list(self._param_names)

    @property
    def n_params(self) -> int:
        return self._n_params

    # ------------------------------------------------------------------
    # Core probability functions
    # ------------------------------------------------------------------

    def log_prior(self, params: dict[str, float]) -> float:
        """Sum of log-prior densities across all parameters."""
        total = 0.0
        for name in self._param_names:
            lp = _log_pdf(self._priors[name], params[name])
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return total

    def log_likelihood(self, params: dict[str, float]) -> float:
        """Sum of log-likelihoods across all calibration targets."""
        total = 0.0
        for target in self._targets:
            predicted = target.model_fn(params)
            total += _dispatch_ll(
                target.likelihood_type,
                target.observed,
                predicted,
                sigma=target.sigma,
                n_trials=target.n_trials,
            )
            if not np.isfinite(total):
                return -np.inf
        return total

    def log_posterior(self, params: dict[str, float]) -> float:
        """Log-prior + log-likelihood (log-posterior up to a constant)."""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(params)
        return lp + ll

    # ------------------------------------------------------------------
    # MAP estimation
    # ------------------------------------------------------------------

    def fit_map(
        self,
        initial_params: dict[str, float] | None = None,
        method: str = "L-BFGS-B",
        options: dict[str, Any] | None = None,
    ) -> CalibrationResult:
        """Find the maximum a posteriori (MAP) estimate.

        Parameters
        ----------
        initial_params : dict or None
            Starting point for the optimiser. Uses prior means if ``None``.
        method : str
            scipy.optimize.minimize method. ``"L-BFGS-B"`` supports bounds.
        options : dict or None
            Passed directly to ``scipy.optimize.minimize``.

        Returns
        -------
        CalibrationResult
        """
        x0 = (
            self._params_to_vec(initial_params)
            if initial_params is not None
            else self._prior_means()
        )
        x0 = _clip_to_bounds(x0, self._bounds)

        scipy_bounds = [
            (lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None)
            for lo, hi in self._bounds
        ]

        def neg_log_post(vec: np.ndarray) -> float:
            params = self._vec_to_params(vec)
            val = self.log_posterior(params)
            return -val if np.isfinite(val) else 1e20

        result = optimize.minimize(
            neg_log_post,
            x0,
            method=method,
            bounds=scipy_bounds,
            options=options or {"maxiter": 2000, "ftol": 1e-12},
        )

        return CalibrationResult(
            method="map",
            params=self._vec_to_params(result.x),
            log_posterior_value=float(-result.fun),
            success=bool(result.success),
            message=str(result.message),
            diagnostics={
                "n_iter": int(result.nit),
                "n_fev": int(result.nfev),
            },
        )

    # ------------------------------------------------------------------
    # MCMC estimation
    # ------------------------------------------------------------------

    def fit_mcmc(
        self,
        n_walkers: int = 32,
        n_steps: int = 5000,
        burn_in: int = 1000,
        thin: int = 5,
        initial_params: dict[str, float] | None = None,
        seed: int = 42,
        progress: bool = False,
    ) -> CalibrationResult:
        """Sample the posterior via emcee ensemble MCMC.

        Parameters
        ----------
        n_walkers : int
            Number of ensemble walkers (must be even and >= 2 * n_params).
        n_steps : int
            Total MCMC steps per walker.
        burn_in : int
            Steps to discard from the start of each chain.
        thin : int
            Keep every *thin*-th sample after burn-in.
        initial_params : dict or None
            Walker initialisation centre. Uses prior means if ``None``.
        seed : int
            Seed for walker initialisation noise.
        progress : bool
            Show emcee progress bar (requires ``tqdm``).

        Returns
        -------
        CalibrationResult

        Raises
        ------
        ImportError
            If emcee is not installed.
        """
        if not _HAS_EMCEE:
            raise ImportError(
                "emcee is required for MCMC. Install with: pip install emcee"
            )
        if n_walkers < 2 * self._n_params:
            raise ValueError(
                f"n_walkers ({n_walkers}) must be >= 2 * n_params "
                f"({2 * self._n_params}) for emcee"
            )
        if burn_in >= n_steps:
            raise ValueError(
                f"burn_in ({burn_in}) must be < n_steps ({n_steps})"
            )

        center = (
            self._params_to_vec(initial_params)
            if initial_params is not None
            else self._prior_means()
        )
        center = _clip_to_bounds(center, self._bounds)

        rng = np.random.default_rng(seed)
        scale = np.where(np.abs(center) > 1e-8, 0.01 * np.abs(center), 1e-4)
        p0 = center[np.newaxis, :] + scale * rng.standard_normal(
            (n_walkers, self._n_params)
        )
        p0 = np.apply_along_axis(_clip_to_bounds, 1, p0, self._bounds)

        def _log_prob(vec: np.ndarray) -> float:
            if not self._in_bounds(vec):
                return -np.inf
            val = self.log_posterior(self._vec_to_params(vec))
            return val if np.isfinite(val) else -np.inf

        sampler = emcee.EnsembleSampler(
            n_walkers, self._n_params, _log_prob
        )
        sampler.run_mcmc(p0, n_steps, progress=progress)

        flat_chain = sampler.get_chain(
            discard=burn_in, thin=thin, flat=True
        )
        flat_lp = sampler.get_log_prob(
            discard=burn_in, thin=thin, flat=True
        )

        best_idx = int(np.argmax(flat_lp))
        best_params = self._vec_to_params(flat_chain[best_idx])

        diagnostics: dict[str, Any] = {
            "n_samples": int(len(flat_chain)),
            "acceptance_fraction": float(
                sampler.acceptance_fraction.mean()
            ),
        }
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            diagnostics["autocorr_time"] = tau.tolist()
            diagnostics["effective_samples"] = float(
                len(flat_chain) / float(np.mean(tau))
            )
        except Exception:
            pass

        return CalibrationResult(
            method="mcmc",
            params=best_params,
            log_posterior_value=float(flat_lp[best_idx]),
            success=True,
            message=(
                f"MCMC complete: {len(flat_chain)} samples "
                f"(acceptance={diagnostics['acceptance_fraction']:.3f})"
            ),
            mcmc_chain=flat_chain,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _params_to_vec(self, params: dict[str, float]) -> np.ndarray:
        return np.array([params[name] for name in self._param_names])

    def _vec_to_params(self, vec: np.ndarray) -> dict[str, float]:
        return {name: float(vec[i]) for i, name in enumerate(self._param_names)}

    def _prior_means(self) -> np.ndarray:
        return np.array(
            [float(self._priors[name].dist.mean()) for name in self._param_names]
        )

    def _in_bounds(self, vec: np.ndarray) -> bool:
        for i, (lo, hi) in enumerate(self._bounds):
            v = vec[i]
            if (lo != -np.inf and v < lo) or (hi != np.inf and v > hi):
                return False
        return True
    