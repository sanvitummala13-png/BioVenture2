"""Update PriorSampler hyperparameters from MAP estimates or MCMC chains.

Two update paths:

**MAP update** — tighten each prior around its MAP point estimate.
  Keeps the same family; raises concentration (Beta) or shifts location
  (Normal/Lognormal) while preserving the scale/shape hyperparameters.

**MCMC update** — fit new distribution hyperparameters directly to
  posterior samples using method of moments.

Both return a *new* PriorSampler; the original is never mutated.
Parameters absent from the MAP dict / chain are carried over unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np

from bioventure.distributions.priors import PriorSampler, PriorSpec


# ---------------------------------------------------------------------------
# MAP-based update
# ---------------------------------------------------------------------------

def update_priors_from_map(
    prior_sampler: PriorSampler,
    map_params: dict[str, float],
    concentration: float = 100.0,
) -> PriorSampler:
    """Return a new PriorSampler with priors tightened around MAP estimates.

    Parameters
    ----------
    prior_sampler : PriorSampler
        Source of all prior specs and their families.
    map_params : dict[str, float]
        MAP point estimates (may be a *subset* of all parameters).
        Parameters not present are carried over unchanged.
    concentration : float
        Controls how tightly Beta priors contract around their MAP value.
        At the MAP estimate p̂, the new Beta has a = p̂ · c, b = (1−p̂) · c,
        so a + b = c.  Higher c → narrower posterior.  Must be > 0.

    Returns
    -------
    PriorSampler
        New sampler with updated hyperparameters.

    Raises
    ------
    ValueError
        If *concentration* <= 0.
    """
    if concentration <= 0:
        raise ValueError(f"concentration must be > 0, got {concentration}")

    new_specs: dict[str, PriorSpec] = {}
    for name, spec in prior_sampler.specs.items():
        if name not in map_params:
            new_specs[name] = spec
        else:
            updated = _update_spec_map(spec, map_params[name], concentration)
            new_specs[name] = updated

    return PriorSampler(new_specs)


def _update_spec_map(
    spec: PriorSpec,
    value: float,
    concentration: float,
) -> PriorSpec:
    """Shift / tighten a single PriorSpec around a MAP scalar value."""
    family = spec.family

    if family == "beta":
        if not 0.0 < value < 1.0:
            warnings.warn(
                f"MAP value {value:.6g} for Beta prior '{spec.name}' is "
                f"outside (0, 1); clamping to avoid invalid hyperparameters.",
                stacklevel=4,
            )
            value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
        a = max(value * concentration, 1e-6)
        b = max((1.0 - value) * concentration, 1e-6)
        return PriorSpec(spec.name, "beta", {"a": a, "b": b})

    if family == "normal":
        return PriorSpec(spec.name, "normal", {
            "loc": float(value),
            "scale": spec.params["scale"],
        })

    if family == "lognormal":
        if value <= 0.0:
            warnings.warn(
                f"MAP value {value:.6g} for Lognormal prior '{spec.name}' "
                f"is <= 0; keeping original prior.",
                stacklevel=4,
            )
            return spec
        return PriorSpec(spec.name, "lognormal", {
            "mu": float(np.log(value)),
            "sigma": spec.params["sigma"],
        })

    # triangular, uniform, bernoulli, poisson: no natural MAP shift; keep as-is
    return spec


# ---------------------------------------------------------------------------
# MCMC-based update
# ---------------------------------------------------------------------------

def update_priors_from_mcmc(
    prior_sampler: PriorSampler,
    chain: np.ndarray,
    param_names: list[str],
    min_samples: int = 50,
) -> PriorSampler:
    """Fit new prior hyperparameters to MCMC posterior samples.

    Uses method of moments for Beta, mean / std of log-samples for
    Lognormal, and mean / std for Normal.  Falls back to the original
    prior when fitting fails or too few samples are available.

    Parameters
    ----------
    prior_sampler : PriorSampler
        Source prior specs (families are preserved; only params change).
    chain : np.ndarray
        Shape ``(n_samples, n_params)`` — flat posterior chain after
        burn-in and thinning (e.g. ``CalibrationResult.mcmc_chain``).
    param_names : list[str]
        Parameter names corresponding to the columns of *chain*.
    min_samples : int
        Minimum number of finite samples required to attempt fitting.
        Parameters with fewer samples are left unchanged.

    Returns
    -------
    PriorSampler
        New sampler with fitted hyperparameters.

    Raises
    ------
    ValueError
        If *chain* is not 2-D or its column count mismatches *param_names*.
    """
    if chain.ndim != 2:
        raise ValueError(f"chain must be 2-D, got {chain.ndim}-D")
    if chain.shape[1] != len(param_names):
        raise ValueError(
            f"chain has {chain.shape[1]} columns but param_names has "
            f"{len(param_names)} entries"
        )

    chain_map: dict[str, np.ndarray] = {
        name: chain[:, i] for i, name in enumerate(param_names)
    }

    new_specs: dict[str, PriorSpec] = {}
    for name, spec in prior_sampler.specs.items():
        if name not in chain_map:
            new_specs[name] = spec
            continue

        samples = chain_map[name]
        finite_samples = samples[np.isfinite(samples)]

        if len(finite_samples) < min_samples:
            warnings.warn(
                f"Only {len(finite_samples)} finite samples for '{name}'; "
                f"keeping original prior.",
                stacklevel=2,
            )
            new_specs[name] = spec
            continue

        updated = _update_spec_mcmc(spec, finite_samples)
        new_specs[name] = updated if updated is not None else spec

    return PriorSampler(new_specs)


def _update_spec_mcmc(
    spec: PriorSpec,
    samples: np.ndarray,
) -> PriorSpec | None:
    """Fit a single PriorSpec to posterior samples.

    Returns ``None`` if fitting fails (caller falls back to original spec).
    """
    family = spec.family

    if family == "beta":
        return _fit_beta(spec, samples)

    if family == "normal":
        return _fit_normal(spec, samples)

    if family == "lognormal":
        return _fit_lognormal(spec, samples)

    # triangular, uniform, bernoulli, poisson: no closed-form update
    return None


def _fit_beta(spec: PriorSpec, samples: np.ndarray) -> PriorSpec | None:
    """Method-of-moments Beta fit: a = m·κ, b = (1−m)·κ, κ = m(1−m)/v − 1."""
    valid = samples[(samples > 0.0) & (samples < 1.0)]
    if len(valid) < 2:
        return None

    m = float(valid.mean())
    v = float(valid.var())

    if v <= 0.0 or not np.isfinite(v):
        return None

    kappa = m * (1.0 - m) / v - 1.0
    if kappa <= 0.0 or not np.isfinite(kappa):
        warnings.warn(
            f"Beta moment-matching failed for '{spec.name}' "
            f"(mean={m:.4f}, var={v:.6f}); keeping original prior.",
            stacklevel=4,
        )
        return None

    a = float(m * kappa)
    b = float((1.0 - m) * kappa)
    if a <= 0.0 or b <= 0.0:
        return None

    return PriorSpec(spec.name, "beta", {"a": a, "b": b})


def _fit_normal(spec: PriorSpec, samples: np.ndarray) -> PriorSpec | None:
    """Fit Normal from sample mean and std."""
    loc = float(samples.mean())
    scale = float(samples.std())
    if scale <= 0.0 or not np.isfinite(loc) or not np.isfinite(scale):
        return None
    return PriorSpec(spec.name, "normal", {"loc": loc, "scale": scale})


def _fit_lognormal(spec: PriorSpec, samples: np.ndarray) -> PriorSpec | None:
    """Fit Lognormal from mean and std of log-samples."""
    pos = samples[samples > 0.0]
    if len(pos) < 2:
        return None

    log_s = np.log(pos)
    mu = float(log_s.mean())
    sigma = float(log_s.std())

    if sigma <= 0.0 or not np.isfinite(mu) or not np.isfinite(sigma):
        return None

    return PriorSpec(spec.name, "lognormal", {"mu": mu, "sigma": sigma})


# ---------------------------------------------------------------------------
# Convenience: apply a CalibrationResult directly
# ---------------------------------------------------------------------------

def apply_calibration_result(
    prior_sampler: PriorSampler,
    result,
    concentration: float = 100.0,
    min_samples: int = 50,
) -> PriorSampler:
    """Update priors from a CalibrationResult, dispatching on method.

    Parameters
    ----------
    prior_sampler : PriorSampler
    result : CalibrationResult
        Output of ``BayesianCalibrator.fit_map()`` or ``.fit_mcmc()``.
    concentration : float
        Passed to :func:`update_priors_from_map` for MAP results.
    min_samples : int
        Passed to :func:`update_priors_from_mcmc` for MCMC results.

    Returns
    -------
    PriorSampler
    """
    if result.method == "map":
        return update_priors_from_map(
            prior_sampler, result.params, concentration=concentration
        )

    if result.method == "mcmc":
        if result.mcmc_chain is None:
            raise ValueError(
                "CalibrationResult.mcmc_chain is None; cannot update from MCMC"
            )
        param_names = sorted(prior_sampler.names)
        return update_priors_from_mcmc(
            prior_sampler,
            result.mcmc_chain,
            param_names,
            min_samples=min_samples,
        )

    raise ValueError(
        f"Unknown CalibrationResult.method '{result.method}'; "
        f"expected 'map' or 'mcmc'"
    )
