"""Tests for calibration subpackage: likelihoods, Bayesian calibrator, prior update."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from bioventure.calibration.bayesian import (
    BayesianCalibrator,
    CalibrationResult,
    CalibrationTarget,
)
from bioventure.calibration.likelihoods import (
    binomial_log_likelihood,
    least_squares_log_likelihood,
    log_likelihood,
    normal_log_likelihood,
)
from bioventure.calibration.prior_update import (
    apply_calibration_result,
    update_priors_from_map,
    update_priors_from_mcmc,
)
from bioventure.distributions.priors import PriorSampler, PriorSpec

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _beta_sampler() -> PriorSampler:
    return PriorSampler({"p": PriorSpec("p", "beta", {"a": 2.0, "b": 5.0})})


def _normal_sampler() -> PriorSampler:
    return PriorSampler({
        "mu": PriorSpec("mu", "normal", {"loc": 0.5, "scale": 0.2})
    })


def _lognormal_sampler() -> PriorSampler:
    return PriorSampler({
        "lam": PriorSpec("lam", "lognormal", {"mu": -1.2, "sigma": 0.4})
    })


def _two_param_sampler() -> PriorSampler:
    return PriorSampler({
        "p": PriorSpec("p", "beta", {"a": 2.0, "b": 5.0}),
        "mu": PriorSpec("mu", "normal", {"loc": 0.5, "scale": 0.2}),
    })


def _binomial_target() -> CalibrationTarget:
    """k=7, n=10 → MAP with Beta(2,5) prior = 8/15."""
    return CalibrationTarget(
        name="binomial",
        likelihood_type="binomial",
        observed=np.array([7.0]),
        model_fn=lambda params: np.array([params["p"]]),
        n_trials=np.array([10.0]),
    )


def _normal_target() -> CalibrationTarget:
    return CalibrationTarget(
        name="normal_obs",
        likelihood_type="normal",
        observed=np.array([0.6]),
        model_fn=lambda params: np.array([params["mu"]]),
        sigma=0.1,
    )


def _ls_target() -> CalibrationTarget:
    return CalibrationTarget(
        name="ls",
        likelihood_type="least_squares",
        observed=np.array([0.3, 0.5, 0.7]),
        model_fn=lambda params: np.array([params["p"]] * 3),
    )


# ===========================================================================
# TestLikelihoods
# ===========================================================================

class TestBinomialLogLikelihood:
    def test_known_value(self):
        # k=7, n=10, p=0.7 → 7*log(0.7)+3*log(0.3)
        expected = 7 * np.log(0.7) + 3 * np.log(0.3)
        assert binomial_log_likelihood(7, 10, 0.7) == pytest.approx(expected)

    def test_all_successes(self):
        # k=n, p=1 → log-likelihood = 0 (after clipping p to 1-eps)
        val = binomial_log_likelihood(10, 10, 1.0 - 1e-12)
        assert np.isfinite(val)

    def test_zero_successes(self):
        val = binomial_log_likelihood(0, 10, 1e-9)
        assert np.isfinite(val)

    def test_clipping_prevents_log_zero(self):
        val = binomial_log_likelihood(5, 10, 0.0)
        assert np.isfinite(val)

    def test_invalid_k_gt_n_raises(self):
        with pytest.raises(ValueError, match="0 <= k <= n"):
            binomial_log_likelihood(11, 10, 0.5)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="n.*must be >= 0"):
            binomial_log_likelihood(3, -1, 0.5)

    def test_vectorised(self):
        k = np.array([3.0, 7.0])
        n = np.array([10.0, 10.0])
        p = np.array([0.3, 0.7])
        val = binomial_log_likelihood(k, n, p)
        assert isinstance(val, float)
        assert np.isfinite(val)


class TestNormalLogLikelihood:
    def test_zero_residual_known_value(self):
        # −0.5*(0/σ)² − log(σ√2π)
        sigma = 2.0
        expected = -np.log(sigma * np.sqrt(2 * np.pi))
        assert normal_log_likelihood(1.0, 1.0, sigma) == pytest.approx(expected)

    def test_larger_residual_lower_likelihood(self):
        ll_close = normal_log_likelihood(1.0, 1.1, 1.0)
        ll_far = normal_log_likelihood(1.0, 5.0, 1.0)
        assert ll_close > ll_far

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            normal_log_likelihood(1.0, 1.0, 0.0)

    def test_heteroscedastic_sigma(self):
        obs = np.array([1.0, 2.0])
        pred = np.array([1.1, 1.9])
        sigma = np.array([0.5, 1.0])
        val = normal_log_likelihood(obs, pred, sigma)
        assert np.isfinite(val)


class TestLeastSquaresLogLikelihood:
    def test_perfect_fit_returns_zero(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert least_squares_log_likelihood(obs, obs) == pytest.approx(0.0)

    def test_known_value(self):
        obs = np.array([0.0, 0.0])
        pred = np.array([1.0, 1.0])
        assert least_squares_log_likelihood(obs, pred) == pytest.approx(-1.0)

    def test_larger_residual_lower(self):
        obs = np.array([0.0])
        ll_small = least_squares_log_likelihood(obs, np.array([0.1]))
        ll_large = least_squares_log_likelihood(obs, np.array([10.0]))
        assert ll_small > ll_large


class TestLogLikelihoodDispatcher:
    def test_dispatch_binomial(self):
        val = log_likelihood(
            "binomial",
            np.array([7.0]),
            np.array([0.7]),
            n_trials=np.array([10.0]),
        )
        expected = binomial_log_likelihood(7, 10, 0.7)
        assert val == pytest.approx(expected)

    def test_dispatch_normal(self):
        val = log_likelihood("normal", np.array([1.0]), np.array([1.0]), sigma=1.0)
        assert np.isfinite(val)

    def test_dispatch_least_squares(self):
        val = log_likelihood(
            "least_squares", np.array([1.0]), np.array([1.0])
        )
        assert val == pytest.approx(0.0)

    def test_dispatch_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown likelihood_type"):
            log_likelihood("poisson", np.array([1.0]), np.array([1.0]))

    def test_dispatch_binomial_missing_n_trials_raises(self):
        with pytest.raises(ValueError, match="n_trials"):
            log_likelihood("binomial", np.array([5.0]), np.array([0.5]))

    def test_dispatch_normal_missing_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            log_likelihood("normal", np.array([1.0]), np.array([1.0]))


# ===========================================================================
# TestCalibrationTarget
# ===========================================================================

class TestCalibrationTarget:
    def test_valid_binomial(self):
        t = _binomial_target()
        assert t.name == "binomial"
        assert t.likelihood_type == "binomial"

    def test_valid_normal(self):
        t = _normal_target()
        assert t.sigma == pytest.approx(0.1)

    def test_valid_least_squares(self):
        t = _ls_target()
        assert t.likelihood_type == "least_squares"

    def test_invalid_likelihood_type_raises(self):
        with pytest.raises(ValueError, match="likelihood_type"):
            CalibrationTarget(
                name="bad", likelihood_type="poisson",
                observed=np.array([1.0]),
                model_fn=lambda p: np.array([1.0]),
            )

    def test_normal_missing_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma required"):
            CalibrationTarget(
                name="x", likelihood_type="normal",
                observed=np.array([1.0]),
                model_fn=lambda p: np.array([1.0]),
            )

    def test_binomial_missing_n_trials_raises(self):
        with pytest.raises(ValueError, match="n_trials required"):
            CalibrationTarget(
                name="x", likelihood_type="binomial",
                observed=np.array([5.0]),
                model_fn=lambda p: np.array([0.5]),
            )

    def test_observed_converted_to_array(self):
        t = _normal_target()
        assert isinstance(t.observed, np.ndarray)


# ===========================================================================
# TestBayesianCalibratorLogFunctions
# ===========================================================================

class TestBayesianCalibratorLogFunctions:
    @pytest.fixture
    def calibrator(self):
        return BayesianCalibrator(_beta_sampler(), [_binomial_target()])

    def test_log_prior_at_mean_finite(self, calibrator):
        params = {"p": 2.0 / 7.0}  # prior mean
        assert np.isfinite(calibrator.log_prior(params))

    def test_log_prior_out_of_bounds_neg_inf(self, calibrator):
        # Beta has support (0, 1); p=1.5 must return -inf
        assert calibrator.log_prior({"p": 1.5}) == -np.inf

    def test_log_likelihood_finite_at_valid_params(self, calibrator):
        ll = calibrator.log_likelihood({"p": 0.5})
        assert np.isfinite(ll)

    def test_log_posterior_equals_prior_plus_likelihood(self, calibrator):
        params = {"p": 0.4}
        lp = calibrator.log_prior(params)
        ll = calibrator.log_likelihood(params)
        lpost = calibrator.log_posterior(params)
        assert lpost == pytest.approx(lp + ll)

    def test_log_posterior_out_of_bounds_neg_inf(self, calibrator):
        assert calibrator.log_posterior({"p": -0.1}) == -np.inf

    def test_no_targets_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            BayesianCalibrator(_beta_sampler(), [])


# ===========================================================================
# TestBayesianCalibratorFitMAP
# ===========================================================================

class TestBayesianCalibratorFitMAP:
    def test_returns_calibration_result(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        result = cal.fit_map()
        assert isinstance(result, CalibrationResult)

    def test_result_method_is_map(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        assert cal.fit_map().method == "map"

    def test_result_success(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        assert cal.fit_map().success is True

    def test_result_log_posterior_finite(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        result = cal.fit_map()
        assert np.isfinite(result.log_posterior_value)

    def test_beta_binomial_analytical_map(self):
        # Prior: Beta(2,5); data: k=7, n=10
        # Posterior Beta(9,8); MAP = (9-1)/(9+8-2) = 8/15
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        result = cal.fit_map()
        assert result.params["p"] == pytest.approx(8.0 / 15.0, abs=1e-4)

    def test_map_with_initial_params(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        result = cal.fit_map(initial_params={"p": 0.5})
        assert result.params["p"] == pytest.approx(8.0 / 15.0, abs=1e-4)

    def test_map_normal_target(self):
        # Normal prior N(0.5, 0.2²); observation 0.6 with sigma=0.1
        # MAP = (obs/σ² + loc/scale²) / (1/σ² + 1/scale²)
        #     = (0.6/0.01 + 0.5/0.04) / (100 + 25)
        #     = (60 + 12.5) / 125 = 72.5 / 125 = 0.58
        cal = BayesianCalibrator(_normal_sampler(), [_normal_target()])
        result = cal.fit_map()
        assert result.params["mu"] == pytest.approx(0.58, abs=2e-3)

    def test_map_least_squares_target(self):
        # Least-squares target with Beta prior: MAP pulled toward data mean
        cal = BayesianCalibrator(_beta_sampler(), [_ls_target()])
        result = cal.fit_map()
        assert 0.0 < result.params["p"] < 1.0

    def test_diagnostics_keys_present(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        result = cal.fit_map()
        assert "n_iter" in result.diagnostics
        assert "n_fev" in result.diagnostics


# ===========================================================================
# TestBayesianCalibratorMCMC  (requires emcee; very small run)
# ===========================================================================

@pytest.mark.slow
class TestBayesianCalibratorMCMC:
    emcee = pytest.importorskip("emcee")

    def test_mcmc_returns_calibration_result(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        result = cal.fit_mcmc(n_walkers=4, n_steps=60, burn_in=10, thin=1)
        assert isinstance(result, CalibrationResult)
        assert result.method == "mcmc"

    def test_mcmc_chain_shape(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        result = cal.fit_mcmc(n_walkers=4, n_steps=60, burn_in=10, thin=1)
        assert result.mcmc_chain is not None
        assert result.mcmc_chain.ndim == 2
        assert result.mcmc_chain.shape[1] == 1  # one parameter

    def test_mcmc_n_walkers_too_small_raises(self):
        cal = BayesianCalibrator(_two_param_sampler(), [_binomial_target()])
        with pytest.raises(ValueError, match="n_walkers"):
            cal.fit_mcmc(n_walkers=2, n_steps=50, burn_in=10)

    def test_mcmc_burn_in_ge_n_steps_raises(self):
        cal = BayesianCalibrator(_beta_sampler(), [_binomial_target()])
        with pytest.raises(ValueError, match="burn_in"):
            cal.fit_mcmc(n_walkers=4, n_steps=50, burn_in=50)


# ===========================================================================
# TestPriorUpdateFromMap
# ===========================================================================

class TestPriorUpdateFromMap:
    def test_beta_prior_updated(self):
        ps = _beta_sampler()
        updated = update_priors_from_map(ps, {"p": 0.6}, concentration=50.0)
        new_spec = updated["p"]
        assert new_spec.params["a"] == pytest.approx(30.0)
        assert new_spec.params["b"] == pytest.approx(20.0)

    def test_beta_mean_shifts_toward_map(self):
        ps = _beta_sampler()
        prior_mean = 2.0 / 7.0
        updated = update_priors_from_map(ps, {"p": 0.8}, concentration=100.0)
        new_mean = float(updated["p"].dist.mean())
        assert new_mean == pytest.approx(0.8, abs=1e-9)

    def test_normal_loc_shifts_to_map(self):
        ps = _normal_sampler()
        updated = update_priors_from_map(ps, {"mu": 0.7})
        assert updated["mu"].params["loc"] == pytest.approx(0.7)

    def test_normal_scale_preserved(self):
        ps = _normal_sampler()
        original_scale = ps["mu"].params["scale"]
        updated = update_priors_from_map(ps, {"mu": 0.7})
        assert updated["mu"].params["scale"] == pytest.approx(original_scale)

    def test_lognormal_mu_updated(self):
        ps = _lognormal_sampler()
        map_val = 0.3
        updated = update_priors_from_map(ps, {"lam": map_val})
        assert updated["lam"].params["mu"] == pytest.approx(np.log(map_val))

    def test_lognormal_sigma_preserved(self):
        ps = _lognormal_sampler()
        original_sigma = ps["lam"].params["sigma"]
        updated = update_priors_from_map(ps, {"lam": 0.3})
        assert updated["lam"].params["sigma"] == pytest.approx(original_sigma)

    def test_unknown_param_unchanged(self):
        ps = _beta_sampler()
        updated = update_priors_from_map(ps, {"nonexistent": 0.5})
        assert updated["p"].params == ps["p"].params

    def test_invalid_concentration_raises(self):
        with pytest.raises(ValueError, match="concentration"):
            update_priors_from_map(_beta_sampler(), {"p": 0.5}, concentration=0.0)

    def test_lognormal_negative_map_warns_and_keeps_original(self):
        ps = _lognormal_sampler()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            updated = update_priors_from_map(ps, {"lam": -0.5})
        assert updated["lam"].params == ps["lam"].params


# ===========================================================================
# TestPriorUpdateFromMCMC
# ===========================================================================

class TestPriorUpdateFromMCMC:
    def _beta_samples(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        return rng.beta(3.0, 7.0, size=n)

    def _normal_samples(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0.6, 0.15, size=n)

    def _lognormal_samples(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        return rng.lognormal(mean=-1.2, sigma=0.4, size=n)

    def test_beta_moments_recovered(self):
        ps = _beta_sampler()
        chain = self._beta_samples()[:, np.newaxis]
        updated = update_priors_from_mcmc(ps, chain, ["p"])
        # True Beta(3, 7): mean = 0.3, a/(a+b) ≈ 0.3
        new_mean = float(updated["p"].dist.mean())
        assert new_mean == pytest.approx(0.3, abs=0.05)

    def test_normal_mean_recovered(self):
        ps = _normal_sampler()
        chain = self._normal_samples()[:, np.newaxis]
        updated = update_priors_from_mcmc(ps, chain, ["mu"])
        new_mean = float(updated["mu"].dist.mean())
        assert new_mean == pytest.approx(0.6, abs=0.05)

    def test_lognormal_mu_recovered(self):
        ps = _lognormal_sampler()
        chain = self._lognormal_samples()[:, np.newaxis]
        updated = update_priors_from_mcmc(ps, chain, ["lam"])
        assert updated["lam"].params["mu"] == pytest.approx(-1.2, abs=0.1)

    def test_param_not_in_chain_unchanged(self):
        ps = _two_param_sampler()
        chain = self._beta_samples()[:, np.newaxis]
        updated = update_priors_from_mcmc(ps, chain, ["p"])
        assert updated["mu"].params == ps["mu"].params

    def test_too_few_samples_warns_and_keeps_original(self):
        ps = _beta_sampler()
        chain = np.array([[0.3], [0.4]])  # only 2 samples < default min_samples=50
        with pytest.warns(UserWarning, match="finite samples"):
            updated = update_priors_from_mcmc(ps, chain, ["p"])
        assert updated["p"].params == ps["p"].params

    def test_wrong_chain_ndim_raises(self):
        ps = _beta_sampler()
        with pytest.raises(ValueError, match="2-D"):
            update_priors_from_mcmc(ps, np.array([0.3, 0.4, 0.5]), ["p"])

    def test_column_mismatch_raises(self):
        ps = _beta_sampler()
        chain = np.ones((100, 2))
        with pytest.raises(ValueError, match="columns"):
            update_priors_from_mcmc(ps, chain, ["p"])


# ===========================================================================
# TestApplyCalibrationResult
# ===========================================================================

class TestApplyCalibrationResult:
    def test_apply_map_result(self):
        ps = _beta_sampler()
        map_result = CalibrationResult(
            method="map",
            params={"p": 0.6},
            log_posterior_value=-1.0,
            success=True,
        )
        updated = apply_calibration_result(ps, map_result, concentration=50.0)
        assert updated["p"].params["a"] == pytest.approx(30.0)

    def test_apply_mcmc_result(self):
        ps = _beta_sampler()
        rng = np.random.default_rng(0)
        chain = rng.beta(3.0, 7.0, size=(500, 1))
        mcmc_result = CalibrationResult(
            method="mcmc",
            params={"p": 0.3},
            log_posterior_value=-1.0,
            success=True,
            mcmc_chain=chain,
        )
        updated = apply_calibration_result(ps, mcmc_result)
        new_mean = float(updated["p"].dist.mean())
        assert new_mean == pytest.approx(0.3, abs=0.05)

    def test_apply_unknown_method_raises(self):
        ps = _beta_sampler()
        bad_result = CalibrationResult(
            method="vi",
            params={"p": 0.5},
            log_posterior_value=-1.0,
            success=True,
        )
        with pytest.raises(ValueError, match="Unknown"):
            apply_calibration_result(ps, bad_result)
            