"""Tests for validation subpackage: backtests, stylized facts, cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

from bioventure.calibration.bayesian import CalibrationTarget
from bioventure.distributions.priors import PriorSampler, PriorSpec
from bioventure.validation.backtests import (
    BacktestResult,
    backtest_coverage,
    coverage_error,
    reliability_diagram_data,
)
from bioventure.validation.cross_validation import (
    CrossValidationResult,
    cross_validate,
    leave_one_out,
)
from bioventure.validation.stylized_facts import (
    StyleResult,
    check_stylized_facts,
    get_fact,
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _gbm_paths(
    n_sims: int = 500,
    n_steps: int = 9,
    s0: float = 55.0,
    mu: float = 0.12,
    sigma: float = 0.18,
    seed: int = 42,
) -> np.ndarray:
    """Exact GBM paths for stylized-fact tests."""
    rng = np.random.default_rng(seed)
    dt = 1.0
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = s0
    z = rng.standard_normal((n_sims, n_steps))
    log_inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_inc, axis=1))
    return paths


def _uniform_sim_paths(
    n_sims: int = 2000,
    n_time: int = 5,
    lo: float = 0.0,
    hi: float = 100.0,
    seed: int = 0,
) -> np.ndarray:
    """Uniform simulated paths: each cell is iid U(lo, hi)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (n_sims, n_time))


def _normal_prior_sampler() -> PriorSampler:
    return PriorSampler({
        "mu": PriorSpec("mu", "normal", {"loc": 0.5, "scale": 0.3})
    })


def _normal_target(n_obs: int = 10, true_mu: float = 0.6) -> CalibrationTarget:
    rng = np.random.default_rng(7)
    observed = rng.normal(true_mu, 0.1, size=n_obs)
    return CalibrationTarget(
        name="normal_cv",
        likelihood_type="normal",
        observed=observed,
        model_fn=lambda params: np.full(n_obs, params["mu"]),
        sigma=0.1,
    )


# ===========================================================================
# TestBacktestCoverage
# ===========================================================================

class TestBacktestCoverage:
    def test_returns_backtest_result(self):
        paths = _uniform_sim_paths()
        obs = np.full(5, 50.0)
        result = backtest_coverage(obs, paths)
        assert isinstance(result, BacktestResult)

    def test_central_obs_covered_at_all_levels(self):
        # Uniform U(0,100); obs=50 should be inside all PIs
        paths = _uniform_sim_paths(n_sims=5000)
        obs = np.full(5, 50.0)
        result = backtest_coverage(obs, paths)
        for label, cov in result.coverage.items():
            assert cov == pytest.approx(1.0), f"{label} coverage={cov}"

    def test_extreme_obs_not_covered(self):
        # Uniform U(0,100); obs=200 should be outside all PIs
        paths = _uniform_sim_paths(n_sims=5000)
        obs = np.full(5, 200.0)
        result = backtest_coverage(obs, paths)
        for label, cov in result.coverage.items():
            assert cov == pytest.approx(0.0), f"{label} coverage={cov}"

    def test_coverage_monotone(self):
        # Higher confidence level → coverage cannot be lower
        rng = np.random.default_rng(1)
        paths = rng.normal(50.0, 10.0, (2000, 8))
        obs = np.linspace(40.0, 60.0, 8)
        result = backtest_coverage(
            obs, paths, confidence_levels=(0.50, 0.80, 0.90, 0.95)
        )
        covs = [result.coverage[f"p{int(cl * 100)}"]
                for cl in (0.50, 0.80, 0.90, 0.95)]
        for i in range(len(covs) - 1):
            assert covs[i] <= covs[i + 1] + 1e-9

    def test_n_observations_recorded(self):
        paths = _uniform_sim_paths()
        obs = np.full(5, 50.0)
        result = backtest_coverage(obs, paths)
        assert result.n_observations == 5

    def test_with_time_indices(self):
        paths = _uniform_sim_paths(n_time=10)
        obs = np.array([50.0, 50.0, 50.0])
        time_idx = np.array([0, 4, 9])
        result = backtest_coverage(obs, paths, time_indices=time_idx)
        assert result.n_observations == 3

    def test_interval_widths_positive(self):
        paths = _uniform_sim_paths()
        obs = np.full(5, 50.0)
        result = backtest_coverage(obs, paths)
        for label, widths in result.interval_widths.items():
            assert np.all(widths >= 0.0), f"{label} has negative widths"

    def test_sharpness_is_p90_mean_width(self):
        paths = _uniform_sim_paths()
        obs = np.full(5, 50.0)
        result = backtest_coverage(obs, paths)
        expected = result.interval_widths["p90"].mean()
        assert result.sharpness == pytest.approx(expected)

    def test_wrong_observed_ndim_raises(self):
        paths = _uniform_sim_paths()
        with pytest.raises(ValueError, match="1-D"):
            backtest_coverage(np.ones((3, 2)), paths)

    def test_wrong_paths_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            backtest_coverage(np.ones(5), np.ones(5))

    def test_n_obs_mismatch_without_indices_raises(self):
        paths = _uniform_sim_paths(n_time=5)
        obs = np.full(3, 50.0)  # 3 != 5
        with pytest.raises(ValueError, match="n_time"):
            backtest_coverage(obs, paths)

    def test_time_indices_out_of_range_raises(self):
        paths = _uniform_sim_paths(n_time=5)
        obs = np.array([50.0])
        with pytest.raises(ValueError, match="time_indices must be in"):
            backtest_coverage(obs, paths, time_indices=np.array([99]))

    def test_invalid_confidence_level_raises(self):
        paths = _uniform_sim_paths()
        obs = np.full(5, 50.0)
        with pytest.raises(ValueError, match="confidence level"):
            backtest_coverage(obs, paths, confidence_levels=(1.5,))


class TestCoverageHelpers:
    @pytest.fixture
    def result_with_perfect_coverage(self):
        paths = _uniform_sim_paths(n_sims=5000)
        obs = np.full(5, 50.0)
        return backtest_coverage(obs, paths, confidence_levels=(0.50, 0.90))

    def test_coverage_error_near_zero_for_central_obs(
        self, result_with_perfect_coverage
    ):
        errors = coverage_error(result_with_perfect_coverage)
        for label, err in errors.items():
            assert err >= -0.05, f"{label} error={err}"

    def test_coverage_error_keys_match_coverage(
        self, result_with_perfect_coverage
    ):
        errors = coverage_error(result_with_perfect_coverage)
        assert set(errors.keys()) == set(
            result_with_perfect_coverage.coverage.keys()
        )

    def test_reliability_diagram_sorted(self, result_with_perfect_coverage):
        nominal, empirical = reliability_diagram_data(
            result_with_perfect_coverage
        )
        assert np.all(np.diff(nominal) > 0)

    def test_reliability_diagram_shape(self, result_with_perfect_coverage):
        nominal, empirical = reliability_diagram_data(
            result_with_perfect_coverage
        )
        n = len(result_with_perfect_coverage.coverage)
        assert nominal.shape == (n,)
        assert empirical.shape == (n,)


# ===========================================================================
# TestStylizedFacts
# ===========================================================================

class TestStylizedFactsGBM:
    @pytest.fixture
    def gbm_result(self):
        paths = _gbm_paths(n_sims=500, n_steps=9)
        return check_stylized_facts(paths)

    def test_returns_style_result(self, gbm_result):
        assert isinstance(gbm_result, StyleResult)

    def test_pass_rate_high_for_gbm(self, gbm_result):
        assert gbm_result.pass_rate >= 0.70

    def test_all_positive_passes_for_gbm(self, gbm_result):
        fact = get_fact(gbm_result, "all_positive")
        assert fact is not None
        assert fact.passed

    def test_terminal_growth_passes_for_gbm(self, gbm_result):
        fact = get_fact(gbm_result, "terminal_growth")
        assert fact is not None
        assert fact.passed

    def test_drift_in_range_passes_for_gbm(self, gbm_result):
        fact = get_fact(gbm_result, "drift_in_range")
        assert fact is not None
        assert fact.passed

    def test_vol_in_range_passes_for_gbm(self, gbm_result):
        fact = get_fact(gbm_result, "vol_in_range")
        assert fact is not None
        assert fact.passed

    def test_positive_drift_passes_for_gbm(self, gbm_result):
        fact = get_fact(gbm_result, "positive_drift")
        assert fact is not None
        assert fact.passed

    def test_n_facts_correct(self, gbm_result):
        assert gbm_result.n_total == len(gbm_result.facts) == 7

    def test_n_passed_plus_failed_eq_total(self, gbm_result):
        n_failed = gbm_result.n_total - gbm_result.n_passed
        assert gbm_result.n_passed + n_failed == gbm_result.n_total

    def test_pass_rate_is_fraction(self, gbm_result):
        assert gbm_result.pass_rate == pytest.approx(
            gbm_result.n_passed / gbm_result.n_total
        )


class TestStylizedFactsEdgeCases:
    def test_all_positive_fails_for_negative_paths(self):
        paths = _gbm_paths(n_sims=100)
        paths[0, 5] = -1.0  # inject a negative value
        result = check_stylized_facts(paths)
        fact = get_fact(result, "all_positive")
        assert fact is not None
        assert not fact.passed

    def test_terminal_growth_fails_for_declining_paths(self):
        # Paths that always decline
        n_sims, n_steps = 100, 9
        paths = np.ones((n_sims, n_steps + 1))
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t - 1] * 0.9
        result = check_stylized_facts(paths)
        fact = get_fact(result, "terminal_growth")
        assert fact is not None
        assert not fact.passed

    def test_get_fact_found(self):
        result = check_stylized_facts(_gbm_paths())
        fact = get_fact(result, "all_positive")
        assert fact is not None
        assert fact.name == "all_positive"

    def test_get_fact_not_found_returns_none(self):
        result = check_stylized_facts(_gbm_paths())
        assert get_fact(result, "nonexistent") is None

    def test_summary_is_string(self):
        result = check_stylized_facts(_gbm_paths())
        s = result.summary()
        assert isinstance(s, str)
        assert "passed" in s.lower()

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            check_stylized_facts(np.ones(10))

    def test_insufficient_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps >= 1"):
            check_stylized_facts(np.ones((10, 1)))

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt must be > 0"):
            check_stylized_facts(_gbm_paths(), dt=-1.0)


# ===========================================================================
# TestCrossValidation
# ===========================================================================

class TestCrossValidate:
    @pytest.fixture
    def cv_result(self):
        ps = _normal_prior_sampler()
        target = _normal_target(n_obs=12, true_mu=0.6)
        return cross_validate(ps, target, n_folds=3, seed=0)

    def test_returns_cv_result(self, cv_result):
        assert isinstance(cv_result, CrossValidationResult)

    def test_n_folds_recorded(self, cv_result):
        assert cv_result.n_folds == 3

    def test_n_completed_equals_n_folds(self, cv_result):
        assert cv_result.n_completed == 3

    def test_fold_results_length(self, cv_result):
        assert len(cv_result.fold_results) == 3

    def test_mean_test_ll_finite(self, cv_result):
        assert np.isfinite(cv_result.mean_test_ll)

    def test_std_test_ll_finite(self, cv_result):
        assert np.isfinite(cv_result.std_test_ll)

    def test_each_fold_has_params(self, cv_result):
        for fold in cv_result.fold_results:
            assert "mu" in fold.params

    def test_each_fold_converged(self, cv_result):
        assert all(f.converged for f in cv_result.fold_results)

    def test_train_test_split_sizes(self, cv_result):
        for fold in cv_result.fold_results:
            assert fold.n_train + fold.n_test == 12

    def test_generalization_gap_is_train_minus_test(self, cv_result):
        assert cv_result.generalization_gap == pytest.approx(
            cv_result.mean_train_ll - cv_result.mean_test_ll
        )

    def test_summary_is_string(self, cv_result):
        s = cv_result.summary()
        assert isinstance(s, str)
        assert "3/3" in s

    def test_n_folds_less_than_2_raises(self):
        ps = _normal_prior_sampler()
        target = _normal_target()
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            cross_validate(ps, target, n_folds=1)

    def test_n_folds_gt_n_obs_raises(self):
        ps = _normal_prior_sampler()
        target = _normal_target(n_obs=5)
        with pytest.raises(ValueError, match="cannot exceed"):
            cross_validate(ps, target, n_folds=10)


class TestLeaveOneOut:
    def test_loo_n_completed_equals_n_obs(self):
        ps = _normal_prior_sampler()
        target = _normal_target(n_obs=5, true_mu=0.6)
        result = leave_one_out(ps, target)
        assert result.n_completed == 5

    def test_loo_fold_test_size_is_one(self):
        ps = _normal_prior_sampler()
        target = _normal_target(n_obs=5, true_mu=0.6)
        result = leave_one_out(ps, target)
        for fold in result.fold_results:
            assert fold.n_test == 1

    def test_loo_train_size_is_n_minus_1(self):
        ps = _normal_prior_sampler()
        target = _normal_target(n_obs=5, true_mu=0.6)
        result = leave_one_out(ps, target)
        for fold in result.fold_results:
            assert fold.n_train == 4
            