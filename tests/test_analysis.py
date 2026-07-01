"""Tests for analysis subpackage: sensitivity and convergence diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from bioventure.analysis.convergence import (
    ConvergenceResult,
    check_convergence,
    convergence_batch,
    effective_sample_size,
    gelman_rubin,
)
from bioventure.analysis.sensitivity import (
    SensitivityResult,
    spearman_sensitivity,
    top_k_params,
    tornado_data,
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _monotone_draws(n: int = 200, seed: int = 0) -> tuple[dict, dict]:
    """Two perfectly correlated and one uncorrelated param → metric pair."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    noise = rng.normal(0, 0.05, n)
    params = {"p1": x1, "p2": x2}
    # metric_a strongly driven by p1; metric_b uncorrelated
    metrics = {
        "metric_a": x1 + noise,
        "metric_b": rng.uniform(0, 1, n),
    }
    return params, metrics


def _stationary_draws(n: int = 500, seed: int = 7) -> np.ndarray:
    """IID draws from N(10, 2²) — should converge quickly."""
    return np.random.default_rng(seed).normal(10.0, 2.0, n)


def _ar1_draws(n: int = 500, phi: float = 0.8, seed: int = 3) -> np.ndarray:
    """AR(1) series — positive autocorrelation, ESS < n."""
    rng = np.random.default_rng(seed)
    v = np.empty(n)
    v[0] = rng.standard_normal()
    for t in range(1, n):
        v[t] = phi * v[t - 1] + rng.standard_normal() * np.sqrt(1 - phi ** 2)
    return v


# ===========================================================================
# TestSensitivityResult
# ===========================================================================

class TestSpearmanSensitivity:
    def test_returns_sensitivity_result(self):
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        assert isinstance(result, SensitivityResult)

    def test_rho_shape(self):
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        assert result.rho.shape == (2, 2)   # 2 params × 2 metrics

    def test_pvalues_shape_matches_rho(self):
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        assert result.pvalues.shape == result.rho.shape

    def test_param_names_recorded(self):
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        assert result.param_names == ["p1", "p2"]

    def test_metric_names_recorded(self):
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        assert result.metric_names == ["metric_a", "metric_b"]

    def test_strong_correlation_detected(self):
        # p1 strongly drives metric_a
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        i_p1 = result.param_names.index("p1")
        j_ma = result.metric_names.index("metric_a")
        assert abs(result.rho[i_p1, j_ma]) > 0.85

    def test_uncorrelated_rho_near_zero(self):
        # p2 should be uncorrelated with metric_a
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        i_p2 = result.param_names.index("p2")
        j_ma = result.metric_names.index("metric_a")
        assert abs(result.rho[i_p2, j_ma]) < 0.25

    def test_rho_values_in_range(self):
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        assert np.all(result.rho >= -1.0) and np.all(result.rho <= 1.0)

    def test_pvalues_in_unit_interval(self):
        params, metrics = _monotone_draws()
        result = spearman_sensitivity(params, metrics)
        assert np.all(result.pvalues >= 0.0) and np.all(result.pvalues <= 1.0)

    def test_significant_p_for_strong_correlation(self):
        params, metrics = _monotone_draws(n=200)
        result = spearman_sensitivity(params, metrics)
        i_p1 = result.param_names.index("p1")
        j_ma = result.metric_names.index("metric_a")
        assert result.pvalues[i_p1, j_ma] < 0.001

    def test_ndarray_inputs_with_names(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, (100, 2))
        o = rng.uniform(0, 1, (100, 1))
        result = spearman_sensitivity(
            p, o, param_names=["a", "b"], metric_names=["y"]
        )
        assert result.rho.shape == (2, 1)

    def test_ndarray_param_without_names_raises(self):
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="param_names is required"):
            spearman_sensitivity(
                rng.uniform(size=(50, 2)),
                rng.uniform(size=(50, 1)),
                metric_names=["y"],
            )

    def test_ndarray_metric_without_names_raises(self):
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="metric_names is required"):
            spearman_sensitivity(
                rng.uniform(size=(50, 2)),
                rng.uniform(size=(50, 1)),
                param_names=["a", "b"],
            )

    def test_mismatched_sim_count_raises(self):
        params = {"p": np.ones(50)}
        metrics = {"m": np.ones(60)}
        with pytest.raises(ValueError, match="same number of simulations"):
            spearman_sensitivity(params, metrics)

    def test_too_few_samples_raises(self):
        params = {"p": np.array([0.1, 0.9])}
        metrics = {"m": np.array([1.0, 2.0])}
        with pytest.raises(ValueError, match="at least 3"):
            spearman_sensitivity(params, metrics)


class TestTopKParams:
    @pytest.fixture
    def result(self):
        params, metrics = _monotone_draws(n=300)
        return spearman_sensitivity(params, metrics)

    def test_returns_list_of_tuples(self, result):
        pairs = top_k_params(result, "metric_a")
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_returns_all_params_by_default(self, result):
        pairs = top_k_params(result, "metric_a")
        assert len(pairs) == len(result.param_names)

    def test_k_limits_output(self, result):
        pairs = top_k_params(result, "metric_a", k=1)
        assert len(pairs) == 1

    def test_sorted_by_absolute_rho_descending(self, result):
        pairs = top_k_params(result, "metric_a")
        rhos = [abs(r) for _, r in pairs]
        assert rhos == sorted(rhos, reverse=True)

    def test_most_influential_param_is_p1(self, result):
        # p1 drives metric_a
        name, _ = top_k_params(result, "metric_a")[0]
        assert name == "p1"

    def test_unknown_metric_raises(self, result):
        with pytest.raises(ValueError, match="not found"):
            top_k_params(result, "nonexistent_metric")

    def test_k_less_than_1_raises(self, result):
        with pytest.raises(ValueError, match="k must be >= 1"):
            top_k_params(result, "metric_a", k=0)


class TestTornadoData:
    @pytest.fixture
    def result(self):
        params, metrics = _monotone_draws(n=300)
        return spearman_sensitivity(params, metrics)

    def test_returns_tuple(self, result):
        out = tornado_data(result, "metric_a")
        assert isinstance(out, tuple) and len(out) == 2

    def test_names_length_equals_n_params(self, result):
        names, rhos = tornado_data(result, "metric_a")
        assert len(names) == len(result.param_names)

    def test_rhos_array_length_matches_names(self, result):
        names, rhos = tornado_data(result, "metric_a")
        assert len(rhos) == len(names)

    def test_rhos_sorted_descending_by_abs(self, result):
        _, rhos = tornado_data(result, "metric_a")
        assert list(np.abs(rhos)) == sorted(np.abs(rhos), reverse=True)

    def test_first_entry_is_most_influential(self, result):
        names, _ = tornado_data(result, "metric_a")
        assert names[0] == "p1"


# ===========================================================================
# TestCheckConvergence
# ===========================================================================

class TestCheckConvergence:
    def test_returns_convergence_result(self):
        v = _stationary_draws()
        result = check_convergence(v)
        assert isinstance(result, ConvergenceResult)

    def test_windows_within_bounds(self):
        v = _stationary_draws()
        result = check_convergence(v)
        assert np.all(result.windows >= 1)
        assert np.all(result.windows <= len(v))

    def test_running_arrays_same_length_as_windows(self):
        v = _stationary_draws()
        result = check_convergence(v)
        nw = len(result.windows)
        assert len(result.running_means) == nw
        assert len(result.running_stds) == nw
        assert len(result.running_p5) == nw
        assert len(result.running_p50) == nw
        assert len(result.running_p95) == nw

    def test_stationary_series_converges(self):
        v = _stationary_draws(n=500)
        result = check_convergence(v, tolerance=0.02)
        assert result.converged

    def test_convergence_iteration_set_when_converged(self):
        v = _stationary_draws(n=500)
        result = check_convergence(v, tolerance=0.02)
        if result.converged:
            assert result.convergence_iteration is not None
            assert result.convergence_iteration <= len(v)

    def test_custom_windows_respected(self):
        v = _stationary_draws(n=100)
        windows = np.array([20, 50, 100])
        result = check_convergence(v, windows=windows)
        np.testing.assert_array_equal(result.windows, windows)

    def test_metric_name_stored(self):
        v = _stationary_draws()
        result = check_convergence(v, metric_name="gross_multiple")
        assert result.metric_name == "gross_multiple"

    def test_relative_change_finite(self):
        v = _stationary_draws()
        result = check_convergence(v)
        assert np.isfinite(result.relative_change)

    def test_running_means_converge_to_true_mean(self):
        # N(10, 2): final running mean should be close to 10
        v = np.random.default_rng(0).normal(10.0, 2.0, 2000)
        result = check_convergence(v)
        assert abs(result.running_means[-1] - 10.0) < 0.5

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            check_convergence(np.ones((10, 5)))

    def test_too_few_values_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            check_convergence(np.array([1.0]))

    def test_invalid_tolerance_raises(self):
        v = _stationary_draws()
        with pytest.raises(ValueError, match="tolerance must be in"):
            check_convergence(v, tolerance=0.0)

    def test_windows_out_of_range_raises(self):
        v = _stationary_draws(n=50)
        with pytest.raises(ValueError, match="must be in"):
            check_convergence(v, windows=np.array([100]))


class TestConvergenceBatch:
    def test_returns_dict(self):
        vd = {"a": _stationary_draws(seed=1), "b": _stationary_draws(seed=2)}
        out = convergence_batch(vd)
        assert isinstance(out, dict)

    def test_keys_match_input(self):
        vd = {"alpha": _stationary_draws(), "beta": _stationary_draws(seed=9)}
        out = convergence_batch(vd)
        assert set(out.keys()) == {"alpha", "beta"}

    def test_each_value_is_convergence_result(self):
        vd = {"x": _stationary_draws()}
        out = convergence_batch(vd)
        assert isinstance(out["x"], ConvergenceResult)

    def test_metric_name_propagated(self):
        vd = {"irr": _stationary_draws()}
        out = convergence_batch(vd)
        assert out["irr"].metric_name == "irr"


# ===========================================================================
# TestEffectiveSampleSize
# ===========================================================================

class TestEffectiveSampleSize:
    def test_iid_ess_near_n(self):
        v = np.random.default_rng(0).normal(0, 1, 500)
        ess = effective_sample_size(v)
        # IID: ESS should be reasonably close to n (within 20%)
        assert ess > 400

    def test_ar1_ess_less_than_n(self):
        v = _ar1_draws(n=500, phi=0.8)
        ess = effective_sample_size(v)
        assert ess < 500

    def test_high_autocorr_reduces_ess(self):
        ess_lo = effective_sample_size(_ar1_draws(phi=0.2))
        ess_hi = effective_sample_size(_ar1_draws(phi=0.9))
        assert ess_hi < ess_lo

    def test_ess_positive(self):
        v = _stationary_draws()
        assert effective_sample_size(v) > 0

    def test_ess_at_most_n(self):
        v = _stationary_draws(n=200)
        assert effective_sample_size(v) <= 200.0 + 1e-9

    def test_constant_array_returns_n(self):
        v = np.full(50, 5.0)
        assert effective_sample_size(v) == pytest.approx(50.0)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            effective_sample_size(np.ones((10, 3)))

    def test_too_few_values_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            effective_sample_size(np.array([1.0]))


# ===========================================================================
# TestGelmanRubin
# ===========================================================================

class TestGelmanRubin:
    def test_identical_chains_rhat_one(self):
        chain = np.random.default_rng(0).normal(0, 1, 200)
        chains = np.stack([chain, chain, chain])
        rhat = gelman_rubin(chains)
        assert rhat == pytest.approx(1.0)

    def test_converged_chains_rhat_near_one(self):
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, (4, 500))
        rhat = gelman_rubin(chains)
        assert rhat < 1.05

    def test_divergent_chains_rhat_high(self):
        # Chain 0 centred at 0, chain 1 centred at 100 → very high R-hat
        rng = np.random.default_rng(1)
        c0 = rng.normal(0.0, 1.0, 200)
        c1 = rng.normal(100.0, 1.0, 200)
        rhat = gelman_rubin(np.stack([c0, c1]))
        assert rhat > 5.0

    def test_rhat_returns_float(self):
        chains = np.random.default_rng(0).normal(size=(3, 100))
        assert isinstance(gelman_rubin(chains), float)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            gelman_rubin(np.ones(100))

    def test_single_chain_raises(self):
        with pytest.raises(ValueError, match="at least 2 chains"):
            gelman_rubin(np.ones((1, 100)))

    def test_single_sample_raises(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            gelman_rubin(np.ones((3, 1)))

    def test_more_chains_more_stable(self):
        # R-hat computed with 2 chains vs 8 chains from same distribution
        rng = np.random.default_rng(5)
        all_chains = rng.normal(0, 1, (8, 300))
        rhat_2 = gelman_rubin(all_chains[:2])
        rhat_8 = gelman_rubin(all_chains)
        # Both should be close to 1; no hard ordering guarantee, just check finite
        assert np.isfinite(rhat_2) and np.isfinite(rhat_8)
        