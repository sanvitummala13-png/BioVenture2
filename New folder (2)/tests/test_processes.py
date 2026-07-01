"""Tests for stochastic processes: shapes, moments, determinism, validation, registry."""

import numpy as np
import pytest

from bioventure.processes.base import StochasticProcess
from bioventure.processes.gbm import GeometricBrownianMotion
from bioventure.processes.jump_diffusion import JumpDiffusion
from bioventure.processes.regime_switching import RegimeSwitching
from bioventure.processes.registry import get_process, list_processes, register_process
from bioventure.rng import RNGFactory

N_PATHS = 50_000
S0 = 100.0
T = 1.0
DT = 0.25
N_STEPS = int(T / DT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gbm_params(drift: float = 0.10, volatility: float = 0.20) -> dict:
    return {"drift": drift, "volatility": volatility}


def _jd_params() -> dict:
    return {
        "drift": 0.10,
        "volatility": 0.15,
        "jump_intensity": 0.5,
        "jump_mean": -0.05,
        "jump_std": 0.10,
    }


def _rs_params() -> dict:
    return {
        "n_regimes": 3,
        "regimes": [
            {"name": "expansion", "drift": 0.18, "volatility": 0.12},
            {"name": "normal", "drift": 0.10, "volatility": 0.18},
            {"name": "contraction", "drift": 0.02, "volatility": 0.30},
        ],
        "transition_matrix": [
            [0.90, 0.08, 0.02],
            [0.05, 0.85, 0.10],
            [0.10, 0.15, 0.75],
        ],
        "initial_regime": 0,
    }


# ---------------------------------------------------------------------------
# GBM
# ---------------------------------------------------------------------------

class TestGBMShape:
    def test_evolve_shape(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        path = proc.evolve(S0, T, DT, rng)
        assert path.shape == (N_STEPS + 1,)

    def test_evolve_batch_shape(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, 200, rng)
        assert paths.shape == (200, N_STEPS + 1)

    def test_starts_at_s0(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        path = proc.evolve(S0, T, DT, rng)
        assert path[0] == pytest.approx(S0)

    def test_batch_starts_at_s0(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, 100, rng)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_all_positive(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, 1000, rng)
        assert np.all(paths > 0)


class TestGBMMoments:
    def test_expected_terminal_value(self):
        mu, sigma = 0.10, 0.20
        proc = GeometricBrownianMotion(_gbm_params(mu, sigma))
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, N_PATHS, rng)
        terminal = paths[:, -1]
        expected_mean = S0 * np.exp(mu * T)
        assert np.abs(terminal.mean() - expected_mean) < 1.0

    def test_terminal_variance(self):
        mu, sigma = 0.10, 0.20
        proc = GeometricBrownianMotion(_gbm_params(mu, sigma))
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, N_PATHS, rng)
        terminal = paths[:, -1]
        expected_var = S0 ** 2 * np.exp(2 * mu * T) * (np.exp(sigma ** 2 * T) - 1)
        assert np.abs(terminal.var() - expected_var) / expected_var < 0.05


class TestGBMDeterminism:
    def test_evolve_deterministic(self):
        proc = GeometricBrownianMotion(_gbm_params())
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        p1 = proc.evolve(S0, T, DT, r1)
        p2 = proc.evolve(S0, T, DT, r2)
        np.testing.assert_array_equal(p1, p2)

    def test_evolve_batch_deterministic(self):
        proc = GeometricBrownianMotion(_gbm_params())
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        p1 = proc.evolve_batch(S0, T, DT, 50, r1)
        p2 = proc.evolve_batch(S0, T, DT, 50, r2)
        np.testing.assert_array_equal(p1, p2)


class TestGBMValidation:
    def test_missing_drift(self):
        with pytest.raises(ValueError, match="drift"):
            GeometricBrownianMotion({"volatility": 0.2})

    def test_missing_volatility(self):
        with pytest.raises(ValueError, match="volatility"):
            GeometricBrownianMotion({"drift": 0.1})

    def test_zero_volatility(self):
        with pytest.raises(ValueError, match="volatility.*> 0"):
            GeometricBrownianMotion({"drift": 0.1, "volatility": 0.0})

    def test_s0_zero(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="s0"):
            proc.evolve(0.0, T, DT, rng)

    def test_negative_t(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="t must be > 0"):
            proc.evolve(S0, -1.0, DT, rng)

    def test_dt_exceeds_t(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="dt.*<= t"):
            proc.evolve(S0, 0.5, 1.0, rng)

    def test_batch_n_paths_zero(self):
        proc = GeometricBrownianMotion(_gbm_params())
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="n_paths"):
            proc.evolve_batch(S0, T, DT, 0, rng)

    def test_params_returns_copy(self):
        proc = GeometricBrownianMotion(_gbm_params())
        p = proc.params
        p["drift"] = 999
        assert proc.drift == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Jump Diffusion
# ---------------------------------------------------------------------------

class TestJumpDiffusionShape:
    def test_evolve_shape(self):
        proc = JumpDiffusion(_jd_params())
        rng = RNGFactory(42).spawn_one()
        path = proc.evolve(S0, T, DT, rng)
        assert path.shape == (N_STEPS + 1,)

    def test_evolve_batch_shape(self):
        proc = JumpDiffusion(_jd_params())
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, 200, rng)
        assert paths.shape == (200, N_STEPS + 1)

    def test_starts_at_s0(self):
        proc = JumpDiffusion(_jd_params())
        rng = RNGFactory(42).spawn_one()
        path = proc.evolve(S0, T, DT, rng)
        assert path[0] == pytest.approx(S0)

    def test_all_positive(self):
        proc = JumpDiffusion(_jd_params())
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, 1000, rng)
        assert np.all(paths > 0)


class TestJumpDiffusionCompensator:
    def test_compensator_value(self):
        proc = JumpDiffusion(_jd_params())
        jm, js = proc.jump_mean, proc.jump_std
        expected = np.exp(jm + 0.5 * js ** 2) - 1.0
        assert proc.compensator == pytest.approx(expected)

    def test_zero_intensity_matches_gbm(self):
        """With λ=0, jump-diffusion should produce identical paths to GBM."""
        params = {
            "drift": 0.10,
            "volatility": 0.20,
            "jump_intensity": 0.0,
            "jump_mean": 0.0,
            "jump_std": 0.01,
        }
        jd = JumpDiffusion(params)
        gbm = GeometricBrownianMotion({"drift": 0.10, "volatility": 0.20})

        rng_jd = RNGFactory(42).spawn_one()
        rng_gbm = RNGFactory(42).spawn_one()

        path_jd = jd.evolve(S0, T, DT, rng_jd)
        path_gbm = gbm.evolve(S0, T, DT, rng_gbm)

        np.testing.assert_allclose(path_jd, path_gbm, rtol=1e-12)


class TestJumpDiffusionDeterminism:
    def test_evolve_deterministic(self):
        proc = JumpDiffusion(_jd_params())
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        p1 = proc.evolve(S0, T, DT, r1)
        p2 = proc.evolve(S0, T, DT, r2)
        np.testing.assert_array_equal(p1, p2)

    def test_evolve_batch_deterministic(self):
        proc = JumpDiffusion(_jd_params())
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        p1 = proc.evolve_batch(S0, T, DT, 50, r1)
        p2 = proc.evolve_batch(S0, T, DT, 50, r2)
        np.testing.assert_array_equal(p1, p2)


class TestJumpDiffusionValidation:
    def test_missing_jump_intensity(self):
        params = _jd_params()
        del params["jump_intensity"]
        with pytest.raises(ValueError, match="jump_intensity"):
            JumpDiffusion(params)

    def test_negative_jump_std(self):
        params = _jd_params()
        params["jump_std"] = -0.1
        with pytest.raises(ValueError, match="jump_std.*> 0"):
            JumpDiffusion(params)

    def test_negative_jump_intensity(self):
        params = _jd_params()
        params["jump_intensity"] = -1.0
        with pytest.raises(ValueError, match="jump_intensity.*>= 0"):
            JumpDiffusion(params)


# ---------------------------------------------------------------------------
# Regime Switching
# ---------------------------------------------------------------------------

class TestRegimeSwitchingShape:
    def test_evolve_shape(self):
        proc = RegimeSwitching(_rs_params())
        rng = RNGFactory(42).spawn_one()
        path = proc.evolve(S0, T, DT, rng)
        assert path.shape == (N_STEPS + 1,)

    def test_evolve_batch_shape(self):
        proc = RegimeSwitching(_rs_params())
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, 200, rng)
        assert paths.shape == (200, N_STEPS + 1)

    def test_starts_at_s0(self):
        proc = RegimeSwitching(_rs_params())
        rng = RNGFactory(42).spawn_one()
        path = proc.evolve(S0, T, DT, rng)
        assert path[0] == pytest.approx(S0)

    def test_all_positive(self):
        proc = RegimeSwitching(_rs_params())
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, 1000, rng)
        assert np.all(paths > 0)


class TestRegimeSwitchingIdenticalRegimes:
    def test_identical_regimes_match_gbm_mean(self):
        """When all regimes have the same parameters, the terminal mean
        should match the GBM theoretical mean."""
        mu, sigma = 0.10, 0.20
        params = {
            "n_regimes": 2,
            "regimes": [
                {"drift": mu, "volatility": sigma},
                {"drift": mu, "volatility": sigma},
            ],
            "transition_matrix": [[0.5, 0.5], [0.5, 0.5]],
            "initial_regime": 0,
        }
        proc = RegimeSwitching(params)
        rng = RNGFactory(42).spawn_one()
        paths = proc.evolve_batch(S0, T, DT, N_PATHS, rng)
        terminal = paths[:, -1]
        expected_mean = S0 * np.exp(mu * T)
        assert np.abs(terminal.mean() - expected_mean) < 1.5


class TestRegimeSwitchingDeterminism:
    def test_evolve_deterministic(self):
        proc = RegimeSwitching(_rs_params())
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        p1 = proc.evolve(S0, T, DT, r1)
        p2 = proc.evolve(S0, T, DT, r2)
        np.testing.assert_array_equal(p1, p2)

    def test_evolve_batch_deterministic(self):
        proc = RegimeSwitching(_rs_params())
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        p1 = proc.evolve_batch(S0, T, DT, 50, r1)
        p2 = proc.evolve_batch(S0, T, DT, 50, r2)
        np.testing.assert_array_equal(p1, p2)


class TestRegimeSwitchingProperties:
    def test_n_regimes(self):
        proc = RegimeSwitching(_rs_params())
        assert proc.n_regimes == 3

    def test_initial_regime(self):
        proc = RegimeSwitching(_rs_params())
        assert proc.initial_regime == 0

    def test_transition_matrix_copy(self):
        proc = RegimeSwitching(_rs_params())
        tm = proc.transition_matrix
        tm[0, 0] = 999.0
        assert proc.transition_matrix[0, 0] == pytest.approx(0.90)


class TestRegimeSwitchingValidation:
    def test_missing_regimes(self):
        params = _rs_params()
        del params["regimes"]
        with pytest.raises(ValueError, match="regimes"):
            RegimeSwitching(params)

    def test_wrong_regime_count(self):
        params = _rs_params()
        params["regimes"] = params["regimes"][:2]
        with pytest.raises(ValueError, match="Expected 3 regimes"):
            RegimeSwitching(params)

    def test_regime_missing_volatility(self):
        params = _rs_params()
        del params["regimes"][1]["volatility"]
        with pytest.raises(ValueError, match="must have 'drift' and 'volatility'"):
            RegimeSwitching(params)

    def test_regime_zero_volatility(self):
        params = _rs_params()
        params["regimes"][0]["volatility"] = 0.0
        with pytest.raises(ValueError, match="volatility must be > 0"):
            RegimeSwitching(params)

    def test_transition_matrix_wrong_shape(self):
        params = _rs_params()
        params["transition_matrix"] = [[0.5, 0.5], [0.5, 0.5]]
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            RegimeSwitching(params)

    def test_transition_matrix_rows_not_sum_one(self):
        params = _rs_params()
        params["transition_matrix"][0] = [0.80, 0.08, 0.02]
        with pytest.raises(ValueError, match="rows must sum to 1"):
            RegimeSwitching(params)

    def test_transition_matrix_negative_entry(self):
        params = _rs_params()
        params["transition_matrix"][0] = [1.10, -0.08, -0.02]
        with pytest.raises(ValueError, match="must be >= 0"):
            RegimeSwitching(params)

    def test_initial_regime_out_of_range(self):
        params = _rs_params()
        params["initial_regime"] = 5
        with pytest.raises(ValueError, match="initial_regime"):
            RegimeSwitching(params)

    def test_n_regimes_too_small(self):
        params = {
            "n_regimes": 1,
            "regimes": [{"drift": 0.1, "volatility": 0.2}],
            "transition_matrix": [[1.0]],
            "initial_regime": 0,
        }
        with pytest.raises(ValueError, match="n_regimes must be >= 2"):
            RegimeSwitching(params)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_processes(self):
        names = list_processes()
        assert "gbm" in names
        assert "jump_diffusion" in names
        assert "regime_switching" in names

    def test_get_gbm(self):
        proc = get_process("gbm", _gbm_params())
        assert isinstance(proc, GeometricBrownianMotion)

    def test_get_jump_diffusion(self):
        proc = get_process("jump_diffusion", _jd_params())
        assert isinstance(proc, JumpDiffusion)

    def test_get_regime_switching(self):
        proc = get_process("regime_switching", _rs_params())
        assert isinstance(proc, RegimeSwitching)

    def test_case_insensitive(self):
        proc = get_process("GBM", _gbm_params())
        assert isinstance(proc, GeometricBrownianMotion)

    def test_strips_whitespace(self):
        proc = get_process("  gbm  ", _gbm_params())
        assert isinstance(proc, GeometricBrownianMotion)

    def test_unknown_process(self):
        with pytest.raises(ValueError, match="Unknown process"):
            get_process("ornstein_uhlenbeck", {})

    def test_get_returns_functional_process(self):
        proc = get_process("gbm", _gbm_params())
        rng = RNGFactory(42).spawn_one()
        path = proc.evolve(S0, T, DT, rng)
        assert path.shape == (N_STEPS + 1,)
        assert path[0] == pytest.approx(S0)


class TestRegisterCustomProcess:
    def test_register_and_retrieve(self):
        class DummyProcess(StochasticProcess):
            def _validate(self) -> None:
                pass

            def evolve(self, s0, t, dt, rng):
                n = self._check_evolve_inputs(s0, t, dt)
                return np.full(n + 1, s0)

            def evolve_batch(self, s0, t, dt, n_paths, rng):
                n = self._check_evolve_inputs(s0, t, dt)
                return np.full((n_paths, n + 1), s0)

        register_process("dummy_test", DummyProcess)
        proc = get_process("dummy_test", {})
        assert isinstance(proc, DummyProcess)

        rng = RNGFactory(42).spawn_one()
        path = proc.evolve(S0, T, DT, rng)
        np.testing.assert_allclose(path, S0)

    def test_register_non_subclass(self):
        with pytest.raises(TypeError, match="StochasticProcess subclass"):
            register_process("bad", dict)

    def test_register_duplicate(self):
        with pytest.raises(ValueError, match="already registered"):
            register_process("gbm", GeometricBrownianMotion)
            