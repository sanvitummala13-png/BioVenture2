"""Tests for simulation subpackage: sampler, recorder, engine, aggregator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bioventure.config import (
    AdoptionConfig,
    BioVentureConfig,
    ClinicalConfig,
    DiscountConfig,
    MarketConfig,
    PayoffConfig,
    PhaseTransition,
    PortfolioConfig,
    RDCompressionConfig,
    SimulationConfig,
    ValuationConfig,
)
from bioventure.distributions.copulas import IndependentCopula
from bioventure.distributions.priors import PriorSampler, PriorSpec
from bioventure.models.market import MarketResult
from bioventure.models.rd_compression import CompressionResult
from bioventure.rng import RNGFactory
from bioventure.simulation.aggregator import Aggregator, AggregationResult
from bioventure.simulation.engine import SimulationEngine
from bioventure.simulation.recorder import Recorder
from bioventure.simulation.sampler import ParameterSampler
from bioventure.valuation.portfolio import PortfolioResult

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
N_STEPS = 9  # 2026–2035


# ---------------------------------------------------------------------------
# Shared builders (not fixtures — called inside fixtures or tests directly)
# ---------------------------------------------------------------------------

def _make_config(n_iterations: int = 20, seed: int = 42) -> BioVentureConfig:
    return BioVentureConfig(
        simulation=SimulationConfig(
            n_iterations=n_iterations, seed=seed,
            start_year=2026, end_year=2035,
        ),
        clinical=ClinicalConfig(
            phases=(
                PhaseTransition("phase_1_to_2", 0.52),
                PhaseTransition("phase_2_to_3", 0.29),
                PhaseTransition("phase_3_to_approval", 0.58),
                PhaseTransition("approval_to_launch", 0.90),
            )
        ),
        rd_compression=RDCompressionConfig(
            baseline_timeline_years=12.0,
            baseline_cost_billion_usd=2.6,
            ai_compression_factor=0.30,
            cost_reduction_factor=0.25,
        ),
        adoption=AdoptionConfig(
            model="bass", bass_p=0.03, bass_q=0.38,
            ceiling=0.8, start_fraction=0.05,
        ),
        market=MarketConfig(
            anchor_2025_billion_usd=55.0,
            process="gbm",
            process_params={"drift": 0.12, "volatility": 0.18},
        ),
        valuation=ValuationConfig(
            payoff=PayoffConfig(
                distribution="lognormal",
                params={"mu": 0.7, "sigma": 1.2},
            ),
            portfolio=PortfolioConfig(n_bets=10, bet_size_million_usd=50.0),
            discount=DiscountConfig(risk_free_rate=0.045, risk_premium=0.08),
        ),
    )


def _small_prior_sampler() -> PriorSampler:
    """Two-parameter PriorSampler for isolated unit tests."""
    return PriorSampler({
        "alpha": PriorSpec("alpha", "beta", {"a": 2.0, "b": 5.0}),
        "beta_param": PriorSpec("beta_param", "beta", {"a": 3.0, "b": 3.0}),
    })


def _market_result(n_pts: int = N_STEPS + 1) -> MarketResult:
    path = np.linspace(55.0, 80.0, n_pts)
    adoption = np.linspace(0.05, 0.5, n_pts)
    return MarketResult(
        total_market=path,
        adoption_fraction=adoption,
        ai_enabled_market=path * adoption,
        terminal_total=float(path[-1]),
        terminal_ai_enabled=float((path * adoption)[-1]),
    )


def _compression_result() -> CompressionResult:
    return CompressionResult(
        ai_compression_factor=0.30,
        cost_reduction_factor=0.25,
        compressed_timeline_years=8.4,
        compressed_cost_billion_usd=1.95,
        time_saved_years=3.6,
        cost_saved_billion_usd=0.65,
    )


def _portfolio_result() -> PortfolioResult:
    return PortfolioResult(
        exit_multiples=np.array([2.5, 0.0, 5.0, 1.2, 0.8,
                                 3.1, 0.0, 0.4, 7.0, 1.1]),
        total_invested_m=500.0,
        total_returned_m=750.0,
        gross_multiple=1.5,
        net_profit_m=250.0,
        success_rate=0.6,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return RNGFactory(42).spawn_one()


@pytest.fixture
def small_config():
    return _make_config(n_iterations=20)


@pytest.fixture
def sampler_from_yaml():
    return ParameterSampler.from_configs(
        CONFIGS_DIR / "priors.yaml",
        CONFIGS_DIR / "correlations.yaml",
    )


@pytest.fixture
def populated_recorder():
    """Recorder filled with n=5 dummy iterations."""
    param_names = ["alpha", "beta_param"]
    rec = Recorder(n_iterations=5, n_steps=N_STEPS, param_names=param_names)
    for i in range(5):
        rec.record(
            i=i,
            params={"alpha": 0.3 + i * 0.01, "beta_param": 0.5 + i * 0.01},
            cumulative_pos=0.08 + i * 0.002,
            compression=_compression_result(),
            market=_market_result(),
            portfolio=_portfolio_result(),
        )
    return rec


@pytest.fixture
def full_recorder(small_config, sampler_from_yaml):
    """Engine-generated recorder (20 iterations) for aggregation tests."""
    return SimulationEngine(small_config, sampler_from_yaml).run()


@pytest.fixture
def aggregator(small_config):
    val = small_config.valuation
    total_m = val.portfolio.n_bets * val.portfolio.bet_size_million_usd
    return Aggregator(
        horizon_years=float(small_config.simulation.n_steps),
        total_invested_m=total_m,
    )


# ===========================================================================
# TestParameterSampler
# ===========================================================================

class TestParameterSamplerDraw:
    def test_draw_returns_all_keys(self, sampler_from_yaml, rng):
        params = sampler_from_yaml.draw(rng)
        assert set(params.keys()) == set(sampler_from_yaml.parameter_names)

    def test_draw_clinical_in_unit_interval(self, sampler_from_yaml, rng):
        params = sampler_from_yaml.draw(rng)
        for key in (
            "clinical.phase_1_to_2",
            "clinical.phase_2_to_3",
            "clinical.phase_3_to_approval",
            "clinical.approval_to_launch",
        ):
            assert 0.0 < params[key] < 1.0, f"{key} = {params[key]}"

    def test_draw_rd_factors_in_unit_interval(self, sampler_from_yaml, rng):
        params = sampler_from_yaml.draw(rng)
        assert 0.0 < params["rd_compression.ai_compression_factor"] < 1.0
        assert 0.0 < params["rd_compression.cost_reduction_factor"] < 1.0

    def test_draw_adoption_positive(self, sampler_from_yaml, rng):
        params = sampler_from_yaml.draw(rng)
        assert params["adoption.bass_p"] > 0.0
        assert params["adoption.bass_q"] > 0.0

    def test_draw_market_volatility_positive(self, sampler_from_yaml, rng):
        params = sampler_from_yaml.draw(rng)
        assert params["market.volatility"] > 0.0

    def test_draw_determinism(self, sampler_from_yaml):
        p1 = sampler_from_yaml.draw(RNGFactory(99).spawn_one())
        p2 = sampler_from_yaml.draw(RNGFactory(99).spawn_one())
        for key in p1:
            assert p1[key] == pytest.approx(p2[key]), key

    def test_draw_different_seeds_differ(self, sampler_from_yaml):
        p1 = sampler_from_yaml.draw(RNGFactory(1).spawn_one())
        p2 = sampler_from_yaml.draw(RNGFactory(2).spawn_one())
        assert any(p1[k] != p2[k] for k in p1)


class TestParameterSamplerBatch:
    def test_draw_batch_shapes(self, sampler_from_yaml, rng):
        n = 50
        batch = sampler_from_yaml.draw_batch(n, rng)
        assert set(batch.keys()) == set(sampler_from_yaml.parameter_names)
        for name, arr in batch.items():
            assert arr.shape == (n,), f"{name}: expected ({n},), got {arr.shape}"

    def test_draw_batch_determinism(self, sampler_from_yaml):
        b1 = sampler_from_yaml.draw_batch(10, RNGFactory(7).spawn_one())
        b2 = sampler_from_yaml.draw_batch(10, RNGFactory(7).spawn_one())
        for key in b1:
            np.testing.assert_array_equal(b1[key], b2[key])

    def test_draw_batch_clinical_in_unit_interval(self, sampler_from_yaml, rng):
        batch = sampler_from_yaml.draw_batch(200, rng)
        for key in (
            "clinical.phase_1_to_2",
            "clinical.phase_2_to_3",
            "clinical.phase_3_to_approval",
            "clinical.approval_to_launch",
        ):
            assert np.all(batch[key] > 0.0) and np.all(batch[key] < 1.0), key

    def test_draw_batch_n_zero_raises(self, sampler_from_yaml, rng):
        with pytest.raises(ValueError, match="n must be >= 1"):
            sampler_from_yaml.draw_batch(0, rng)


class TestParameterSamplerProperties:
    def test_correlated_names_match_yaml(self, sampler_from_yaml):
        expected = {
            "rd_compression.ai_compression_factor",
            "adoption.bass_p",
            "adoption.bass_q",
            "market.drift",
            "market.volatility",
        }
        assert set(sampler_from_yaml.correlated_names) == expected

    def test_independent_names_not_in_correlated(self, sampler_from_yaml):
        corr = set(sampler_from_yaml.correlated_names)
        for name in sampler_from_yaml.independent_names:
            assert name not in corr

    def test_n_parameters_equals_total(self, sampler_from_yaml):
        total = (
            len(sampler_from_yaml.correlated_names)
            + len(sampler_from_yaml.independent_names)
        )
        assert sampler_from_yaml.n_parameters == total

    def test_from_configs_has_12_params(self):
        s = ParameterSampler.from_configs(
            CONFIGS_DIR / "priors.yaml",
            CONFIGS_DIR / "correlations.yaml",
        )
        assert s.n_parameters == 12

    def test_independent_copula_works(self):
        prior_sampler = _small_prior_sampler()
        copula = IndependentCopula(n_dim=2)
        sampler = ParameterSampler(
            prior_sampler, copula, ["alpha", "beta_param"]
        )
        rng = np.random.default_rng(0)
        params = sampler.draw(rng)
        assert "alpha" in params
        assert "beta_param" in params


class TestParameterSamplerValidation:
    def test_missing_correlated_params_raises(self):
        prior_sampler = _small_prior_sampler()
        copula = IndependentCopula(n_dim=2)
        with pytest.raises(ValueError, match="correlated_params is required"):
            ParameterSampler(prior_sampler, copula, correlated_params=None)

    def test_length_mismatch_raises(self):
        prior_sampler = _small_prior_sampler()
        copula = IndependentCopula(n_dim=2)
        with pytest.raises(ValueError, match="correlated_params has"):
            ParameterSampler(prior_sampler, copula, ["alpha"])

    def test_unknown_param_in_corr_raises(self):
        prior_sampler = _small_prior_sampler()
        copula = IndependentCopula(n_dim=2)
        with pytest.raises(ValueError, match="not found in priors"):
            ParameterSampler(prior_sampler, copula, ["alpha", "nonexistent"])


# ===========================================================================
# TestRecorder
# ===========================================================================

class TestRecorderInit:
    def test_param_arrays_shape(self):
        rec = Recorder(n_iterations=10, n_steps=N_STEPS, param_names=["x", "y"])
        assert rec.params["x"].shape == (10,)
        assert rec.params["y"].shape == (10,)

    def test_scalar_arrays_shape(self):
        rec = Recorder(n_iterations=10, n_steps=N_STEPS, param_names=["x"])
        assert rec.cumulative_pos.shape == (10,)
        assert rec.gross_multiple.shape == (10,)
        assert rec.terminal_ai_enabled.shape == (10,)

    def test_path_arrays_shape(self):
        n_pts = N_STEPS + 1
        rec = Recorder(n_iterations=10, n_steps=N_STEPS, param_names=["x"])
        assert rec.total_market_paths.shape == (10, n_pts)
        assert rec.ai_enabled_paths.shape == (10, n_pts)
        assert rec.adoption_curves.shape == (10, n_pts)

    def test_is_complete_false_initially(self):
        rec = Recorder(n_iterations=5, n_steps=N_STEPS, param_names=["x"])
        assert not rec.is_complete
        assert rec.n_recorded == 0

    def test_invalid_n_iterations_raises(self):
        with pytest.raises(ValueError, match="n_iterations"):
            Recorder(n_iterations=0, n_steps=N_STEPS, param_names=["x"])

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            Recorder(n_iterations=5, n_steps=0, param_names=["x"])


class TestRecorderRecord:
    def test_record_fills_cumulative_pos(self, populated_recorder):
        assert populated_recorder.cumulative_pos[0] == pytest.approx(0.08)
        assert populated_recorder.cumulative_pos[4] == pytest.approx(0.088)

    def test_record_fills_params(self, populated_recorder):
        assert populated_recorder.params["alpha"][0] == pytest.approx(0.30)
        assert populated_recorder.params["alpha"][4] == pytest.approx(0.34)

    def test_record_fills_gross_multiple(self, populated_recorder):
        assert np.all(populated_recorder.gross_multiple == pytest.approx(1.5))

    def test_record_fills_paths(self, populated_recorder):
        assert populated_recorder.total_market_paths.shape == (5, N_STEPS + 1)
        assert populated_recorder.total_market_paths[0, 0] == pytest.approx(55.0)

    def test_n_recorded_after_all(self, populated_recorder):
        assert populated_recorder.n_recorded == 5

    def test_is_complete_after_all(self, populated_recorder):
        assert populated_recorder.is_complete

    def test_record_out_of_range_raises(self):
        rec = Recorder(n_iterations=3, n_steps=N_STEPS, param_names=["x"])
        with pytest.raises(IndexError):
            rec.record(
                i=5,
                params={"x": 0.5},
                cumulative_pos=0.08,
                compression=_compression_result(),
                market=_market_result(),
                portfolio=_portfolio_result(),
            )


class TestRecorderDataFrame:
    def test_to_dataframe_shape(self, populated_recorder):
        df = populated_recorder.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_param_columns_prefixed(self, populated_recorder):
        df = populated_recorder.to_dataframe()
        assert "param_alpha" in df.columns
        assert "param_beta_param" in df.columns

    def test_outcome_columns_present(self, populated_recorder):
        df = populated_recorder.to_dataframe()
        for col in (
            "cumulative_pos", "gross_multiple", "net_profit_m",
            "terminal_ai_enabled", "success_rate",
        ):
            assert col in df.columns, f"Missing: {col}"

    def test_no_path_arrays_in_dataframe(self, populated_recorder):
        df = populated_recorder.to_dataframe()
        for col in df.columns:
            assert "path" not in col


# ===========================================================================
# TestSimulationEngine
# ===========================================================================

class TestSimulationEngineRun:
    def test_run_returns_recorder(self, small_config, sampler_from_yaml):
        recorder = SimulationEngine(small_config, sampler_from_yaml).run()
        assert isinstance(recorder, Recorder)

    def test_run_recorder_is_complete(self, small_config, sampler_from_yaml):
        recorder = SimulationEngine(small_config, sampler_from_yaml).run()
        assert recorder.is_complete

    def test_run_n_iterations_matches_config(self, small_config, sampler_from_yaml):
        recorder = SimulationEngine(small_config, sampler_from_yaml).run()
        assert recorder.n_recorded == small_config.simulation.n_iterations

    def test_run_determinism(self, small_config, sampler_from_yaml):
        r1 = SimulationEngine(small_config, sampler_from_yaml).run()
        r2 = SimulationEngine(small_config, sampler_from_yaml).run()
        np.testing.assert_array_almost_equal(r1.gross_multiple, r2.gross_multiple)
        np.testing.assert_array_almost_equal(
            r1.terminal_ai_enabled, r2.terminal_ai_enabled
        )

    def test_run_different_seeds_differ(self, sampler_from_yaml):
        r1 = SimulationEngine(_make_config(seed=42), sampler_from_yaml).run()
        r2 = SimulationEngine(_make_config(seed=999), sampler_from_yaml).run()
        assert not np.allclose(r1.gross_multiple, r2.gross_multiple)

    def test_progress_callback_called_per_iteration(
        self, small_config, sampler_from_yaml
    ):
        calls: list[int] = []
        SimulationEngine(small_config, sampler_from_yaml).run(
            progress_callback=lambda i, n: calls.append(i)
        )
        assert len(calls) == small_config.simulation.n_iterations
        assert calls[0] == 0
        assert calls[-1] == small_config.simulation.n_iterations - 1

    def test_gross_multiple_non_negative(self, small_config, sampler_from_yaml):
        recorder = SimulationEngine(small_config, sampler_from_yaml).run()
        assert np.all(recorder.gross_multiple >= 0.0)

    def test_terminal_ai_enabled_positive(self, small_config, sampler_from_yaml):
        recorder = SimulationEngine(small_config, sampler_from_yaml).run()
        assert np.all(recorder.terminal_ai_enabled > 0.0)

    def test_cumulative_pos_in_unit_interval(
        self, small_config, sampler_from_yaml
    ):
        recorder = SimulationEngine(small_config, sampler_from_yaml).run()
        assert np.all(recorder.cumulative_pos > 0.0)
        assert np.all(recorder.cumulative_pos < 1.0)

    def test_from_configs_classmethod_builds_engine(self):
        engine = SimulationEngine.from_configs(
            CONFIGS_DIR / "base.yaml",
            CONFIGS_DIR / "priors.yaml",
            CONFIGS_DIR / "correlations.yaml",
        )
        assert engine.n_iterations == 50000


# ===========================================================================
# TestAggregator
# ===========================================================================

class TestAggregatorSummarise:
    def test_returns_aggregation_result(self, full_recorder, aggregator):
        assert isinstance(aggregator.summarise(full_recorder), AggregationResult)

    def test_financial_keys_present(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        for key in (
            "moic_mean", "moic_median", "moic_p5", "moic_p95",
            "var_5", "cvar_5", "probability_of_loss",
        ):
            assert key in result.financial, f"Missing financial key: {key}"

    def test_outcome_keys_present(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        for col in (
            "cumulative_pos", "terminal_ai_enabled", "gross_multiple",
            "net_profit_m", "success_rate", "compressed_timeline",
        ):
            assert col in result.outcomes, f"Missing outcome key: {col}"

    def test_outcome_percentile_stats_present(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        pct = result.outcomes["gross_multiple"]
        for stat in ("p5", "p25", "p50", "p75", "p95", "mean", "std"):
            assert stat in pct, f"Missing stat: {stat}"

    def test_parameter_keys_match_recorder(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        assert set(result.parameters.keys()) == set(full_recorder.param_names)

    def test_market_paths_shape(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        n_pts = N_STEPS + 1
        for key in ("total_market_p50", "ai_enabled_p50", "adoption_p50"):
            assert key in result.market_paths, f"Missing: {key}"
            assert result.market_paths[key].shape == (n_pts,), key

    def test_sensitivity_rho_in_bounds(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        assert "gross_multiple" in result.sensitivity
        for rho in result.sensitivity["gross_multiple"].values():
            assert -1.0 <= rho <= 1.0

    def test_sensitivity_covers_all_params(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        rhos = result.sensitivity["gross_multiple"]
        assert set(rhos.keys()) == set(full_recorder.param_names)

    def test_irr_summary_percentile_keys(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        for stat in ("p5", "p50", "p95", "mean"):
            assert stat in result.irr_summary, f"Missing IRR stat: {stat}"

    def test_n_iterations_in_result(self, full_recorder, aggregator):
        result = aggregator.summarise(full_recorder)
        assert result.n_iterations == 20


class TestAggregatorFlatDict:
    def test_no_arrays_in_flat_dict(self, full_recorder, aggregator):
        flat = aggregator.to_flat_dict(aggregator.summarise(full_recorder))
        for val in flat.values():
            assert not isinstance(val, np.ndarray), f"Array in flat dict"

    def test_flat_dict_financial_keys(self, full_recorder, aggregator):
        flat = aggregator.to_flat_dict(aggregator.summarise(full_recorder))
        assert "financial.moic_mean" in flat
        assert "financial.probability_of_loss" in flat

    def test_flat_dict_n_iterations(self, full_recorder, aggregator):
        flat = aggregator.to_flat_dict(aggregator.summarise(full_recorder))
        assert flat["n_iterations"] == 20


class TestAggregatorValidation:
    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon_years"):
            Aggregator(horizon_years=0.0, total_invested_m=500.0)

    def test_invalid_total_invested_raises(self):
        with pytest.raises(ValueError, match="total_invested_m"):
            Aggregator(horizon_years=9.0, total_invested_m=-1.0)

    def test_partial_recorder_warns(self):
        agg = Aggregator(horizon_years=9.0, total_invested_m=500.0)
        partial = Recorder(n_iterations=5, n_steps=N_STEPS, param_names=["x"])
        with pytest.warns(UserWarning, match="partial"):
            agg.summarise(partial)
            