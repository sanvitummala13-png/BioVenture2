"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from bioventure.config import (
    AdoptionConfig,
    BioVentureConfig,
    MarketConfig,
    PhaseTransition,
    RDCompressionConfig,
    SimulationConfig,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestLoadBaseConfig:
    def test_loads_successfully(self, base_config):
        assert base_config.simulation.n_iterations == 50000
        assert base_config.simulation.seed == 42

    def test_simulation_horizon(self, base_config):
        assert base_config.simulation.start_year == 2026
        assert base_config.simulation.end_year == 2035
        assert base_config.simulation.n_steps == 9
        assert len(base_config.simulation.years) == 10
        assert base_config.simulation.years[0] == 2026
        assert base_config.simulation.years[-1] == 2035

    def test_clinical_phases(self, base_config):
        assert len(base_config.clinical.phases) == 4
        expected_cum = 0.52 * 0.29 * 0.58 * 0.90
        assert base_config.clinical.cumulative_probability == pytest.approx(expected_cum, rel=1e-9)

    def test_market_process(self, base_config):
        assert base_config.market.process == "gbm"
        assert base_config.market.process_params["drift"] == pytest.approx(0.12)
        assert base_config.market.process_params["volatility"] == pytest.approx(0.18)

    def test_valuation_discount_rate(self, base_config):
        assert base_config.valuation.discount.discount_rate == pytest.approx(0.125)

    def test_config_hash_deterministic(self):
        cfg1 = BioVentureConfig.from_yaml(CONFIGS_DIR / "base.yaml")
        cfg2 = BioVentureConfig.from_yaml(CONFIGS_DIR / "base.yaml")
        assert cfg1.config_hash() == cfg2.config_hash()
        assert len(cfg1.config_hash()) == 16


class TestValidation:
    def test_invalid_phase_probability_high(self):
        with pytest.raises(ValueError, match="must be in"):
            PhaseTransition(name="bad", probability=1.5)

    def test_invalid_phase_probability_zero(self):
        with pytest.raises(ValueError, match="must be in"):
            PhaseTransition(name="bad", probability=0.0)

    def test_invalid_n_iterations(self):
        with pytest.raises(ValueError, match="n_iterations"):
            SimulationConfig(n_iterations=0, seed=1, start_year=2026, end_year=2035)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="end_year"):
            SimulationConfig(n_iterations=100, seed=1, start_year=2035, end_year=2026)

    def test_invalid_compression_factor(self):
        with pytest.raises(ValueError, match="ai_compression_factor"):
            RDCompressionConfig(
                baseline_timeline_years=12.0,
                baseline_cost_billion_usd=2.6,
                ai_compression_factor=1.5,
                cost_reduction_factor=0.25,
            )

    def test_invalid_adoption_model(self):
        with pytest.raises(ValueError, match="Unknown adoption model"):
            AdoptionConfig(model="logistic", bass_p=0.03, bass_q=0.38, ceiling=0.8,
                           start_fraction=0.05)

    def test_invalid_process_type(self):
        with pytest.raises(ValueError, match="Unknown process"):
            MarketConfig(anchor_2025_billion_usd=55.0, process="ou", process_params={})

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            BioVentureConfig.from_yaml("nonexistent.yaml")


class TestImmutability:
    def test_frozen_simulation(self, base_config):
        with pytest.raises(AttributeError):
            base_config.simulation.n_iterations = 999

    def test_frozen_top_level(self, base_config):
        with pytest.raises(AttributeError):
            base_config.simulation = None
