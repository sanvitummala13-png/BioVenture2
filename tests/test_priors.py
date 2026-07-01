"""Tests for prior distribution sampling: shapes, bounds, determinism, validation."""

from pathlib import Path

import numpy as np
import pytest

from bioventure.distributions.priors import PriorSampler, PriorSpec, _build_distribution
from bioventure.rng import RNGFactory

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
N = 10_000


# ---------------------------------------------------------------------------
# PriorSpec: individual distribution tests
# ---------------------------------------------------------------------------

class TestPriorSpecNormal:
    def test_shape(self):
        spec = PriorSpec("x", "normal", {"loc": 0.0, "scale": 1.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert samples.shape == (N,)

    def test_mean_approx(self):
        spec = PriorSpec("x", "normal", {"loc": 5.0, "scale": 0.5})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.abs(samples.mean() - 5.0) < 0.05

    def test_scalar(self):
        spec = PriorSpec("x", "normal", {"loc": 0.0, "scale": 1.0})
        rng = RNGFactory(42).spawn_one()
        val = spec.sample_scalar(rng)
        assert isinstance(val, float)


class TestPriorSpecLognormal:
    def test_all_positive(self):
        spec = PriorSpec("x", "lognormal", {"mu": 0.0, "sigma": 1.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.all(samples > 0)

    def test_median_approx(self):
        spec = PriorSpec("x", "lognormal", {"mu": -3.51, "sigma": 0.5})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.abs(np.median(samples) - np.exp(-3.51)) < 0.01


class TestPriorSpecBeta:
    def test_bounds(self):
        spec = PriorSpec("x", "beta", {"a": 2.0, "b": 5.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_mean_approx(self):
        a, b = 52.0, 48.0
        spec = PriorSpec("x", "beta", {"a": a, "b": b})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.abs(samples.mean() - a / (a + b)) < 0.01


class TestPriorSpecTriangular:
    def test_bounds(self):
        spec = PriorSpec("x", "triangular", {"left": 1.0, "mode": 3.0, "right": 5.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.all(samples >= 1.0)
        assert np.all(samples <= 5.0)

    def test_mode_near_peak(self):
        spec = PriorSpec("x", "triangular", {"left": 0.0, "mode": 0.5, "right": 1.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        hist, edges = np.histogram(samples, bins=10)
        peak_bin = hist.argmax()
        peak_center = (edges[peak_bin] + edges[peak_bin + 1]) / 2
        assert np.abs(peak_center - 0.5) < 0.15


class TestPriorSpecBernoulli:
    def test_values_binary(self):
        spec = PriorSpec("x", "bernoulli", {"p": 0.7})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert set(np.unique(samples)).issubset({0.0, 1.0})

    def test_mean_approx(self):
        spec = PriorSpec("x", "bernoulli", {"p": 0.3})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.abs(samples.mean() - 0.3) < 0.03


class TestPriorSpecPoisson:
    def test_non_negative_integer(self):
        spec = PriorSpec("x", "poisson", {"mu": 4.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.all(samples >= 0)
        assert np.all(samples == np.floor(samples))

    def test_mean_approx(self):
        spec = PriorSpec("x", "poisson", {"mu": 7.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.abs(samples.mean() - 7.0) < 0.2


class TestPriorSpecUniform:
    def test_bounds(self):
        spec = PriorSpec("x", "uniform", {"low": 2.0, "high": 8.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 8.0)

    def test_mean_approx(self):
        spec = PriorSpec("x", "uniform", {"low": 0.0, "high": 10.0})
        rng = RNGFactory(42).spawn_one()
        samples = spec.sample(N, rng)
        assert np.abs(samples.mean() - 5.0) < 0.15


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_samples(self):
        spec = PriorSpec("x", "beta", {"a": 3.0, "b": 7.0})
        r1 = RNGFactory(99).spawn_one()
        r2 = RNGFactory(99).spawn_one()
        s1 = spec.sample(100, r1)
        s2 = spec.sample(100, r2)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seed_different_samples(self):
        spec = PriorSpec("x", "beta", {"a": 3.0, "b": 7.0})
        r1 = RNGFactory(99).spawn_one()
        r2 = RNGFactory(100).spawn_one()
        s1 = spec.sample(100, r1)
        s2 = spec.sample(100, r2)
        assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_family(self):
        with pytest.raises(ValueError, match="Unknown distribution family"):
            PriorSpec("x", "cauchy", {"loc": 0, "scale": 1})

    def test_missing_params(self):
        with pytest.raises(ValueError, match="missing"):
            PriorSpec("x", "normal", {"loc": 0.0})

    def test_normal_scale_zero(self):
        with pytest.raises(ValueError, match="scale"):
            PriorSpec("x", "normal", {"loc": 0.0, "scale": 0.0})

    def test_lognormal_sigma_negative(self):
        with pytest.raises(ValueError, match="sigma"):
            PriorSpec("x", "lognormal", {"mu": 0.0, "sigma": -1.0})

    def test_beta_a_zero(self):
        with pytest.raises(ValueError, match="must be > 0"):
            PriorSpec("x", "beta", {"a": 0.0, "b": 1.0})

    def test_triangular_wrong_order(self):
        with pytest.raises(ValueError, match="left <= mode <= right"):
            PriorSpec("x", "triangular", {"left": 5.0, "mode": 2.0, "right": 8.0})

    def test_triangular_degenerate(self):
        with pytest.raises(ValueError, match="left < right"):
            PriorSpec("x", "triangular", {"left": 3.0, "mode": 3.0, "right": 3.0})

    def test_bernoulli_p_out_of_range(self):
        with pytest.raises(ValueError, match="must be in"):
            PriorSpec("x", "bernoulli", {"p": 1.5})

    def test_poisson_mu_negative(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            PriorSpec("x", "poisson", {"mu": -1.0})

    def test_uniform_low_ge_high(self):
        with pytest.raises(ValueError, match="low < high"):
            PriorSpec("x", "uniform", {"low": 5.0, "high": 5.0})

    def test_sample_size_zero(self):
        spec = PriorSpec("x", "normal", {"loc": 0.0, "scale": 1.0})
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="must be >= 1"):
            spec.sample(0, rng)


# ---------------------------------------------------------------------------
# PriorSampler: batch loading and sampling
# ---------------------------------------------------------------------------

class TestPriorSamplerFromYaml:
    def test_loads_priors_yaml(self):
        sampler = PriorSampler.from_yaml(CONFIGS_DIR / "priors.yaml")
        assert len(sampler) > 0
        assert "clinical.phase_1_to_2" in sampler.names
        assert "market.drift" in sampler.names

    def test_dotted_names(self):
        sampler = PriorSampler.from_yaml(CONFIGS_DIR / "priors.yaml")
        for name in sampler.names:
            assert "." in name, f"Expected dotted name, got '{name}'"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PriorSampler.from_yaml("nonexistent.yaml")


class TestPriorSamplerBatch:
    def test_sample_shapes(self):
        sampler = PriorSampler.from_yaml(CONFIGS_DIR / "priors.yaml")
        rng = RNGFactory(42).spawn_one()
        draws = sampler.sample(500, rng)
        for name, arr in draws.items():
            assert arr.shape == (500,), f"{name} has wrong shape: {arr.shape}"

    def test_sample_scalar_types(self):
        sampler = PriorSampler.from_yaml(CONFIGS_DIR / "priors.yaml")
        rng = RNGFactory(42).spawn_one()
        draws = sampler.sample_scalar(rng)
        for name, val in draws.items():
            assert isinstance(val, float), f"{name} is {type(val)}, expected float"

    def test_batch_determinism(self):
        sampler = PriorSampler.from_yaml(CONFIGS_DIR / "priors.yaml")
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        d1 = sampler.sample(200, r1)
        d2 = sampler.sample(200, r2)
        for name in sampler.names:
            np.testing.assert_array_equal(d1[name], d2[name])

    def test_getitem(self):
        sampler = PriorSampler.from_yaml(CONFIGS_DIR / "priors.yaml")
        spec = sampler["clinical.phase_1_to_2"]
        assert isinstance(spec, PriorSpec)
        assert spec.family == "beta"

    def test_getitem_missing(self):
        sampler = PriorSampler.from_yaml(CONFIGS_DIR / "priors.yaml")
        with pytest.raises(KeyError, match="Unknown prior"):
            sampler["nonexistent.param"]


class TestPriorSamplerValidation:
    def test_empty_specs_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            PriorSampler({})

    def test_malformed_yaml_no_family(self):
        raw = {"group": {"param": {"params": {"a": 1}}}}
        with pytest.raises(ValueError, match="must have 'family'"):
            PriorSampler._from_dict(raw)

    def test_malformed_yaml_non_dict_entry(self):
        raw = {"group": {"param": "not_a_dict"}}
        with pytest.raises(ValueError, match="Expected dict"):
            PriorSampler._from_dict(raw)

    def test_malformed_yaml_non_dict_category(self):
        raw = {"group": "not_a_dict"}
        with pytest.raises(ValueError, match="Expected dict"):
            PriorSampler._from_dict(raw)
            