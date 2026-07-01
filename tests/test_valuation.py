"""Tests for valuation: payoff distributions, portfolio, discounting, metrics."""

import numpy as np
import pytest

from bioventure.rng import RNGFactory
from bioventure.valuation.payoff import (
    LognormalPayoff,
    ParetoPayoff,
    build_payoff,
)
from bioventure.valuation.portfolio import PortfolioModel, PortfolioResult
from bioventure.valuation.discount import DiscountModel
from bioventure.valuation.metrics import (
    compute_summary,
    cvar,
    moic,
    percentile_summary,
    probability_of_loss,
    simple_irr,
    solve_irr,
    var,
)

N = 50_000


# ===========================================================================
# Payoff: Lognormal
# ===========================================================================

class TestLognormalPayoff:
    def test_shape(self):
        p = LognormalPayoff(mu=0.7, sigma=1.2)
        rng = RNGFactory(42).spawn_one()
        samples = p.sample(500, rng)
        assert samples.shape == (500,)

    def test_all_positive(self):
        p = LognormalPayoff(mu=0.7, sigma=1.2)
        rng = RNGFactory(42).spawn_one()
        samples = p.sample(N, rng)
        assert np.all(samples > 0)

    def test_mean_approx(self):
        mu, sigma = 0.7, 1.2
        p = LognormalPayoff(mu=mu, sigma=sigma)
        rng = RNGFactory(42).spawn_one()
        samples = p.sample(N, rng)
        expected = np.exp(mu + 0.5 * sigma ** 2)
        assert np.abs(samples.mean() - expected) / expected < 0.05

    def test_median_approx(self):
        mu, sigma = 0.7, 1.2
        p = LognormalPayoff(mu=mu, sigma=sigma)
        rng = RNGFactory(42).spawn_one()
        samples = p.sample(N, rng)
        expected = np.exp(mu)
        assert np.abs(np.median(samples) - expected) / expected < 0.05

    def test_theoretical_mean(self):
        p = LognormalPayoff(mu=0.5, sigma=0.8)
        assert p.mean() == pytest.approx(np.exp(0.5 + 0.5 * 0.64))

    def test_theoretical_median(self):
        p = LognormalPayoff(mu=0.5, sigma=0.8)
        assert p.median() == pytest.approx(np.exp(0.5))

    def test_determinism(self):
        p = LognormalPayoff(mu=0.7, sigma=1.2)
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        np.testing.assert_array_equal(p.sample(100, r1), p.sample(100, r2))

    def test_sigma_zero(self):
        with pytest.raises(ValueError, match="sigma.*> 0"):
            LognormalPayoff(mu=0.7, sigma=0.0)

    def test_n_zero(self):
        p = LognormalPayoff(mu=0.7, sigma=1.2)
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="n must be >= 1"):
            p.sample(0, rng)


# ===========================================================================
# Payoff: Pareto
# ===========================================================================

class TestParetoPayoff:
    def test_shape(self):
        p = ParetoPayoff(alpha=2.5)
        rng = RNGFactory(42).spawn_one()
        samples = p.sample(500, rng)
        assert samples.shape == (500,)

    def test_all_non_negative(self):
        p = ParetoPayoff(alpha=2.5)
        rng = RNGFactory(42).spawn_one()
        samples = p.sample(N, rng)
        assert np.all(samples >= 0)

    def test_mean_approx(self):
        alpha = 3.0
        p = ParetoPayoff(alpha=alpha)
        rng = RNGFactory(42).spawn_one()
        samples = p.sample(N, rng)
        expected = 1.0 / (alpha - 1.0)
        assert np.abs(samples.mean() - expected) / expected < 0.05

    def test_theoretical_mean(self):
        p = ParetoPayoff(alpha=3.0)
        assert p.mean() == pytest.approx(0.5)

    def test_theoretical_median(self):
        p = ParetoPayoff(alpha=3.0)
        assert p.median() == pytest.approx(2.0 ** (1.0 / 3.0) - 1.0)

    def test_alpha_one(self):
        with pytest.raises(ValueError, match="alpha must be > 1"):
            ParetoPayoff(alpha=1.0)

    def test_alpha_less_than_one(self):
        with pytest.raises(ValueError, match="alpha must be > 1"):
            ParetoPayoff(alpha=0.5)


# ===========================================================================
# Payoff: Factory
# ===========================================================================

class TestBuildPayoff:
    def test_build_lognormal(self):
        p = build_payoff("lognormal", {"mu": 0.7, "sigma": 1.2})
        assert isinstance(p, LognormalPayoff)

    def test_build_pareto(self):
        p = build_payoff("pareto", {"alpha": 2.5})
        assert isinstance(p, ParetoPayoff)

    def test_case_insensitive(self):
        p = build_payoff("LOGNORMAL", {"mu": 0.7, "sigma": 1.2})
        assert isinstance(p, LognormalPayoff)

    def test_unknown_distribution(self):
        with pytest.raises(ValueError, match="Unknown payoff"):
            build_payoff("gaussian", {})

    def test_missing_lognormal_params(self):
        with pytest.raises(ValueError, match="mu.*sigma"):
            build_payoff("lognormal", {"mu": 0.7})

    def test_missing_pareto_params(self):
        with pytest.raises(ValueError, match="alpha"):
            build_payoff("pareto", {})


# ===========================================================================
# Portfolio
# ===========================================================================

def _make_portfolio(n_bets: int = 20) -> PortfolioModel:
    payoff = LognormalPayoff(mu=0.7, sigma=1.2)
    return PortfolioModel(n_bets=n_bets, bet_size_million_usd=50.0, payoff=payoff)


class TestPortfolioSimulate:
    def test_result_type(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        result = port.simulate(rng)
        assert isinstance(result, PortfolioResult)

    def test_exit_multiples_shape(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        result = port.simulate(rng)
        assert result.exit_multiples.shape == (20,)

    def test_total_invested(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        result = port.simulate(rng)
        assert result.total_invested_m == pytest.approx(20 * 50.0)

    def test_gross_multiple_consistent(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        result = port.simulate(rng)
        expected_gm = result.total_returned_m / result.total_invested_m
        assert result.gross_multiple == pytest.approx(expected_gm)

    def test_net_profit_consistent(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        result = port.simulate(rng)
        assert result.net_profit_m == pytest.approx(
            result.total_returned_m - result.total_invested_m
        )

    def test_determinism(self):
        port = _make_portfolio()
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        res1 = port.simulate(r1)
        res2 = port.simulate(r2)
        np.testing.assert_array_equal(res1.exit_multiples, res2.exit_multiples)


class TestPortfolioSuccessProbability:
    def test_zero_success_all_zero(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        result = port.simulate(rng, success_probability=0.0)
        assert np.all(result.exit_multiples == 0)
        assert result.gross_multiple == pytest.approx(0.0)

    def test_full_success_no_zeroing(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        result = port.simulate(rng, success_probability=1.0)
        assert np.all(result.exit_multiples > 0)

    def test_partial_success_reduces_returns(self):
        port = _make_portfolio(n_bets=100)
        rng1 = RNGFactory(42).spawn_one()
        rng2 = RNGFactory(42).spawn_one()
        full = port.simulate(rng1, success_probability=1.0)
        partial = port.simulate(rng2, success_probability=0.10)
        assert partial.gross_multiple < full.gross_multiple

    def test_invalid_probability(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="must be in"):
            port.simulate(rng, success_probability=1.5)


class TestPortfolioBatch:
    def test_batch_shapes(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        batch = port.simulate_batch(200, rng)
        assert batch["exit_multiples"].shape == (200, 20)
        assert batch["gross_multiple"].shape == (200,)
        assert batch["net_profit_m"].shape == (200,)
        assert batch["total_returned_m"].shape == (200,)
        assert batch["success_rate"].shape == (200,)

    def test_batch_with_scalar_prob(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        batch = port.simulate_batch(100, rng, success_probabilities=0.5)
        assert batch["gross_multiple"].shape == (100,)

    def test_batch_with_array_prob(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        probs = np.full(100, 0.10)
        batch = port.simulate_batch(100, rng, success_probabilities=probs)
        assert batch["gross_multiple"].shape == (100,)

    def test_batch_prob_length_mismatch(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="shape"):
            port.simulate_batch(100, rng, success_probabilities=np.ones(50))

    def test_batch_determinism(self):
        port = _make_portfolio()
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        b1 = port.simulate_batch(50, r1)
        b2 = port.simulate_batch(50, r2)
        np.testing.assert_array_equal(b1["gross_multiple"], b2["gross_multiple"])

    def test_batch_n_zero(self):
        port = _make_portfolio()
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="n must be >= 1"):
            port.simulate_batch(0, rng)


class TestPortfolioValidation:
    def test_zero_bets(self):
        with pytest.raises(ValueError, match="n_bets"):
            PortfolioModel(0, 50.0, LognormalPayoff(0.7, 1.2))

    def test_zero_bet_size(self):
        with pytest.raises(ValueError, match="bet_size"):
            PortfolioModel(20, 0.0, LognormalPayoff(0.7, 1.2))


# ===========================================================================
# Discount
# ===========================================================================

class TestDiscountFactor:
    def test_at_zero(self):
        dm = DiscountModel(0.045, 0.08)
        assert dm.discount_factor(0) == pytest.approx(1.0)

    def test_at_one_year(self):
        dm = DiscountModel(0.045, 0.08)
        r = 0.125
        assert dm.discount_factor(1) == pytest.approx(1.0 / (1.0 + r))

    def test_at_ten_years(self):
        dm = DiscountModel(0.045, 0.08)
        r = 0.125
        assert dm.discount_factor(10) == pytest.approx((1.0 + r) ** -10)

    def test_negative_time(self):
        dm = DiscountModel(0.045, 0.08)
        with pytest.raises(ValueError, match="t must be >= 0"):
            dm.discount_factor(-1.0)


class TestDiscountFactors:
    def test_shape(self):
        dm = DiscountModel(0.05, 0.05)
        times = np.arange(11, dtype=float)
        df = dm.discount_factors(times)
        assert df.shape == (11,)

    def test_decreasing(self):
        dm = DiscountModel(0.05, 0.05)
        times = np.arange(11, dtype=float)
        df = dm.discount_factors(times)
        assert np.all(np.diff(df) < 0)

    def test_starts_at_one(self):
        dm = DiscountModel(0.05, 0.05)
        times = np.arange(5, dtype=float)
        df = dm.discount_factors(times)
        assert df[0] == pytest.approx(1.0)


class TestNPV:
    def test_known_npv(self):
        dm = DiscountModel(risk_free_rate=0.10, risk_premium=0.0)
        cashflows = np.array([-100.0, 60.0, 60.0])
        times = np.array([0.0, 1.0, 2.0])
        result = dm.npv(cashflows, times)
        expected = -100.0 + 60.0 / 1.1 + 60.0 / 1.21
        assert result == pytest.approx(expected, rel=1e-10)

    def test_single_cashflow(self):
        dm = DiscountModel(0.05, 0.05)
        result = dm.npv(np.array([1000.0]), np.array([5.0]))
        expected = dm.present_value(1000.0, 5.0)
        assert result == pytest.approx(expected)

    def test_length_mismatch(self):
        dm = DiscountModel(0.05, 0.05)
        with pytest.raises(ValueError, match="lengths must match"):
            dm.npv(np.array([100.0, 200.0]), np.array([1.0]))


class TestRNPV:
    def test_full_probability_equals_npv(self):
        dm = DiscountModel(0.10, 0.0)
        cf = np.array([-100.0, 60.0, 60.0])
        t = np.array([0.0, 1.0, 2.0])
        p = np.ones(3)
        assert dm.rnpv(cf, t, p) == pytest.approx(dm.npv(cf, t))

    def test_zero_probability_is_zero_return(self):
        dm = DiscountModel(0.10, 0.0)
        cf = np.array([100.0, 200.0])
        t = np.array([1.0, 2.0])
        p = np.zeros(2)
        assert dm.rnpv(cf, t, p) == pytest.approx(0.0)

    def test_half_probability(self):
        dm = DiscountModel(0.10, 0.0)
        cf = np.array([100.0])
        t = np.array([1.0])
        p = np.array([0.5])
        expected = 0.5 * 100.0 / 1.1
        assert dm.rnpv(cf, t, p) == pytest.approx(expected)

    def test_invalid_probability(self):
        dm = DiscountModel(0.10, 0.0)
        with pytest.raises(ValueError, match="probabilities must be in"):
            dm.rnpv(np.array([100.0]), np.array([1.0]), np.array([1.5]))


class TestNPVBatch:
    def test_batch_shape(self):
        dm = DiscountModel(0.10, 0.0)
        cf_matrix = np.array([
            [-100.0, 60.0, 60.0],
            [-100.0, 50.0, 80.0],
        ])
        times = np.array([0.0, 1.0, 2.0])
        result = dm.npv_batch(cf_matrix, times)
        assert result.shape == (2,)

    def test_batch_matches_single(self):
        dm = DiscountModel(0.10, 0.0)
        cf = np.array([-100.0, 60.0, 60.0])
        times = np.array([0.0, 1.0, 2.0])
        single = dm.npv(cf, times)
        batch = dm.npv_batch(cf.reshape(1, -1), times)
        assert batch[0] == pytest.approx(single)

    def test_batch_column_mismatch(self):
        dm = DiscountModel(0.10, 0.0)
        with pytest.raises(ValueError, match="columns"):
            dm.npv_batch(np.ones((5, 3)), np.array([1.0, 2.0]))


class TestDiscountValidation:
    def test_negative_rf(self):
        with pytest.raises(ValueError, match="risk_free_rate"):
            DiscountModel(-0.01, 0.08)

    def test_negative_rp(self):
        with pytest.raises(ValueError, match="risk_premium"):
            DiscountModel(0.045, -0.01)

    def test_discount_rate_property(self):
        dm = DiscountModel(0.045, 0.08)
        assert dm.discount_rate == pytest.approx(0.125)

    def test_from_config(self):
        dm = DiscountModel.from_config(0.045, 0.08)
        assert dm.risk_free_rate == pytest.approx(0.045)
        assert dm.risk_premium == pytest.approx(0.08)


# ===========================================================================
# Metrics: IRR
# ===========================================================================

class TestSimpleIRR:
    def test_double_in_ten_years(self):
        irr = simple_irr(
            np.array([2000.0]), total_invested=1000.0, horizon_years=10.0
        )
        assert irr[0] == pytest.approx(2.0 ** 0.1 - 1.0, rel=1e-10)

    def test_break_even(self):
        irr = simple_irr(
            np.array([1000.0]), total_invested=1000.0, horizon_years=5.0
        )
        assert irr[0] == pytest.approx(0.0)

    def test_total_loss(self):
        irr = simple_irr(
            np.array([0.0]), total_invested=1000.0, horizon_years=5.0
        )
        assert irr[0] == pytest.approx(-1.0)

    def test_vectorised(self):
        returns = np.array([500.0, 1000.0, 2000.0])
        irr = simple_irr(returns, total_invested=1000.0, horizon_years=10.0)
        assert irr.shape == (3,)
        assert irr[0] < 0
        assert irr[1] == pytest.approx(0.0)
        assert irr[2] > 0

    def test_invalid_invested(self):
        with pytest.raises(ValueError, match="total_invested"):
            simple_irr(np.array([100.0]), 0.0, 5.0)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="horizon_years"):
            simple_irr(np.array([100.0]), 100.0, 0.0)


class TestSolveIRR:
    def test_matches_simple_irr(self):
        cashflows = np.array([-1000.0, 2000.0])
        times = np.array([0.0, 10.0])
        result = solve_irr(cashflows, times)
        expected = 2.0 ** 0.1 - 1.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_known_npv_zero(self):
        cashflows = np.array([-100.0, 110.0])
        times = np.array([0.0, 1.0])
        result = solve_irr(cashflows, times)
        assert result == pytest.approx(0.10, rel=1e-6)

    def test_no_solution_returns_nan(self):
        cashflows = np.array([100.0, 200.0])
        times = np.array([0.0, 1.0])
        result = solve_irr(cashflows, times)
        assert np.isnan(result)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            solve_irr(np.array([-100.0]), np.array([0.0, 1.0]))


# ===========================================================================
# Metrics: MOIC
# ===========================================================================

class TestMOIC:
    def test_simple(self):
        result = moic(np.array([2000.0, 500.0]), 1000.0)
        np.testing.assert_allclose(result, [2.0, 0.5])

    def test_zero_invested(self):
        with pytest.raises(ValueError, match="total_invested"):
            moic(np.array([100.0]), 0.0)


# ===========================================================================
# Metrics: VaR / CVaR
# ===========================================================================

class TestVaR:
    def test_all_profitable_zero_var(self):
        gm = np.full(1000, 2.0)
        assert var(gm, 0.05) == pytest.approx(0.0)

    def test_all_loss(self):
        gm = np.full(1000, 0.5)
        assert var(gm, 0.05) == pytest.approx(0.5)

    def test_var_positive_for_mixed(self):
        rng = RNGFactory(42).spawn_one()
        gm = rng.lognormal(0.0, 1.0, size=N)
        v = var(gm, 0.05)
        assert v > 0

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            var(np.ones(100), 0.0)

    def test_invalid_alpha_one(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            var(np.ones(100), 1.0)


class TestCVaR:
    def test_all_same(self):
        gm = np.full(1000, 0.6)
        assert cvar(gm, 0.05) == pytest.approx(0.4)

    def test_cvar_ge_var(self):
        rng = RNGFactory(42).spawn_one()
        gm = rng.lognormal(-0.5, 1.5, size=N)
        v = var(gm, 0.05)
        cv = cvar(gm, 0.05)
        assert cv >= v - 1e-10

    def test_all_profitable_zero_cvar(self):
        gm = np.full(1000, 3.0)
        assert cvar(gm, 0.05) == pytest.approx(0.0)


# ===========================================================================
# Metrics: Probability of loss
# ===========================================================================

class TestProbabilityOfLoss:
    def test_no_losses(self):
        gm = np.array([1.5, 2.0, 3.0])
        assert probability_of_loss(gm) == pytest.approx(0.0)

    def test_all_losses(self):
        gm = np.array([0.5, 0.8, 0.9])
        assert probability_of_loss(gm) == pytest.approx(1.0)

    def test_half_losses(self):
        gm = np.array([0.5, 0.8, 1.5, 2.0])
        assert probability_of_loss(gm) == pytest.approx(0.5)

    def test_breakeven_not_loss(self):
        gm = np.array([1.0, 1.0, 1.0])
        assert probability_of_loss(gm) == pytest.approx(0.0)


# ===========================================================================
# Metrics: Percentile summary
# ===========================================================================

class TestPercentileSummary:
    def test_keys(self):
        result = percentile_summary(np.arange(100, dtype=float))
        assert "p5" in result
        assert "p25" in result
        assert "p50" in result
        assert "p75" in result
        assert "p95" in result
        assert "mean" in result
        assert "std" in result

    def test_median_correct(self):
        values = np.arange(1001, dtype=float)
        result = percentile_summary(values)
        assert result["p50"] == pytest.approx(500.0)

    def test_custom_percentiles(self):
        values = np.arange(100, dtype=float)
        result = percentile_summary(values, percentiles=(10, 90))
        assert "p10" in result
        assert "p90" in result
        assert "p50" not in result

    def test_invalid_percentile(self):
        with pytest.raises(ValueError, match="Percentile must be in"):
            percentile_summary(np.ones(10), percentiles=(105,))


# ===========================================================================
# Metrics: Compute summary
# ===========================================================================

class TestComputeSummary:
    def test_all_keys_present(self):
        rng = RNGFactory(42).spawn_one()
        gm = rng.lognormal(0.3, 0.8, size=1000)
        result = compute_summary(gm, total_invested=1000.0, horizon_years=10.0)
        expected_keys = {
            "moic_mean", "moic_std", "moic_p5", "moic_p25", "moic_median",
            "moic_p75", "moic_p95", "irr_mean", "irr_median", "irr_p5",
            "irr_p95", "var_5", "cvar_5", "probability_of_loss",
        }
        assert set(result.keys()) == expected_keys

    def test_moic_median_matches_percentile(self):
        rng = RNGFactory(42).spawn_one()
        gm = rng.lognormal(0.3, 0.8, size=5000)
        result = compute_summary(gm, 1000.0, 10.0)
        assert result["moic_median"] == pytest.approx(
            float(np.percentile(gm, 50))
        )

    def test_loss_probability_consistent(self):
        gm = np.array([0.5, 0.8, 1.2, 1.5, 2.0])
        result = compute_summary(gm, 1000.0, 10.0)
        assert result["probability_of_loss"] == pytest.approx(0.4)
        