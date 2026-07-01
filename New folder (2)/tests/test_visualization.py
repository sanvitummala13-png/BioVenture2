"""Tests for visualization subpackage.

Uses the Agg (non-interactive) backend so tests run headless.
All tests are skipped automatically when matplotlib is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from bioventure.analysis.convergence import check_convergence  # noqa: E402
from bioventure.analysis.sensitivity import spearman_sensitivity  # noqa: E402
from bioventure.visualization.market_plots import (  # noqa: E402
    adoption_curve_plot,
    fan_chart,
    terminal_distribution,
)
from bioventure.visualization.returns_plots import (  # noqa: E402
    convergence_plot,
    irr_distribution,
    moic_distribution,
    reliability_diagram,
)
from bioventure.visualization.sensitivity_plots import (  # noqa: E402
    sensitivity_heatmap,
    tornado_chart,
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test."""
    yield
    plt.close("all")


def _paths(n_sims: int = 100, n_steps: int = 9, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s0 = 50.0
    z = rng.standard_normal((n_sims, n_steps))
    log_inc = 0.08 * 1.0 + 0.15 * z
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_inc, axis=1))
    return paths


def _terminal(n: int = 200, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).lognormal(4.0, 0.5, n)


def _gross_multiples(n: int = 300, seed: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).lognormal(0.5, 0.8, n)


def _irr(n: int = 300, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.18, 0.25, n)


@pytest.fixture
def sensitivity_result():
    rng = np.random.default_rng(7)
    n = 200
    p1 = rng.uniform(0, 1, n)
    p2 = rng.uniform(0, 1, n)
    params = {"adoption_p": p1, "drift_mu": p2}
    metrics = {"terminal_market": p1 + rng.normal(0, 0.05, n),
               "gross_multiple": rng.uniform(0, 5, n)}
    return spearman_sensitivity(params, metrics)


@pytest.fixture
def convergence_result():
    v = np.random.default_rng(0).normal(10.0, 2.0, 300)
    return check_convergence(v, metric_name="gross_multiple")


# ===========================================================================
# TestFanChart
# ===========================================================================

class TestFanChart:
    def test_returns_axes(self):
        ax = fan_chart(_paths())
        assert isinstance(ax, Axes)

    def test_creates_figure_when_no_ax(self):
        ax = fan_chart(_paths())
        assert ax.figure is not None

    def test_accepts_existing_ax(self):
        _, existing = plt.subplots()
        ax = fan_chart(_paths(), ax=existing)
        assert ax is existing

    def test_title_set(self):
        ax = fan_chart(_paths(), title="Total Market")
        assert ax.get_title() == "Total Market"

    def test_xlabel_set(self):
        ax = fan_chart(_paths(), xlabel="Simulation Year")
        assert ax.get_xlabel() == "Simulation Year"

    def test_ylabel_set(self):
        ax = fan_chart(_paths(), ylabel="$B")
        assert ax.get_ylabel() == "$B"

    def test_time_labels_accepted(self):
        paths = _paths()
        labels = np.arange(2025, 2025 + paths.shape[1])
        ax = fan_chart(paths, time_labels=labels)
        assert isinstance(ax, Axes)

    def test_time_labels_wrong_length_raises(self):
        paths = _paths(n_steps=9)
        with pytest.raises(ValueError, match="time_labels length"):
            fan_chart(paths, time_labels=np.arange(5))

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            fan_chart(np.ones(100))

    def test_legend_present(self):
        ax = fan_chart(_paths())
        assert ax.get_legend() is not None


# ===========================================================================
# TestTerminalDistribution
# ===========================================================================

class TestTerminalDistribution:
    def test_returns_axes(self):
        ax = terminal_distribution(_terminal())
        assert isinstance(ax, Axes)

    def test_creates_figure_when_no_ax(self):
        ax = terminal_distribution(_terminal())
        assert ax.figure is not None

    def test_accepts_existing_ax(self):
        _, existing = plt.subplots()
        ax = terminal_distribution(_terminal(), ax=existing)
        assert ax is existing

    def test_title_set(self):
        ax = terminal_distribution(_terminal(), title="Terminal Market")
        assert ax.get_title() == "Terminal Market"

    def test_vlines_accepted(self):
        vl = {"p5": 80.0, "p95": 350.0}
        ax = terminal_distribution(_terminal(), vlines=vl)
        assert ax.get_legend() is not None

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            terminal_distribution(np.ones((50, 2)))


# ===========================================================================
# TestAdoptionCurvePlot
# ===========================================================================

class TestAdoptionCurvePlot:
    def test_returns_axes(self):
        adoption = _paths() / _paths().max()   # normalise to [0,1] range
        ax = adoption_curve_plot(adoption)
        assert isinstance(ax, Axes)

    def test_ylabel_default(self):
        ax = adoption_curve_plot(_paths())
        assert ax.get_ylabel() == "AI Adoption Fraction"


# ===========================================================================
# TestTornadoChart
# ===========================================================================

class TestTornadoChart:
    def test_returns_axes(self, sensitivity_result):
        ax = tornado_chart(sensitivity_result, "terminal_market")
        assert isinstance(ax, Axes)

    def test_creates_figure_when_no_ax(self, sensitivity_result):
        ax = tornado_chart(sensitivity_result, "terminal_market")
        assert ax.figure is not None

    def test_accepts_existing_ax(self, sensitivity_result):
        _, existing = plt.subplots()
        ax = tornado_chart(sensitivity_result, "terminal_market", ax=existing)
        assert ax is existing

    def test_title_defaults_to_metric_name(self, sensitivity_result):
        ax = tornado_chart(sensitivity_result, "terminal_market")
        assert "terminal_market" in ax.get_title()

    def test_custom_title(self, sensitivity_result):
        ax = tornado_chart(sensitivity_result, "terminal_market", title="My Chart")
        assert ax.get_title() == "My Chart"

    def test_k_limits_bars(self, sensitivity_result):
        ax = tornado_chart(sensitivity_result, "terminal_market", k=1)
        # One horizontal bar → one container with one patch
        n_bars = sum(len(c) for c in ax.containers)
        assert n_bars == 1

    def test_unknown_metric_raises(self, sensitivity_result):
        with pytest.raises(ValueError, match="not found"):
            tornado_chart(sensitivity_result, "nonexistent")

    def test_k_zero_raises(self, sensitivity_result):
        with pytest.raises(ValueError, match="k must be >= 1"):
            tornado_chart(sensitivity_result, "terminal_market", k=0)


# ===========================================================================
# TestSensitivityHeatmap
# ===========================================================================

class TestSensitivityHeatmap:
    def test_returns_axes(self, sensitivity_result):
        ax = sensitivity_heatmap(sensitivity_result)
        assert isinstance(ax, Axes)

    def test_creates_figure_when_no_ax(self, sensitivity_result):
        ax = sensitivity_heatmap(sensitivity_result)
        assert ax.figure is not None

    def test_accepts_existing_ax(self, sensitivity_result):
        _, existing = plt.subplots()
        ax = sensitivity_heatmap(sensitivity_result, ax=existing)
        assert ax is existing

    def test_title_set(self, sensitivity_result):
        ax = sensitivity_heatmap(sensitivity_result, title="Rho Matrix")
        assert ax.get_title() == "Rho Matrix"

    def test_annotate_false_accepted(self, sensitivity_result):
        ax = sensitivity_heatmap(sensitivity_result, annotate=False)
        assert isinstance(ax, Axes)


# ===========================================================================
# TestMoicDistribution
# ===========================================================================

class TestMoicDistribution:
    def test_returns_axes(self):
        ax = moic_distribution(_gross_multiples())
        assert isinstance(ax, Axes)

    def test_creates_figure_when_no_ax(self):
        ax = moic_distribution(_gross_multiples())
        assert ax.figure is not None

    def test_accepts_existing_ax(self):
        _, existing = plt.subplots()
        ax = moic_distribution(_gross_multiples(), ax=existing)
        assert ax is existing

    def test_title_default(self):
        ax = moic_distribution(_gross_multiples())
        assert "MOIC" in ax.get_title()

    def test_custom_title(self):
        ax = moic_distribution(_gross_multiples(), title="Returns")
        assert ax.get_title() == "Returns"

    def test_show_var_false_accepted(self):
        ax = moic_distribution(_gross_multiples(), show_var=False)
        assert isinstance(ax, Axes)

    def test_invalid_var_alpha_raises(self):
        with pytest.raises(ValueError, match="var_alpha must be in"):
            moic_distribution(_gross_multiples(), var_alpha=0.0)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            moic_distribution(np.ones((10, 3)))


# ===========================================================================
# TestIrrDistribution
# ===========================================================================

class TestIrrDistribution:
    def test_returns_axes(self):
        ax = irr_distribution(_irr())
        assert isinstance(ax, Axes)

    def test_title_default(self):
        ax = irr_distribution(_irr())
        assert "IRR" in ax.get_title()

    def test_show_zero_false_accepted(self):
        ax = irr_distribution(_irr(), show_zero=False)
        assert isinstance(ax, Axes)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            irr_distribution(np.ones((10, 2)))


# ===========================================================================
# TestConvergencePlot
# ===========================================================================

class TestConvergencePlot:
    def test_returns_axes(self, convergence_result):
        ax = convergence_plot(convergence_result)
        assert isinstance(ax, Axes)

    def test_creates_figure_when_no_ax(self, convergence_result):
        ax = convergence_plot(convergence_result)
        assert ax.figure is not None

    def test_accepts_existing_ax(self, convergence_result):
        _, existing = plt.subplots()
        ax = convergence_plot(convergence_result, ax=existing)
        assert ax is existing

    def test_title_contains_metric_name(self, convergence_result):
        ax = convergence_plot(convergence_result)
        assert convergence_result.metric_name in ax.get_title()

    def test_custom_title(self, convergence_result):
        ax = convergence_plot(convergence_result, title="MC Convergence")
        assert ax.get_title() == "MC Convergence"

    def test_convergence_line_shown_when_converged(self, convergence_result):
        ax = convergence_plot(convergence_result, show_convergence_line=True)
        assert isinstance(ax, Axes)

    def test_legend_present(self, convergence_result):
        ax = convergence_plot(convergence_result)
        assert ax.get_legend() is not None


# ===========================================================================
# TestReliabilityDiagram
# ===========================================================================

class TestReliabilityDiagram:
    @pytest.fixture
    def nom_emp(self):
        nom = np.array([0.50, 0.80, 0.90, 0.95])
        emp = np.array([0.52, 0.79, 0.88, 0.96])
        return nom, emp

    def test_returns_axes(self, nom_emp):
        ax = reliability_diagram(*nom_emp)
        assert isinstance(ax, Axes)

    def test_creates_figure_when_no_ax(self, nom_emp):
        ax = reliability_diagram(*nom_emp)
        assert ax.figure is not None

    def test_accepts_existing_ax(self, nom_emp):
        _, existing = plt.subplots()
        ax = reliability_diagram(*nom_emp, ax=existing)
        assert ax is existing

    def test_title_default(self, nom_emp):
        ax = reliability_diagram(*nom_emp)
        assert "Reliability" in ax.get_title()

    def test_custom_title(self, nom_emp):
        ax = reliability_diagram(*nom_emp, title="Calibration")
        assert ax.get_title() == "Calibration"

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            reliability_diagram(np.ones((3, 2)), np.ones(3))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            reliability_diagram(np.array([0.5, 0.9]), np.array([0.5]))
            