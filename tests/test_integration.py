"""End-to-end integration tests for the full BioVenture simulation pipeline.

Verifies that all subpackages compose correctly: engine → aggregation →
analysis → IO → reports.  Uses the ``full_recorder`` fixture from
conftest.py (20 iterations, seeded) for most tests.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCALAR_FIELDS = (
    "cumulative_pos",
    "compressed_timeline",
    "compressed_cost",
    "terminal_total_market",
    "terminal_ai_enabled",
    "gross_multiple",
    "net_profit_m",
    "success_rate",
)

_PATH_FIELDS = (
    "total_market_paths",
    "ai_enabled_paths",
    "adoption_curves",
)

_FINANCIAL_KEYS = (
    "moic_mean", "moic_median", "moic_p5", "moic_p95",
    "irr_mean", "irr_median", "var_5", "cvar_5",
    "probability_of_loss",
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_recorder(two_recorders):
    r1, _ = two_recorders
    return r1


@pytest.fixture(scope="module")
def agg_result(two_recorder):
    from bioventure.simulation import Aggregator
    return Aggregator().summarise(two_recorder)


@pytest.fixture(scope="module")
def two_recorders():
    """Run engine twice from the same config+seed → must match exactly."""
    try:
        conftest = importlib.import_module("conftest")
        cfg = conftest._make_config(n_iterations=30)
    except (ImportError, AttributeError, TypeError):
        pytest.skip("_make_config not accessible from conftest")

    from bioventure.simulation import SimulationEngine
    r1 = SimulationEngine(cfg).run()
    r2 = SimulationEngine(cfg).run()
    return r1, r2


# ===========================================================================
# TestRecorderValidity
# ===========================================================================

class TestRecorderValidity:
    """All scalar outputs must be finite and satisfy domain invariants."""

    @pytest.mark.parametrize("field", _SCALAR_FIELDS)
    def test_scalar_field_finite(self, full_recorder, field):
        arr = getattr(full_recorder, field)
        assert np.all(np.isfinite(arr)), f"{field} contains non-finite values"

    @pytest.mark.parametrize("field", _PATH_FIELDS)
    def test_path_field_finite(self, full_recorder, field):
        arr = getattr(full_recorder, field)
        assert np.all(np.isfinite(arr)), f"{field} contains non-finite values"

    def test_cumulative_pos_in_unit_interval(self, full_recorder):
        cp = full_recorder.cumulative_pos
        assert np.all(cp >= 0.0) and np.all(cp <= 1.0)

    def test_success_rate_in_unit_interval(self, full_recorder):
        sr = full_recorder.success_rate
        assert np.all(sr >= 0.0) and np.all(sr <= 1.0)

    def test_terminal_total_market_positive(self, full_recorder):
        assert np.all(full_recorder.terminal_total_market > 0)

    def test_terminal_ai_enabled_positive(self, full_recorder):
        assert np.all(full_recorder.terminal_ai_enabled > 0)

    def test_terminal_ai_enabled_le_total(self, full_recorder):
        assert np.all(
            full_recorder.terminal_ai_enabled
            <= full_recorder.terminal_total_market + 1e-6
        )

    def test_market_paths_positive(self, full_recorder):
        assert np.all(full_recorder.total_market_paths > 0)

    def test_ai_enabled_paths_positive(self, full_recorder):
        assert np.all(full_recorder.ai_enabled_paths > 0)

    def test_adoption_curves_in_unit_interval(self, full_recorder):
        ac = full_recorder.adoption_curves
        assert np.all(ac >= 0.0) and np.all(ac <= 1.0 + 1e-9)

    def test_adoption_curves_monotone_non_decreasing(self, full_recorder):
        diffs = np.diff(full_recorder.adoption_curves, axis=1)
        assert np.all(diffs >= -1e-9)

    def test_dataframe_has_all_scalar_columns(self, full_recorder):
        df = full_recorder.to_dataframe()
        for field in _SCALAR_FIELDS:
            assert field in df.columns, f"Missing column: {field}"

    def test_dataframe_has_no_nans(self, full_recorder):
        df = full_recorder.to_dataframe()
        assert not df.isnull().any().any()

    def test_dataframe_row_count_matches_n_iterations(self, full_recorder):
        df = full_recorder.to_dataframe()
        n = len(full_recorder.gross_multiple)
        assert len(df) == n


# ===========================================================================
# TestAggregationIntegration
# ===========================================================================

class TestAggregationIntegration:
    def test_result_has_financial_dict(self, agg_result):
        assert isinstance(agg_result.financial, dict)

    @pytest.mark.parametrize("key", _FINANCIAL_KEYS)
    def test_financial_key_present(self, agg_result, key):
        assert key in agg_result.financial

    @pytest.mark.parametrize("key", _FINANCIAL_KEYS)
    def test_financial_key_finite(self, agg_result, key):
        assert np.isfinite(agg_result.financial[key])

    def test_probability_of_loss_in_unit_interval(self, agg_result):
        pol = agg_result.financial["probability_of_loss"]
        assert 0.0 <= pol <= 1.0

    def test_outcomes_has_scalar_fields(self, agg_result):
        for field in _SCALAR_FIELDS:
            assert field in agg_result.outcomes, f"Missing outcome: {field}"

    def test_each_outcome_has_percentile_keys(self, agg_result):
        for field, pct_dict in agg_result.outcomes.items():
            for key in ("p5", "p50", "p95", "mean", "std"):
                assert key in pct_dict, f"{field} missing {key}"

    def test_market_paths_structure(self, agg_result):
        mp = agg_result.market_paths
        for segment in ("total_market", "ai_enabled", "adoption"):
            assert segment in mp, f"Missing market_paths segment: {segment}"

    def test_sensitivity_has_outcome_keys(self, agg_result):
        sens = agg_result.sensitivity
        assert isinstance(sens, dict)
        assert len(sens) > 0

    def test_sensitivity_rho_in_range(self, agg_result):
        for metric, param_dict in agg_result.sensitivity.items():
            for param, rho in param_dict.items():
                assert -1.0 <= rho <= 1.0, (
                    f"sensitivity[{metric}][{param}] = {rho} outside [-1, 1]"
                )

    def test_to_flat_dict_returns_scalars_only(self, agg_result):
        flat = agg_result.to_flat_dict()
        assert isinstance(flat, dict)
        for k, v in flat.items():
            assert not isinstance(v, np.ndarray), (
                f"to_flat_dict() contained ndarray for key '{k}'"
            )

    def test_to_flat_dict_all_finite(self, agg_result):
        flat = agg_result.to_flat_dict()
        for k, v in flat.items():
            try:
                assert np.isfinite(float(v)), f"flat_dict['{k}'] = {v} not finite"
            except (TypeError, ValueError):
                pass  # non-numeric values are fine


# ===========================================================================
# TestAnalysisIntegration
# ===========================================================================

class TestAnalysisIntegration:
    def test_spearman_sensitivity_runs(self, full_recorder, agg_result):
        from bioventure.analysis import spearman_sensitivity
        param_draws = agg_result.parameters
        if not param_draws:
            pytest.skip("No parameter draws in aggregation result")
        outcome_draws = {
            "gross_multiple": full_recorder.gross_multiple,
        }
        result = spearman_sensitivity(param_draws, outcome_draws)
        assert result.rho.shape[1] == 1

    def test_check_convergence_runs(self, full_recorder):
        from bioventure.analysis import check_convergence
        result = check_convergence(
            full_recorder.gross_multiple,
            metric_name="gross_multiple",
        )
        assert result.metric_name == "gross_multiple"
        assert len(result.windows) > 0

    def test_convergence_batch_covers_all_scalars(self, full_recorder):
        from bioventure.analysis import convergence_batch
        values_dict = {f: getattr(full_recorder, f) for f in _SCALAR_FIELDS}
        results = convergence_batch(values_dict)
        assert set(results.keys()) == set(_SCALAR_FIELDS)

    def test_effective_sample_size_positive(self, full_recorder):
        from bioventure.analysis import effective_sample_size
        ess = effective_sample_size(full_recorder.gross_multiple)
        assert ess > 0

    def test_effective_sample_size_at_most_n(self, full_recorder):
        from bioventure.analysis import effective_sample_size
        n = len(full_recorder.gross_multiple)
        ess = effective_sample_size(full_recorder.gross_multiple)
        assert ess <= n + 1e-9


# ===========================================================================
# TestIOIntegration
# ===========================================================================

class TestIOIntegration:
    def test_save_recorder_round_trip(self, full_recorder, tmp_path):
        from bioventure.io import load_recorder_arrays, save_recorder
        p = save_recorder(full_recorder, tmp_path / "rec")
        arrays = load_recorder_arrays(p)
        np.testing.assert_allclose(
            arrays["gross_multiple"], full_recorder.gross_multiple
        )

    def test_all_scalar_fields_round_trip(self, full_recorder, tmp_path):
        from bioventure.io import load_recorder_arrays, save_recorder
        p = save_recorder(full_recorder, tmp_path / "rec")
        arrays = load_recorder_arrays(p)
        for field in _SCALAR_FIELDS:
            np.testing.assert_allclose(
                arrays[field], getattr(full_recorder, field),
                err_msg=f"Round-trip mismatch for {field}",
            )

    def test_all_path_fields_round_trip(self, full_recorder, tmp_path):
        from bioventure.io import load_recorder_arrays, save_recorder
        p = save_recorder(full_recorder, tmp_path / "rec")
        arrays = load_recorder_arrays(p)
        for field in _PATH_FIELDS:
            np.testing.assert_allclose(
                arrays[field], getattr(full_recorder, field),
                err_msg=f"Round-trip mismatch for {field}",
            )

    def test_save_summary_round_trip(self, agg_result, tmp_path):
        from bioventure.io import load_summary, save_summary
        flat = agg_result.to_flat_dict()
        p = save_summary(flat, tmp_path / "summary")
        loaded = load_summary(p)
        for k, v in flat.items():
            if isinstance(v, float):
                assert loaded[k] == pytest.approx(v, rel=1e-6)

    def test_recorder_npz_exists_after_save(self, full_recorder, tmp_path):
        from bioventure.io import save_recorder
        p = save_recorder(full_recorder, tmp_path / "rec")
        assert p.exists() and p.suffix == ".npz"

    def test_summary_json_is_valid(self, agg_result, tmp_path):
        import json
        from bioventure.io import save_summary
        p = save_summary(agg_result.to_flat_dict(), tmp_path / "summary")
        data = json.loads(p.read_text())
        assert isinstance(data, dict) and len(data) > 0


# ===========================================================================
# TestReportIntegration
# ===========================================================================

class TestReportIntegration:
    def test_text_report_non_empty(self, agg_result):
        from bioventure.io import generate_text_report
        txt = generate_text_report(agg_result)
        assert len(txt) > 100

    def test_markdown_report_non_empty(self, agg_result):
        from bioventure.io import generate_markdown_report
        md = generate_markdown_report(agg_result)
        assert len(md) > 100

    def test_text_report_saved_to_disk(self, agg_result, tmp_path):
        from bioventure.io import generate_text_report, save_report
        txt = generate_text_report(agg_result)
        p = save_report(txt, tmp_path / "report.txt")
        assert p.exists()
        assert len(p.read_text()) > 0

    def test_markdown_report_saved_to_disk(self, agg_result, tmp_path):
        from bioventure.io import generate_markdown_report, save_report
        md = generate_markdown_report(agg_result)
        p = save_report(md, tmp_path / "report.md")
        assert p.exists()

    def test_text_report_contains_moic(self, agg_result):
        from bioventure.io import generate_text_report
        assert "MOIC" in generate_text_report(agg_result)

    def test_markdown_report_starts_with_heading(self, agg_result):
        from bioventure.io import generate_markdown_report
        assert generate_markdown_report(agg_result).startswith("# ")

    def test_report_with_metadata(self, agg_result):
        from bioventure.io import generate_text_report
        txt = generate_text_report(
            agg_result, metadata={"n_iterations": 20, "seed": 42}
        )
        assert "20" in txt and "42" in txt


# ===========================================================================
# TestReproducibility
# ===========================================================================

class TestReproducibility:
    def test_gross_multiple_identical_across_runs(self, two_recorders):
        r1, r2 = two_recorders
        np.testing.assert_array_equal(r1.gross_multiple, r2.gross_multiple)

    def test_terminal_market_identical_across_runs(self, two_recorders):
        r1, r2 = two_recorders
        np.testing.assert_array_equal(
            r1.terminal_total_market, r2.terminal_total_market
        )

    def test_market_paths_identical_across_runs(self, two_recorders):
        r1, r2 = two_recorders
        np.testing.assert_array_equal(
            r1.total_market_paths, r2.total_market_paths
        )

    def test_adoption_curves_identical_across_runs(self, two_recorders):
        r1, r2 = two_recorders
        np.testing.assert_array_equal(
            r1.adoption_curves, r2.adoption_curves
        )
