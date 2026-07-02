"""Tests for IO subpackage: results persistence and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bioventure.io.results import (
    load_recorder_arrays,
    load_summary,
    save_recorder,
    save_summary,
)
from bioventure.io.reports import (
    generate_markdown_report,
    generate_text_report,
    save_report,
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_SCALAR_FIELDS = (
    "cumulative_pos", "compressed_timeline", "compressed_cost",
    "terminal_total_market", "terminal_ai_enabled",
    "gross_multiple", "net_profit_m", "success_rate",
)
_PATH_FIELDS = ("total_market_paths", "ai_enabled_paths", "adoption_curves")


def _mock_recorder(n: int = 40, n_pts: int = 10) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    attrs = {f: rng.uniform(0, 1, n) for f in _SCALAR_FIELDS}
    attrs["total_market_paths"] = rng.uniform(100, 500, (n, n_pts))
    attrs["ai_enabled_paths"]   = rng.uniform(10,  100, (n, n_pts))
    attrs["adoption_curves"]    = rng.uniform(0,   1,   (n, n_pts))
    return SimpleNamespace(**attrs)


def _mock_result() -> SimpleNamespace:
    fin = {
        "moic_mean": 2.5,    "moic_median": 2.0,
        "moic_p5": 0.3,      "moic_p25": 1.0,
        "moic_p75": 3.5,     "moic_p95": 6.0,
        "moic_std": 1.8,
        "irr_mean": 0.18,    "irr_median": 0.15,
        "irr_p5": -0.20,     "irr_p95": 0.55,
        "var_5": 0.70,       "cvar_5": 0.85,
        "probability_of_loss": 0.25,
    }
    outcomes = {
        "terminal_total_market": {
            "mean": 280.0, "std": 80.0,
            "p5": 150.0, "p25": 210.0, "p50": 275.0,
            "p75": 340.0, "p95": 430.0,
        },
        "terminal_ai_enabled": {
            "mean": 55.0, "std": 20.0,
            "p5": 25.0, "p25": 40.0, "p50": 52.0,
            "p75": 68.0, "p95": 95.0,
        },
        "gross_multiple": {
            "mean": 2.5, "std": 1.8,
            "p5": 0.3, "p25": 1.0, "p50": 2.0,
            "p75": 3.5, "p95": 6.0,
        },
        "success_rate": {
            "mean": 0.30, "std": 0.10,
            "p5": 0.10, "p25": 0.22, "p50": 0.29,
            "p75": 0.38, "p95": 0.50,
        },
    }
    sensitivity = {
        "gross_multiple": {
            "adoption_p": 0.42,
            "drift_mu": 0.18,
            "clinical_p1": -0.35,
            "payoff_mu": 0.28,
            "payoff_sigma": 0.15,
        }
    }
    ns = SimpleNamespace(
        financial=fin,
        outcomes=outcomes,
        parameters={},
        market_paths={},
        irr_summary={},
        sensitivity=sensitivity,
    )
    ns.to_flat_dict = lambda: dict(fin)
    return ns


# ===========================================================================
# TestSaveRecorder
# ===========================================================================

class TestSaveRecorder:
    def test_writes_npz_file(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        assert p.exists()
        assert p.suffix == ".npz"

    def test_adds_npz_suffix_automatically(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        assert p.name == "rec.npz"

    def test_no_double_suffix_when_given(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec.npz")
        assert p.name == "rec.npz"

    def test_creates_parent_directories(self, tmp_path):
        rec = _mock_recorder()
        deep = tmp_path / "a" / "b" / "c" / "rec"
        p = save_recorder(rec, deep)
        assert p.exists()

    def test_returns_resolved_path(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        assert p.is_absolute()

    def test_npz_contains_scalar_fields(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        with np.load(p) as data:
            for field in _SCALAR_FIELDS:
                assert field in data.files

    def test_npz_contains_path_fields(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        with np.load(p) as data:
            for field in _PATH_FIELDS:
                assert field in data.files


# ===========================================================================
# TestLoadRecorderArrays
# ===========================================================================

class TestLoadRecorderArrays:
    def test_returns_dict(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        out = load_recorder_arrays(p)
        assert isinstance(out, dict)

    def test_scalar_fields_present(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        out = load_recorder_arrays(p)
        for field in _SCALAR_FIELDS:
            assert field in out

    def test_path_fields_present(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        out = load_recorder_arrays(p)
        for field in _PATH_FIELDS:
            assert field in out

    def test_scalar_values_round_trip(self, tmp_path):
        rec = _mock_recorder()
        p = save_recorder(rec, tmp_path / "rec")
        out = load_recorder_arrays(p)
        np.testing.assert_allclose(out["gross_multiple"], rec.gross_multiple)

    def test_path_shapes_preserved(self, tmp_path):
        rec = _mock_recorder(n=40, n_pts=10)
        p = save_recorder(rec, tmp_path / "rec")
        out = load_recorder_arrays(p)
        assert out["total_market_paths"].shape == (40, 10)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_recorder_arrays(tmp_path / "missing.npz")


# ===========================================================================
# TestSaveSummary
# ===========================================================================

class TestSaveSummary:
    def test_writes_json_file(self, tmp_path):
        p = save_summary({"moic_mean": 2.5}, tmp_path / "summary")
        assert p.exists()
        assert p.suffix == ".json"

    def test_adds_json_suffix_automatically(self, tmp_path):
        p = save_summary({}, tmp_path / "summary")
        assert p.name == "summary.json"

    def test_creates_parent_directories(self, tmp_path):
        p = save_summary({"x": 1.0}, tmp_path / "deep" / "nested" / "summary")
        assert p.exists()

    def test_output_is_valid_json(self, tmp_path):
        d = {"a": 1.0, "b": 2.5, "c": -0.3}
        p = save_summary(d, tmp_path / "s")
        loaded = json.loads(p.read_text())
        assert loaded == pytest.approx(d)

    def test_numpy_scalars_serialised(self, tmp_path):
        d = {"x": np.float64(3.14), "n": np.int32(7)}
        p = save_summary(d, tmp_path / "s")
        loaded = json.loads(p.read_text())
        assert loaded["x"] == pytest.approx(3.14)
        assert loaded["n"] == 7

    def test_ndarray_values_skipped(self, tmp_path):
        d = {"scalar": 1.0, "arr": np.ones(5)}
        p = save_summary(d, tmp_path / "s")
        loaded = json.loads(p.read_text())
        assert "scalar" in loaded
        assert "arr" not in loaded

    def test_returns_resolved_path(self, tmp_path):
        p = save_summary({}, tmp_path / "s")
        assert p.is_absolute()


# ===========================================================================
# TestLoadSummary
# ===========================================================================

class TestLoadSummary:
    def test_returns_dict(self, tmp_path):
        save_summary({"x": 1.0}, tmp_path / "s")
        out = load_summary(tmp_path / "s.json")
        assert isinstance(out, dict)

    def test_round_trip(self, tmp_path):
        d = {"moic_mean": 2.5, "irr_median": 0.15, "var_5": 0.7}
        save_summary(d, tmp_path / "s")
        out = load_summary(tmp_path / "s.json")
        assert out == pytest.approx(d)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_summary(tmp_path / "nonexistent.json")


# ===========================================================================
# TestGenerateTextReport
# ===========================================================================

class TestGenerateTextReport:
    @pytest.fixture
    def result(self):
        return _mock_result()

    def test_returns_string(self, result):
        assert isinstance(generate_text_report(result), str)

    def test_contains_moic_values(self, result):
        txt = generate_text_report(result)
        assert "2.500" in txt or "2.0" in txt   # mean or median

    def test_contains_irr_section(self, result):
        txt = generate_text_report(result)
        assert "IRR" in txt

    def test_contains_market_section(self, result):
        txt = generate_text_report(result)
        assert "Market" in txt

    def test_contains_sensitivity_section(self, result):
        txt = generate_text_report(result)
        assert "Sensitivity" in txt or "adoption_p" in txt

    def test_metadata_keys_appear(self, result):
        txt = generate_text_report(result, metadata={"n_iterations": 5000})
        assert "5000" in txt

    def test_none_metadata_accepted(self, result):
        txt = generate_text_report(result, metadata=None)
        assert isinstance(txt, str)

    def test_custom_title_appears(self, result):
        txt = generate_text_report(result, title="My Report")
        assert "My Report" in txt

    def test_var_appears(self, result):
        txt = generate_text_report(result)
        assert "VaR" in txt

    def test_probability_of_loss_appears(self, result):
        txt = generate_text_report(result)
        assert "loss" in txt.lower()


# ===========================================================================
# TestGenerateMarkdownReport
# ===========================================================================

class TestGenerateMarkdownReport:
    @pytest.fixture
    def result(self):
        return _mock_result()

    def test_returns_string(self, result):
        assert isinstance(generate_markdown_report(result), str)

    def test_contains_h1_heading(self, result):
        md = generate_markdown_report(result)
        assert md.startswith("# ")

    def test_contains_h2_sections(self, result):
        md = generate_markdown_report(result)
        assert "## " in md

    def test_contains_table_separator(self, result):
        md = generate_markdown_report(result)
        assert "---" in md

    def test_contains_financial_section(self, result):
        md = generate_markdown_report(result)
        assert "Financial" in md

    def test_contains_market_section(self, result):
        md = generate_markdown_report(result)
        assert "Market" in md

    def test_metadata_table_included(self, result):
        md = generate_markdown_report(result, metadata={"seed": 42})
        assert "42" in md

    def test_none_metadata_accepted(self, result):
        md = generate_markdown_report(result, metadata=None)
        assert isinstance(md, str)

    def test_custom_title_appears(self, result):
        md = generate_markdown_report(result, title="Custom Title")
        assert "Custom Title" in md

    def test_sensitivity_section_present(self, result):
        md = generate_markdown_report(result)
        assert "Sensitivity" in md or "adoption_p" in md


# ===========================================================================
# TestSaveReport
# ===========================================================================

class TestSaveReport:
    def test_writes_file(self, tmp_path):
        p = save_report("hello", tmp_path / "report.txt")
        assert p.exists()

    def test_content_round_trip(self, tmp_path):
        content = "Line 1\nLine 2\n"
        p = save_report(content, tmp_path / "report.txt")
        assert p.read_text(encoding="utf-8") == content

    def test_creates_parent_directories(self, tmp_path):
        p = save_report("x", tmp_path / "deep" / "dir" / "report.txt")
        assert p.exists()

    def test_returns_resolved_path(self, tmp_path):
        p = save_report("x", tmp_path / "report.txt")
        assert p.is_absolute()
        