"""Validation subpackage: backtests, stylized facts, cross-validation."""

from bioventure.validation.backtests import (
    BacktestResult,
    backtest_coverage,
    coverage_error,
    reliability_diagram_data,
)
from bioventure.validation.cross_validation import (
    CrossValidationResult,
    FoldResult,
    cross_validate,
    leave_one_out,
)
from bioventure.validation.stylized_facts import (
    StyleFact,
    StyleResult,
    check_stylized_facts,
    get_fact,
)

__all__ = [
    "BacktestResult",
    "CrossValidationResult",
    "FoldResult",
    "StyleFact",
    "StyleResult",
    "backtest_coverage",
    "check_stylized_facts",
    "coverage_error",
    "cross_validate",
    "get_fact",
    "leave_one_out",
    "reliability_diagram_data",
]
