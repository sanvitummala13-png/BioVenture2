"""Human-readable report generation from aggregation results.

Provides plain-text and Markdown formatters that summarise the key
outputs of a BioVenture simulation run: financial performance, market
size distributions, portfolio outcomes, and top sensitivity drivers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_text_report(
    result: Any,
    title: str = "BioVenture Simulation Report",
    metadata: dict | None = None,
) -> str:
    """Format an ``AggregationResult`` as a fixed-width text report.

    Parameters
    ----------
    result : AggregationResult
        Output of :meth:`~bioventure.simulation.Aggregator.summarise`.
    title : str
        Report heading.
    metadata : dict, optional
        Arbitrary key-value pairs printed in the header block
        (e.g. ``{"n_iterations": 10_000, "horizon_years": 9}``).

    Returns
    -------
    str
        Multi-line text suitable for printing or writing to ``.txt``.
    """
    w = 70
    sep = "=" * w
    thin = "-" * w
    lines: list[str] = []

    def _h(text: str) -> None:
        lines.append(sep)
        lines.append(_centre(text, w))
        lines.append(sep)

    def _section(text: str) -> None:
        lines.append("")
        lines.append(thin)
        lines.append(f"  {text}")
        lines.append(thin)

    def _kv(label: str, value: str, indent: int = 4) -> None:
        lines.append(f"{' ' * indent}{label:<35}{value}")

    _h(title)

    # --- Metadata ---
    if metadata:
        _section("Run Parameters")
        for k, v in metadata.items():
            _kv(str(k), str(v))

    # --- Financial performance ---
    fin = result.financial
    _section("Financial Performance")
    _kv("MOIC — mean",        _fmt(fin.get("moic_mean")))
    _kv("MOIC — median",      _fmt(fin.get("moic_median")))
    _kv("MOIC — p5 / p95",
        f"{_fmt(fin.get('moic_p5'))} / {_fmt(fin.get('moic_p95'))}")
    _kv("IRR — mean",         _pct(fin.get("irr_mean")))
    _kv("IRR — median",       _pct(fin.get("irr_median")))
    _kv("IRR — p5 / p95",
        f"{_pct(fin.get('irr_p5'))} / {_pct(fin.get('irr_p95'))}")
    _kv("VaR 5%",             _fmt(fin.get("var_5")))
    _kv("CVaR 5%",            _fmt(fin.get("cvar_5")))
    _kv("Probability of loss", _pct(fin.get("probability_of_loss")))

    # --- Market outcomes ---
    _section("Terminal Market Size (B USD)")
    mkt = result.outcomes.get("terminal_total_market", {})
    if mkt:
        _kv("Mean",           _fmt(mkt.get("mean")))
        _kv("Median (p50)",   _fmt(mkt.get("p50")))
        _kv("p5 / p95",
            f"{_fmt(mkt.get('p5'))} / {_fmt(mkt.get('p95'))}")

    ai = result.outcomes.get("terminal_ai_enabled", {})
    _section("Terminal AI-Enabled Market (B USD)")
    if ai:
        _kv("Mean",           _fmt(ai.get("mean")))
        _kv("Median (p50)",   _fmt(ai.get("p50")))
        _kv("p5 / p95",
            f"{_fmt(ai.get('p5'))} / {_fmt(ai.get('p95'))}")

    # --- Portfolio outcomes ---
    _section("Portfolio Outcomes")
    gm = result.outcomes.get("gross_multiple", {})
    sr = result.outcomes.get("success_rate", {})
    if gm:
        _kv("Gross multiple — mean",   _fmt(gm.get("mean")))
        _kv("Gross multiple — median", _fmt(gm.get("p50")))
    if sr:
        _kv("Success rate — mean",     _pct(sr.get("mean")))
        _kv("Success rate — median",   _pct(sr.get("p50")))

    # --- Sensitivity (top 5 params → gross_multiple) ---
    sens = result.sensitivity
    if sens and "gross_multiple" in sens:
        _section("Top Sensitivity Drivers  →  gross_multiple  (Spearman ρ)")
        gm_sens = sens["gross_multiple"]
        top = sorted(gm_sens.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for param, rho in top:
            _kv(param, f"{rho:+.3f}")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def generate_markdown_report(
    result: Any,
    title: str = "BioVenture Simulation Report",
    metadata: dict | None = None,
) -> str:
    """Format an ``AggregationResult`` as GitHub-flavoured Markdown.

    Parameters
    ----------
    result : AggregationResult
    title : str
    metadata : dict, optional

    Returns
    -------
    str
        Markdown string.
    """
    lines: list[str] = []

    def _h1(t: str) -> None:
        lines.append(f"# {t}\n")

    def _h2(t: str) -> None:
        lines.append(f"\n## {t}\n")

    def _table_row(*cols: str) -> str:
        return "| " + " | ".join(cols) + " |"

    def _table_header(*cols: str) -> None:
        lines.append(_table_row(*cols))
        lines.append(_table_row(*("---" for _ in cols)))

    _h1(title)

    if metadata:
        _h2("Run Parameters")
        _table_header("Parameter", "Value")
        for k, v in metadata.items():
            lines.append(_table_row(str(k), str(v)))

    fin = result.financial
    _h2("Financial Performance")
    _table_header("Metric", "Value")
    rows = [
        ("MOIC — mean",         _fmt(fin.get("moic_mean"))),
        ("MOIC — median",       _fmt(fin.get("moic_median"))),
        ("MOIC — p5",           _fmt(fin.get("moic_p5"))),
        ("MOIC — p95",          _fmt(fin.get("moic_p95"))),
        ("IRR — mean",          _pct(fin.get("irr_mean"))),
        ("IRR — median",        _pct(fin.get("irr_median"))),
        ("VaR 5%",              _fmt(fin.get("var_5"))),
        ("CVaR 5%",             _fmt(fin.get("cvar_5"))),
        ("Probability of loss", _pct(fin.get("probability_of_loss"))),
    ]
    for label, val in rows:
        lines.append(_table_row(label, val))

    _h2("Terminal Market Size (B USD)")
    mkt = result.outcomes.get("terminal_total_market", {})
    ai  = result.outcomes.get("terminal_ai_enabled", {})
    _table_header("Segment", "Mean", "p50", "p5", "p95")
    if mkt:
        lines.append(_table_row(
            "Total market",
            _fmt(mkt.get("mean")), _fmt(mkt.get("p50")),
            _fmt(mkt.get("p5")),   _fmt(mkt.get("p95")),
        ))
    if ai:
        lines.append(_table_row(
            "AI-enabled",
            _fmt(ai.get("mean")), _fmt(ai.get("p50")),
            _fmt(ai.get("p5")),   _fmt(ai.get("p95")),
        ))

    _h2("Portfolio Outcomes")
    gm = result.outcomes.get("gross_multiple", {})
    sr = result.outcomes.get("success_rate", {})
    _table_header("Metric", "Mean", "p50", "p5", "p95")
    if gm:
        lines.append(_table_row(
            "Gross multiple",
            _fmt(gm.get("mean")), _fmt(gm.get("p50")),
            _fmt(gm.get("p5")),   _fmt(gm.get("p95")),
        ))
    if sr:
        lines.append(_table_row(
            "Success rate",
            _pct(sr.get("mean")), _pct(sr.get("p50")),
            _pct(sr.get("p5")),   _pct(sr.get("p95")),
        ))

    sens = result.sensitivity
    if sens and "gross_multiple" in sens:
        _h2("Top Sensitivity Drivers → `gross_multiple`")
        _table_header("Parameter", "Spearman ρ")
        gm_sens = sens["gross_multiple"]
        top = sorted(gm_sens.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for param, rho in top:
            lines.append(_table_row(f"`{param}`", f"{rho:+.3f}"))

    lines.append("")
    return "\n".join(lines)


def save_report(text: str, path: str | Path) -> Path:
    """Write a report string to a file.

    Parameters
    ----------
    text : str
        Report content (text or Markdown).
    path : str or Path
        Output path. Created (including parent directories) if absent.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p.resolve()


# ---------------------------------------------------------------------------
# Internal formatters
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v: Any, decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return str(v)


def _centre(text: str, width: int) -> str:
    return text.center(width)
