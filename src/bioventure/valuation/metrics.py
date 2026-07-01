"""Financial performance metrics for venture portfolio simulations.

All functions operate on arrays of simulation outcomes (gross multiples,
net profits, or cashflow streams) and return scalar or dict summaries.

Conventions:
- **Gross multiple (MOIC)**: total_returned / total_invested.
  1.0 = break-even, <1.0 = loss, >1.0 = profit.
- **Net return**: gross_multiple - 1.  Negative = loss.
- **VaR / CVaR**: expressed as positive loss magnitudes at a given
  confidence level (e.g. VaR₅ = loss exceeded in only 5% of cases).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


# ---------------------------------------------------------------------------
# IRR
# ---------------------------------------------------------------------------

def simple_irr(
    total_returned: np.ndarray,
    total_invested: float,
    horizon_years: float,
) -> np.ndarray:
    """Annualised IRR for single invest-then-exit cashflow pattern.

    IRR = (total_returned / total_invested)^(1/T) - 1

    Parameters
    ----------
    total_returned : np.ndarray
        Shape ``(n,)`` — terminal portfolio value per simulation.
    total_invested : float
        Total capital deployed (same for all simulations).
    horizon_years : float
        Investment holding period in years.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` — annualised IRR.  Negative when
        total_returned < total_invested.
    """
    total_returned = np.asarray(total_returned, dtype=np.float64)
    if total_invested <= 0:
        raise ValueError(f"total_invested must be > 0, got {total_invested}")
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be > 0, got {horizon_years}")

    ratio = total_returned / total_invested
    sign = np.sign(ratio)
    return sign * np.abs(ratio) ** (1.0 / horizon_years) - 1.0


def solve_irr(
    cashflows: np.ndarray,
    times: np.ndarray,
    guess: float = 0.1,
    bounds: tuple[float, float] = (-0.99, 10.0),
) -> float:
    """Numerically solve for IRR of an arbitrary cashflow stream.

    Finds *r* such that  Σ cf_i / (1+r)^t_i = 0.

    Parameters
    ----------
    cashflows : np.ndarray
        Shape ``(k,)`` — signed cashflows (negative = outflow).
    times : np.ndarray
        Shape ``(k,)`` — year of each cashflow.
    guess : float
        Initial guess for the solver.
    bounds : tuple[float, float]
        Search interval for Brent's method.

    Returns
    -------
    float
        Internal rate of return, or ``nan`` if no solution found.
    """
    cf = np.asarray(cashflows, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if len(cf) != len(t):
        raise ValueError(
            f"cashflows and times must have same length, "
            f"got {len(cf)} and {len(t)}"
        )

    def npv_at(r: float) -> float:
        return float(np.sum(cf * (1.0 + r) ** (-t)))

    try:
        return float(optimize.brentq(npv_at, bounds[0], bounds[1]))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# MOIC / TVPI / DPI
# ---------------------------------------------------------------------------

def moic(total_returned: np.ndarray, total_invested: float) -> np.ndarray:
    """Multiple on Invested Capital (= TVPI for fully realised funds).

    Parameters
    ----------
    total_returned : np.ndarray
        Shape ``(n,)`` — total exit proceeds per simulation.
    total_invested : float
        Total capital deployed.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` — gross multiples.
    """
    if total_invested <= 0:
        raise ValueError(f"total_invested must be > 0, got {total_invested}")
    return np.asarray(total_returned, dtype=np.float64) / total_invested


# ---------------------------------------------------------------------------
# Tail risk: VaR, CVaR
# ---------------------------------------------------------------------------

def var(gross_multiples: np.ndarray, alpha: float = 0.05) -> float:
    """Value at Risk at confidence level (1 - α).

    VaR_α = 1 - quantile(gross_multiples, α)

    Interpretation: the portfolio loses at least this fraction of
    invested capital in the worst α share of simulations.

    Parameters
    ----------
    gross_multiples : np.ndarray
        Shape ``(n,)`` — MOIC per simulation.
    alpha : float
        Left-tail probability (e.g. 0.05 for 5th percentile).

    Returns
    -------
    float
        Positive loss magnitude (0 means no loss at this level).
    """
    _validate_alpha(alpha)
    q = float(np.percentile(gross_multiples, alpha * 100))
    return max(1.0 - q, 0.0)


def cvar(gross_multiples: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional VaR (Expected Shortfall) at level α.

    CVaR_α = 1 - mean(gross_multiples | gross_multiples <= quantile(α))

    Interpretation: expected loss given that we are in the worst
    α share of outcomes.

    Parameters
    ----------
    gross_multiples : np.ndarray
        Shape ``(n,)`` — MOIC per simulation.
    alpha : float
        Left-tail probability.

    Returns
    -------
    float
        Positive expected loss magnitude.
    """
    _validate_alpha(alpha)
    gm = np.asarray(gross_multiples, dtype=np.float64)
    threshold = np.percentile(gm, alpha * 100)
    tail = gm[gm <= threshold]
    if len(tail) == 0:
        return 0.0
    return max(1.0 - float(tail.mean()), 0.0)


# ---------------------------------------------------------------------------
# Probability of loss
# ---------------------------------------------------------------------------

def probability_of_loss(gross_multiples: np.ndarray) -> float:
    """Fraction of simulations where gross multiple < 1.0 (capital loss).

    Parameters
    ----------
    gross_multiples : np.ndarray
        Shape ``(n,)`` — MOIC per simulation.

    Returns
    -------
    float
        In [0, 1].
    """
    return float(np.mean(np.asarray(gross_multiples) < 1.0))


# ---------------------------------------------------------------------------
# Percentile summary
# ---------------------------------------------------------------------------

def percentile_summary(
    values: np.ndarray,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> dict[str, float]:
    """Compute named percentiles of a distribution.

    Parameters
    ----------
    values : np.ndarray
        Shape ``(n,)`` — simulation outcomes.
    percentiles : tuple[float, ...]
        Percentile levels (0–100 scale).

    Returns
    -------
    dict[str, float]
        Keys like ``"p5"``, ``"p25"``, etc.
    """
    values = np.asarray(values, dtype=np.float64)
    result: dict[str, float] = {}
    for p in percentiles:
        if not 0 <= p <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {p}")
        key = f"p{int(p)}"
        result[key] = float(np.percentile(values, p))
    result["mean"] = float(values.mean())
    result["std"] = float(values.std())
    return result


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------

def compute_summary(
    gross_multiples: np.ndarray,
    total_invested: float,
    horizon_years: float,
) -> dict[str, float]:
    """One-shot summary of all key financial metrics.

    Parameters
    ----------
    gross_multiples : np.ndarray
        Shape ``(n,)`` — portfolio MOIC per simulation.
    total_invested : float
        Total capital deployed.
    horizon_years : float
        Investment holding period.

    Returns
    -------
    dict[str, float]
        Comprehensive metrics dictionary.
    """
    gm = np.asarray(gross_multiples, dtype=np.float64)
    total_returned = gm * total_invested
    irr_values = simple_irr(total_returned, total_invested, horizon_years)

    pct = percentile_summary(gm)
    irr_pct = percentile_summary(irr_values)

    return {
        "moic_mean": pct["mean"],
        "moic_std": pct["std"],
        "moic_p5": pct["p5"],
        "moic_p25": pct["p25"],
        "moic_median": pct["p50"],
        "moic_p75": pct["p75"],
        "moic_p95": pct["p95"],
        "irr_mean": irr_pct["mean"],
        "irr_median": irr_pct["p50"],
        "irr_p5": irr_pct["p5"],
        "irr_p95": irr_pct["p95"],
        "var_5": var(gm, 0.05),
        "cvar_5": cvar(gm, 0.05),
        "probability_of_loss": probability_of_loss(gm),
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    