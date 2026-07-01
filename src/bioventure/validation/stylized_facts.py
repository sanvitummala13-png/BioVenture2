"""Stylized-facts checks for simulated market paths.

Stylized facts are well-documented empirical regularities in financial
time series.  A simulation that cannot reproduce them is mis-specified
regardless of how well it fits calibration targets.

Checks implemented
------------------
1. **positive_drift**   — mean annualised log-return > 0.
2. **drift_in_range**   — mean annualised log-return within a plausible band.
3. **vol_in_range**     — mean annualised volatility within a plausible band.
4. **low_autocorr**     — |lag-1 return autocorrelation| below threshold.
5. **all_positive**     — no path ever reaches zero or below.
6. **terminal_growth**  — mean terminal value exceeds mean initial value.
7. **return_skew**      — log-return skewness within an expected range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class StyleFact:
    """Result of a single stylized-fact check.

    Attributes
    ----------
    name : str
        Short identifier.
    description : str
        Human-readable description of what is being checked.
    passed : bool
        Whether the check passed.
    value : float
        The measured statistic.
    expected_range : tuple[float, float]
        The (low, high) interval that defines a pass.
    """

    name: str
    description: str
    passed: bool
    value: float
    expected_range: tuple[float, float]


@dataclass
class StyleResult:
    """Aggregated stylized-fact validation report.

    Attributes
    ----------
    facts : list[StyleFact]
        One entry per check.
    n_passed : int
        Number of checks that passed.
    n_total : int
        Total number of checks run.
    pass_rate : float
        ``n_passed / n_total``.
    """

    facts: list[StyleFact] = field(default_factory=list)
    n_passed: int = 0
    n_total: int = 0
    pass_rate: float = 0.0

    def summary(self) -> str:
        """Return a compact text summary."""
        lines = [
            f"Stylized facts: {self.n_passed}/{self.n_total} passed "
            f"({self.pass_rate:.0%})",
        ]
        for fact in self.facts:
            status = "PASS" if fact.passed else "FAIL"
            lines.append(
                f"  [{status}] {fact.name}: "
                f"{fact.value:.4f} "
                f"(expected {fact.expected_range[0]:.4f}–"
                f"{fact.expected_range[1]:.4f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_stylized_facts(
    paths: np.ndarray,
    dt: float = 1.0,
    drift_range: tuple[float, float] = (0.0, 0.35),
    vol_range: tuple[float, float] = (0.05, 0.55),
    autocorr_threshold: float = 0.25,
    skew_range: tuple[float, float] = (-3.0, 3.0),
) -> StyleResult:
    """Run all stylized-fact checks on simulated market paths.

    Parameters
    ----------
    paths : np.ndarray
        Shape ``(n_sims, n_steps + 1)`` — simulated price / market-size
        paths.  The first column is the common initial value ``S₀``.
    dt : float
        Step size in years (default 1.0).
    drift_range : tuple[float, float]
        Acceptable range for mean annualised log-return.
    vol_range : tuple[float, float]
        Acceptable range for mean annualised volatility.
    autocorr_threshold : float
        Maximum absolute lag-1 return autocorrelation for a pass.
    skew_range : tuple[float, float]
        Acceptable range for cross-path log-return skewness.

    Returns
    -------
    StyleResult
    """
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2-D, got {paths.ndim}-D")
    if paths.shape[1] < 2:
        raise ValueError(
            f"paths must have >= 2 columns (n_steps >= 1), "
            f"got {paths.shape[1]}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    n_sims, n_pts = paths.shape
    n_steps = n_pts - 1

    log_returns = np.log(paths[:, 1:] / paths[:, :-1])  # (n_sims, n_steps)

    facts: list[StyleFact] = []

    facts.append(_check_positive_drift(log_returns, dt))
    facts.append(_check_drift_in_range(log_returns, dt, drift_range))
    facts.append(_check_vol_in_range(log_returns, dt, vol_range))
    facts.append(_check_autocorrelation(log_returns, n_steps, autocorr_threshold))
    facts.append(_check_all_positive(paths))
    facts.append(_check_terminal_growth(paths))
    facts.append(_check_return_skew(log_returns, skew_range))

    n_passed = sum(f.passed for f in facts)
    n_total = len(facts)

    return StyleResult(
        facts=facts,
        n_passed=n_passed,
        n_total=n_total,
        pass_rate=n_passed / n_total,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_positive_drift(
    log_returns: np.ndarray,
    dt: float,
) -> StyleFact:
    """Mean annualised log-return should be positive."""
    mean_annual = float(log_returns.mean() / dt)
    lo, hi = 0.0, np.inf
    return StyleFact(
        name="positive_drift",
        description="Mean annualised log-return > 0",
        passed=mean_annual > 0.0,
        value=mean_annual,
        expected_range=(lo, hi),
    )


def _check_drift_in_range(
    log_returns: np.ndarray,
    dt: float,
    drift_range: tuple[float, float],
) -> StyleFact:
    """Mean annualised log-return within a plausible economic band."""
    mean_annual = float(log_returns.mean() / dt)
    lo, hi = drift_range
    return StyleFact(
        name="drift_in_range",
        description=f"Mean annualised log-return in [{lo:.2f}, {hi:.2f}]",
        passed=lo <= mean_annual <= hi,
        value=mean_annual,
        expected_range=(lo, hi),
    )


def _check_vol_in_range(
    log_returns: np.ndarray,
    dt: float,
    vol_range: tuple[float, float],
) -> StyleFact:
    """Mean annualised volatility within a plausible range."""
    per_path_vol = log_returns.std(axis=1) / np.sqrt(dt)
    mean_vol = float(per_path_vol.mean())
    lo, hi = vol_range
    return StyleFact(
        name="vol_in_range",
        description=f"Mean annualised volatility in [{lo:.2f}, {hi:.2f}]",
        passed=lo <= mean_vol <= hi,
        value=mean_vol,
        expected_range=(lo, hi),
    )


def _check_autocorrelation(
    log_returns: np.ndarray,
    n_steps: int,
    threshold: float,
) -> StyleFact:
    """Lag-1 return autocorrelation near zero (no predictability)."""
    if n_steps < 3:
        return StyleFact(
            name="low_autocorr",
            description="Lag-1 autocorrelation below threshold (skipped: < 3 steps)",
            passed=True,
            value=0.0,
            expected_range=(-threshold, threshold),
        )

    autocorrs: list[float] = []
    for i in range(len(log_returns)):
        r = log_returns[i]
        if r.std() > 0:
            ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            if np.isfinite(ac):
                autocorrs.append(ac)

    if not autocorrs:
        mean_ac = 0.0
    else:
        mean_ac = float(np.mean(autocorrs))

    return StyleFact(
        name="low_autocorr",
        description=f"|lag-1 autocorr| < {threshold:.2f}",
        passed=abs(mean_ac) < threshold,
        value=mean_ac,
        expected_range=(-threshold, threshold),
    )


def _check_all_positive(paths: np.ndarray) -> StyleFact:
    """No simulated path reaches zero or below."""
    min_val = float(paths.min())
    return StyleFact(
        name="all_positive",
        description="All simulated values > 0",
        passed=min_val > 0.0,
        value=min_val,
        expected_range=(0.0, np.inf),
    )


def _check_terminal_growth(paths: np.ndarray) -> StyleFact:
    """Mean terminal value exceeds mean initial value."""
    mean_initial = float(paths[:, 0].mean())
    mean_terminal = float(paths[:, -1].mean())
    growth = mean_terminal / mean_initial if mean_initial > 0 else 0.0
    return StyleFact(
        name="terminal_growth",
        description="Mean terminal > mean initial (growth ratio > 1)",
        passed=growth > 1.0,
        value=growth,
        expected_range=(1.0, np.inf),
    )


def _check_return_skew(
    log_returns: np.ndarray,
    skew_range: tuple[float, float],
) -> StyleFact:
    """Pooled log-return skewness within an expected range."""
    flat = log_returns.ravel()
    n = len(flat)
    if n < 4:
        return StyleFact(
            name="return_skew",
            description="Log-return skewness in expected range (skipped: too few obs)",
            passed=True,
            value=0.0,
            expected_range=skew_range,
        )
    mean = flat.mean()
    std = flat.std()
    if std == 0:
        skew = 0.0
    else:
        skew = float(np.mean(((flat - mean) / std) ** 3))
    lo, hi = skew_range
    return StyleFact(
        name="return_skew",
        description=f"Log-return skewness in [{lo:.1f}, {hi:.1f}]",
        passed=lo <= skew <= hi,
        value=skew,
        expected_range=(lo, hi),
    )


# ---------------------------------------------------------------------------
# Convenience: check a single named fact
# ---------------------------------------------------------------------------

def get_fact(result: StyleResult, name: str) -> StyleFact | None:
    """Look up a StyleFact by name from a StyleResult."""
    for fact in result.facts:
        if fact.name == name:
            return fact
    return None
