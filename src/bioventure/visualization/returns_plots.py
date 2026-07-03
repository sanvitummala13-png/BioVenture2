"""Return distribution and convergence visualisation utilities."""

from __future__ import annotations

import numpy as np

from bioventure.analysis.convergence import ConvergenceResult

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes as _Axes
    _MPL = True
except ImportError:  # pragma: no cover
    _MPL = False


def _require_mpl() -> None:
    if not _MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def moic_distribution(
    gross_multiples: np.ndarray,
    ax: "_Axes | None" = None,
    bins: int = 60,
    color: str = "#1f77b4",
    title: str | None = None,
    xlabel: str = "Gross Multiple (MOIC)",
    ylabel: str = "Frequency",
    show_var: bool = True,
    var_alpha: float = 0.05,
) -> "_Axes":
    """Histogram of portfolio gross multiples with optional VaR annotation.

    Parameters
    ----------
    gross_multiples : np.ndarray
        Shape ``(n_sims,)`` — MOIC per simulation.
    ax : Axes, optional
        Existing axes. Creates a new figure if ``None``.
    bins : int
    color : str
    title : str, optional
    xlabel : str
    ylabel : str
    show_var : bool
        If ``True``, draw a vertical line at the VaR quantile and a shaded
        left-tail region.
    var_alpha : float
        Left-tail probability for VaR (default 0.05 = 5th percentile).

    Returns
    -------
    Axes
    """
    _require_mpl()
    gm = np.asarray(gross_multiples, dtype=np.float64)
    if gm.ndim != 1:
        raise ValueError(f"gross_multiples must be 1-D, got {gm.ndim}-D")
    if not 0.0 < var_alpha < 1.0:
        raise ValueError(f"var_alpha must be in (0, 1), got {var_alpha}")

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(gm, bins=bins, color=color, alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", label="Break-even (1×)")

    if show_var:
        q = float(np.percentile(gm, var_alpha * 100))
        ax.axvline(
            q, color="#d62728", linewidth=1.5, linestyle=":",
            label=f"VaR {int(var_alpha * 100)}% ({q:.2f}×)",
        )
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], q, alpha=0.08, color="#d62728")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else "MOIC Distribution")
    ax.legend()

    return ax


def irr_distribution(
    irr_values: np.ndarray,
    ax: "_Axes | None" = None,
    bins: int = 60,
    color: str = "#ff7f0e",
    title: str | None = None,
    xlabel: str = "Annualised IRR",
    ylabel: str = "Frequency",
    show_zero: bool = True,
) -> "_Axes":
    """Histogram of annualised IRR values.

    Parameters
    ----------
    irr_values : np.ndarray
        Shape ``(n_sims,)`` — IRR per simulation (e.g. 0.25 = 25 %).
    ax : Axes, optional
    bins : int
    color : str
    title : str, optional
    xlabel : str
    ylabel : str
    show_zero : bool
        If ``True``, draw a vertical line at IRR = 0 (break-even).

    Returns
    -------
    Axes
    """
    _require_mpl()
    irr = np.asarray(irr_values, dtype=np.float64)
    if irr.ndim != 1:
        raise ValueError(f"irr_values must be 1-D, got {irr.ndim}-D")

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(irr, bins=bins, color=color, alpha=0.7, edgecolor="white")

    if show_zero:
        ax.axvline(
            0.0, color="black", linewidth=1.0,
            linestyle="--", label="IRR = 0",
        )
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else "IRR Distribution")

    return ax


def convergence_plot(
    result: ConvergenceResult,
    ax: "_Axes | None" = None,
    color: str = "#1f77b4",
    band_alpha: float = 0.20,
    title: str | None = None,
    xlabel: str = "Iterations",
    ylabel: str | None = None,
    show_convergence_line: bool = True,
) -> "_Axes":
    """Running-mean ± 1 std band plot for a single convergence result.

    Parameters
    ----------
    result : ConvergenceResult
        Output of :func:`~bioventure.analysis.convergence.check_convergence`.
    ax : Axes, optional
    color : str
    band_alpha : float
        Opacity of the ±1 std shaded band.
    title : str, optional
        Defaults to ``"Convergence: <metric_name>"``.
    xlabel : str
    ylabel : str, optional
        Defaults to the result's metric name.
    show_convergence_line : bool
        If ``True`` and the series converged, draw a vertical line at
        ``result.convergence_iteration``.

    Returns
    -------
    Axes
    """
    _require_mpl()

    x = result.windows
    mean = result.running_means
    std  = result.running_stds

    if ax is None:
        _, ax = plt.subplots()

    ax.fill_between(
        x, mean - std, mean + std,
        color=color, alpha=band_alpha, label="±1 std",
    )
    ax.plot(x, mean, color=color, linewidth=2, label="Running mean")

    if show_convergence_line and result.converged and result.convergence_iteration is not None:
        ax.axvline(
            result.convergence_iteration,
            color="#2ca02c", linewidth=1.5, linestyle="--",
            label=f"Converged @ {result.convergence_iteration}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else result.metric_name)
    ax.set_title(
        title if title is not None else f"Convergence: {result.metric_name}"
    )
    ax.legend()

    return ax


def reliability_diagram(
    nominal: np.ndarray,
    empirical: np.ndarray,
    ax: "_Axes | None" = None,
    color: str = "#1f77b4",
    title: str | None = None,
    xlabel: str = "Nominal coverage",
    ylabel: str = "Empirical coverage",
) -> "_Axes":
    """Reliability (calibration) diagram for prediction interval coverage.

    Plots empirical vs. nominal coverage levels. The diagonal represents
    perfect calibration; points above indicate over-coverage (conservative
    intervals) and points below indicate under-coverage (too narrow).

    Parameters
    ----------
    nominal : np.ndarray
        Shape ``(k,)`` — stated confidence levels (e.g. 0.50, 0.90, 0.95).
    empirical : np.ndarray
        Shape ``(k,)`` — observed coverage fractions from backtest.
    ax : Axes, optional
    color : str
    title : str, optional
    xlabel : str
    ylabel : str

    Returns
    -------
    Axes
    """
    _require_mpl()
    nom = np.asarray(nominal, dtype=np.float64)
    emp = np.asarray(empirical, dtype=np.float64)
    if nom.ndim != 1 or emp.ndim != 1:
        raise ValueError("nominal and empirical must be 1-D arrays")
    if len(nom) != len(emp):
        raise ValueError(
            f"nominal and empirical must have the same length, "
            f"got {len(nom)} and {len(emp)}"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot([0, 1], [0, 1], color="black", linewidth=1.0,
            linestyle="--", label="Perfect calibration")
    ax.scatter(nom, emp, color=color, zorder=3, label="Observed")
    ax.plot(nom, emp, color=color, alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else "Reliability Diagram")
    ax.legend()

    return ax
