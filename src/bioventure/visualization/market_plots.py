"""Market-size and adoption visualisation utilities."""

from __future__ import annotations

import numpy as np

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


def fan_chart(
    paths: np.ndarray,
    time_labels: np.ndarray | None = None,
    ax: "_Axes | None" = None,
    outer_percentiles: tuple[float, float] = (5.0, 95.0),
    inner_percentiles: tuple[float, float] = (25.0, 75.0),
    color: str = "#1f77b4",
    outer_alpha: float = 0.15,
    inner_alpha: float = 0.30,
    title: str | None = None,
    xlabel: str = "Year",
    ylabel: str = "Market Size (B USD)",
) -> "_Axes":
    """Fan chart of simulated market paths showing percentile bands.

    Parameters
    ----------
    paths : np.ndarray
        Shape ``(n_sims, n_steps + 1)`` — simulated paths.
    time_labels : np.ndarray, optional
        Shape ``(n_steps + 1,)`` — x-axis values. Defaults to 0, 1, …, n_steps.
    ax : Axes, optional
        Existing axes. Creates a new figure if ``None``.
    outer_percentiles : tuple[float, float]
        (lo, hi) for the wide band (default p5–p95).
    inner_percentiles : tuple[float, float]
        (lo, hi) for the narrow band (default p25–p75).
    color : str
        Base colour applied to all bands and the median line.
    outer_alpha : float
        Fill opacity for the outer band.
    inner_alpha : float
        Fill opacity for the inner band.
    title : str, optional
    xlabel : str
    ylabel : str

    Returns
    -------
    Axes
    """
    _require_mpl()
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2:
        raise ValueError(
            f"paths must be 2-D (n_sims, n_steps+1), got {paths.ndim}-D"
        )

    n_pts = paths.shape[1]
    x = np.arange(n_pts) if time_labels is None else np.asarray(time_labels)
    if len(x) != n_pts:
        raise ValueError(
            f"time_labels length ({len(x)}) must match paths columns ({n_pts})"
        )

    p_lo_out = np.percentile(paths, outer_percentiles[0], axis=0)
    p_hi_out = np.percentile(paths, outer_percentiles[1], axis=0)
    p_lo_in  = np.percentile(paths, inner_percentiles[0], axis=0)
    p_hi_in  = np.percentile(paths, inner_percentiles[1], axis=0)
    p_med    = np.percentile(paths, 50.0, axis=0)

    if ax is None:
        _, ax = plt.subplots()

    ax.fill_between(
        x, p_lo_out, p_hi_out, color=color, alpha=outer_alpha,
        label=f"p{int(outer_percentiles[0])}–p{int(outer_percentiles[1])}",
    )
    ax.fill_between(
        x, p_lo_in, p_hi_in, color=color, alpha=inner_alpha,
        label=f"p{int(inner_percentiles[0])}–p{int(inner_percentiles[1])}",
    )
    ax.plot(x, p_med, color=color, linewidth=2, label="Median")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper left")

    return ax


def terminal_distribution(
    terminal_values: np.ndarray,
    ax: "_Axes | None" = None,
    bins: int = 50,
    color: str = "#1f77b4",
    title: str | None = None,
    xlabel: str = "Terminal Market Size (B USD)",
    ylabel: str = "Frequency",
    vlines: dict[str, float] | None = None,
) -> "_Axes":
    """Histogram of terminal market sizes with optional vertical reference lines.

    Parameters
    ----------
    terminal_values : np.ndarray
        Shape ``(n_sims,)`` — terminal value per simulation.
    ax : Axes, optional
    bins : int
    color : str
    title : str, optional
    xlabel : str
    ylabel : str
    vlines : dict[str, float], optional
        Label → value mapping for reference lines
        (e.g. ``{"p5": 120.0, "median": 280.0}``).

    Returns
    -------
    Axes
    """
    _require_mpl()
    v = np.asarray(terminal_values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"terminal_values must be 1-D, got {v.ndim}-D")

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(v, bins=bins, color=color, alpha=0.7, edgecolor="white")

    if vlines:
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
        for i, (label, val) in enumerate(vlines.items()):
            ax.axvline(
                val, color=colors[i % len(colors)],
                linestyle="--", linewidth=1.5, label=label,
            )
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return ax


def adoption_curve_plot(
    adoption_paths: np.ndarray,
    time_labels: np.ndarray | None = None,
    ax: "_Axes | None" = None,
    color: str = "#2ca02c",
    title: str | None = None,
    xlabel: str = "Year",
    ylabel: str = "AI Adoption Fraction",
) -> "_Axes":
    """Fan chart of AI adoption fraction across simulations.

    Delegates to :func:`fan_chart` with adoption-appropriate defaults.

    Parameters
    ----------
    adoption_paths : np.ndarray
        Shape ``(n_sims, n_steps + 1)`` — adoption fractions in ``[0, 1]``.
    time_labels : np.ndarray, optional
    ax : Axes, optional
    color : str
    title : str, optional
    xlabel : str
    ylabel : str

    Returns
    -------
    Axes
    """
    return fan_chart(
        adoption_paths,
        time_labels=time_labels,
        ax=ax,
        color=color,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
