"""Sensitivity analysis visualisation: tornado charts and heatmaps."""

from __future__ import annotations

import numpy as np

from bioventure.analysis.sensitivity import SensitivityResult

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


def tornado_chart(
    result: SensitivityResult,
    metric: str,
    ax: "_Axes | None" = None,
    k: int | None = None,
    color_pos: str = "#1f77b4",
    color_neg: str = "#d62728",
    title: str | None = None,
    xlabel: str = "Spearman ρ",
) -> "_Axes":
    """Horizontal bar chart of Spearman ρ values for one output metric.

    Bars are sorted so the most influential parameter (highest |ρ|) appears
    at the top. Positive ρ uses *color_pos*; negative ρ uses *color_neg*.

    Parameters
    ----------
    result : SensitivityResult
        Output of :func:`~bioventure.analysis.sensitivity.spearman_sensitivity`.
    metric : str
        Name of the output metric to visualise.
    ax : Axes, optional
        Existing axes. Creates a new figure if ``None``.
    k : int, optional
        Show only the top *k* parameters by |ρ|. ``None`` shows all.
    color_pos : str
        Bar colour for positive correlations.
    color_neg : str
        Bar colour for negative correlations.
    title : str, optional
        Axes title. Defaults to ``"Sensitivity: <metric>"``.
    xlabel : str

    Returns
    -------
    Axes
    """
    _require_mpl()

    if metric not in result.metric_names:
        raise ValueError(
            f"Metric '{metric}' not found. Available: {result.metric_names}"
        )
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    j = result.metric_names.index(metric)
    rho_col = result.rho[:, j]

    order = np.argsort(np.abs(rho_col))          # ascending |ρ| → bottom to top
    if k is not None:
        order = order[-k:]

    names = [result.param_names[i] for i in order]
    rhos  = rho_col[order]
    colors = [color_pos if r >= 0 else color_neg for r in rhos]

    if ax is None:
        fig_h = max(3.0, 0.4 * len(names))
        _, ax = plt.subplots(figsize=(7, fig_h))

    y = np.arange(len(names))
    ax.barh(y, rhos, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title if title is not None else f"Sensitivity: {metric}")

    return ax


def sensitivity_heatmap(
    result: SensitivityResult,
    ax: "_Axes | None" = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    title: str | None = None,
    annotate: bool = True,
) -> "_Axes":
    """Heatmap of the full Spearman ρ matrix (params × metrics).

    Parameters
    ----------
    result : SensitivityResult
        Output of :func:`~bioventure.analysis.sensitivity.spearman_sensitivity`.
    ax : Axes, optional
        Existing axes. Creates a new figure if ``None``.
    cmap : str
        Matplotlib colourmap name. Diverging maps (``"RdBu_r"``) work best
        since ρ is signed.
    vmin : float
        Colour scale minimum (default −1).
    vmax : float
        Colour scale maximum (default +1).
    title : str, optional
    annotate : bool
        If ``True``, write the ρ value inside each cell.

    Returns
    -------
    Axes
    """
    _require_mpl()

    rho = result.rho  # (n_params, n_metrics)
    n_params, n_metrics = rho.shape

    if ax is None:
        fig_w = max(4.0, 1.2 * n_metrics)
        fig_h = max(3.0, 0.5 * n_params)
        _, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(rho, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(result.metric_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_params))
    ax.set_yticklabels(result.param_names)

    if annotate:
        for i in range(n_params):
            for j in range(n_metrics):
                val = rho[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=8, color=text_color,
                )

    ax.set_title(title if title is not None else "Sensitivity Heatmap")

    return ax
