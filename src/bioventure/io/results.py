"""Persistence utilities for simulation outputs.

Provides symmetric save/load for:
- Recorder contents  → compressed NumPy archive (.npz)
- Aggregation summaries → JSON flat dict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_SCALAR_FIELDS: tuple[str, ...] = (
    "cumulative_pos",
    "compressed_timeline",
    "compressed_cost",
    "terminal_total_market",
    "terminal_ai_enabled",
    "gross_multiple",
    "net_profit_m",
    "success_rate",
)

_PATH_FIELDS: tuple[str, ...] = (
    "total_market_paths",
    "ai_enabled_paths",
    "adoption_curves",
)


def save_recorder(recorder, path: str | Path) -> Path:
    """Save all recorder arrays to a compressed .npz archive.

    Parameters
    ----------
    recorder : Recorder
        A fully populated ``Recorder`` instance.
    path : str or Path
        Output path. The ``.npz`` extension is added if absent.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    p = _ensure_suffix(Path(path), ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for field in _SCALAR_FIELDS:
        arrays[field] = np.asarray(getattr(recorder, field))
    for field in _PATH_FIELDS:
        arrays[field] = np.asarray(getattr(recorder, field))

    np.savez_compressed(p, **arrays)
    return p.resolve()


def load_recorder_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load recorder arrays from a .npz archive.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file written by :func:`save_recorder`.

    Returns
    -------
    dict[str, np.ndarray]
        All arrays keyed by field name.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")

    with np.load(p) as data:
        return {k: data[k].copy() for k in data.files}


def save_summary(summary: dict, path: str | Path) -> Path:
    """Persist an aggregation flat-dict to a JSON file.

    NumPy scalar types are automatically coerced to Python natives so the
    output is always valid JSON.

    Parameters
    ----------
    summary : dict
        Output of ``AggregationResult.to_flat_dict()`` or any flat scalar
        mapping.  ndarray values are silently skipped.
    path : str or Path
        Output path. The ``.json`` extension is added if absent.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    p = _ensure_suffix(Path(path), ".json")
    p.parent.mkdir(parents=True, exist_ok=True)

    serialisable = {
        k: v for k, v in summary.items()
        if not isinstance(v, np.ndarray)
    }

    def _default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serialisable"
        )

    p.write_text(
        json.dumps(serialisable, indent=2, default=_default),
        encoding="utf-8",
    )
    return p.resolve()


def load_summary(path: str | Path) -> dict:
    """Load an aggregation summary from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.json`` file written by :func:`save_summary`.

    Returns
    -------
    dict
        Flat scalar dictionary.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Summary file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _ensure_suffix(p: Path, suffix: str) -> Path:
    return p if p.suffix == suffix else p.with_suffix(suffix)
