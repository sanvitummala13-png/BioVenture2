"""Process registry: map config strings to stochastic process classes."""

from __future__ import annotations

from typing import Any

from bioventure.processes.base import StochasticProcess
from bioventure.processes.gbm import GeometricBrownianMotion
from bioventure.processes.jump_diffusion import JumpDiffusion
from bioventure.processes.regime_switching import RegimeSwitching

_REGISTRY: dict[str, type[StochasticProcess]] = {
    "gbm": GeometricBrownianMotion,
    "jump_diffusion": JumpDiffusion,
    "regime_switching": RegimeSwitching,
}


def list_processes() -> list[str]:
    """Return sorted list of registered process names."""
    return sorted(_REGISTRY)


def get_process(name: str, params: dict[str, Any]) -> StochasticProcess:
    """Look up a process by config name and instantiate it.

    Parameters
    ----------
    name : str
        Process identifier (e.g. ``"gbm"``, ``"jump_diffusion"``,
        ``"regime_switching"``).
    params : dict[str, Any]
        Process-specific parameters passed to the constructor.

    Returns
    -------
    StochasticProcess
        An initialised, ready-to-simulate process instance.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown process '{name}'. Available: {list_processes()}"
        )
    return _REGISTRY[name](params)


def register_process(name: str, cls: type[StochasticProcess]) -> None:
    """Add a custom process to the registry at runtime.

    Parameters
    ----------
    name : str
        Short identifier (used in YAML configs).
    cls : type[StochasticProcess]
        A concrete subclass of ``StochasticProcess``.

    Raises
    ------
    TypeError
        If *cls* is not a subclass of ``StochasticProcess``.
    ValueError
        If *name* is already registered.
    """
    if not (isinstance(cls, type) and issubclass(cls, StochasticProcess)):
        raise TypeError(
            f"Expected a StochasticProcess subclass, got {cls!r}"
        )
    name = name.lower().strip()
    if name in _REGISTRY:
        raise ValueError(
            f"Process '{name}' is already registered "
            f"({_REGISTRY[name].__name__})"
        )
    _REGISTRY[name] = cls
    