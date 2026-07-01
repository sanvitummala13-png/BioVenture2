"""Prior distribution sampler: builds frozen scipy distributions from priors.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.random import Generator
from scipy import stats


# ---------------------------------------------------------------------------
# Supported distribution families and their required parameters
# ---------------------------------------------------------------------------

_REQUIRED_PARAMS: dict[str, set[str]] = {
    "normal": {"loc", "scale"},
    "lognormal": {"mu", "sigma"},
    "beta": {"a", "b"},
    "triangular": {"left", "mode", "right"},
    "bernoulli": {"p"},
    "poisson": {"mu"},
    "uniform": {"low", "high"},
}


def _build_distribution(family: str, params: dict[str, float]) -> stats.rv_frozen:
    """Convert a family name + parameter dict into a frozen scipy distribution.

    Parameters
    ----------
    family : str
        One of the supported distribution families.
    params : dict[str, float]
        Hyperparameters matching the family's required set.

    Returns
    -------
    scipy.stats.rv_frozen
        A frozen distribution ready for ``.rvs()`` calls.

    Raises
    ------
    ValueError
        If the family is unknown or required parameters are missing/invalid.
    """
    family = family.lower()
    if family not in _REQUIRED_PARAMS:
        raise ValueError(
            f"Unknown distribution family '{family}'. "
            f"Supported: {sorted(_REQUIRED_PARAMS)}"
        )

    required = _REQUIRED_PARAMS[family]
    missing = required - set(params)
    if missing:
        raise ValueError(
            f"Distribution '{family}' requires parameters {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )

    if family == "normal":
        if params["scale"] <= 0:
            raise ValueError(f"normal 'scale' must be > 0, got {params['scale']}")
        return stats.norm(loc=params["loc"], scale=params["scale"])

    if family == "lognormal":
        if params["sigma"] <= 0:
            raise ValueError(f"lognormal 'sigma' must be > 0, got {params['sigma']}")
        return stats.lognorm(s=params["sigma"], scale=np.exp(params["mu"]))

    if family == "beta":
        if params["a"] <= 0 or params["b"] <= 0:
            raise ValueError(f"beta 'a' and 'b' must be > 0, got a={params['a']}, b={params['b']}")
        return stats.beta(a=params["a"], b=params["b"])

    if family == "triangular":
        left, mode, right = params["left"], params["mode"], params["right"]
        if not left <= mode <= right:
            raise ValueError(
                f"triangular requires left <= mode <= right, "
                f"got left={left}, mode={mode}, right={right}"
            )
        if left == right:
            raise ValueError("triangular requires left < right")
        c = (mode - left) / (right - left)
        return stats.triang(c=c, loc=left, scale=right - left)

    if family == "bernoulli":
        p = params["p"]
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"bernoulli 'p' must be in [0, 1], got {p}")
        return stats.bernoulli(p=p)

    if family == "poisson":
        mu = params["mu"]
        if mu < 0:
            raise ValueError(f"poisson 'mu' must be >= 0, got {mu}")
        return stats.poisson(mu=mu)

    # uniform
    low, high = params["low"], params["high"]
    if low >= high:
        raise ValueError(f"uniform requires low < high, got low={low}, high={high}")
    return stats.uniform(loc=low, scale=high - low)


# ---------------------------------------------------------------------------
# PriorSpec: one named prior
# ---------------------------------------------------------------------------

class PriorSpec:
    """A single named prior distribution backed by a frozen scipy object."""

    __slots__ = ("name", "family", "params", "_dist")

    def __init__(self, name: str, family: str, params: dict[str, float]) -> None:
        self.name = name
        self.family = family.lower()
        self.params = dict(params)
        self._dist = _build_distribution(self.family, self.params)

    @property
    def dist(self) -> stats.rv_frozen:
        return self._dist

    def sample(self, n: int, rng: Generator) -> np.ndarray:
        """Draw *n* independent samples.

        Parameters
        ----------
        n : int
            Number of samples (must be >= 1).
        rng : numpy.random.Generator
            Random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Shape ``(n,)`` array of samples.
        """
        if n < 1:
            raise ValueError(f"Sample size must be >= 1, got {n}")
        return self._dist.rvs(size=n, random_state=rng)

    def sample_scalar(self, rng: Generator) -> float:
        """Draw a single scalar sample."""
        return float(self._dist.rvs(random_state=rng))

    def __repr__(self) -> str:
        return f"PriorSpec(name={self.name!r}, family={self.family!r}, params={self.params})"


# ---------------------------------------------------------------------------
# PriorSampler: loads a full priors.yaml and exposes batch sampling
# ---------------------------------------------------------------------------

class PriorSampler:
    """Load priors.yaml and sample all uncertain parameters as a batch.

    Parameters
    ----------
    specs : dict[str, PriorSpec]
        Mapping of dotted parameter name (e.g. ``"clinical.phase_1_to_2"``)
        to its ``PriorSpec``.
    """

    def __init__(self, specs: dict[str, PriorSpec]) -> None:
        if not specs:
            raise ValueError("PriorSampler requires at least one PriorSpec")
        self._specs = dict(specs)

    @property
    def names(self) -> list[str]:
        """Sorted list of parameter names."""
        return sorted(self._specs)

    @property
    def specs(self) -> dict[str, PriorSpec]:
        return dict(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, name: str) -> PriorSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown prior '{name}'. Available: {self.names}")
        return self._specs[name]

    def sample(self, n: int, rng: Generator) -> dict[str, np.ndarray]:
        """Draw *n* independent samples for every prior.

        Parameters
        ----------
        n : int
            Number of draws per parameter.
        rng : Generator
            Parent generator; a substream is spawned per parameter for
            reproducibility regardless of iteration order.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of parameter name to shape ``(n,)`` sample array.
        """
        if n < 1:
            raise ValueError(f"Sample size must be >= 1, got {n}")
        sub_seeds = rng.bit_generator.seed_seq.spawn(len(self._specs))
        result: dict[str, np.ndarray] = {}
        for (name, spec), ss in zip(sorted(self._specs.items()), sub_seeds):
            child_rng = np.random.default_rng(ss)
            result[name] = spec.sample(n, child_rng)
        return result

    def sample_scalar(self, rng: Generator) -> dict[str, float]:
        """Draw one scalar per prior (used inside the MC loop)."""
        sub_seeds = rng.bit_generator.seed_seq.spawn(len(self._specs))
        result: dict[str, float] = {}
        for (name, spec), ss in zip(sorted(self._specs.items()), sub_seeds):
            child_rng = np.random.default_rng(ss)
            result[name] = spec.sample_scalar(child_rng)
        return result

    @classmethod
    def from_yaml(cls, path: str | Path) -> PriorSampler:
        """Load a priors.yaml file and construct a PriorSampler.

        The YAML structure is nested by category::

            clinical:
              phase_1_to_2:
                family: "beta"
                params: {a: 52, b: 48}

        Parameter names are flattened to dotted form:
        ``"clinical.phase_1_to_2"``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Priors file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> PriorSampler:
        specs: dict[str, PriorSpec] = {}
        for category, entries in raw.items():
            if not isinstance(entries, dict):
                raise ValueError(f"Expected dict under '{category}', got {type(entries).__name__}")
            for param_name, definition in entries.items():
                if not isinstance(definition, dict):
                    raise ValueError(
                        f"Expected dict for '{category}.{param_name}', "
                        f"got {type(definition).__name__}"
                    )
                if "family" not in definition or "params" not in definition:
                    raise ValueError(
                        f"Prior '{category}.{param_name}' must have 'family' and 'params' keys"
                    )
                dotted = f"{category}.{param_name}"
                specs[dotted] = PriorSpec(
                    name=dotted,
                    family=definition["family"],
                    params=definition["params"],
                )
        return cls(specs)
    