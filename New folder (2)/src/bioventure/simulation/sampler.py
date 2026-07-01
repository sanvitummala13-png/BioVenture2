"""Parameter sampler: draw correlated parameter vectors via copula + inverse CDF.

Pipeline per draw:
1. Copula produces correlated uniform samples U ∈ [0,1]^d.
2. Each uniform is transformed through its prior's inverse CDF (ppf)
   to produce the parameter value with the correct marginal distribution.
3. Parameters not in the copula are drawn independently from their priors.

This preserves each marginal distribution exactly while introducing the
dependence structure specified by the copula's correlation matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.random import Generator

from bioventure.distributions.copulas import (
    CopulaBase,
    GaussianCopula,
    IndependentCopula,
)
from bioventure.distributions.priors import PriorSampler


class ParameterSampler:
    """Draw correlated or independent parameter vectors for Monte Carlo.

    Parameters
    ----------
    prior_sampler : PriorSampler
        Contains all uncertain parameter priors.
    copula : CopulaBase or None
        Copula for correlated parameters.  ``None`` means all
        parameters are drawn independently.
    correlated_params : list[str] or None
        Ordered list of parameter names matching copula dimensions.
        Required when *copula* is not ``None``.
    """

    def __init__(
        self,
        prior_sampler: PriorSampler,
        copula: CopulaBase | None = None,
        correlated_params: list[str] | None = None,
    ) -> None:
        self._priors = prior_sampler

        if copula is not None:
            if correlated_params is None:
                raise ValueError(
                    "correlated_params is required when copula is provided"
                )
            if len(correlated_params) != copula.n_dim:
                raise ValueError(
                    f"correlated_params has {len(correlated_params)} entries "
                    f"but copula has {copula.n_dim} dimensions"
                )
            for name in correlated_params:
                if name not in prior_sampler.specs:
                    raise ValueError(
                        f"Correlated param '{name}' not found in priors. "
                        f"Available: {prior_sampler.names}"
                    )
            self._copula = copula
            self._corr_params = list(correlated_params)
            corr_set = set(correlated_params)
            self._uncorr_params = sorted(
                n for n in prior_sampler.names if n not in corr_set
            )
        else:
            self._copula = None
            self._corr_params = []
            self._uncorr_params = sorted(prior_sampler.names)

    @property
    def parameter_names(self) -> list[str]:
        """All parameter names (sorted)."""
        return sorted(self._corr_params + self._uncorr_params)

    @property
    def correlated_names(self) -> list[str]:
        """Parameter names coupled through the copula."""
        return list(self._corr_params)

    @property
    def independent_names(self) -> list[str]:
        """Parameter names drawn independently."""
        return list(self._uncorr_params)

    @property
    def n_parameters(self) -> int:
        return len(self._corr_params) + len(self._uncorr_params)

    def draw(self, rng: Generator) -> dict[str, float]:
        """Draw one parameter vector.

        Parameters
        ----------
        rng : Generator
            NumPy random generator (state is advanced).

        Returns
        -------
        dict[str, float]
            Parameter name → scalar value.
        """
        result: dict[str, float] = {}

        if self._copula is not None and self._corr_params:
            u = self._copula.sample(1, rng)[0]
            for i, name in enumerate(self._corr_params):
                spec = self._priors[name]
                result[name] = float(spec.dist.ppf(u[i]))

        for name in self._uncorr_params:
            result[name] = self._priors[name].sample_scalar(rng)

        return result

    def draw_batch(self, n: int, rng: Generator) -> dict[str, np.ndarray]:
        """Draw *n* parameter vectors (vectorised).

        Parameters
        ----------
        n : int
            Number of draws.
        rng : Generator
            NumPy random generator.

        Returns
        -------
        dict[str, np.ndarray]
            Parameter name → shape ``(n,)`` array.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        result: dict[str, np.ndarray] = {}

        if self._copula is not None and self._corr_params:
            u = self._copula.sample(n, rng)
            for i, name in enumerate(self._corr_params):
                spec = self._priors[name]
                result[name] = spec.dist.ppf(u[:, i])

        if self._uncorr_params:
            sub_seeds = rng.bit_generator.seed_seq.spawn(len(self._uncorr_params))
            for name, ss in zip(self._uncorr_params, sub_seeds):
                child_rng = np.random.default_rng(ss)
                result[name] = self._priors[name].sample(n, child_rng)

        return result

    @classmethod
    def from_configs(
        cls,
        priors_path: str | Path,
        correlations_path: str | Path,
    ) -> ParameterSampler:
        """Build a sampler from YAML config files.

        Parameters
        ----------
        priors_path : path-like
            Path to ``priors.yaml``.
        correlations_path : path-like
            Path to ``correlations.yaml``.
        """
        prior_sampler = PriorSampler.from_yaml(priors_path)

        corr_path = Path(correlations_path)
        if not corr_path.exists():
            raise FileNotFoundError(
                f"Correlations file not found: {corr_path}"
            )
        with open(corr_path) as f:
            corr_cfg = yaml.safe_load(f)

        copula_type = corr_cfg.get("copula_type", "gaussian").lower()
        matrix = np.array(corr_cfg["matrix"], dtype=np.float64)
        param_names = corr_cfg["parameters"]

        if copula_type == "gaussian":
            copula = GaussianCopula(matrix)
        elif copula_type == "independent":
            copula = IndependentCopula(n_dim=len(param_names))
        else:
            raise ValueError(
                f"Unknown copula_type '{copula_type}'. "
                f"Supported: 'gaussian', 'independent'"
            )

        return cls(prior_sampler, copula, param_names)
    