"""Distributions subpackage: prior sampling and copula-based dependence."""

from bioventure.distributions.copulas import (
    CopulaBase,
    GaussianCopula,
    IndependentCopula,
    recover_correlation,
    validate_correlation_matrix,
)
from bioventure.distributions.priors import PriorSampler, PriorSpec

__all__ = [
    "CopulaBase",
    "GaussianCopula",
    "IndependentCopula",
    "PriorSampler",
    "PriorSpec",
    "recover_correlation",
    "validate_correlation_matrix",
]
