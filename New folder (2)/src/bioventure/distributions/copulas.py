"""Pluggable copula layer: correlate uniform marginals for Monte Carlo sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from scipy import stats


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_correlation_matrix(matrix: np.ndarray) -> None:
    """Check that a matrix is a valid correlation matrix.

    A valid correlation matrix is:
    - 2-D and square
    - Symmetric
    - Has ones on the diagonal
    - Positive semi-definite (all eigenvalues >= 0)

    Parameters
    ----------
    matrix : np.ndarray
        Candidate correlation matrix.

    Raises
    ------
    ValueError
        If any condition is violated.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Correlation matrix must be 2-D, got {matrix.ndim}-D")

    n, m = matrix.shape
    if n != m:
        raise ValueError(f"Correlation matrix must be square, got shape ({n}, {m})")

    if n < 2:
        raise ValueError(f"Correlation matrix must be at least 2x2, got ({n}, {m})")

    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Correlation matrix must be symmetric")

    diag = np.diag(matrix)
    if not np.allclose(diag, 1.0, atol=1e-10):
        offenders = np.where(~np.isclose(diag, 1.0, atol=1e-10))[0]
        raise ValueError(
            f"Correlation matrix diagonal must be 1.0, "
            f"violations at indices {offenders.tolist()}: {diag[offenders].tolist()}"
        )

    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues < -1e-8):
        min_eig = float(eigenvalues.min())
        raise ValueError(
            f"Correlation matrix must be positive semi-definite, "
            f"smallest eigenvalue = {min_eig:.6e}"
        )


def recover_correlation(samples: np.ndarray) -> np.ndarray:
    """Compute the Pearson correlation matrix of a samples array.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(n_samples, n_dimensions)`` array.

    Returns
    -------
    np.ndarray
        Shape ``(d, d)`` empirical correlation matrix.
    """
    if samples.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {samples.ndim}-D")
    return np.corrcoef(samples, rowvar=False)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CopulaBase(ABC):
    """Protocol for copula samplers.

    All copulas produce uniform marginals on [0, 1] with a specified
    dependence structure.
    """

    @property
    @abstractmethod
    def n_dim(self) -> int:
        """Number of dimensions (marginals)."""

    @abstractmethod
    def sample(self, n: int, rng: Generator) -> np.ndarray:
        """Draw correlated uniform samples.

        Parameters
        ----------
        n : int
            Number of samples (rows).
        rng : Generator
            NumPy random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Shape ``(n, n_dim)`` array with values in [0, 1].
        """


# ---------------------------------------------------------------------------
# Independent copula
# ---------------------------------------------------------------------------

class IndependentCopula(CopulaBase):
    """Independence copula: marginals are uncorrelated uniform draws.

    Useful as a baseline to isolate the effect of correlation on outcomes.

    Parameters
    ----------
    n_dim : int
        Number of dimensions. Must be >= 2.
    """

    def __init__(self, n_dim: int) -> None:
        if n_dim < 2:
            raise ValueError(f"n_dim must be >= 2, got {n_dim}")
        self._n_dim = n_dim

    @property
    def n_dim(self) -> int:
        return self._n_dim

    def sample(self, n: int, rng: Generator) -> np.ndarray:
        """Draw independent uniform samples.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : Generator
            NumPy random generator.

        Returns
        -------
        np.ndarray
            Shape ``(n, n_dim)`` of independent U(0,1) draws.
        """
        if n < 1:
            raise ValueError(f"Sample count must be >= 1, got {n}")
        return rng.random((n, self._n_dim))


# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------

class GaussianCopula(CopulaBase):
    """Gaussian copula: symmetric dependence, no tail dependence.

    Algorithm:
    1. Cholesky-decompose the correlation matrix: ``L = chol(Σ)``
    2. Draw ``Z ~ N(0, I)`` of shape ``(n, d)``
    3. Correlate: ``Y = Z @ L.T``  (now ``Y ~ N(0, Σ)``)
    4. Map to uniforms: ``U = Φ(Y)``

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Shape ``(d, d)`` valid correlation matrix.
    """

    def __init__(self, correlation_matrix: np.ndarray) -> None:
        matrix = np.asarray(correlation_matrix, dtype=np.float64)
        validate_correlation_matrix(matrix)
        self._corr = matrix
        self._n_dim = matrix.shape[0]
        self._cholesky = np.linalg.cholesky(matrix)

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Return a copy of the target correlation matrix."""
        return self._corr.copy()

    def sample(self, n: int, rng: Generator) -> np.ndarray:
        """Draw correlated uniform samples via the Gaussian copula.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : Generator
            NumPy random generator.

        Returns
        -------
        np.ndarray
            Shape ``(n, n_dim)`` with values in [0, 1].
        """
        if n < 1:
            raise ValueError(f"Sample count must be >= 1, got {n}")

        z = rng.standard_normal((n, self._n_dim))
        y = z @ self._cholesky.T
        u = stats.norm.cdf(y)
        return u
    