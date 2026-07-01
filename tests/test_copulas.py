"""Tests for copulas: validation, shapes, determinism, correlation recovery."""

import numpy as np
import pytest

from bioventure.distributions.copulas import (
    GaussianCopula,
    IndependentCopula,
    recover_correlation,
    validate_correlation_matrix,
)
from bioventure.rng import RNGFactory

N = 50_000


# ---------------------------------------------------------------------------
# validate_correlation_matrix
# ---------------------------------------------------------------------------

class TestValidateCorrelationMatrix:
    def test_valid_identity(self):
        validate_correlation_matrix(np.eye(3))

    def test_valid_with_offdiag(self):
        m = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
        ])
        validate_correlation_matrix(m)

    def test_not_2d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            validate_correlation_matrix(np.ones(5))

    def test_not_square(self):
        with pytest.raises(ValueError, match="must be square"):
            validate_correlation_matrix(np.ones((3, 4)))

    def test_too_small(self):
        with pytest.raises(ValueError, match="at least 2x2"):
            validate_correlation_matrix(np.ones((1, 1)))

    def test_not_symmetric(self):
        m = np.array([
            [1.0, 0.3],
            [0.5, 1.0],
        ])
        with pytest.raises(ValueError, match="symmetric"):
            validate_correlation_matrix(m)

    def test_diagonal_not_one(self):
        m = np.array([
            [1.0, 0.2],
            [0.2, 0.9],
        ])
        with pytest.raises(ValueError, match="diagonal must be 1.0"):
            validate_correlation_matrix(m)

    def test_not_positive_semidefinite(self):
        m = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ])
        with pytest.raises(ValueError, match="positive semi-definite"):
            validate_correlation_matrix(m)


# ---------------------------------------------------------------------------
# recover_correlation
# ---------------------------------------------------------------------------

class TestRecoverCorrelation:
    def test_shape(self):
        samples = np.random.default_rng(42).random((100, 4))
        corr = recover_correlation(samples)
        assert corr.shape == (4, 4)

    def test_diagonal_ones(self):
        samples = np.random.default_rng(42).random((500, 3))
        corr = recover_correlation(samples)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            recover_correlation(np.ones(10))


# ---------------------------------------------------------------------------
# IndependentCopula
# ---------------------------------------------------------------------------

class TestIndependentCopula:
    def test_shape(self):
        cop = IndependentCopula(n_dim=4)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(100, rng)
        assert samples.shape == (100, 4)

    def test_n_dim_property(self):
        cop = IndependentCopula(n_dim=5)
        assert cop.n_dim == 5

    def test_uniform_bounds(self):
        cop = IndependentCopula(n_dim=3)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_marginal_uniformity(self):
        cop = IndependentCopula(n_dim=3)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        for d in range(3):
            col = samples[:, d]
            assert np.abs(col.mean() - 0.5) < 0.02
            assert np.abs(col.std() - (1.0 / np.sqrt(12))) < 0.02

    def test_near_zero_correlation(self):
        cop = IndependentCopula(n_dim=4)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        corr = recover_correlation(samples)
        offdiag = corr[np.triu_indices_from(corr, k=1)]
        assert np.all(np.abs(offdiag) < 0.03)

    def test_determinism(self):
        cop = IndependentCopula(n_dim=3)
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        s1 = cop.sample(50, r1)
        s2 = cop.sample(50, r2)
        np.testing.assert_array_equal(s1, s2)

    def test_n_dim_too_small(self):
        with pytest.raises(ValueError, match="n_dim must be >= 2"):
            IndependentCopula(n_dim=1)

    def test_sample_count_zero(self):
        cop = IndependentCopula(n_dim=2)
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="must be >= 1"):
            cop.sample(0, rng)


# ---------------------------------------------------------------------------
# GaussianCopula
# ---------------------------------------------------------------------------

class TestGaussianCopulaBasic:
    @pytest.fixture()
    def corr_3x3(self) -> np.ndarray:
        return np.array([
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ])

    def test_shape(self, corr_3x3):
        cop = GaussianCopula(corr_3x3)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(200, rng)
        assert samples.shape == (200, 3)

    def test_n_dim_property(self, corr_3x3):
        cop = GaussianCopula(corr_3x3)
        assert cop.n_dim == 3

    def test_uniform_bounds(self, corr_3x3):
        cop = GaussianCopula(corr_3x3)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_marginal_uniformity(self, corr_3x3):
        cop = GaussianCopula(corr_3x3)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        for d in range(3):
            col = samples[:, d]
            assert np.abs(col.mean() - 0.5) < 0.02
            assert np.abs(col.std() - (1.0 / np.sqrt(12))) < 0.02

    def test_determinism(self, corr_3x3):
        cop = GaussianCopula(corr_3x3)
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        s1 = cop.sample(100, r1)
        s2 = cop.sample(100, r2)
        np.testing.assert_array_equal(s1, s2)

    def test_sample_count_zero(self, corr_3x3):
        cop = GaussianCopula(corr_3x3)
        rng = RNGFactory(42).spawn_one()
        with pytest.raises(ValueError, match="must be >= 1"):
            cop.sample(0, rng)

    def test_correlation_matrix_copy(self, corr_3x3):
        cop = GaussianCopula(corr_3x3)
        retrieved = cop.correlation_matrix
        retrieved[0, 1] = 999.0
        assert cop.correlation_matrix[0, 1] == pytest.approx(0.6)


class TestGaussianCopulaCorrelationRecovery:
    """Verify that empirical correlations approximately match the target."""

    def test_2d_high_correlation(self):
        target = np.array([
            [1.0, 0.8],
            [0.8, 1.0],
        ])
        cop = GaussianCopula(target)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        recovered = recover_correlation(samples)
        np.testing.assert_allclose(recovered, target, atol=0.03)

    def test_2d_negative_correlation(self):
        target = np.array([
            [1.0, -0.6],
            [-0.6, 1.0],
        ])
        cop = GaussianCopula(target)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        recovered = recover_correlation(samples)
        np.testing.assert_allclose(recovered, target, atol=0.03)

    def test_5d_mixed_correlation(self):
        target = np.array([
            [ 1.00,  0.50,  0.30,  0.20, -0.10],
            [ 0.50,  1.00,  0.40,  0.15, -0.05],
            [ 0.30,  0.40,  1.00,  0.10, -0.05],
            [ 0.20,  0.15,  0.10,  1.00,  0.30],
            [-0.10, -0.05, -0.05,  0.30,  1.00],
        ])
        cop = GaussianCopula(target)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        recovered = recover_correlation(samples)
        np.testing.assert_allclose(recovered, target, atol=0.03)

    def test_identity_gives_near_independent(self):
        target = np.eye(4)
        cop = GaussianCopula(target)
        rng = RNGFactory(42).spawn_one()
        samples = cop.sample(N, rng)
        recovered = recover_correlation(samples)
        offdiag = recovered[np.triu_indices_from(recovered, k=1)]
        assert np.all(np.abs(offdiag) < 0.03)


class TestGaussianCopulaValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            GaussianCopula(np.ones((2, 3)))

    def test_rejects_asymmetric(self):
        m = np.array([[1.0, 0.3], [0.5, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            GaussianCopula(m)

    def test_rejects_bad_diagonal(self):
        m = np.array([[0.9, 0.2], [0.2, 1.0]])
        with pytest.raises(ValueError, match="diagonal"):
            GaussianCopula(m)

    def test_rejects_not_psd(self):
        m = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ])
        with pytest.raises(ValueError, match="positive semi-definite"):
            GaussianCopula(m)

    def test_accepts_list_input(self):
        m = [[1.0, 0.5], [0.5, 1.0]]
        cop = GaussianCopula(m)
        assert cop.n_dim == 2
        