"""Tests for transport matrix analysis features (NMF, frequency, sparsity)."""

import numpy as np
import pytest

from src.core.transport import TransportMatrix


@pytest.fixture
def transport_with_structure():
    """Create a transport matrix with realistic structure (not random)."""
    rng = np.random.default_rng(42)
    # Low-rank + sparse structure mimics real transport
    n = 64
    rank = 8
    U = np.abs(rng.standard_normal((n, rank)))
    V = np.abs(rng.standard_normal((rank, n)))
    T = U @ V
    T /= T.max()
    # Add some sparsity
    T[T < 0.1] = 0
    return TransportMatrix(T=T, cam_shape=(8, 8), proj_shape=(8, 8))


class TestFrequencyAnalysis:
    def test_returns_expected_keys(self, transport_with_structure):
        result = transport_with_structure.frequency_analysis()
        assert "power_spectrum" in result
        assert "radial_profile" in result
        assert "dc_fraction" in result
        assert "high_freq_fraction" in result

    def test_dc_fraction_range(self, transport_with_structure):
        result = transport_with_structure.frequency_analysis()
        assert 0 <= result["dc_fraction"] <= 1

    def test_high_freq_fraction_range(self, transport_with_structure):
        result = transport_with_structure.frequency_analysis()
        assert 0 <= result["high_freq_fraction"] <= 1


class TestSparsity:
    def test_returns_expected_keys(self, transport_with_structure):
        result = transport_with_structure.sparsity()
        assert "nnz" in result
        assert "nnz_fraction" in result
        assert "density_per_column" in result

    def test_nnz_consistent(self, transport_with_structure):
        result = transport_with_structure.sparsity()
        assert result["nnz"] <= transport_with_structure.T.size
        assert 0 <= result["nnz_fraction"] <= 1


class TestReciprocity:
    def test_symmetric_matrix_has_zero_error(self):
        """A symmetric T has zero reciprocity error."""
        T = np.array([[1, 2], [2, 3]], dtype=float)
        tm = TransportMatrix(T=T, cam_shape=(1, 2), proj_shape=(1, 2))
        assert tm.reciprocity_error() < 1e-10

    def test_asymmetric_has_positive_error(self):
        """An asymmetric T has positive reciprocity error."""
        T = np.array([[1, 0], [5, 1]], dtype=float)
        tm = TransportMatrix(T=T, cam_shape=(1, 2), proj_shape=(1, 2))
        assert tm.reciprocity_error() > 0.1


class TestNMF:
    def test_nmf_reconstruction(self, transport_with_structure):
        """NMF produces a reasonable reconstruction."""
        W, H = transport_with_structure.compute_nmf(n_components=8)
        T_approx = W @ H
        T_orig = np.maximum(transport_with_structure.T, 0)
        rel_error = np.linalg.norm(T_orig - T_approx) / (np.linalg.norm(T_orig) + 1e-10)
        assert rel_error < 0.5  # Reasonable reconstruction

    def test_nmf_non_negative(self, transport_with_structure):
        """NMF factors are non-negative."""
        W, H = transport_with_structure.compute_nmf(n_components=4)
        assert np.all(W >= -1e-10)
        assert np.all(H >= -1e-10)

    def test_nmf_shapes(self, transport_with_structure):
        """NMF returns correct shapes."""
        W, H = transport_with_structure.compute_nmf(n_components=8)
        assert W.shape == (64, 8)
        assert H.shape == (8, 64)
