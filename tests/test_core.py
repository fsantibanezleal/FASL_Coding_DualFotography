"""Tests for the dual photography core engine.

Validates transport matrix computation, SVD decomposition, dual image
generation, and pattern generation correctness.
"""

import numpy as np
import pytest

from src.core.dual import DualPhotographer
from src.core.patterns import (
    PatternType,
    generate_bernoulli_patterns,
    generate_canonical_patterns,
    generate_hadamard_patterns,
    generate_patterns,
)
from src.core.transport import TransportMatrix


class TestPatternGeneration:
    """Tests for illumination pattern generators."""

    def test_canonical_patterns_count(self):
        """Canonical patterns produce exactly height*width patterns."""
        patterns = list(generate_canonical_patterns(4, 4))
        assert len(patterns) == 16

    def test_canonical_patterns_are_one_hot(self):
        """Each canonical pattern has exactly one pixel on."""
        for pattern in generate_canonical_patterns(3, 3):
            assert pattern.sum() == 1.0
            assert pattern.shape == (3, 3)

    def test_hadamard_patterns_shape(self):
        """Hadamard patterns have correct shape."""
        patterns = list(generate_hadamard_patterns(4, 4))
        assert len(patterns) == 16
        for p in patterns:
            assert p.shape == (4, 4)
            assert set(np.unique(p)).issubset({0.0, 1.0})

    def test_bernoulli_patterns_count(self):
        """Bernoulli generator produces requested number of patterns."""
        patterns = list(generate_bernoulli_patterns(4, 4, n_patterns=10))
        assert len(patterns) == 10

    def test_bernoulli_reproducibility(self):
        """Same seed produces identical patterns."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = list(generate_bernoulli_patterns(4, 4, 5, rng1))
        p2 = list(generate_bernoulli_patterns(4, 4, 5, rng2))
        for a, b in zip(p1, p2):
            np.testing.assert_array_equal(a, b)

    def test_generate_patterns_convenience(self):
        """Convenience function dispatches correctly."""
        patterns = generate_patterns(4, 4, PatternType.CANONICAL)
        assert len(patterns) == 16


class TestTransportMatrix:
    """Tests for transport matrix computation and operations."""

    @pytest.fixture
    def simple_transport(self):
        """Create a simple known transport matrix."""
        # 4x4 camera, 4x4 projector -> T is (16, 16)
        rng = np.random.default_rng(42)
        T = rng.random((16, 16))
        return TransportMatrix(
            T=T, cam_shape=(4, 4), proj_shape=(4, 4)
        )

    def test_forward_shape(self, simple_transport):
        """Forward pass produces correct camera image shape."""
        pattern = np.ones((4, 4))
        result = simple_transport.forward(pattern)
        assert result.shape == (4, 4)

    def test_dual_shape(self, simple_transport):
        """Dual image has projector shape."""
        illum = np.ones((4, 4))
        result = simple_transport.dual(illum)
        assert result.shape == (4, 4)

    def test_svd_decomposition(self, simple_transport):
        """SVD decomposes and reconstructs correctly."""
        simple_transport.compute_svd()
        T_recon = (
            simple_transport.U
            @ np.diag(simple_transport.S)
            @ simple_transport.Vt
        )
        np.testing.assert_allclose(T_recon, simple_transport.T, atol=1e-10)

    def test_truncated_svd(self, simple_transport):
        """Truncated SVD produces lower-rank approximation."""
        simple_transport.compute_svd(rank=5)
        assert len(simple_transport.S) == 5
        T_approx = simple_transport.get_truncated_T(5)
        assert T_approx.shape == simple_transport.T.shape

    def test_from_canonical_patterns(self):
        """Transport matrix computed from canonical patterns matches original."""
        rng = np.random.default_rng(42)
        T_true = rng.random((9, 9))
        # Generate canonical patterns and simulate captures
        patterns = list(generate_canonical_patterns(3, 3))
        captures = [T_true @ p.flatten() for p in patterns]
        captures = [c.reshape(3, 3) for c in captures]

        tm = TransportMatrix.from_pattern_captures(
            patterns=patterns,
            captures=captures,
            proj_shape=(3, 3),
            cam_shape=(3, 3),
            method="direct",
        )
        np.testing.assert_allclose(tm.T, T_true, atol=1e-10)

    def test_save_load(self, simple_transport, tmp_path):
        """Save and load preserves the transport matrix."""
        path = str(tmp_path / "test_tm.npz")
        simple_transport.compute_svd(rank=5)
        simple_transport.save(path)
        loaded = TransportMatrix.load(path)
        np.testing.assert_allclose(loaded.T, simple_transport.T)
        assert loaded.cam_shape == simple_transport.cam_shape
        assert loaded.proj_shape == simple_transport.proj_shape

    def test_helmholtz_reciprocity_symmetry(self):
        """For a symmetric scene, T should be approximately symmetric."""
        # Create a symmetric transport matrix (symmetric BRDF)
        rng = np.random.default_rng(42)
        A = rng.random((9, 9))
        T_sym = (A + A.T) / 2  # Force symmetry
        tm = TransportMatrix(T=T_sym, cam_shape=(3, 3), proj_shape=(3, 3))

        # For symmetric T, dual should equal forward with same input
        pattern = rng.random((3, 3))
        forward = tm.forward(pattern)
        dual = tm.dual(pattern)
        np.testing.assert_allclose(forward, dual, atol=1e-10)


class TestDualPhotographer:
    """Tests for the high-level DualPhotographer orchestrator."""

    def test_simulation_workflow(self):
        """Full simulation workflow produces valid results."""
        photographer = DualPhotographer(
            proj_shape=(4, 4),
            cam_shape=(4, 4),
        )
        rng = np.random.default_rng(42)
        T = rng.random((16, 16))
        photographer.compute_transport_from_simulation(T)

        dual = photographer.dual_image()
        assert dual.shape == (4, 4)
        assert dual.min() >= 0

    def test_analyze_transport(self):
        """Analysis returns expected keys."""
        photographer = DualPhotographer(proj_shape=(4, 4), cam_shape=(4, 4))
        rng = np.random.default_rng(42)
        T = rng.random((16, 16))
        photographer.compute_transport_from_simulation(T)

        analysis = photographer.analyze_transport()
        assert "singular_values" in analysis
        assert "condition_number" in analysis
        assert "effective_rank_90" in analysis
        assert "effective_rank_99" in analysis
        assert "energy_cumulative" in analysis

    def test_relight(self):
        """Relighting produces valid camera image."""
        photographer = DualPhotographer(proj_shape=(4, 4), cam_shape=(4, 4))
        rng = np.random.default_rng(42)
        T = rng.random((16, 16))
        photographer.compute_transport_from_simulation(T)

        pattern = np.ones((4, 4))
        relighted = photographer.relight(pattern)
        assert relighted.shape == (4, 4)
        assert relighted.min() >= 0

    def test_error_before_compute(self):
        """Operations fail gracefully before transport is computed."""
        photographer = DualPhotographer(proj_shape=(4, 4), cam_shape=(4, 4))
        with pytest.raises(RuntimeError, match="Transport matrix not computed"):
            photographer.dual_image()
