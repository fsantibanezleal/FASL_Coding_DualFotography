"""Tests for the simulation module.

Validates synthetic scene generation via ray-casting, transport matrix
properties, and the virtual renderer pipeline.
"""

import numpy as np
import pytest

from src.simulation.renderer import VirtualRenderer
from src.simulation.scene import SceneType, SyntheticScene


class TestSyntheticScene:
    """Tests for ray-cast synthetic scene generation."""

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_transport_matrix_shape(self, scene_type):
        """Transport matrix has correct dimensions for all scene types."""
        scene = SyntheticScene(
            scene_type=scene_type, proj_shape=(8, 8), cam_shape=(8, 8)
        )
        T = scene.compute_transport_matrix()
        assert T.shape == (64, 64)

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_transport_matrix_non_negative(self, scene_type):
        """Transport matrix entries are non-negative (physical constraint)."""
        scene = SyntheticScene(
            scene_type=scene_type, proj_shape=(8, 8), cam_shape=(8, 8)
        )
        T = scene.compute_transport_matrix()
        assert np.all(T >= -1e-10), f"Negative entries in T for scene {scene_type}"

    def test_transport_not_diagonal(self):
        """Box+Wall scene produces a non-diagonal transport matrix."""
        scene = SyntheticScene(
            scene_type=SceneType.BOX_AND_WALL,
            proj_shape=(16, 16),
            cam_shape=(16, 16),
        )
        T = scene.compute_transport_matrix()
        # Check that off-diagonal entries exist (matrix is not identity-like)
        diag = np.diag(T)
        off_diag_energy = np.sum(T ** 2) - np.sum(diag ** 2)
        total_energy = np.sum(T ** 2)
        if total_energy > 0:
            off_diag_ratio = off_diag_energy / total_energy
            # Should have significant off-diagonal energy for 3D scenes
            assert off_diag_ratio > 0.1, (
                f"Transport matrix too diagonal (off-diag ratio = {off_diag_ratio:.4f}). "
                "This means primal and dual images will look nearly identical."
            )

    def test_primal_dual_differ(self):
        """Primal and dual images should be visually different for 3D scenes."""
        scene = SyntheticScene(
            scene_type=SceneType.BOX_AND_WALL,
            proj_shape=(16, 16),
            cam_shape=(16, 16),
        )
        T = scene.compute_transport_matrix()

        # Primal: uniform illumination
        primal = T @ np.ones(256)
        # Dual: T^T with uniform illumination
        dual = T.T @ np.ones(256)

        # They should not be identical
        if np.linalg.norm(primal) > 0 and np.linalg.norm(dual) > 0:
            # Normalize both
            p_norm = primal / (np.linalg.norm(primal) + 1e-10)
            d_norm = dual / (np.linalg.norm(dual) + 1e-10)
            correlation = np.dot(p_norm, d_norm)
            assert correlation < 0.99, (
                f"Primal and dual too similar (correlation = {correlation:.4f})"
            )


class TestVirtualRenderer:
    """Tests for the virtual renderer pipeline."""

    def test_run_simulation_returns_result(self):
        """Simulation produces all expected outputs."""
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8))
        result = renderer.run_simulation(scene_type=SceneType.BOX_AND_WALL)

        assert result.primal_image.shape == (8, 8)
        assert result.dual_image.shape == (8, 8)
        assert result.transport is not None
        assert "singular_values" in result.analysis

    def test_compare_svd_ranks(self):
        """SVD rank comparison produces decreasing error."""
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8))
        result = renderer.run_simulation(scene_type=SceneType.TWO_PLANES)

        comparisons = renderer.compare_svd_ranks(
            result.transport, ranks=[4, 16, 64]
        )
        errors = [err for _, _, err in comparisons]
        assert errors[-1] <= errors[0] + 1e-10

    def test_relighting_produces_different_results(self):
        """Relighting with different patterns produces different images."""
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8))
        result = renderer.run_simulation(scene_type=SceneType.SPHERE_ON_PLANE)

        p1 = np.ones((8, 8))
        p2 = np.zeros((8, 8))
        p2[:4, :] = 1.0

        r1 = renderer.run_relighting(result.transport, p1)
        r2 = renderer.run_relighting(result.transport, p2)
        assert not np.allclose(r1, r2)

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_all_scenes_produce_nonzero_images(self, scene_type):
        """Every scene type produces non-zero primal and dual images."""
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8))
        result = renderer.run_simulation(scene_type=scene_type)
        assert result.primal_image.max() > 0, f"Zero primal for {scene_type}"
        assert result.dual_image.max() > 0, f"Zero dual for {scene_type}"
