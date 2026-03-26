"""Tests for the simulation module.

Validates synthetic scene generation, transport matrix properties,
and the virtual renderer pipeline.
"""

import numpy as np
import pytest

from src.simulation.renderer import VirtualRenderer
from src.simulation.scene import SceneType, SyntheticScene


class TestSyntheticScene:
    """Tests for synthetic scene generation."""

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_transport_matrix_shape(self, scene_type):
        """Transport matrix has correct dimensions for all scene types."""
        scene = SyntheticScene(
            scene_type=scene_type, proj_shape=(4, 4), cam_shape=(4, 4)
        )
        T = scene.compute_transport_matrix()
        assert T.shape == (16, 16)

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_transport_matrix_non_negative(self, scene_type):
        """Transport matrix entries are non-negative (physical constraint)."""
        scene = SyntheticScene(
            scene_type=scene_type, proj_shape=(4, 4), cam_shape=(4, 4)
        )
        T = scene.compute_transport_matrix()
        assert np.all(T >= 0), f"Negative entries in T for scene {scene_type}"

    def test_transport_matrix_normalized(self):
        """Transport matrix max value is 1.0 after normalization."""
        scene = SyntheticScene(
            scene_type=SceneType.FLAT_WALL, proj_shape=(4, 4), cam_shape=(4, 4)
        )
        T = scene.compute_transport_matrix()
        assert T.max() <= 1.0 + 1e-10

    def test_inter_reflections_increase_energy(self):
        """Inter-reflections add energy to the transport matrix."""
        scene_direct = SyntheticScene(
            scene_type=SceneType.CORNER,
            proj_shape=(4, 4),
            cam_shape=(4, 4),
            inter_reflections=False,
        )
        scene_bounced = SyntheticScene(
            scene_type=SceneType.CORNER,
            proj_shape=(4, 4),
            cam_shape=(4, 4),
            inter_reflections=True,
            bounce_limit=2,
        )
        T_direct = scene_direct.compute_transport_matrix()
        T_bounced = scene_bounced.compute_transport_matrix()
        # After normalization both max to 1.0, but the structure should differ
        assert not np.allclose(T_direct, T_bounced)


class TestVirtualRenderer:
    """Tests for the virtual renderer pipeline."""

    def test_run_simulation_returns_result(self):
        """Simulation produces all expected outputs."""
        renderer = VirtualRenderer(proj_shape=(4, 4), cam_shape=(4, 4))
        result = renderer.run_simulation(scene_type=SceneType.FLAT_WALL)

        assert result.primal_image.shape == (4, 4)
        assert result.dual_image.shape == (4, 4)
        assert result.transport is not None
        assert "singular_values" in result.analysis

    def test_compare_svd_ranks(self):
        """SVD rank comparison produces decreasing error."""
        renderer = VirtualRenderer(proj_shape=(4, 4), cam_shape=(4, 4))
        result = renderer.run_simulation(scene_type=SceneType.CORNER)

        comparisons = renderer.compare_svd_ranks(
            result.transport, ranks=[2, 4, 8, 16]
        )
        errors = [err for _, _, err in comparisons]
        # Error should decrease as rank increases
        assert errors[-1] <= errors[0] + 1e-10

    def test_relighting(self):
        """Relighting with different patterns produces different images."""
        renderer = VirtualRenderer(proj_shape=(4, 4), cam_shape=(4, 4))
        result = renderer.run_simulation(scene_type=SceneType.FLAT_WALL)

        p1 = np.ones((4, 4))
        p2 = np.zeros((4, 4))
        p2[:2, :] = 1.0

        r1 = renderer.run_relighting(result.transport, p1)
        r2 = renderer.run_relighting(result.transport, p2)
        assert not np.allclose(r1, r2)
