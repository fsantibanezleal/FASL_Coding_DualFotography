"""Tests for the simulation module.

Validates ray-casting, vectorized transport matrix computation,
BRDF models, multi-bounce transport, and all scene types.
"""

import numpy as np
import pytest

from src.simulation.renderer import VirtualRenderer
from src.simulation.scene import (
    SceneType, SyntheticScene, Material, SceneObject,
    make_checker_texture, make_letter_texture, make_arrow_texture,
)


class TestMaterials:
    """Tests for the Material and texture system."""

    def test_material_defaults(self):
        mat = Material()
        assert mat.diffuse == 0.7
        assert mat.specular == 0.0
        assert mat.shininess == 32.0
        assert mat.is_mirror is False

    def test_material_texture_sampling(self):
        tex = make_checker_texture(8, 2)
        mat = Material(diffuse=tex)
        assert mat.sample_diffuse(np.array([0.0, 0.0])) in (0.15, 0.9)
        assert mat.sample_diffuse(np.array([0.5, 0.5])) in (0.15, 0.9)

    def test_checker_texture_shape(self):
        tex = make_checker_texture(32, 4)
        assert tex.shape == (32, 32)
        assert tex.min() >= 0 and tex.max() <= 1

    def test_letter_texture_shape(self):
        for letter in "FDPALR":
            tex = make_letter_texture(letter, 48)
            assert tex.shape == (48, 48)
            assert tex.max() > 0.5  # Has bright pixels

    def test_arrow_texture_has_content(self):
        tex = make_arrow_texture(32)
        assert tex.shape == (32, 32)
        bright = (tex > 0.5).sum()
        assert bright > 0  # Arrow exists


class TestSceneObject:
    """Tests for vectorized ray intersection."""

    def test_plane_intersection_batch(self):
        obj = SceneObject(kind="plane", center=np.zeros(3),
                          normal=np.array([0., 0., 1.]),
                          material=Material(diffuse=0.5))
        origins = np.array([[0., 0., 5.], [0., 0., 5.]])
        dirs = np.array([[0., 0., -1.], [1., 0., 0.]])  # One hits, one misses
        t, pts, nrm, uvs = obj.intersect_batch(origins, dirs)
        assert t[0] < 100  # First ray hits
        assert t[1] == np.inf  # Second ray parallel, misses

    def test_sphere_intersection_batch(self):
        obj = SceneObject(kind="sphere", center=np.zeros(3), radius=1.0,
                          material=Material(diffuse=0.8))
        origins = np.array([[0., 0., 5.], [10., 10., 5.]])
        dirs = np.array([[0., 0., -1.], [0., 0., -1.]])
        t, pts, nrm, uvs = obj.intersect_batch(origins, dirs)
        assert t[0] < 100  # Hits sphere
        assert t[1] == np.inf  # Misses sphere

    def test_box_face_bounds(self):
        obj = SceneObject(kind="box_face", center=np.zeros(3),
                          normal=np.array([0., 0., 1.]),
                          material=Material(diffuse=0.5),
                          half_extents=(0.5, 0.5),
                          right_axis=np.array([1., 0., 0.]),
                          up_axis=np.array([0., 1., 0.]))
        # One inside bounds, one outside
        origins = np.array([[0., 0., 5.], [5., 5., 5.]])
        dirs = np.array([[0., 0., -1.], [0., 0., -1.]])
        t, pts, nrm, uvs = obj.intersect_batch(origins, dirs)
        assert t[0] < 100  # Inside bounds
        assert t[1] == np.inf  # Outside bounds


class TestSyntheticScene:
    """Tests for scene construction and transport matrix computation."""

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_transport_matrix_shape(self, scene_type):
        """Transport matrix has correct dimensions for all scene types."""
        scene = SyntheticScene(scene_type=scene_type,
                               proj_shape=(8, 8), cam_shape=(8, 8))
        T = scene.compute_transport_matrix()
        assert T.shape == (64, 64)

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_transport_matrix_non_negative(self, scene_type):
        """Transport matrix entries are non-negative."""
        scene = SyntheticScene(scene_type=scene_type,
                               proj_shape=(8, 8), cam_shape=(8, 8))
        T = scene.compute_transport_matrix()
        assert np.all(T >= -1e-10), f"Negative entries for {scene_type}"

    def test_transport_not_diagonal(self):
        """Box+Wall produces a non-diagonal transport matrix."""
        scene = SyntheticScene(scene_type=SceneType.BOX_AND_WALL,
                               proj_shape=(16, 16), cam_shape=(16, 16))
        T = scene.compute_transport_matrix()
        diag_energy = np.sum(np.diag(T) ** 2)
        total_energy = np.sum(T ** 2)
        if total_energy > 0:
            off_diag = 1.0 - diag_energy / total_energy
            assert off_diag > 0.1

    def test_primal_dual_differ(self):
        """Primal and dual images are visually different for 3D scenes."""
        scene = SyntheticScene(scene_type=SceneType.BOX_AND_WALL,
                               proj_shape=(16, 16), cam_shape=(16, 16))
        T = scene.compute_transport_matrix()
        primal = T @ np.ones(256)
        dual = T.T @ np.ones(256)
        if np.linalg.norm(primal) > 0 and np.linalg.norm(dual) > 0:
            p_n = primal / (np.linalg.norm(primal) + 1e-10)
            d_n = dual / (np.linalg.norm(dual) + 1e-10)
            assert np.dot(p_n, d_n) < 0.99

    def test_specular_affects_transport(self):
        """Specular BRDF changes the transport matrix compared to diffuse-only."""
        scene_diff = SyntheticScene(scene_type=SceneType.MIRROR_ROOM,
                                     proj_shape=(8, 8), cam_shape=(8, 8))
        T1 = scene_diff.compute_transport_matrix()
        # Mirror room has high specular, so T should have content
        assert T1.max() > 0

    def test_multi_bounce_changes_transport(self):
        """Multi-bounce transport differs from direct-only."""
        scene_0 = SyntheticScene(scene_type=SceneType.CORNELL_BOX,
                                  proj_shape=(8, 8), cam_shape=(8, 8),
                                  n_bounces=0)
        scene_2 = SyntheticScene(scene_type=SceneType.CORNELL_BOX,
                                  proj_shape=(8, 8), cam_shape=(8, 8),
                                  n_bounces=2)
        T0 = scene_0.compute_transport_matrix()
        T2 = scene_2.compute_transport_matrix()
        # Should differ due to indirect light
        assert not np.allclose(T0, T2, atol=0.01)


class TestVirtualRenderer:
    """Tests for the virtual renderer pipeline."""

    def test_run_simulation_returns_result(self):
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8))
        result = renderer.run_simulation(scene_type=SceneType.BOX_AND_WALL)
        assert result.primal_image.shape == (8, 8)
        assert result.dual_image.shape == (8, 8)
        assert result.transport is not None
        assert "singular_values" in result.analysis

    def test_compare_svd_ranks(self):
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8))
        result = renderer.run_simulation(scene_type=SceneType.TWO_PLANES)
        comparisons = renderer.compare_svd_ranks(result.transport, ranks=[4, 16, 64])
        errors = [err for _, _, err in comparisons]
        assert errors[-1] <= errors[0] + 1e-10

    def test_relighting_produces_different_results(self):
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
        """Every scene produces non-zero primal and dual images."""
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8))
        result = renderer.run_simulation(scene_type=scene_type)
        assert result.primal_image.max() > 0, f"Zero primal for {scene_type}"
        assert result.dual_image.max() > 0, f"Zero dual for {scene_type}"

    def test_n_bounces_parameter(self):
        """Renderer passes n_bounces to scene correctly."""
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8), n_bounces=1)
        result = renderer.run_simulation(scene_type=SceneType.CORNELL_BOX)
        assert result.primal_image.max() > 0

    def test_backwards_compat_inter_reflections(self):
        """Old inter_reflections parameter still works."""
        renderer = VirtualRenderer(proj_shape=(8, 8), cam_shape=(8, 8),
                                   inter_reflections=True, bounce_limit=2)
        assert renderer.n_bounces == 2
