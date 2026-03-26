"""Tests for the Dash frontend components and callbacks.

Uses Dash's built-in testing utilities to verify that callbacks
produce correct outputs for given inputs.
"""

import numpy as np
import pytest

from src.frontend.app import (
    _make_relight_pattern,
    _numpy_to_b64_img,
    run_simulation,
    run_relighting,
    update_svd_options,
)


class TestHelperFunctions:
    """Tests for frontend helper/utility functions."""

    def test_numpy_to_b64_returns_data_uri(self):
        """Base64 conversion returns a valid data URI."""
        arr = np.random.default_rng(0).random((8, 8))
        result = _numpy_to_b64_img(arr)
        assert result.startswith("data:image/png;base64,")
        assert len(result) > 50

    def test_numpy_to_b64_handles_constant_image(self):
        """Handles a constant (uniform) image without division by zero."""
        arr = np.ones((8, 8)) * 0.5
        result = _numpy_to_b64_img(arr)
        assert result.startswith("data:image/png;base64,")

    def test_numpy_to_b64_handles_zeros(self):
        """Handles an all-zero image."""
        arr = np.zeros((4, 4))
        result = _numpy_to_b64_img(arr)
        assert result.startswith("data:image/png;base64,")

    @pytest.mark.parametrize("pattern_name", [
        "white", "left", "right", "top", "bottom",
        "spot", "h_stripes", "v_stripes", "diagonal", "random",
    ])
    def test_relight_pattern_shape(self, pattern_name):
        """All relight patterns produce correct shape."""
        p = _make_relight_pattern(pattern_name, 8, 8)
        assert p.shape == (8, 8)
        assert p.min() >= 0.0
        assert p.max() <= 1.0

    def test_relight_pattern_white_is_ones(self):
        """White pattern is all ones."""
        p = _make_relight_pattern("white", 4, 4)
        np.testing.assert_array_equal(p, np.ones((4, 4)))

    def test_relight_pattern_left_right_complement(self):
        """Left and right patterns are complementary."""
        left = _make_relight_pattern("left", 8, 8)
        right = _make_relight_pattern("right", 8, 8)
        combined = left + right
        np.testing.assert_array_equal(combined, np.ones((8, 8)))

    def test_relight_unknown_defaults_to_white(self):
        """Unknown pattern name defaults to all-white."""
        p = _make_relight_pattern("nonexistent", 4, 4)
        np.testing.assert_array_equal(p, np.ones((4, 4)))


class TestCallbacks:
    """Tests for Dash callback functions (invoked directly, not via HTTP)."""

    def test_update_svd_options_returns_valid_list(self):
        """SVD options callback returns a list with Full option first."""
        options = update_svd_options("8")
        assert isinstance(options, list)
        assert options[0]["value"] == "0"
        assert options[0]["label"] == "Full (no truncation)"
        # Should include some power-of-2 ranks
        values = [o["value"] for o in options]
        assert "1" in values
        assert "2" in values

    def test_update_svd_options_scales_with_resolution(self):
        """Higher resolution produces more SVD rank options."""
        opts_8 = update_svd_options("8")
        opts_16 = update_svd_options("16")
        assert len(opts_16) > len(opts_8)

    def test_run_simulation_produces_outputs(self):
        """Simulation callback returns all 6 outputs with valid types."""
        primal, dual, fig, analysis, status, store = run_simulation(
            n_clicks=1,
            scene_type="flat_wall",
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            inter_reflections=False,
            proj_x_str="-0.3",
            cam_x_str="0.3",
        )
        assert primal.startswith("data:image/png;base64,")
        assert dual.startswith("data:image/png;base64,")
        assert fig is not None
        assert isinstance(analysis, list)
        assert len(analysis) >= 5
        assert store is not None
        assert "T" in store
        assert "proj_shape" in store

    def test_run_simulation_prevents_update_on_zero_clicks(self):
        """Simulation is not triggered when n_clicks is 0."""
        from dash.exceptions import PreventUpdate
        with pytest.raises(PreventUpdate):
            run_simulation(0, "flat_wall", "8", "0.8", "0", False, "-0.3", "0.3")

    def test_run_simulation_with_svd_rank(self):
        """Simulation works with SVD truncation enabled."""
        primal, dual, fig, analysis, status, store = run_simulation(
            n_clicks=1,
            scene_type="corner",
            resolution_str="8",
            albedo_str="0.5",
            svd_rank_str="4",
            inter_reflections=False,
            proj_x_str="-0.5",
            cam_x_str="0.5",
        )
        assert primal.startswith("data:image/png;base64,")
        # Check that SVD rank alert is present in analysis
        has_rank_alert = any("SVD rank" in str(child) for child in analysis)
        assert has_rank_alert

    def test_run_simulation_with_inter_reflections(self):
        """Simulation works with inter-reflections enabled."""
        primal, dual, fig, analysis, status, store = run_simulation(
            n_clicks=1,
            scene_type="corner",
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            inter_reflections=True,
            proj_x_str="-0.3",
            cam_x_str="0.3",
        )
        assert primal.startswith("data:image/png;base64,")

    def test_run_relighting_produces_images(self):
        """Relighting callback produces two valid images."""
        # First run a simulation to get store_data
        _, _, _, _, _, store = run_simulation(
            n_clicks=1,
            scene_type="flat_wall",
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            inter_reflections=False,
            proj_x_str="-0.3",
            cam_x_str="0.3",
        )
        pattern_img, result_img = run_relighting(
            n_clicks=1,
            pattern_name="left",
            store_data=store,
        )
        assert pattern_img.startswith("data:image/png;base64,")
        assert result_img.startswith("data:image/png;base64,")

    def test_run_relighting_prevents_update_without_store(self):
        """Relighting fails gracefully when no simulation has been run."""
        from dash.exceptions import PreventUpdate
        with pytest.raises(PreventUpdate):
            run_relighting(n_clicks=1, pattern_name="white", store_data=None)

    @pytest.mark.parametrize("scene", [
        "flat_wall", "corner", "v_groove", "sphere", "checkerboard", "textured_wall"
    ])
    def test_all_scene_types_work(self, scene):
        """Every scene type can be simulated without errors."""
        primal, dual, fig, analysis, status, store = run_simulation(
            n_clicks=1,
            scene_type=scene,
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            inter_reflections=False,
            proj_x_str="-0.3",
            cam_x_str="0.3",
        )
        assert primal.startswith("data:image/png;base64,")
        assert store is not None
