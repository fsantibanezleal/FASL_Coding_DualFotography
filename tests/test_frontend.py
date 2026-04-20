"""Tests for the Dash frontend components and callbacks.

Uses Dash's built-in testing utilities to verify that callbacks
produce correct outputs for given inputs.
"""

import numpy as np
import pytest

from src.frontend.app import (
    HELP_MARKDOWN,
    SCENE_OPTIONS,
    TOOLTIP_TARGETS,
    _make_relight_pattern,
    _numpy_to_b64_img,
    run_simulation,
    run_relighting,
    toggle_help_modal,
    update_status_bar,
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
        """Simulation callback returns all 8 outputs with valid types."""
        primal, dual, fig, analysis, status, store, _log, _log_ui, status_data = run_simulation(
            n_clicks=1,
            scene_type="box_and_wall",
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            n_bounces_str="0",
            proj_x_str="-1.5",
            cam_x_str="1.5",
            log_data=[],
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
            run_simulation(0, "flat_textured", "8", "0.8", "0", "0", "-1.5", "1.5", [])

    def test_run_simulation_with_svd_rank(self):
        """Simulation works with SVD truncation enabled."""
        primal, dual, fig, analysis, status, store, _log, _log_ui, status_data = run_simulation(
            n_clicks=1,
            scene_type="two_planes",
            resolution_str="8",
            albedo_str="0.5",
            svd_rank_str="4",
            n_bounces_str="0",
            proj_x_str="-1.5",
            cam_x_str="1.5",
            log_data=[],
        )
        assert primal.startswith("data:image/png;base64,")
        # Check that SVD rank alert is present in analysis
        has_rank_alert = any("SVD rank" in str(child) for child in analysis)
        assert has_rank_alert

    def test_run_simulation_with_inter_reflections(self):
        """Simulation works with inter-reflections enabled."""
        primal, dual, fig, analysis, status, store, _log, _log_ui, status_data = run_simulation(
            n_clicks=1,
            scene_type="corner_room",
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            n_bounces_str="1",
            proj_x_str="-1.5",
            cam_x_str="1.5",
            log_data=[],
        )
        assert primal.startswith("data:image/png;base64,")

    def test_run_relighting_produces_images(self):
        """Relighting callback produces two valid images."""
        # First run a simulation to get store_data
        _, _, _, _, _, store, _, _, _ = run_simulation(
            n_clicks=1,
            scene_type="box_and_wall",
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            n_bounces_str="0",
            proj_x_str="-1.5",
            cam_x_str="1.5",
            log_data=[],
        )
        pattern_img, result_img, _, _ = run_relighting(
            n_clicks=1,
            pattern_name="left",
            store_data=store,
            log_data=[],
        )
        assert pattern_img.startswith("data:image/png;base64,")
        assert result_img.startswith("data:image/png;base64,")

    def test_run_relighting_prevents_update_without_store(self):
        """Relighting fails gracefully when no simulation has been run."""
        from dash.exceptions import PreventUpdate
        with pytest.raises(PreventUpdate):
            run_relighting(n_clicks=1, pattern_name="white", store_data=None, log_data=[])

    @pytest.mark.parametrize("scene", [
        "box_and_wall", "cornell_box", "gallery", "staircase", "mirror_room",
        "sphere_on_plane", "corner_room", "two_planes", "cylinder_text", "flat_textured"
    ])
    def test_all_scene_types_work(self, scene):
        """Every scene type can be simulated without errors."""
        primal, dual, fig, analysis, status, store, _log, _log_ui, status_data = run_simulation(
            n_clicks=1,
            scene_type=scene,
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            n_bounces_str="0",
            proj_x_str="-1.5",
            cam_x_str="1.5",
            log_data=[],
        )
        assert primal.startswith("data:image/png;base64,")
        assert store is not None


class TestHelpersAndStatusBar:
    """Tests covering the Help modal, status bar and tooltip wiring (issue #25)."""

    def test_help_markdown_is_non_trivial(self):
        """Help content has headings and shortcut table."""
        assert "Quick Start" in HELP_MARKDOWN
        assert "Keyboard shortcuts" in HELP_MARKDOWN
        assert "?" in HELP_MARKDOWN

    def test_tooltip_targets_cover_every_select(self):
        """Every interactive select/button id in the layout has a tooltip."""
        ids = {t[0] for t in TOOLTIP_TARGETS}
        required = {
            "scene-type", "resolution", "albedo", "svd-rank", "n-bounces",
            "proj-x", "cam-x", "run-btn", "relight-pattern", "relight-btn",
            "help-open-btn",
        }
        missing = required - ids
        assert not missing, f"Missing tooltips for: {missing}"

    def test_tooltip_descriptions_are_non_empty(self):
        """Tooltip descriptions are descriptive, not placeholders."""
        for target, text in TOOLTIP_TARGETS:
            assert len(text) >= 15, f"Tooltip for {target} is too short: {text}"

    def test_update_status_bar_idle_when_empty(self):
        """Status bar shows the idle string when no simulation has run."""
        text = update_status_bar(None)
        assert "Idle" in text

    def test_update_status_bar_idle_on_empty_dict(self):
        """Status bar shows idle when scene is None (fresh store)."""
        text = update_status_bar({"scene": None, "resolution": None,
                                   "bounces": None, "elapsed_s": None})
        assert "Idle" in text

    def test_update_status_bar_formats_last_run(self):
        """Status bar renders scene/resolution/bounces/elapsed for a run."""
        text = update_status_bar({
            "scene": "box_and_wall", "resolution": 16,
            "bounces": 0, "elapsed_s": 1.23,
        })
        assert "box_and_wall" in text
        assert "16x16" in text
        assert "bounces=0" in text
        assert "1.23s" in text

    def test_run_simulation_emits_status_data(self):
        """`run_simulation` populates the status-store payload."""
        *_, status_data = run_simulation(
            n_clicks=1,
            scene_type="box_and_wall",
            resolution_str="8",
            albedo_str="0.8",
            svd_rank_str="0",
            n_bounces_str="0",
            proj_x_str="-1.5",
            cam_x_str="1.5",
            log_data=[],
        )
        assert status_data["scene"] == "box_and_wall"
        assert status_data["resolution"] == 8
        assert status_data["bounces"] == 0
        assert isinstance(status_data["elapsed_s"], float)
        assert status_data["elapsed_s"] >= 0

    def test_scene_options_have_tooltip(self):
        """Every declared scene type has a human-readable label."""
        for opt in SCENE_OPTIONS:
            assert "label" in opt and "value" in opt
            assert len(opt["label"]) >= 3


class TestHelpModalToggle:
    """Tests for the help-modal open/close callback."""

    def test_toggle_opens_on_button_click(self):
        """Clicking the open button opens the modal."""
        from unittest.mock import patch
        # The callback reads `ctx.triggered_id` from dash; patch it.
        with patch("src.frontend.app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "help-open-btn"
            assert toggle_help_modal(1, 0, "", False) is True

    def test_toggle_closes_on_close_button(self):
        """Close button closes the modal."""
        from unittest.mock import patch
        with patch("src.frontend.app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "help-close-btn"
            assert toggle_help_modal(1, 1, "", True) is False

    def test_toggle_opens_on_help_key(self):
        """Keyboard shim with 'help:' prefix opens the modal."""
        from unittest.mock import patch
        with patch("src.frontend.app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "kbd-shim"
            assert toggle_help_modal(0, 0, "help:12345", False) is True

    def test_toggle_closes_on_escape(self):
        """Keyboard shim with 'close:' prefix closes the modal."""
        from unittest.mock import patch
        with patch("src.frontend.app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "kbd-shim"
            assert toggle_help_modal(0, 0, "close:12345", True) is False

    def test_toggle_ignores_unrelated_shim(self):
        """Unknown shim values are ignored via PreventUpdate."""
        from dash.exceptions import PreventUpdate
        from unittest.mock import patch
        with patch("src.frontend.app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "kbd-shim"
            with pytest.raises(PreventUpdate):
                toggle_help_modal(0, 0, "garbage", False)
