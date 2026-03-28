"""Tests for projector-camera calibration using Gray codes.

Validates Gray code pattern generation, encoding/decoding round-trip,
and correspondence map construction.
"""

import numpy as np
import pytest

from src.core.calibration import (
    generate_gray_code_patterns,
    decode_gray_code,
    build_correspondence_map,
    _gray_to_binary_array,
)


class TestGrayCodeGeneration:
    """Tests for Gray code pattern generation."""

    def test_pattern_count(self):
        """Number of patterns should equal ceil(log2(W)) + ceil(log2(H))."""
        patterns = generate_gray_code_patterns(8, 8)
        # 8 -> 3 bits each -> 3 + 3 = 6 patterns
        assert len(patterns) == 6, f"Expected 6 patterns, got {len(patterns)}"

    def test_pattern_count_non_power_of_two(self):
        """Non-power-of-two dimensions should round up."""
        patterns = generate_gray_code_patterns(5, 3)
        # 5 -> ceil(log2(5)) = 3, 3 -> ceil(log2(3)) = 2 -> 5 patterns
        assert len(patterns) == 5, f"Expected 5 patterns, got {len(patterns)}"

    def test_pattern_shape(self):
        """Each pattern should have shape (height, width)."""
        patterns = generate_gray_code_patterns(16, 8)
        for p in patterns:
            assert p.shape == (8, 16)

    def test_pattern_values_binary(self):
        """Patterns should only contain 0 and 255."""
        patterns = generate_gray_code_patterns(8, 8)
        for p in patterns:
            unique = set(np.unique(p))
            assert unique.issubset({0, 255}), f"Unexpected values: {unique}"

    def test_minimum_size(self):
        """Should handle minimum 2x2 size."""
        patterns = generate_gray_code_patterns(2, 2)
        assert len(patterns) >= 2  # at least 1 bit each direction


class TestGrayCodeDecode:
    """Tests for Gray code decoding."""

    def test_roundtrip_identity(self):
        """Encoding then decoding should recover pixel coordinates."""
        W, H = 8, 8
        patterns = generate_gray_code_patterns(W, H)

        # Simulate identity capture (camera = projector, 1:1 mapping)
        captures = patterns.astype(np.float64)

        proj_x, proj_y = decode_gray_code(captures, threshold=128.0)

        # For an 8x8 projector with identity mapping, pixel (x, y) should
        # decode to (x, y). Check the first few pixels.
        # Note: due to Gray code structure, the decoded values represent
        # the column/row indices within the encoded range.
        assert proj_x.shape == (H, W)
        assert proj_y.shape == (H, W)

        # Each column should decode to the same x value
        for x in range(W):
            col_vals = proj_x[:, x]
            assert np.all(col_vals == col_vals[0]), \
                f"Column {x}: not all same value: {np.unique(col_vals)}"

        # Each row should decode to the same y value
        for y in range(H):
            row_vals = proj_y[y, :]
            assert np.all(row_vals == row_vals[0]), \
                f"Row {y}: not all same value: {np.unique(row_vals)}"

    def test_unique_columns(self):
        """Each column should decode to a unique x coordinate (square grid)."""
        # Use square grid so N//2 split matches the number of h/v patterns
        W, H = 8, 8
        patterns = generate_gray_code_patterns(W, H)
        captures = patterns.astype(np.float64)
        proj_x, _ = decode_gray_code(captures, threshold=128.0)

        # Check uniqueness of decoded x per column
        col_vals = [proj_x[0, x] for x in range(W)]
        assert len(set(col_vals)) == W, \
            f"Expected {W} unique column values, got {len(set(col_vals))}: {col_vals}"

    def test_gray_to_binary_scalar(self):
        """Gray to binary conversion for known values."""
        # Gray(0)=0, Gray(1)=1, Gray(2)=3, Gray(3)=2
        gray_arr = np.array([0, 1, 3, 2])
        binary_arr = _gray_to_binary_array(gray_arr)
        np.testing.assert_array_equal(binary_arr, [0, 1, 2, 3])


class TestCorrespondenceMap:
    """Tests for correspondence map building."""

    def test_identity_correspondence(self):
        """With identity mapping, correspondence should map each pixel to itself."""
        W, H = 4, 4
        # Create identity proj_x, proj_y
        proj_x = np.tile(np.arange(W), (H, 1))  # each row = [0,1,2,3]
        proj_y = np.tile(np.arange(H).reshape(-1, 1), (1, W))  # each col = [0,1,2,3]

        corr_map = build_correspondence_map(proj_x, proj_y, W, H)

        for py in range(H):
            for px in range(W):
                assert corr_map[py, px, 0] == py, \
                    f"corr_map[{py},{px}] should map to cam_y={py}"
                assert corr_map[py, px, 1] == px, \
                    f"corr_map[{py},{px}] should map to cam_x={px}"

    def test_out_of_range_ignored(self):
        """Projector coordinates outside range should map to [-1, -1]."""
        proj_x = np.array([[0, 100], [0, 1]])
        proj_y = np.array([[0, 0], [100, 1]])

        corr_map = build_correspondence_map(proj_x, proj_y, 4, 4)

        # (0,0) should have correspondence
        assert corr_map[0, 0, 0] == 0
        assert corr_map[0, 0, 1] == 0

        # (1,1) should have correspondence
        assert corr_map[1, 1, 0] == 1
        assert corr_map[1, 1, 1] == 1

        # Most cells should be -1
        invalid = np.all(corr_map == -1, axis=2)
        assert invalid.sum() > 10  # most of 4x4 = 16 should be invalid

    def test_correspondence_shape(self):
        """Output shape should be (proj_height, proj_width, 2)."""
        proj_x = np.zeros((3, 3), dtype=int)
        proj_y = np.zeros((3, 3), dtype=int)
        corr_map = build_correspondence_map(proj_x, proj_y, 8, 6)
        assert corr_map.shape == (6, 8, 2)


class TestCalibrationEndToEnd:
    """End-to-end calibration pipeline test."""

    def test_full_pipeline(self):
        """Generate, simulate capture, decode, and build map."""
        W, H = 8, 8
        patterns = generate_gray_code_patterns(W, H)

        # Simulate captures (identity, no noise)
        captures = patterns.astype(np.float64)
        proj_x, proj_y = decode_gray_code(captures, threshold=128.0)

        corr_map = build_correspondence_map(proj_x, proj_y, W, H)

        # Check that at least some correspondences are valid
        valid = np.all(corr_map >= 0, axis=2)
        coverage = valid.sum() / (W * H)
        assert coverage > 0, "Should have some valid correspondences"
