"""Pattern generation for light transport matrix acquisition.

Implements multiple pattern strategies used in dual photography:
- Canonical basis (one-pixel-at-a-time, brute-force)
- Hadamard patterns (multiplexed, optimal SNR)
- Random Bernoulli patterns (compressed sensing)
- Gray code / binary structured light patterns

References:
    Sen et al., "Dual Photography", SIGGRAPH 2005
    Sen & Darabi, "Compressive Dual Photography", Eurographics 2009
"""

from __future__ import annotations

from enum import Enum
from typing import Generator

import numpy as np
from scipy.linalg import hadamard


class PatternType(Enum):
    """Enumeration of supported illumination pattern types."""

    CANONICAL = "canonical"
    HADAMARD = "hadamard"
    BERNOULLI = "bernoulli"
    GRAY_CODE = "gray_code"


def generate_canonical_patterns(
    height: int, width: int
) -> Generator[np.ndarray, None, None]:
    """Generate canonical basis patterns (one pixel on at a time).

    Each pattern has exactly one pixel set to 1.0, all others 0.0.
    This is the brute-force approach: requires height*width captures
    but produces the exact transport matrix.

    Args:
        height: Projector vertical resolution.
        width: Projector horizontal resolution.

    Yields:
        Pattern array of shape (height, width) with float64 values in [0, 1].
    """
    total = height * width
    for idx in range(total):
        pattern = np.zeros(total, dtype=np.float64)
        pattern[idx] = 1.0
        yield pattern.reshape(height, width)


def generate_hadamard_patterns(
    height: int, width: int
) -> Generator[np.ndarray, None, None]:
    """Generate Hadamard-based multiplexed patterns.

    Hadamard patterns provide optimal signal-to-noise ratio for
    multiplexed illumination. The Hadamard matrix H has entries {-1, +1};
    we map them to {0, 1} for physical display via (H + 1) / 2.

    The matrix size must be a power of 2. If height*width is not a power
    of 2, the next larger Hadamard matrix is used and truncated.

    Args:
        height: Projector vertical resolution.
        width: Projector horizontal resolution.

    Yields:
        Pattern array of shape (height, width) with float64 values in {0, 1}.
    """
    total = height * width
    # Find next power of 2
    n = 1
    while n < total:
        n *= 2
    H = hadamard(n)
    # Map from {-1, +1} to {0, 1}
    H_positive = (H + 1) / 2.0
    for row_idx in range(total):
        pattern = H_positive[row_idx, :total]
        yield pattern.reshape(height, width)


def generate_bernoulli_patterns(
    height: int,
    width: int,
    n_patterns: int,
    rng: np.random.Generator | None = None,
) -> Generator[np.ndarray, None, None]:
    """Generate random Bernoulli (binary) patterns for compressed sensing.

    Each pixel is independently set to 0 or 1 with equal probability.
    Used in compressive dual photography (Sen & Darabi, 2009) where
    far fewer patterns than projector pixels are needed for reconstruction.

    Args:
        height: Projector vertical resolution.
        width: Projector horizontal resolution.
        n_patterns: Number of random patterns to generate.
        rng: NumPy random generator for reproducibility. If None, a new one is created.

    Yields:
        Pattern array of shape (height, width) with float64 values in {0, 1}.
    """
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(n_patterns):
        pattern = rng.integers(0, 2, size=(height, width)).astype(np.float64)
        yield pattern


def generate_gray_code_patterns(
    height: int, width: int
) -> Generator[np.ndarray, None, None]:
    """Generate Gray code structured light patterns.

    Produces horizontal and vertical Gray code stripe patterns for
    establishing projector-camera pixel correspondences. The number
    of patterns is 2 * ceil(log2(max(height, width))).

    Args:
        height: Projector vertical resolution.
        width: Projector horizontal resolution.

    Yields:
        Pattern array of shape (height, width) with float64 values in {0, 1}.
    """
    n_bits_h = int(np.ceil(np.log2(max(height, 2))))
    n_bits_w = int(np.ceil(np.log2(max(width, 2))))

    # Vertical stripes (encode columns)
    for bit in range(n_bits_w):
        pattern = np.zeros((height, width), dtype=np.float64)
        for col in range(width):
            gray = col ^ (col >> 1)
            if (gray >> bit) & 1:
                pattern[:, col] = 1.0
        yield pattern

    # Horizontal stripes (encode rows)
    for bit in range(n_bits_h):
        pattern = np.zeros((height, width), dtype=np.float64)
        for row in range(height):
            gray = row ^ (row >> 1)
            if (gray >> bit) & 1:
                pattern[row, :] = 1.0
        yield pattern


def generate_patterns(
    height: int,
    width: int,
    pattern_type: PatternType = PatternType.HADAMARD,
    n_patterns: int | None = None,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate a complete set of illumination patterns.

    Convenience function that dispatches to the appropriate pattern
    generator and returns all patterns as a list.

    Args:
        height: Projector vertical resolution.
        width: Projector horizontal resolution.
        pattern_type: Type of pattern to generate.
        n_patterns: Number of patterns (only used for BERNOULLI type).
            For other types, the full set is generated.
        seed: Random seed for reproducibility (BERNOULLI only).

    Returns:
        List of pattern arrays, each of shape (height, width).

    Raises:
        ValueError: If n_patterns is not specified for BERNOULLI type.
    """
    if pattern_type == PatternType.CANONICAL:
        return list(generate_canonical_patterns(height, width))
    elif pattern_type == PatternType.HADAMARD:
        return list(generate_hadamard_patterns(height, width))
    elif pattern_type == PatternType.BERNOULLI:
        if n_patterns is None:
            raise ValueError("n_patterns must be specified for BERNOULLI patterns.")
        rng = np.random.default_rng(seed)
        return list(generate_bernoulli_patterns(height, width, n_patterns, rng))
    elif pattern_type == PatternType.GRAY_CODE:
        return list(generate_gray_code_patterns(height, width))
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
