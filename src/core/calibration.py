"""
Projector-camera geometric calibration using structured light.

Implements Gray code + phase shifting calibration to establish
the geometric correspondence between projector and camera pixels.
This is required for physical dual photography capture.

The calibration pipeline:
1. Project Gray code patterns (horizontal + vertical)
2. Capture camera images of each pattern
3. Decode patterns to get projector->camera pixel mapping
4. Build the correspondence map (homography or per-pixel lookup)
"""
import numpy as np
from typing import Tuple


def generate_gray_code_patterns(width: int, height: int) -> np.ndarray:
    """Generate horizontal and vertical Gray code patterns.

    Gray codes are used instead of binary codes because only
    one bit changes between consecutive patterns, making the
    decoding robust to slight misalignment or timing errors.

    Args:
        width: Projector width in pixels.
        height: Projector height in pixels.

    Returns:
        (N_patterns, height, width) array of binary patterns.
    """
    def to_gray(n):
        return n ^ (n >> 1)

    n_bits_h = int(np.ceil(np.log2(max(width, 2))))
    n_bits_v = int(np.ceil(np.log2(max(height, 2))))

    patterns = []

    # Vertical stripes (decode horizontal position)
    for bit in range(n_bits_h):
        pattern = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            gray = to_gray(x)
            pattern[:, x] = 255 if (gray >> bit) & 1 else 0
        patterns.append(pattern)

    # Horizontal stripes (decode vertical position)
    for bit in range(n_bits_v):
        pattern = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            gray = to_gray(y)
            pattern[y, :] = 255 if (gray >> bit) & 1 else 0
        patterns.append(pattern)

    return np.array(patterns, dtype=np.uint8)


def decode_gray_code(captures: np.ndarray, threshold: float = 128.0) -> Tuple[np.ndarray, np.ndarray]:
    """Decode captured Gray code images to get projector coordinates.

    Args:
        captures: (N_patterns, cam_H, cam_W) captured images.
        threshold: Binarization threshold.

    Returns:
        (proj_x, proj_y): (cam_H, cam_W) arrays of decoded projector coordinates.
    """
    N = captures.shape[0]
    cam_H, cam_W = captures.shape[1], captures.shape[2]

    # Binarize
    binary = (captures > threshold).astype(int)

    # Assume first half are horizontal (x), second half vertical (y)
    n_h = N // 2
    n_v = N - n_h

    # Decode horizontal (projector x)
    proj_x = np.zeros((cam_H, cam_W), dtype=int)
    for bit in range(n_h):
        proj_x |= binary[bit] << bit
    # Convert from Gray to binary
    proj_x = _gray_to_binary_array(proj_x)

    # Decode vertical (projector y)
    proj_y = np.zeros((cam_H, cam_W), dtype=int)
    for bit in range(n_v):
        proj_y |= binary[n_h + bit] << bit
    proj_y = _gray_to_binary_array(proj_y)

    return proj_x, proj_y


def _gray_to_binary_array(gray: np.ndarray) -> np.ndarray:
    """Convert Gray code to binary (vectorized)."""
    binary = gray.copy()
    shift = 1
    while shift < 32:
        binary ^= (binary >> shift)
        shift <<= 1
    return binary


def build_correspondence_map(proj_x, proj_y, proj_width, proj_height):
    """Build a lookup table mapping projector pixels to camera pixels.

    Returns:
        (proj_height, proj_width, 2) array where [py, px] = [cam_y, cam_x]
        or [-1, -1] if no correspondence found.
    """
    cam_H, cam_W = proj_x.shape
    corr_map = np.full((proj_height, proj_width, 2), -1, dtype=int)

    for cy in range(cam_H):
        for cx in range(cam_W):
            px = proj_x[cy, cx]
            py = proj_y[cy, cx]
            if 0 <= px < proj_width and 0 <= py < proj_height:
                corr_map[py, px] = [cy, cx]

    return corr_map
