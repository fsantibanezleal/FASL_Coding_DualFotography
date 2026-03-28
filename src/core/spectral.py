"""
Spectral transport matrix for wavelength-dependent dual photography.

Computes transport matrices at multiple wavelengths, enabling:
- Spectral relighting (change illumination color)
- Material classification (different materials reflect differently per wavelength)
- Chromatic dual photography (color-accurate projector-view reconstruction)

The spectral transport is a set of N transport matrices T_lambda, one per
wavelength channel. Each T_lambda captures how light at wavelength lambda
propagates from projector to camera through the scene.

For RGB simplification: T_R, T_G, T_B (3 transport matrices).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


class SpectralTransportMatrix:
    """Wavelength-dependent transport matrix collection."""

    def __init__(self, resolution: int = 32):
        self.resolution = resolution
        self.channels: Dict[str, np.ndarray] = {}  # name -> T matrix

    def add_channel(self, name: str, transport_matrix: np.ndarray):
        """Add a transport matrix for a wavelength channel."""
        self.channels[name] = transport_matrix.copy()

    def compute_rgb_from_scene(self, scene_reflectance_rgb: Dict[str, np.ndarray],
                                projector_pattern: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute camera image for each color channel.

        Args:
            scene_reflectance_rgb: Dict mapping channel name to reflectance.
            projector_pattern: 1D illumination pattern.

        Returns:
            Dict mapping channel name to camera image (1D).
        """
        result = {}
        for name, T in self.channels.items():
            if name in scene_reflectance_rgb:
                # Element-wise multiply T by reflectance, then project
                T_weighted = T * scene_reflectance_rgb[name][np.newaxis, :]
                result[name] = T_weighted @ projector_pattern
            else:
                result[name] = T @ projector_pattern
        return result

    def dual_image_spectral(self, channel: str, virtual_illumination: np.ndarray) -> np.ndarray:
        """Compute spectral dual image for a single channel.

        Dual image: I_dual = T^T @ virtual_illumination
        """
        if channel not in self.channels:
            raise ValueError(f"Unknown channel: {channel}")
        return self.channels[channel].T @ virtual_illumination

    def get_state(self) -> dict:
        """Serialize for API response."""
        return {
            'resolution': self.resolution,
            'channels': list(self.channels.keys()),
            'shapes': {name: list(T.shape) for name, T in self.channels.items()},
        }


def simulate_rgb_transport(resolution: int = 32,
                            seed: int = 42) -> SpectralTransportMatrix:
    """Simulate RGB transport matrices for a scene.

    Each channel has slightly different transport to model
    wavelength-dependent material properties.

    Args:
        resolution: Grid resolution (NxN).
        seed: Random seed for reproducibility.

    Returns:
        SpectralTransportMatrix with R, G, B channels.
    """
    n = resolution * resolution
    rng = np.random.default_rng(seed)

    # Generate a base transport matrix (diffuse scene)
    base_T = rng.random((n, n)) * 0.5

    spectral = SpectralTransportMatrix(resolution)

    # Red channel: warm materials reflect more
    T_r = base_T * rng.uniform(0.8, 1.2, base_T.shape)
    # Green channel: neutral
    T_g = base_T.copy()
    # Blue channel: cool materials reflect more
    T_b = base_T * rng.uniform(0.7, 1.1, base_T.shape)

    spectral.add_channel('red', T_r)
    spectral.add_channel('green', T_g)
    spectral.add_channel('blue', T_b)

    return spectral
