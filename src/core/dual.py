"""High-level dual photography orchestrator.

Provides the DualPhotographer class that ties together pattern generation,
transport matrix acquisition (simulated or physical), and dual/relighting
image synthesis.

References:
    Sen et al., "Dual Photography", SIGGRAPH 2005
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.core.patterns import PatternType, generate_patterns
from src.core.transport import TransportMatrix


@dataclass
class DualPhotographer:
    """Orchestrates the full dual photography pipeline.

    This class manages the end-to-end workflow:
    1. Generate illumination patterns
    2. Acquire responses (simulated or physical capture)
    3. Compute the transport matrix T
    4. Generate dual images via T^T
    5. Perform relighting via T with new patterns

    Attributes:
        proj_shape: Projector resolution as (height, width).
        cam_shape: Camera image resolution as (height, width).
        pattern_type: Type of illumination pattern to use.
        n_patterns: Number of patterns (for compressed sensing modes).
        seed: Random seed for reproducibility.
        transport: Computed TransportMatrix instance (available after acquire).
    """

    proj_shape: tuple[int, int]
    cam_shape: tuple[int, int]
    pattern_type: PatternType = PatternType.HADAMARD
    n_patterns: int | None = None
    seed: int | None = 42
    transport: TransportMatrix | None = None

    def generate_patterns(self) -> list[np.ndarray]:
        """Generate the set of illumination patterns for acquisition.

        Returns:
            List of pattern arrays, each of shape proj_shape.
        """
        return generate_patterns(
            height=self.proj_shape[0],
            width=self.proj_shape[1],
            pattern_type=self.pattern_type,
            n_patterns=self.n_patterns,
            seed=self.seed,
        )

    def compute_transport(
        self,
        patterns: list[np.ndarray],
        captures: list[np.ndarray],
        method: Literal["pseudoinverse", "direct"] = "pseudoinverse",
    ) -> TransportMatrix:
        """Compute the transport matrix from pattern-capture pairs.

        Args:
            patterns: List of projected patterns.
            captures: List of captured camera images.
            method: Computation method ("pseudoinverse" or "direct").

        Returns:
            The computed TransportMatrix.
        """
        self.transport = TransportMatrix.from_pattern_captures(
            patterns=patterns,
            captures=captures,
            proj_shape=self.proj_shape,
            cam_shape=self.cam_shape,
            method=method,
        )
        return self.transport

    def compute_transport_from_simulation(
        self, T: np.ndarray, n_channels: int = 1
    ) -> TransportMatrix:
        """Set the transport matrix directly from a simulation.

        Args:
            T: Pre-computed transport matrix of shape (cam_pixels, proj_pixels).
            n_channels: Number of color channels.

        Returns:
            The TransportMatrix wrapper.
        """
        self.transport = TransportMatrix(
            T=T,
            cam_shape=self.cam_shape,
            proj_shape=self.proj_shape,
            n_channels=n_channels,
        )
        return self.transport

    def dual_image(
        self,
        illumination: np.ndarray | None = None,
        svd_rank: int | None = None,
    ) -> np.ndarray:
        """Generate the dual photograph (scene viewed from projector position).

        Args:
            illumination: Virtual illumination from camera position. If None,
                uniform white illumination is used.
            svd_rank: If specified, use a rank-k SVD approximation of T.

        Returns:
            Dual image of shape proj_shape.

        Raises:
            RuntimeError: If transport matrix has not been computed yet.
        """
        if self.transport is None:
            raise RuntimeError("Transport matrix not computed. Call compute_transport first.")

        if svd_rank is not None:
            self.transport.compute_svd(svd_rank)

        if illumination is None:
            illumination = np.ones(
                (self.cam_shape[0], self.cam_shape[1]), dtype=np.float64
            )

        return self.transport.dual(illumination)

    def relight(self, pattern: np.ndarray) -> np.ndarray:
        """Relight the scene with a new illumination pattern.

        Args:
            pattern: New projector pattern of shape proj_shape.

        Returns:
            Relighted camera image.

        Raises:
            RuntimeError: If transport matrix has not been computed yet.
        """
        if self.transport is None:
            raise RuntimeError("Transport matrix not computed. Call compute_transport first.")
        return self.transport.relight(pattern)

    def analyze_transport(self, rank: int | None = None) -> dict:
        """Analyze the transport matrix properties.

        Returns summary statistics including singular value spectrum,
        effective rank, condition number, and energy distribution.

        Args:
            rank: SVD truncation rank for analysis.

        Returns:
            Dictionary with analysis results:
                - singular_values: Array of singular values.
                - condition_number: Ratio of largest to smallest singular value.
                - energy_cumulative: Cumulative fraction of total energy per rank.
                - effective_rank_90: Rank capturing 90% of total energy.
                - effective_rank_99: Rank capturing 99% of total energy.
        """
        if self.transport is None:
            raise RuntimeError("Transport matrix not computed.")

        self.transport.compute_svd(rank)
        sv = self.transport.singular_value_spectrum()

        energy = sv**2
        total_energy = energy.sum()
        cumulative = np.cumsum(energy) / total_energy if total_energy > 0 else np.zeros_like(energy)

        rank_90 = int(np.searchsorted(cumulative, 0.9)) + 1
        rank_99 = int(np.searchsorted(cumulative, 0.99)) + 1

        return {
            "singular_values": sv,
            "condition_number": sv[0] / sv[-1] if sv[-1] > 0 else float("inf"),
            "energy_cumulative": cumulative,
            "effective_rank_90": min(rank_90, len(sv)),
            "effective_rank_99": min(rank_99, len(sv)),
        }
