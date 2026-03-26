"""Virtual renderer for dual photography simulation.

Provides a complete simulation pipeline that generates synthetic transport
matrices and produces primal, dual, and relighted images without any
physical hardware.

This is the primary entry point for testing dual photography algorithms
in a controlled environment with known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.dual import DualPhotographer
from src.core.transport import TransportMatrix
from src.simulation.scene import SceneType, SyntheticScene


@dataclass
class SimulationResult:
    """Container for dual photography simulation results.

    Attributes:
        transport: The computed TransportMatrix.
        primal_image: Image as seen by the camera under uniform projector illumination.
        dual_image: Image as seen from the projector position (via T^T).
        scene_config: The scene configuration used.
        analysis: Transport matrix analysis (singular values, rank, etc.).
    """

    transport: TransportMatrix
    primal_image: np.ndarray
    dual_image: np.ndarray
    scene_config: dict
    analysis: dict


class VirtualRenderer:
    """Simulates the full dual photography pipeline without hardware.

    Generates a synthetic scene, computes its transport matrix analytically,
    and produces primal and dual images for visualization and algorithm
    validation.

    Example:
        >>> renderer = VirtualRenderer(proj_shape=(16, 16), cam_shape=(16, 16))
        >>> result = renderer.run_simulation(scene_type=SceneType.CORNER)
        >>> print(result.dual_image.shape)  # (16, 16)
        >>> print(result.analysis['effective_rank_90'])
    """

    def __init__(
        self,
        proj_shape: tuple[int, int] = (16, 16),
        cam_shape: tuple[int, int] = (16, 16),
        albedo: float = 0.8,
        inter_reflections: bool = False,
        bounce_limit: int = 3,
    ):
        """Initialize the virtual renderer.

        Args:
            proj_shape: Virtual projector resolution (height, width).
            cam_shape: Virtual camera resolution (height, width).
            albedo: Default surface albedo for synthetic scenes.
            inter_reflections: Enable inter-reflection simulation.
            bounce_limit: Maximum number of light bounces.
        """
        self.proj_shape = proj_shape
        self.cam_shape = cam_shape
        self.albedo = albedo
        self.inter_reflections = inter_reflections
        self.bounce_limit = bounce_limit

    def run_simulation(
        self,
        scene_type: SceneType = SceneType.FLAT_WALL,
        proj_pos: np.ndarray | None = None,
        cam_pos: np.ndarray | None = None,
        svd_rank: int | None = None,
        illumination: np.ndarray | None = None,
    ) -> SimulationResult:
        """Run a complete dual photography simulation.

        Steps:
        1. Create synthetic scene with specified geometry
        2. Compute transport matrix T analytically
        3. Generate primal image (camera view under uniform illumination)
        4. Generate dual image (projector view via T^T)
        5. Analyze transport matrix properties

        Args:
            scene_type: Type of synthetic scene geometry.
            proj_pos: Custom projector position (3D). If None, uses default.
            cam_pos: Custom camera position (3D). If None, uses default.
            svd_rank: If specified, use truncated SVD for dual image.
            illumination: Custom illumination for dual image. If None, uniform.

        Returns:
            SimulationResult with all computed images and analysis.
        """
        scene_kwargs = {
            "scene_type": scene_type,
            "proj_shape": self.proj_shape,
            "cam_shape": self.cam_shape,
            "albedo": self.albedo,
            "inter_reflections": self.inter_reflections,
            "bounce_limit": self.bounce_limit,
        }
        if proj_pos is not None:
            scene_kwargs["proj_pos"] = proj_pos
        if cam_pos is not None:
            scene_kwargs["cam_pos"] = cam_pos

        scene = SyntheticScene(**scene_kwargs)
        T = scene.compute_transport_matrix()

        photographer = DualPhotographer(
            proj_shape=self.proj_shape,
            cam_shape=self.cam_shape,
        )
        photographer.compute_transport_from_simulation(T)

        # Primal: camera image under uniform projector illumination
        uniform_pattern = np.ones(self.proj_shape, dtype=np.float64)
        primal = photographer.transport.forward(uniform_pattern)

        # Dual: scene as seen from projector position
        dual = photographer.dual_image(illumination=illumination, svd_rank=svd_rank)

        # Analysis
        analysis = photographer.analyze_transport()

        scene_config = {
            "scene_type": scene_type.value,
            "proj_shape": self.proj_shape,
            "cam_shape": self.cam_shape,
            "albedo": self.albedo,
            "inter_reflections": self.inter_reflections,
            "proj_pos": scene.proj_pos.tolist(),
            "cam_pos": scene.cam_pos.tolist(),
        }

        return SimulationResult(
            transport=photographer.transport,
            primal_image=primal,
            dual_image=dual,
            scene_config=scene_config,
            analysis=analysis,
        )

    def run_relighting(
        self,
        transport: TransportMatrix,
        pattern: np.ndarray,
    ) -> np.ndarray:
        """Relight a scene using an existing transport matrix.

        Args:
            transport: Previously computed TransportMatrix.
            pattern: New projector pattern of shape proj_shape.

        Returns:
            Relighted camera image.
        """
        return transport.relight(pattern)

    def compare_svd_ranks(
        self,
        transport: TransportMatrix,
        ranks: list[int],
        illumination: np.ndarray | None = None,
    ) -> list[tuple[int, np.ndarray, float]]:
        """Compare dual images at different SVD truncation ranks.

        Useful for visualizing the effect of low-rank approximation
        on dual image quality.

        Args:
            transport: The transport matrix to analyze.
            ranks: List of SVD ranks to compare.
            illumination: Virtual illumination. Uniform if None.

        Returns:
            List of (rank, dual_image, reconstruction_error) tuples.
        """
        if illumination is None:
            cam_size = self.cam_shape[0] * self.cam_shape[1]
            illumination = np.ones(cam_size, dtype=np.float64)

        illum_flat = illumination.flatten()
        full_dual = transport.T.T @ illum_flat

        results = []
        for rank in ranks:
            transport.compute_svd(rank)
            T_approx = transport.get_truncated_T(rank)
            dual_approx = T_approx.T @ illum_flat
            error = np.linalg.norm(full_dual - dual_approx) / (np.linalg.norm(full_dual) + 1e-10)
            dual_image = np.clip(dual_approx, 0, None).reshape(self.proj_shape)
            results.append((rank, dual_image, error))

        return results
