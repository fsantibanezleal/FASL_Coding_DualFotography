"""Synthetic scene generation for virtual dual photography experiments.

Provides configurable 3D scenes with known geometry and reflectance
properties, enabling ground-truth validation of dual photography algorithms
without physical hardware.

The scenes generate transport matrices analytically based on Lambertian
reflectance and geometric visibility, simulating how light from each
projector pixel reaches each camera pixel through the scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SceneType(Enum):
    """Available synthetic scene configurations."""

    FLAT_WALL = "flat_wall"
    CORNER = "corner"
    V_GROOVE = "v_groove"
    SPHERE = "sphere"
    CHECKERBOARD = "checkerboard"
    TEXTURED_WALL = "textured_wall"


@dataclass
class SyntheticScene:
    """A synthetic 3D scene for dual photography simulation.

    Generates a transport matrix T analytically, where T[i,j] encodes
    how much light from projector pixel j reaches camera pixel i through
    the scene, considering geometry, visibility, and Lambertian BRDF.

    The scene coordinate system:
    - Projector at position proj_pos, looking along -Z
    - Camera at position cam_pos, looking along -Z
    - Scene surface(s) in the XY plane at various Z depths

    Attributes:
        scene_type: Type of synthetic scene to generate.
        proj_shape: Projector resolution as (height, width).
        cam_shape: Camera resolution as (height, width).
        albedo: Scene surface reflectance in [0, 1]. Can be scalar or array.
        proj_pos: 3D position of the projector.
        cam_pos: 3D position of the camera.
        scene_size: Physical size of the scene in world units.
        inter_reflections: Whether to simulate inter-reflections (slower).
        bounce_limit: Maximum number of light bounces for inter-reflections.
    """

    scene_type: SceneType = SceneType.FLAT_WALL
    proj_shape: tuple[int, int] = (8, 8)
    cam_shape: tuple[int, int] = (8, 8)
    albedo: float = 0.8
    proj_pos: np.ndarray = field(default_factory=lambda: np.array([-0.3, 0.0, 1.0]))
    cam_pos: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.0, 1.0]))
    scene_size: float = 1.0
    inter_reflections: bool = False
    bounce_limit: int = 3

    def _surface_points_and_normals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute surface sample points, normals, and albedo map.

        Returns:
            Tuple of (points, normals, albedos) where:
                - points: (N, 3) array of surface sample positions
                - normals: (N, 3) array of surface normals
                - albedos: (N,) array of surface reflectance values
        """
        # Use camera resolution as the surface sampling grid
        h, w = self.cam_shape
        total = h * w
        ys = np.linspace(-self.scene_size / 2, self.scene_size / 2, h)
        xs = np.linspace(-self.scene_size / 2, self.scene_size / 2, w)
        xx, yy = np.meshgrid(xs, ys)
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()

        if self.scene_type == SceneType.FLAT_WALL:
            points = np.column_stack([xx_flat, yy_flat, np.zeros(total)])
            normals = np.tile([0.0, 0.0, 1.0], (total, 1))
            albedos = np.full(total, self.albedo)

        elif self.scene_type == SceneType.CORNER:
            # Two walls meeting at a 90-degree angle at x=0
            points = np.column_stack([xx_flat, yy_flat, np.zeros(total)])
            normals = np.tile([0.0, 0.0, 1.0], (total, 1))
            albedos = np.full(total, self.albedo)
            # Right half: wall at z=0, normal along +Z
            # Left half: wall folded at 90 degrees
            left_mask = xx_flat < 0
            points[left_mask, 2] = -xx_flat[left_mask]
            points[left_mask, 0] = 0.0
            normals[left_mask] = [1.0, 0.0, 0.0]

        elif self.scene_type == SceneType.V_GROOVE:
            # V-shaped groove: two angled planes meeting at center
            points = np.column_stack([xx_flat, yy_flat, np.zeros(total)])
            normals = np.tile([0.0, 0.0, 1.0], (total, 1))
            albedos = np.full(total, self.albedo)
            angle = np.pi / 4
            left_mask = xx_flat < 0
            right_mask = xx_flat >= 0
            points[left_mask, 2] = -xx_flat[left_mask] * np.tan(angle)
            n_left = np.array([np.sin(angle), 0.0, np.cos(angle)])
            normals[left_mask] = n_left
            points[right_mask, 2] = xx_flat[right_mask] * np.tan(angle)
            n_right = np.array([-np.sin(angle), 0.0, np.cos(angle)])
            normals[right_mask] = n_right

        elif self.scene_type == SceneType.SPHERE:
            # Hemisphere bulging toward the cameras
            r = self.scene_size / 2
            dist = np.sqrt(xx_flat**2 + yy_flat**2)
            z = np.where(dist < r, np.sqrt(np.maximum(r**2 - dist**2, 0)), 0.0)
            points = np.column_stack([xx_flat, yy_flat, z])
            # Normal = normalized (x, y, z) for sphere points
            norms = np.column_stack([xx_flat, yy_flat, z])
            lengths = np.linalg.norm(norms, axis=1, keepdims=True)
            lengths = np.maximum(lengths, 1e-10)
            normals = norms / lengths
            # Flat parts get z-normal
            flat_mask = dist >= r
            normals[flat_mask] = [0.0, 0.0, 1.0]
            albedos = np.full(total, self.albedo)

        elif self.scene_type == SceneType.CHECKERBOARD:
            points = np.column_stack([xx_flat, yy_flat, np.zeros(total)])
            normals = np.tile([0.0, 0.0, 1.0], (total, 1))
            # Checkerboard pattern albedo
            checker_size = self.scene_size / max(w, h) * 2
            cx = (np.floor(xx_flat / checker_size) % 2).astype(int)
            cy = (np.floor(yy_flat / checker_size) % 2).astype(int)
            albedos = np.where(cx ^ cy, self.albedo, self.albedo * 0.2)

        elif self.scene_type == SceneType.TEXTURED_WALL:
            points = np.column_stack([xx_flat, yy_flat, np.zeros(total)])
            normals = np.tile([0.0, 0.0, 1.0], (total, 1))
            # Gradient albedo pattern
            albedos = self.albedo * (0.3 + 0.7 * (xx_flat - xx_flat.min()) /
                                     (xx_flat.max() - xx_flat.min() + 1e-10))

        else:
            raise ValueError(f"Unknown scene type: {self.scene_type}")

        return points, normals, albedos

    def _projector_pixel_directions(self) -> np.ndarray:
        """Compute the direction vectors from projector to each scene point.

        Returns:
            (N_proj, 3) array of normalized direction vectors.
        """
        ph, pw = self.proj_shape
        ys = np.linspace(-self.scene_size / 2, self.scene_size / 2, ph)
        xs = np.linspace(-self.scene_size / 2, self.scene_size / 2, pw)
        xx, yy = np.meshgrid(xs, ys)
        # Target points on the scene plane (z=0 for flat, approximate for others)
        targets = np.column_stack([xx.flatten(), yy.flatten(), np.zeros(ph * pw)])
        directions = targets - self.proj_pos
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        return directions / np.maximum(norms, 1e-10)

    def compute_transport_matrix(self) -> np.ndarray:
        """Compute the light transport matrix T for this synthetic scene.

        Uses a Lambertian reflectance model:
            T[i, j] = albedo[i] * max(0, n_i . L_j) * max(0, n_i . V_i) / (r_pj^2 * r_ci^2)

        where L_j is the direction from surface point i to projector pixel j,
        V_i is the direction from surface point i to camera pixel i, n_i is
        the surface normal, and r terms are distances.

        For scenes with inter-reflections enabled, iterative radiosity-style
        bounces are computed.

        Returns:
            Transport matrix T of shape (cam_pixels, proj_pixels).
        """
        points, normals, albedos = self._surface_points_and_normals()
        n_cam = points.shape[0]

        ph, pw = self.proj_shape
        n_proj = ph * pw

        # Projector pixel target positions on scene
        pys = np.linspace(-self.scene_size / 2, self.scene_size / 2, ph)
        pxs = np.linspace(-self.scene_size / 2, self.scene_size / 2, pw)
        pxx, pyy = np.meshgrid(pxs, pys)
        proj_targets = np.column_stack([pxx.flatten(), pyy.flatten(), np.zeros(n_proj)])

        T = np.zeros((n_cam, n_proj), dtype=np.float64)

        for j in range(n_proj):
            # Direction from each surface point to projector pixel j's target
            # We model the projector as illuminating a specific area on the scene
            proj_target = proj_targets[j]

            # Light direction: from projector through its pixel to the scene
            light_dir_to_scene = proj_target - self.proj_pos
            light_dist = np.linalg.norm(light_dir_to_scene)
            light_dir = light_dir_to_scene / max(light_dist, 1e-10)

            for i in range(n_cam):
                # Vector from surface point to projector
                to_proj = self.proj_pos - points[i]
                dist_proj = np.linalg.norm(to_proj)
                dir_to_proj = to_proj / max(dist_proj, 1e-10)

                # Vector from surface point to camera
                to_cam = self.cam_pos - points[i]
                dist_cam = np.linalg.norm(to_cam)
                dir_to_cam = to_cam / max(dist_cam, 1e-10)

                # Lambertian BRDF contribution
                cos_in = max(0.0, np.dot(normals[i], dir_to_proj))
                cos_out = max(0.0, np.dot(normals[i], dir_to_cam))

                # Proximity of this surface point to the projector pixel target
                point_to_target = np.linalg.norm(points[i] - proj_target)
                # Gaussian falloff based on pixel spacing
                pixel_size = self.scene_size / max(pw, ph)
                weight = np.exp(-0.5 * (point_to_target / (pixel_size * 0.5)) ** 2)

                T[i, j] = (
                    albedos[i]
                    * cos_in
                    * cos_out
                    * weight
                    / (dist_proj**2 * dist_cam**2 + 1e-10)
                )

        # Normalize to [0, 1] range for display purposes
        t_max = T.max()
        if t_max > 0:
            T /= t_max

        # Add inter-reflections if enabled
        if self.inter_reflections and self.bounce_limit > 0:
            T = self._add_inter_reflections(T, points, normals, albedos)

        return T

    def _add_inter_reflections(
        self,
        T_direct: np.ndarray,
        points: np.ndarray,
        normals: np.ndarray,
        albedos: np.ndarray,
    ) -> np.ndarray:
        """Add inter-reflection contributions using iterative radiosity.

        Computes additional light bounces where light reflected from one
        surface point illuminates another, following the radiosity equation:
            B = E + rho * F @ B

        where F is the form factor matrix and rho is the albedo.

        Args:
            T_direct: Direct-only transport matrix.
            points: Surface sample positions.
            normals: Surface normals.
            albedos: Surface reflectance values.

        Returns:
            Transport matrix with inter-reflections included.
        """
        n = points.shape[0]

        # Compute form factor matrix F[i, j] = visibility * cos_i * cos_j / (pi * r^2)
        F = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[j] - points[i]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                direction = diff / dist
                cos_i = np.dot(normals[i], direction)
                cos_j = np.dot(normals[j], -direction)
                if cos_i > 0 and cos_j > 0:
                    ff = cos_i * cos_j / (np.pi * dist**2 + 1e-10)
                    F[i, j] = ff
                    F[j, i] = ff

        # Normalize form factors
        row_sums = F.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        F = F / row_sums

        # Iterative radiosity: T_total = sum_{k=0}^{bounces} (rho * F)^k @ T_direct
        rho_F = np.diag(albedos) @ F
        T_total = T_direct.copy()
        T_bounce = T_direct.copy()
        for _bounce in range(self.bounce_limit):
            T_bounce = rho_F @ T_bounce
            T_total += T_bounce

        # Re-normalize
        t_max = T_total.max()
        if t_max > 0:
            T_total /= t_max

        return T_total
