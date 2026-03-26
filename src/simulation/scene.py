"""Synthetic scene generation with proper perspective ray-casting.

Implements physically-based transport matrix computation using ray-casting
from both projector and camera positions, with proper perspective projection,
occlusion testing, and support for 3D scenes with depth variation.

The transport matrix T[i,j] is computed by:
1. Casting a ray from the projector through pixel j to find the hit point on the scene
2. Evaluating the Lambertian BRDF at the hit point
3. Casting a ray from the hit point toward the camera pixel i
4. Checking visibility (occlusion) along both paths

This produces a transport matrix that is far from diagonal when the scene
has 3D geometry, giving visually distinct primal and dual images.

References:
    Sen et al., "Dual Photography", SIGGRAPH 2005
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np


class SceneType(Enum):
    """Available synthetic scene configurations."""

    FLAT_TEXTURED = "flat_textured"
    CORNER_ROOM = "corner_room"
    BOX_AND_WALL = "box_and_wall"
    CYLINDER_TEXT = "cylinder_text"
    SPHERE_ON_PLANE = "sphere_on_plane"
    TWO_PLANES = "two_planes"


class HitResult(NamedTuple):
    """Result of a ray-scene intersection test."""

    hit: bool
    t: float          # Ray parameter at hit point
    point: np.ndarray  # 3D hit position
    normal: np.ndarray  # Surface normal at hit
    albedo: float      # Surface reflectance at hit
    obj_id: int        # Object identifier


@dataclass
class SceneObject:
    """A primitive object in the scene.

    Supports planes (infinite or bounded) and spheres.

    Attributes:
        kind: "plane", "box_face", or "sphere".
        center: Center position for sphere, or point on plane.
        normal: Surface normal (for planes).
        radius: Radius (for spheres).
        albedo: Reflectance value or 2D albedo texture array.
        half_extents: Half-size in each axis for bounded planes (x, y).
        up_axis: Up direction for texture mapping on planes.
        right_axis: Right direction for texture mapping on planes.
        obj_id: Unique object identifier.
    """

    kind: str
    center: np.ndarray
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    radius: float = 0.5
    albedo: float | np.ndarray = 0.7
    half_extents: tuple[float, float] = (0.5, 0.5)
    up_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    right_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    obj_id: int = 0

    def intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> HitResult:
        """Test ray intersection with this object.

        Args:
            ray_origin: 3D ray origin.
            ray_dir: 3D normalized ray direction.

        Returns:
            HitResult with intersection data, or HitResult(hit=False, ...) if no hit.
        """
        no_hit = HitResult(False, 1e30, np.zeros(3), np.zeros(3), 0.0, -1)

        if self.kind in ("plane", "box_face"):
            return self._intersect_plane(ray_origin, ray_dir)
        elif self.kind == "sphere":
            return self._intersect_sphere(ray_origin, ray_dir)
        return no_hit

    def _intersect_plane(self, ro: np.ndarray, rd: np.ndarray) -> HitResult:
        """Ray-plane intersection with optional bounding."""
        no_hit = HitResult(False, 1e30, np.zeros(3), np.zeros(3), 0.0, -1)
        denom = np.dot(rd, self.normal)
        if abs(denom) < 1e-10:
            return no_hit
        t = np.dot(self.center - ro, self.normal) / denom
        if t < 1e-4:
            return no_hit

        point = ro + t * rd

        # Check bounds for box_face
        if self.kind == "box_face":
            local = point - self.center
            u = np.dot(local, self.right_axis)
            v = np.dot(local, self.up_axis)
            if abs(u) > self.half_extents[0] or abs(v) > self.half_extents[1]:
                return no_hit

        # Compute albedo (texture lookup if array)
        albedo_val = self._sample_albedo(point)
        return HitResult(True, t, point, self.normal.copy(), albedo_val, self.obj_id)

    def _intersect_sphere(self, ro: np.ndarray, rd: np.ndarray) -> HitResult:
        """Ray-sphere intersection."""
        no_hit = HitResult(False, 1e30, np.zeros(3), np.zeros(3), 0.0, -1)
        oc = ro - self.center
        a = np.dot(rd, rd)
        b = 2.0 * np.dot(oc, rd)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return no_hit
        sqrt_disc = np.sqrt(disc)
        t = (-b - sqrt_disc) / (2 * a)
        if t < 1e-4:
            t = (-b + sqrt_disc) / (2 * a)
        if t < 1e-4:
            return no_hit

        point = ro + t * rd
        normal = (point - self.center) / self.radius
        albedo_val = self._sample_albedo(point)
        return HitResult(True, t, point, normal, albedo_val, self.obj_id)

    def _sample_albedo(self, point: np.ndarray) -> float:
        """Sample the albedo at a surface point.

        If albedo is a scalar, returns it directly. If it is a 2D array
        (texture), performs UV mapping and bilinear lookup.
        """
        if isinstance(self.albedo, (int, float)):
            return float(self.albedo)

        # UV mapping for textured surfaces
        local = point - self.center
        u = np.dot(local, self.right_axis)
        v = np.dot(local, self.up_axis)

        tex = self.albedo
        h, w = tex.shape

        # Map to [0, 1] based on half_extents
        u_norm = (u / self.half_extents[0] + 1.0) / 2.0
        v_norm = (v / self.half_extents[1] + 1.0) / 2.0
        u_norm = np.clip(u_norm, 0, 1)
        v_norm = np.clip(v_norm, 0, 1)

        px = int(u_norm * (w - 1))
        py = int((1.0 - v_norm) * (h - 1))  # Flip vertical
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        return float(tex[py, px])


class SyntheticScene:
    """A 3D scene for dual photography simulation via ray-casting.

    Computes the transport matrix by casting rays from the projector and
    camera through their respective pixel grids, intersecting with scene
    geometry, and evaluating Lambertian shading with occlusion testing.

    This produces transport matrices that are far from diagonal when the
    scene has 3D depth variation, giving visually distinct primal and
    dual images that demonstrate Helmholtz reciprocity.

    Attributes:
        scene_type: Configuration of the 3D scene.
        proj_shape: Projector resolution as (height, width).
        cam_shape: Camera resolution as (height, width).
        proj_pos: 3D position of the projector.
        cam_pos: 3D position of the camera.
        proj_fov: Projector field of view in degrees.
        cam_fov: Camera field of view in degrees.
        proj_target: Point the projector aims at.
        cam_target: Point the camera aims at.
    """

    def __init__(
        self,
        scene_type: SceneType = SceneType.BOX_AND_WALL,
        proj_shape: tuple[int, int] = (16, 16),
        cam_shape: tuple[int, int] = (16, 16),
        proj_pos: np.ndarray | None = None,
        cam_pos: np.ndarray | None = None,
        proj_fov: float = 60.0,
        cam_fov: float = 60.0,
        proj_target: np.ndarray | None = None,
        cam_target: np.ndarray | None = None,
        albedo: float = 0.8,
    ):
        self.scene_type = scene_type
        self.proj_shape = proj_shape
        self.cam_shape = cam_shape
        self.proj_pos = proj_pos if proj_pos is not None else np.array([-1.5, 0.5, 2.0])
        self.cam_pos = cam_pos if cam_pos is not None else np.array([1.5, 0.5, 2.0])
        self.proj_fov = proj_fov
        self.cam_fov = cam_fov
        self.proj_target = proj_target if proj_target is not None else np.array([0.0, 0.0, 0.0])
        self.cam_target = cam_target if cam_target is not None else np.array([0.0, 0.0, 0.0])
        self.albedo = albedo
        self.objects: list[SceneObject] = []
        self._build_scene()

    def _build_scene(self) -> None:
        """Construct scene objects based on scene_type."""
        if self.scene_type == SceneType.FLAT_TEXTURED:
            self._build_flat_textured()
        elif self.scene_type == SceneType.CORNER_ROOM:
            self._build_corner_room()
        elif self.scene_type == SceneType.BOX_AND_WALL:
            self._build_box_and_wall()
        elif self.scene_type == SceneType.CYLINDER_TEXT:
            self._build_cylinder_text()
        elif self.scene_type == SceneType.SPHERE_ON_PLANE:
            self._build_sphere_on_plane()
        elif self.scene_type == SceneType.TWO_PLANES:
            self._build_two_planes()

    def _make_checker_texture(self, size: int = 32, n_checks: int = 4) -> np.ndarray:
        """Generate a checkerboard texture.

        Args:
            size: Texture resolution.
            n_checks: Number of checks per side.

        Returns:
            2D array with values 0.2 and 0.9.
        """
        tex = np.zeros((size, size), dtype=np.float64)
        block = size // n_checks
        for i in range(size):
            for j in range(size):
                if ((i // block) + (j // block)) % 2 == 0:
                    tex[i, j] = 0.9
                else:
                    tex[i, j] = 0.2
        return tex

    def _make_letter_texture(self, letter: str = "F", size: int = 32) -> np.ndarray:
        """Generate a texture with a letter rendered as pixels.

        Creates recognizable letters that clearly show viewpoint changes
        (e.g., 'F' appears mirrored from the other side).

        Args:
            letter: Single character to render.
            size: Texture resolution.

        Returns:
            2D array with values in [0.1, 0.95].
        """
        tex = np.full((size, size), 0.1, dtype=np.float64)

        # Simple bitmap font for common letters
        fonts = {
            "F": [
                "########",
                "#.......",
                "#.......",
                "######..",
                "#.......",
                "#.......",
                "#.......",
                "#.......",
            ],
            "D": [
                "######..",
                "#.....#.",
                "#......#",
                "#......#",
                "#......#",
                "#......#",
                "#.....#.",
                "######..",
            ],
            "P": [
                "######..",
                "#.....#.",
                "#.....#.",
                "######..",
                "#.......",
                "#.......",
                "#.......",
                "#.......",
            ],
            "L": [
                "#.......",
                "#.......",
                "#.......",
                "#.......",
                "#.......",
                "#.......",
                "#.......",
                "########",
            ],
            "A": [
                "..###...",
                ".#...#..",
                "#.....#.",
                "#.....#.",
                "#######.",
                "#.....#.",
                "#.....#.",
                "#.....#.",
            ],
            "R": [
                "######..",
                "#.....#.",
                "#.....#.",
                "######..",
                "#..#....",
                "#...#...",
                "#....#..",
                "#.....#.",
            ],
        }

        bitmap = fonts.get(letter, fonts["F"])
        bh = len(bitmap)
        bw = len(bitmap[0])
        margin = size // 8
        cell_h = (size - 2 * margin) / bh
        cell_w = (size - 2 * margin) / bw

        for row_idx, row_str in enumerate(bitmap):
            for col_idx, ch in enumerate(row_str):
                if ch == "#":
                    r0 = int(margin + row_idx * cell_h)
                    r1 = int(margin + (row_idx + 1) * cell_h)
                    c0 = int(margin + col_idx * cell_w)
                    c1 = int(margin + (col_idx + 1) * cell_w)
                    tex[r0:r1, c0:c1] = 0.95

        return tex

    def _make_arrow_texture(self, size: int = 32) -> np.ndarray:
        """Generate a right-pointing arrow texture.

        Arrow direction makes viewpoint swap immediately obvious.
        """
        tex = np.full((size, size), 0.1, dtype=np.float64)
        mid = size // 2
        # Shaft
        shaft_h = size // 6
        for r in range(mid - shaft_h, mid + shaft_h):
            for c in range(size // 6, size * 2 // 3):
                tex[r, c] = 0.95
        # Arrowhead
        head_start = size * 2 // 3
        for c in range(head_start, size - size // 8):
            spread = int((c - head_start) * 0.8)
            for r in range(mid - size // 3 + spread, mid + size // 3 - spread):
                if 0 <= r < size:
                    tex[r, c] = 0.95
        return tex

    def _build_flat_textured(self) -> None:
        """Scene: A textured back wall viewed from two angles."""
        tex = self._make_letter_texture("F", 48)
        self.objects.append(SceneObject(
            kind="plane",
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            albedo=tex,
            half_extents=(1.5, 1.5),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=0,
        ))

    def _build_corner_room(self) -> None:
        """Scene: Two walls meeting at a 90-degree corner with textures.

        Creates a concave L-shaped room where inter-reflections can occur
        and each viewpoint sees a different wall predominantly.
        """
        # Back wall (Z=0 plane) with checkerboard
        checker = self._make_checker_texture(48, 6)
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            albedo=checker,
            half_extents=(1.5, 1.5),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=0,
        ))
        # Right wall with letter
        letter = self._make_letter_texture("R", 48)
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([1.5, 0.0, 1.0]),
            normal=np.array([-1.0, 0.0, 0.0]),
            albedo=letter,
            half_extents=(1.0, 1.5),
            right_axis=np.array([0.0, 0.0, -1.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=1,
        ))
        # Floor
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([0.0, -1.5, 0.5]),
            normal=np.array([0.0, 1.0, 0.0]),
            albedo=0.4,
            half_extents=(1.5, 1.0),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 0.0, -1.0]),
            obj_id=2,
        ))

    def _build_box_and_wall(self) -> None:
        """Scene: A box in front of a textured wall.

        This is the most compelling demo: the box creates occlusion.
        From the camera side, you see one face of the box and the wall
        behind it. From the projector side (dual), you see the OTHER
        face of the box and different parts of the wall.
        """
        # Back wall with arrow texture (direction shows viewpoint swap)
        arrow = self._make_arrow_texture(48)
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([0.0, 0.0, -0.5]),
            normal=np.array([0.0, 0.0, 1.0]),
            albedo=arrow,
            half_extents=(2.0, 1.5),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=0,
        ))
        # Box: 6 faces of a cube at the center
        box_center = np.array([0.0, 0.0, 0.5])
        box_size = 0.4
        # Front face (+Z)
        self.objects.append(SceneObject(
            kind="box_face",
            center=box_center + np.array([0, 0, box_size]),
            normal=np.array([0.0, 0.0, 1.0]),
            albedo=0.9,
            half_extents=(box_size, box_size),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=1,
        ))
        # Back face (-Z)
        self.objects.append(SceneObject(
            kind="box_face",
            center=box_center + np.array([0, 0, -box_size]),
            normal=np.array([0.0, 0.0, -1.0]),
            albedo=0.9,
            half_extents=(box_size, box_size),
            right_axis=np.array([-1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=2,
        ))
        # Left face (-X)
        letter_l = self._make_letter_texture("L", 32)
        self.objects.append(SceneObject(
            kind="box_face",
            center=box_center + np.array([-box_size, 0, 0]),
            normal=np.array([-1.0, 0.0, 0.0]),
            albedo=letter_l,
            half_extents=(box_size, box_size),
            right_axis=np.array([0.0, 0.0, -1.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=3,
        ))
        # Right face (+X)
        letter_r = self._make_letter_texture("R", 32)
        self.objects.append(SceneObject(
            kind="box_face",
            center=box_center + np.array([box_size, 0, 0]),
            normal=np.array([1.0, 0.0, 0.0]),
            albedo=letter_r,
            half_extents=(box_size, box_size),
            right_axis=np.array([0.0, 0.0, 1.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=4,
        ))
        # Top face (+Y)
        self.objects.append(SceneObject(
            kind="box_face",
            center=box_center + np.array([0, box_size, 0]),
            normal=np.array([0.0, 1.0, 0.0]),
            albedo=0.95,
            half_extents=(box_size, box_size),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 0.0, -1.0]),
            obj_id=5,
        ))
        # Bottom face (-Y)
        self.objects.append(SceneObject(
            kind="box_face",
            center=box_center + np.array([0, -box_size, 0]),
            normal=np.array([0.0, -1.0, 0.0]),
            albedo=0.3,
            half_extents=(box_size, box_size),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 0.0, 1.0]),
            obj_id=6,
        ))
        # Floor
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([0.0, -1.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            albedo=0.35,
            half_extents=(2.0, 2.0),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 0.0, -1.0]),
            obj_id=7,
        ))

    def _build_cylinder_text(self) -> None:
        """Scene: Approximated cylinder with text using polygon strip.

        Text on a curved surface appears mirrored from different viewpoints,
        providing the most intuitive dual photography demonstration.
        Uses a polygon strip to approximate a half-cylinder.
        """
        # Back wall
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([0.0, 0.0, -1.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            albedo=0.3,
            half_extents=(2.0, 1.5),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=0,
        ))
        # Approximate half-cylinder with 12 flat faces
        n_segments = 12
        radius = 0.6
        letter_tex = self._make_letter_texture("D", 48)
        for i in range(n_segments):
            theta0 = np.pi * 0.2 + (np.pi * 0.6) * i / n_segments
            theta1 = np.pi * 0.2 + (np.pi * 0.6) * (i + 1) / n_segments
            theta_mid = (theta0 + theta1) / 2.0

            cx = radius * np.cos(theta_mid)
            cz = radius * np.sin(theta_mid)
            nx = np.cos(theta_mid)
            nz = np.sin(theta_mid)

            seg_width = radius * (theta1 - theta0) / 2.0

            # Sample texture column for this segment
            tex_col_start = int(i / n_segments * 48)
            tex_col_end = int((i + 1) / n_segments * 48)
            seg_tex = letter_tex[:, tex_col_start:max(tex_col_end, tex_col_start + 1)]
            if seg_tex.shape[1] < 2:
                seg_tex = np.tile(seg_tex, (1, 2))
            # Use average albedo for this strip
            avg_albedo = float(seg_tex.mean())

            self.objects.append(SceneObject(
                kind="box_face",
                center=np.array([cx, 0.0, cz]),
                normal=np.array([nx, 0.0, nz]),
                albedo=avg_albedo,
                half_extents=(seg_width, 0.8),
                right_axis=np.array([-nz, 0.0, nx]),  # Tangent
                up_axis=np.array([0.0, 1.0, 0.0]),
                obj_id=10 + i,
            ))

    def _build_sphere_on_plane(self) -> None:
        """Scene: A sphere sitting on a checkerboard floor.

        The sphere creates curved reflections and self-shadowing.
        Different viewpoints see different parts of the sphere surface.
        """
        # Checkerboard floor
        checker = self._make_checker_texture(48, 6)
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([0.0, -0.5, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            albedo=checker,
            half_extents=(2.0, 2.0),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 0.0, -1.0]),
            obj_id=0,
        ))
        # Sphere
        self.objects.append(SceneObject(
            kind="sphere",
            center=np.array([0.0, 0.1, 0.3]),
            radius=0.55,
            albedo=0.85,
            obj_id=1,
        ))
        # Back wall
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([0.0, 0.5, -1.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            albedo=0.5,
            half_extents=(2.0, 1.5),
            right_axis=np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=2,
        ))

    def _build_two_planes(self) -> None:
        """Scene: Two angled planes with different textures.

        The angle between them means each viewpoint sees a different
        mix of the two surfaces. Simple but effective.
        """
        # Left plane angled to face right
        letter_a = self._make_letter_texture("A", 48)
        angle = np.radians(30)
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([-0.6, 0.0, 0.0]),
            normal=np.array([np.sin(angle), 0.0, np.cos(angle)]),
            albedo=letter_a,
            half_extents=(1.0, 1.0),
            right_axis=np.array([np.cos(angle), 0.0, -np.sin(angle)]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=0,
        ))
        # Right plane angled to face left
        letter_p = self._make_letter_texture("P", 48)
        self.objects.append(SceneObject(
            kind="box_face",
            center=np.array([0.6, 0.0, 0.0]),
            normal=np.array([-np.sin(angle), 0.0, np.cos(angle)]),
            albedo=letter_p,
            half_extents=(1.0, 1.0),
            right_axis=np.array([-np.cos(angle), 0.0, np.sin(angle)]),
            up_axis=np.array([0.0, 1.0, 0.0]),
            obj_id=1,
        ))

    def _build_camera_basis(
        self, pos: np.ndarray, target: np.ndarray, fov_deg: float, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ray directions for a pinhole camera/projector.

        Args:
            pos: Camera/projector position.
            target: Look-at target point.
            fov_deg: Field of view in degrees.
            shape: Image resolution as (height, width).

        Returns:
            Tuple of (forward, right, up) unit vectors scaled by FOV.
        """
        forward = target - pos
        forward = forward / (np.linalg.norm(forward) + 1e-12)

        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 0.0, 1.0])

        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-12)

        aspect = shape[1] / shape[0]
        half_h = np.tan(np.radians(fov_deg) / 2.0)
        half_w = half_h * aspect

        return forward, right * half_w, up * half_h

    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray) -> HitResult:
        """Cast a ray into the scene and return the closest hit.

        Tests intersection against all scene objects, returns the
        nearest valid hit (front-face only for planes).

        Args:
            origin: Ray origin.
            direction: Normalized ray direction.

        Returns:
            Nearest HitResult, or HitResult(hit=False) if nothing hit.
        """
        best = HitResult(False, 1e30, np.zeros(3), np.zeros(3), 0.0, -1)
        for obj in self.objects:
            hit = obj.intersect(origin, direction)
            if hit.hit and hit.t < best.t:
                best = hit
        return best

    def _pixel_ray(
        self,
        pos: np.ndarray,
        forward: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        row: int,
        col: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Compute the ray direction for a specific pixel.

        Args:
            pos: Camera/projector position.
            forward, right, up: Camera basis vectors (from _build_camera_basis).
            row, col: Pixel coordinates.
            shape: Image shape as (height, width).

        Returns:
            Normalized ray direction vector.
        """
        h, w = shape
        # Map pixel to normalized device coordinates [-1, 1]
        u = (2.0 * (col + 0.5) / w) - 1.0
        v = 1.0 - (2.0 * (row + 0.5) / h)  # Flip Y: top row = +v

        direction = forward + u * right + v * up
        return direction / (np.linalg.norm(direction) + 1e-12)

    def compute_transport_matrix(self) -> np.ndarray:
        """Compute the light transport matrix via ray-casting.

        For each projector pixel j, casts a ray from the projector to find
        where it hits the scene. Then, for each camera pixel i, evaluates
        how much of that reflected light reaches camera pixel i, considering
        Lambertian BRDF and geometric factors.

        The key insight: projector pixel j illuminates a specific surface
        point. That point scatters light in all directions. Camera pixel i
        receives light proportional to the albedo, the cosine toward the
        camera, and the solid angle subtended by camera pixel i.

        For efficiency with small images, we use a "splatting" approach:
        each projector pixel's light contribution is evaluated at every
        camera pixel by checking what the camera sees and whether it
        matches the illuminated point.

        Returns:
            Transport matrix T of shape (cam_h * cam_w, proj_h * proj_w).
        """
        ph, pw = self.proj_shape
        ch, cw = self.cam_shape
        n_proj = ph * pw
        n_cam = ch * cw

        # Build projector and camera ray bases
        p_fwd, p_right, p_up = self._build_camera_basis(
            self.proj_pos, self.proj_target, self.proj_fov, self.proj_shape
        )
        c_fwd, c_right, c_up = self._build_camera_basis(
            self.cam_pos, self.cam_target, self.cam_fov, self.cam_shape
        )

        # Pre-compute camera ray hits (what each camera pixel sees)
        cam_hits: list[HitResult] = []
        for ci in range(ch):
            for cj in range(cw):
                ray_dir = self._pixel_ray(
                    self.cam_pos, c_fwd, c_right, c_up, ci, cj, self.cam_shape
                )
                hit = self._cast_ray(self.cam_pos, ray_dir)
                cam_hits.append(hit)

        T = np.zeros((n_cam, n_proj), dtype=np.float64)

        for pj_row in range(ph):
            for pj_col in range(pw):
                j = pj_row * pw + pj_col

                # Cast ray from projector pixel j
                proj_ray = self._pixel_ray(
                    self.proj_pos, p_fwd, p_right, p_up, pj_row, pj_col, self.proj_shape
                )
                proj_hit = self._cast_ray(self.proj_pos, proj_ray)

                if not proj_hit.hit:
                    continue

                # The projector illuminates point proj_hit.point on the scene.
                # Now compute how much of this light reaches each camera pixel.
                hit_point = proj_hit.point
                hit_normal = proj_hit.normal
                hit_albedo = proj_hit.albedo

                # Cosine of incidence angle (projector -> surface)
                to_proj = self.proj_pos - hit_point
                dist_proj = np.linalg.norm(to_proj)
                dir_to_proj = to_proj / (dist_proj + 1e-12)
                cos_in = max(0.0, np.dot(hit_normal, dir_to_proj))

                if cos_in < 1e-6:
                    continue

                for i in range(n_cam):
                    cam_hit = cam_hits[i]
                    if not cam_hit.hit:
                        continue

                    # Check if this camera pixel sees approximately the
                    # same point that the projector illuminated.
                    # Use distance threshold based on scene scale.
                    dist_to_illuminated = np.linalg.norm(cam_hit.point - hit_point)

                    # Adaptive threshold: proportional to the solid angle
                    # of one projector pixel at the hit distance
                    pixel_footprint = dist_proj * np.tan(np.radians(self.proj_fov) / max(ph, pw))
                    threshold = pixel_footprint * 1.5

                    if dist_to_illuminated > threshold:
                        continue

                    # Cosine of viewing angle (surface -> camera)
                    to_cam = self.cam_pos - hit_point
                    dist_cam = np.linalg.norm(to_cam)
                    dir_to_cam = to_cam / (dist_cam + 1e-12)
                    cos_out = max(0.0, np.dot(hit_normal, dir_to_cam))

                    if cos_out < 1e-6:
                        continue

                    # Check visibility: is the path from hit_point to camera
                    # blocked by any other object?
                    shadow_dir = dir_to_cam
                    shadow_hit = self._cast_ray(
                        hit_point + hit_normal * 1e-3, shadow_dir
                    )
                    if shadow_hit.hit and shadow_hit.t < dist_cam - 0.01:
                        continue  # Occluded

                    # Lambertian BRDF: T[i,j] = albedo * cos_in * cos_out / r^2
                    T[i, j] = (
                        hit_albedo * cos_in * cos_out
                        / (dist_cam ** 2 + 1e-6)
                    )

        # Normalize to [0, 1]
        t_max = T.max()
        if t_max > 0:
            T /= t_max

        return T
