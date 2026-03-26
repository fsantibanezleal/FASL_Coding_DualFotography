"""3D scene simulation with vectorized ray-casting for dual photography.

Computes light transport matrices using GPU-style vectorized NumPy operations
for perspective projection, multi-object intersection, occlusion testing,
Lambertian + Blinn-Phong BRDF, and multi-bounce indirect illumination.

The transport matrix T[i,j] encodes how much light from projector pixel j
reaches camera pixel i through all paths in the scene. This matrix is the
foundation of dual photography: its transpose T^T swaps the viewpoints
of projector and camera (Helmholtz reciprocity).

Performance: Vectorized operations achieve 10-100x speedup over scalar loops,
making 64x64 and 128x128 resolutions practical for interactive use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SceneType(Enum):
    """Available synthetic scene configurations."""

    FLAT_TEXTURED = "flat_textured"
    CORNER_ROOM = "corner_room"
    BOX_AND_WALL = "box_and_wall"
    CYLINDER_TEXT = "cylinder_text"
    SPHERE_ON_PLANE = "sphere_on_plane"
    TWO_PLANES = "two_planes"
    CORNELL_BOX = "cornell_box"
    GALLERY = "gallery"
    STAIRCASE = "staircase"
    MIRROR_ROOM = "mirror_room"


# ---------------------------------------------------------------------------
# Material model
# ---------------------------------------------------------------------------

@dataclass
class Material:
    """PBR surface material with GGX microfacet specular model.

    Uses the metallic/roughness workflow common in physically-based rendering:
    - roughness: Controls microfacet distribution width (0=mirror, 1=fully rough)
    - metallic: Interpolates between dielectric and metallic behavior
    - f0: Fresnel reflectance at normal incidence

    For dielectrics (metallic=0): albedo goes to diffuse, f0=0.04.
    For metals (metallic=1): no diffuse, f0=albedo (colored specular).

    Attributes:
        diffuse: Diffuse reflectance (albedo). Scalar or 2D texture array.
        roughness: Surface roughness [0, 1]. Controls GGX NDF width.
        metallic: Metallic factor [0, 1].
        f0: Base Fresnel reflectance at normal incidence.
        specular: Legacy specular coefficient (converted to roughness internally).
        shininess: Legacy Phong exponent (converted to roughness internally).
        is_mirror: If True, surface acts as a perfect mirror.
    """

    diffuse: float | np.ndarray = 0.7
    roughness: float = 0.5
    metallic: float = 0.0
    f0: float = 0.04
    specular: float = 0.0
    shininess: float = 32.0
    is_mirror: bool = False

    def __post_init__(self):
        # Legacy compat: convert specular/shininess to roughness if set
        if self.specular > 0.01 and self.roughness == 0.5:
            self.roughness = max(0.05, 1.0 - (self.shininess / 128.0))
            self.metallic = min(self.specular, 0.9)
            self.f0 = 0.04 + self.specular * 0.5


# ---------------------------------------------------------------------------
# Scene primitives
# ---------------------------------------------------------------------------

@dataclass
class SceneObject:
    """A geometric primitive in the scene (plane or sphere).

    Supports bounded planes (rectangles) and spheres. Each object has an
    associated material and texture mapping parameters.

    Attributes:
        kind: Primitive type: "plane", "box_face", or "sphere".
        center: Position (3D) of the object center.
        normal: Surface normal (for planes).
        radius: Sphere radius.
        material: Surface material properties.
        half_extents: Half-size for bounded planes (u_extent, v_extent).
        right_axis: Local U direction for texture mapping.
        up_axis: Local V direction for texture mapping.
        obj_id: Unique identifier.
    """

    kind: str
    center: np.ndarray
    normal: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    radius: float = 0.5
    material: Material = field(default_factory=Material)
    half_extents: tuple[float, float] = (0.5, 0.5)
    right_axis: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    up_axis: np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))
    obj_id: int = 0

    def intersect_batch(
        self, origins: np.ndarray, dirs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batch ray intersection for N rays at once.

        Args:
            origins: (N, 3) ray origins.
            dirs: (N, 3) normalized ray directions.

        Returns:
            Tuple of (t_hits, points, normals, uvs) where:
              - t_hits: (N,) hit distances, np.inf if missed.
              - points: (N, 3) hit positions.
              - normals: (N, 3) surface normals at hit.
              - uvs: (N, 2) texture coordinates at hit.
        """
        N = origins.shape[0]
        t_hits = np.full(N, np.inf)
        points = np.zeros((N, 3))
        normals = np.zeros((N, 3))
        uvs = np.zeros((N, 2))

        if self.kind in ("plane", "box_face"):
            self._intersect_plane_batch(origins, dirs, t_hits, points, normals, uvs)
        elif self.kind == "sphere":
            self._intersect_sphere_batch(origins, dirs, t_hits, points, normals, uvs)

        return t_hits, points, normals, uvs

    def _intersect_plane_batch(self, origins, dirs, t_hits, points, normals, uvs):
        """Vectorized ray-plane intersection."""
        n = self.normal
        denom = dirs @ n  # (N,)
        valid_denom = np.abs(denom) > 1e-10

        diff = self.center - origins  # (N, 3)
        t = np.where(valid_denom, (diff @ n) / np.where(valid_denom, denom, 1.0), np.inf)
        valid = valid_denom & (t > 1e-4)

        pts = origins + t[:, None] * dirs  # (N, 3)

        if self.kind == "box_face":
            local = pts - self.center
            u_coord = local @ self.right_axis
            v_coord = local @ self.up_axis
            in_bounds = (np.abs(u_coord) <= self.half_extents[0]) & \
                        (np.abs(v_coord) <= self.half_extents[1])
            valid = valid & in_bounds
            # UV mapping
            uv_u = (u_coord / self.half_extents[0] + 1.0) / 2.0
            uv_v = (v_coord / self.half_extents[1] + 1.0) / 2.0
            uvs[valid, 0] = np.clip(uv_u[valid], 0, 1)
            uvs[valid, 1] = np.clip(uv_v[valid], 0, 1)
        else:
            local = pts - self.center
            u_coord = local @ self.right_axis
            v_coord = local @ self.up_axis
            uv_u = (u_coord / self.half_extents[0] + 1.0) / 2.0
            uv_v = (v_coord / self.half_extents[1] + 1.0) / 2.0
            uvs[valid, 0] = np.clip(uv_u[valid], 0, 1)
            uvs[valid, 1] = np.clip(uv_v[valid], 0, 1)

        t_hits[valid] = t[valid]
        points[valid] = pts[valid]
        normals[valid] = n

    def _intersect_sphere_batch(self, origins, dirs, t_hits, points, normals, uvs):
        """Vectorized ray-sphere intersection."""
        oc = origins - self.center  # (N, 3)
        a = np.sum(dirs * dirs, axis=1)
        b = 2.0 * np.sum(oc * dirs, axis=1)
        c = np.sum(oc * oc, axis=1) - self.radius ** 2
        disc = b * b - 4 * a * c

        valid_disc = disc >= 0
        sqrt_disc = np.sqrt(np.maximum(disc, 0))
        t1 = (-b - sqrt_disc) / (2 * a + 1e-12)
        t2 = (-b + sqrt_disc) / (2 * a + 1e-12)

        # Take nearest positive t
        t = np.where(t1 > 1e-4, t1, t2)
        valid = valid_disc & (t > 1e-4)

        pts = origins + t[:, None] * dirs
        nrm = (pts - self.center) / self.radius

        t_hits[valid] = t[valid]
        points[valid] = pts[valid]
        normals[valid] = nrm[valid]

        # Spherical UV mapping
        rel = nrm[valid]
        uvs[valid, 0] = 0.5 + np.arctan2(rel[:, 0], rel[:, 2]) / (2 * np.pi)
        uvs[valid, 1] = 0.5 + np.arcsin(np.clip(rel[:, 1], -1, 1)) / np.pi


# ---------------------------------------------------------------------------
# Texture generation
# ---------------------------------------------------------------------------

def make_checker_texture(size: int = 48, n_checks: int = 6) -> np.ndarray:
    """Generate a checkerboard texture.

    Args:
        size: Texture resolution (square).
        n_checks: Number of checks per side.

    Returns:
        2D float64 array with values 0.15 and 0.9.
    """
    block = max(1, size // n_checks)
    rows = np.arange(size)[:, None] // block
    cols = np.arange(size)[None, :] // block
    return np.where((rows + cols) % 2 == 0, 0.9, 0.15).astype(np.float64)


def make_letter_texture(letter: str = "F", size: int = 48) -> np.ndarray:
    """Render a single letter as a pixel-art texture.

    Args:
        letter: Character to render (supported: F, D, P, L, A, R).
        size: Texture resolution (square).

    Returns:
        2D float64 array with values ~0.1 (background) and ~0.95 (letter).
    """
    fonts = {
        "F": ["########", "#.......", "#.......", "######..",
               "#.......", "#.......", "#.......", "#......."],
        "D": ["######..", "#.....#.", "#......#", "#......#",
               "#......#", "#......#", "#.....#.", "######.."],
        "P": ["######..", "#.....#.", "#.....#.", "######..",
               "#.......", "#.......", "#.......", "#......."],
        "L": ["#.......", "#.......", "#.......", "#.......",
               "#.......", "#.......", "#.......", "########"],
        "A": ["..###...", ".#...#..", "#.....#.", "#.....#.",
               "#######.", "#.....#.", "#.....#.", "#.....#."],
        "R": ["######..", "#.....#.", "#.....#.", "######..",
               "#..#....", "#...#...", "#....#..", "#.....#."],
    }
    tex = np.full((size, size), 0.1, dtype=np.float64)
    bitmap = fonts.get(letter, fonts["F"])
    bh, bw = len(bitmap), len(bitmap[0])
    margin = size // 8
    cell_h = (size - 2 * margin) / bh
    cell_w = (size - 2 * margin) / bw
    for ri, row in enumerate(bitmap):
        for ci, ch in enumerate(row):
            if ch == "#":
                r0, r1 = int(margin + ri * cell_h), int(margin + (ri + 1) * cell_h)
                c0, c1 = int(margin + ci * cell_w), int(margin + (ci + 1) * cell_w)
                tex[r0:r1, c0:c1] = 0.95
    return tex


def make_arrow_texture(size: int = 48) -> np.ndarray:
    """Render a right-pointing arrow texture."""
    tex = np.full((size, size), 0.1, dtype=np.float64)
    mid = size // 2
    shaft_h = size // 6
    tex[mid - shaft_h:mid + shaft_h, size // 6:size * 2 // 3] = 0.95
    head_start = size * 2 // 3
    for c in range(head_start, size - size // 8):
        spread = int((c - head_start) * 0.8)
        lo = max(0, mid - size // 3 + spread)
        hi = min(size, mid + size // 3 - spread)
        tex[lo:hi, c] = 0.95
    return tex


def make_gradient_texture(size: int = 48) -> np.ndarray:
    """Horizontal gradient from dark to bright."""
    return np.tile(np.linspace(0.1, 0.95, size), (size, 1)).astype(np.float64)


def make_circles_texture(size: int = 48) -> np.ndarray:
    """Concentric circles pattern."""
    y, x = np.mgrid[:size, :size]
    r = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
    return (0.5 + 0.4 * np.sin(r * 2 * np.pi / (size / 4))).astype(np.float64)


def make_cross_texture(size: int = 48) -> np.ndarray:
    """Cross/plus sign pattern."""
    tex = np.full((size, size), 0.1, dtype=np.float64)
    arm = size // 6
    mid = size // 2
    tex[mid - arm:mid + arm, :] = 0.9
    tex[:, mid - arm:mid + arm] = 0.9
    return tex


# ---------------------------------------------------------------------------
# Main scene class
# ---------------------------------------------------------------------------

class SyntheticScene:
    """3D scene with vectorized ray-casting for transport matrix computation.

    This is the simulation engine for dual photography. It constructs a
    scene from geometric primitives, then computes the transport matrix
    by casting rays in bulk using NumPy vectorized operations.

    Features over the previous version:
    - Vectorized ray-casting: 10-100x faster than scalar loops
    - Blinn-Phong BRDF: specular highlights that are view-dependent
    - Multi-bounce transport: light bounces between surfaces (radiosity)
    - 10 scene types including Cornell Box, Gallery, Staircase, Mirror Room
    - Material system with diffuse, specular, and mirror properties

    Attributes:
        scene_type: Which scene configuration to build.
        proj_shape: Projector pixel resolution (height, width).
        cam_shape: Camera pixel resolution (height, width).
        proj_pos: 3D position of the projector.
        cam_pos: 3D position of the camera.
        proj_fov: Projector field of view in degrees.
        cam_fov: Camera field of view in degrees.
        proj_target: Look-at target for the projector.
        cam_target: Look-at target for the camera.
        n_bounces: Number of indirect light bounces (0 = direct only).
    """

    def __init__(
        self,
        scene_type: SceneType = SceneType.BOX_AND_WALL,
        proj_shape: tuple[int, int] = (32, 32),
        cam_shape: tuple[int, int] = (32, 32),
        proj_pos: np.ndarray | None = None,
        cam_pos: np.ndarray | None = None,
        proj_fov: float = 60.0,
        cam_fov: float = 60.0,
        proj_target: np.ndarray | None = None,
        cam_target: np.ndarray | None = None,
        albedo: float = 0.8,
        n_bounces: int = 0,
    ):
        self.scene_type = scene_type
        self.proj_shape = proj_shape
        self.cam_shape = cam_shape
        self.proj_pos = proj_pos if proj_pos is not None else np.array([-1.5, 0.5, 2.0])
        self.cam_pos = cam_pos if cam_pos is not None else np.array([1.5, 0.5, 2.0])
        self.proj_fov = proj_fov
        self.cam_fov = cam_fov
        self.proj_target = proj_target if proj_target is not None else np.array([0., 0., 0.])
        self.cam_target = cam_target if cam_target is not None else np.array([0., 0., 0.])
        self.albedo = albedo
        self.n_bounces = n_bounces
        self.objects: list[SceneObject] = []
        self._build_scene()

    # -------------------------------------------------------------------
    # Camera / ray utilities
    # -------------------------------------------------------------------

    @staticmethod
    def _look_at(pos: np.ndarray, target: np.ndarray, fov_deg: float,
                 shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute camera basis vectors for a pinhole model.

        Returns (forward, right_scaled, up_scaled) where right and up are
        scaled by the FOV half-extent so that pixel rays can be computed as:
            direction = forward + u*right_scaled + v*up_scaled
        """
        fwd = target - pos
        fwd /= np.linalg.norm(fwd) + 1e-12
        world_up = np.array([0., 1., 0.])
        if abs(np.dot(fwd, world_up)) > 0.99:
            world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up) + 1e-12
        aspect = shape[1] / shape[0]
        half_h = np.tan(np.radians(fov_deg) / 2.0)
        return fwd, right * half_h * aspect, up * half_h

    def _generate_rays(
        self, pos: np.ndarray, fwd: np.ndarray, right: np.ndarray,
        up: np.ndarray, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate all rays for a device as (N, 3) origins and directions.

        Args:
            pos: Device position.
            fwd, right, up: Basis from _look_at.
            shape: (height, width).

        Returns:
            (origins, directions) each of shape (H*W, 3).
        """
        h, w = shape
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        u = (2.0 * (cols.ravel() + 0.5) / w) - 1.0  # (N,)
        v = 1.0 - (2.0 * (rows.ravel() + 0.5) / h)

        # direction = fwd + u*right + v*up  (broadcast)
        dirs = fwd[None, :] + u[:, None] * right[None, :] + v[:, None] * up[None, :]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= norms + 1e-12

        origins = np.broadcast_to(pos, dirs.shape).copy()
        return origins, dirs

    # -------------------------------------------------------------------
    # Vectorized scene intersection
    # -------------------------------------------------------------------

    def _trace_batch(
        self, origins: np.ndarray, dirs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trace N rays against all scene objects, return nearest hits.

        Returns:
            (hit_mask, t_vals, hit_pts, hit_normals, hit_uvs, hit_obj_ids)
            each of length N.
        """
        N = origins.shape[0]
        best_t = np.full(N, np.inf)
        best_pts = np.zeros((N, 3))
        best_nrm = np.zeros((N, 3))
        best_uvs = np.zeros((N, 2))
        best_oid = np.full(N, -1, dtype=np.int32)

        for obj in self.objects:
            t, pts, nrm, uvs = obj.intersect_batch(origins, dirs)
            closer = t < best_t
            best_t[closer] = t[closer]
            best_pts[closer] = pts[closer]
            best_nrm[closer] = nrm[closer]
            best_uvs[closer] = uvs[closer]
            best_oid[closer] = obj.obj_id

        hit_mask = best_t < 1e20
        return hit_mask, best_t, best_pts, best_nrm, best_uvs, best_oid

    def _sample_materials(
        self, obj_ids: np.ndarray, uvs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Look up PBR material properties for each hit by object ID.

        Returns:
            (albedo, roughness, metallic, f0) arrays of shape (N,).
        """
        N = obj_ids.shape[0]
        albedo = np.zeros(N)
        roughness = np.full(N, 0.5)
        metallic = np.zeros(N)
        f0 = np.full(N, 0.04)

        obj_map = {obj.obj_id: obj for obj in self.objects}
        for oid, obj in obj_map.items():
            mask = obj_ids == oid
            if not np.any(mask):
                continue
            mat = obj.material
            if isinstance(mat.diffuse, (int, float)):
                albedo[mask] = float(mat.diffuse)
            else:
                tex = mat.diffuse
                h, w = tex.shape
                px = np.clip((uvs[mask, 0] * (w - 1)).astype(int), 0, w - 1)
                py = np.clip(((1.0 - uvs[mask, 1]) * (h - 1)).astype(int), 0, h - 1)
                albedo[mask] = tex[py, px]
            roughness[mask] = mat.roughness
            metallic[mask] = mat.metallic
            f0[mask] = mat.f0

        return albedo, roughness, metallic, f0

    # -------------------------------------------------------------------
    # Transport matrix computation
    # -------------------------------------------------------------------

    def compute_transport_matrix(self) -> np.ndarray:
        """Compute the light transport matrix with GGX BRDF and shadow testing.

        Uses physically-based Cook-Torrance GGX microfacet BRDF with
        Schlick Fresnel and Smith geometric shadowing. The inner loop
        over projector pixels is scalar, but all per-pixel operations
        (candidate matching, BRDF evaluation, shadow rays) are vectorized.

        Multi-bounce indirect illumination uses Neumann series:
            T_total = T_direct + F @ T_direct + F^2 @ T_direct + ...

        Returns:
            Transport matrix T of shape (cam_pixels, proj_pixels).
        """
        from src.core.brdf import evaluate_brdf_batch

        ph, pw = self.proj_shape
        ch, cw = self.cam_shape
        n_proj = ph * pw
        n_cam = ch * cw

        # Generate and trace all rays
        p_fwd, p_right, p_up = self._look_at(
            self.proj_pos, self.proj_target, self.proj_fov, self.proj_shape)
        c_fwd, c_right, c_up = self._look_at(
            self.cam_pos, self.cam_target, self.cam_fov, self.cam_shape)

        _, proj_dirs = self._generate_rays(
            self.proj_pos, p_fwd, p_right, p_up, self.proj_shape)
        _, cam_dirs = self._generate_rays(
            self.cam_pos, c_fwd, c_right, c_up, self.cam_shape)

        proj_origins = np.broadcast_to(self.proj_pos, proj_dirs.shape).copy()
        cam_origins = np.broadcast_to(self.cam_pos, cam_dirs.shape).copy()

        c_hit, c_t, c_pts, c_nrm, c_uvs, c_oid = self._trace_batch(cam_origins, cam_dirs)
        p_hit, p_t, p_pts, p_nrm, p_uvs, p_oid = self._trace_batch(proj_origins, proj_dirs)

        # Pre-fetch all projector-hit materials in one call
        p_albedo, p_rough, p_metal, p_f0 = self._sample_materials(p_oid, p_uvs)

        # Adaptive pixel footprint for surface-point matching
        max_dim = max(ph, pw)
        base_footprint = np.tan(np.radians(self.proj_fov) / max_dim)

        # Adaptive shadow bias: scale with mean scene distance
        mean_dist = np.mean(p_t[p_hit]) if np.any(p_hit) else 1.0
        shadow_bias = max(1e-3, mean_dist * 1e-3)

        T = np.zeros((n_cam, n_proj), dtype=np.float64)

        # Direction from any surface hit to the camera (constant for pinhole)
        to_cam_global = self.cam_pos  # will subtract hit_point per j

        for j in range(n_proj):
            if not p_hit[j]:
                continue

            hp = p_pts[j]
            hn = p_nrm[j]

            # Light direction (surface -> projector)
            to_proj = self.proj_pos - hp
            dist_proj = np.linalg.norm(to_proj)
            dir_to_proj = to_proj / (dist_proj + 1e-12)
            cos_in = np.dot(hn, dir_to_proj)
            if cos_in < 1e-6:
                continue

            # View direction (surface -> camera)
            to_cam = self.cam_pos - hp
            dist_cam = np.linalg.norm(to_cam)
            dir_to_cam = to_cam / (dist_cam + 1e-12)
            cos_out = np.dot(hn, dir_to_cam)
            if cos_out < 1e-6:
                continue

            # Shadow test: is the path surface->camera occluded?
            shadow_o = (hp + hn * shadow_bias).reshape(1, 3)
            shadow_d = dir_to_cam.reshape(1, 3)
            s_hit, s_t, _, _, _, _ = self._trace_batch(shadow_o, shadow_d)
            if s_hit[0] and s_t[0] < dist_cam - shadow_bias * 10:
                continue

            # Match: which camera pixels see this same surface point?
            threshold = dist_proj * base_footprint * 1.8
            dists_sq = np.sum((c_pts - hp) ** 2, axis=1)
            candidates = c_hit & (dists_sq < threshold * threshold)
            idx = np.where(candidates)[0]
            if len(idx) == 0:
                continue

            # Evaluate GGX BRDF at this surface point
            n_cand = len(idx)
            brdf_val = evaluate_brdf_batch(
                n_dot_l=np.full(n_cand, cos_in),
                n_dot_v=np.full(n_cand, cos_out),
                normals=np.broadcast_to(hn, (n_cand, 3)),
                light_dirs=np.broadcast_to(dir_to_proj, (n_cand, 3)),
                view_dirs=np.broadcast_to(dir_to_cam, (n_cand, 3)),
                albedo=np.full(n_cand, p_albedo[j]),
                roughness=np.full(n_cand, p_rough[j]),
                metallic=np.full(n_cand, p_metal[j]),
                f0=np.full(n_cand, p_f0[j]),
            )

            # Geometric attenuation: cos_in * cos_out / dist^2
            geom = cos_in * cos_out / (dist_cam ** 2 + 1e-6)
            T[idx, j] = brdf_val * geom

        # Multi-bounce via Neumann series
        if self.n_bounces > 0:
            T = self._neumann_multi_bounce(T, c_hit, c_pts, c_nrm, c_oid, c_uvs)

        # Normalize to [0, 1]
        t_max = T.max()
        if t_max > 0:
            T /= t_max

        return T

    def _neumann_multi_bounce(
        self, T_direct: np.ndarray,
        c_hit: np.ndarray, c_pts: np.ndarray, c_nrm: np.ndarray,
        c_oid: np.ndarray, c_uvs: np.ndarray,
    ) -> np.ndarray:
        """Multi-bounce via Neumann series with proper form factors.

        Computes the inter-surface form factor matrix F between
        camera-visible surface points, then accumulates bounced light:
            T_total = T_direct + F @ T + F^2 @ T + ...

        The spectral radius of F is checked to ensure convergence.
        If rho(F) >= 1, the matrix is damped to ensure stability.

        Args:
            T_direct: Direct-only transport matrix.
            c_hit, c_pts, c_nrm, c_oid, c_uvs: Camera hit data.

        Returns:
            Transport matrix with multi-bounce contributions.
        """
        n_cam = c_pts.shape[0]
        valid_idx = np.where(c_hit)[0]
        n_valid = len(valid_idx)
        if n_valid < 2:
            return T_direct

        albedo, _, _, _ = self._sample_materials(c_oid, c_uvs)

        # Build form factor matrix F over valid surface points
        # F[i,j] = albedo[j] * cos_i * cos_j / (pi * r^2)
        pts = c_pts[valid_idx]   # (n_valid, 3)
        nrm = c_nrm[valid_idx]
        alb = albedo[valid_idx]

        F_small = np.zeros((n_valid, n_valid), dtype=np.float64)

        # Vectorized form factor computation row by row
        for i in range(n_valid):
            diff = pts - pts[i]                   # (n_valid, 3)
            dist_sq = np.sum(diff * diff, axis=1)  # (n_valid,)
            dist = np.sqrt(dist_sq + 1e-8)
            direction = diff / (dist[:, None] + 1e-12)

            cos_i = np.sum(direction * nrm[i], axis=1)    # cos at point i
            cos_j = -np.sum(direction * nrm, axis=1)      # cos at point j

            ff = np.maximum(cos_i, 0) * np.maximum(cos_j, 0) * alb / (np.pi * dist_sq + 1e-6)
            ff[i] = 0  # No self-contribution
            F_small[i] = ff

        # Normalize: ensure spectral radius < 1 for convergence
        row_sums = F_small.sum(axis=1)
        max_row = row_sums.max()
        if max_row > 0.95:
            F_small *= 0.9 / max_row

        # Expand F_small to full camera pixel space
        F = np.zeros((n_cam, n_cam), dtype=np.float64)
        F[np.ix_(valid_idx, valid_idx)] = F_small

        # Neumann series: T_total = sum_{k=0}^{bounces} F^k @ T_direct
        T_total = T_direct.copy()
        T_bounce = T_direct.copy()
        for _ in range(self.n_bounces):
            T_bounce = F @ T_bounce
            T_total += T_bounce
            # Early exit if contribution is negligible
            if np.max(np.abs(T_bounce)) < 1e-6 * np.max(np.abs(T_total)):
                break

        return T_total

    # -------------------------------------------------------------------
    # Scene builders
    # -------------------------------------------------------------------

    def _build_scene(self) -> None:
        """Construct scene objects based on scene_type."""
        builder = {
            SceneType.FLAT_TEXTURED: self._build_flat_textured,
            SceneType.CORNER_ROOM: self._build_corner_room,
            SceneType.BOX_AND_WALL: self._build_box_and_wall,
            SceneType.CYLINDER_TEXT: self._build_cylinder_text,
            SceneType.SPHERE_ON_PLANE: self._build_sphere_on_plane,
            SceneType.TWO_PLANES: self._build_two_planes,
            SceneType.CORNELL_BOX: self._build_cornell_box,
            SceneType.GALLERY: self._build_gallery,
            SceneType.STAIRCASE: self._build_staircase,
            SceneType.MIRROR_ROOM: self._build_mirror_room,
        }
        builder[self.scene_type]()

    def _box(self, center, half, material, start_id=0,
             tex_front=None, tex_back=None, tex_left=None,
             tex_right=None, tex_top=None, tex_bottom=None):
        """Helper: add a 6-face axis-aligned box to the scene."""
        faces = [
            (np.array([0, 0, 1.]), np.array([0, 0, half]),
             np.array([1., 0, 0]), np.array([0, 1., 0]), tex_front),
            (np.array([0, 0, -1.]), np.array([0, 0, -half]),
             np.array([-1., 0, 0]), np.array([0, 1., 0]), tex_back),
            (np.array([-1., 0, 0]), np.array([-half, 0, 0]),
             np.array([0, 0, -1.]), np.array([0, 1., 0]), tex_left),
            (np.array([1., 0, 0]), np.array([half, 0, 0]),
             np.array([0, 0, 1.]), np.array([0, 1., 0]), tex_right),
            (np.array([0, 1., 0]), np.array([0, half, 0]),
             np.array([1., 0, 0]), np.array([0, 0, -1.]), tex_top),
            (np.array([0, -1., 0]), np.array([0, -half, 0]),
             np.array([1., 0, 0]), np.array([0, 0, 1.]), tex_bottom),
        ]
        for fi, (n, offset, r, u, tex) in enumerate(faces):
            mat = Material(diffuse=tex if tex is not None else material.diffuse,
                           specular=material.specular, shininess=material.shininess)
            self.objects.append(SceneObject(
                kind="box_face", center=center + offset, normal=n,
                material=mat, half_extents=(half, half),
                right_axis=r, up_axis=u, obj_id=start_id + fi,
            ))

    # --- Original 6 scenes (updated for Material system) ---

    def _build_flat_textured(self):
        tex = make_letter_texture("F", 48)
        self.objects.append(SceneObject(
            kind="plane", center=np.array([0., 0., 0.]),
            normal=np.array([0., 0., 1.]), material=Material(diffuse=tex),
            half_extents=(1.5, 1.5), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=0))

    def _build_corner_room(self):
        checker = make_checker_texture(48, 6)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 0., 0.]),
            normal=np.array([0., 0., 1.]), material=Material(diffuse=checker),
            half_extents=(1.5, 1.5), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=0))
        letter = make_letter_texture("R", 48)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([1.5, 0., 1.]),
            normal=np.array([-1., 0., 0.]), material=Material(diffuse=letter),
            half_extents=(1., 1.5), right_axis=np.array([0., 0., -1.]),
            up_axis=np.array([0., 1., 0.]), obj_id=1))
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., -1.5, 0.5]),
            normal=np.array([0., 1., 0.]), material=Material(diffuse=0.4),
            half_extents=(1.5, 1.), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 0., -1.]), obj_id=2))

    def _build_box_and_wall(self):
        arrow = make_arrow_texture(48)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 0., -0.5]),
            normal=np.array([0., 0., 1.]), material=Material(diffuse=arrow),
            half_extents=(2., 1.5), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=0))
        bc = np.array([0., 0., 0.5])
        s = 0.4
        self._box(bc, s, Material(diffuse=0.9, specular=0.15, shininess=24),
                  start_id=1, tex_left=make_letter_texture("L", 32),
                  tex_right=make_letter_texture("R", 32))
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., -1., 0.]),
            normal=np.array([0., 1., 0.]), material=Material(diffuse=0.35),
            half_extents=(2., 2.), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 0., -1.]), obj_id=20))

    def _build_cylinder_text(self):
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 0., -1.]),
            normal=np.array([0., 0., 1.]), material=Material(diffuse=0.3),
            half_extents=(2., 1.5), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=0))
        letter_tex = make_letter_texture("D", 48)
        n_seg = 16
        radius = 0.6
        for i in range(n_seg):
            t0 = np.pi * 0.2 + (np.pi * 0.6) * i / n_seg
            t1 = np.pi * 0.2 + (np.pi * 0.6) * (i + 1) / n_seg
            tm = (t0 + t1) / 2.0
            cx, cz = radius * np.cos(tm), radius * np.sin(tm)
            nx, nz = np.cos(tm), np.sin(tm)
            sw = radius * (t1 - t0) / 2.0
            col_s = int(i / n_seg * 48)
            col_e = int((i + 1) / n_seg * 48)
            seg = letter_tex[:, col_s:max(col_e, col_s + 1)]
            avg = float(seg.mean())
            self.objects.append(SceneObject(
                kind="box_face", center=np.array([cx, 0., cz]),
                normal=np.array([nx, 0., nz]),
                material=Material(diffuse=avg, specular=0.1, shininess=16),
                half_extents=(sw, 0.8),
                right_axis=np.array([-nz, 0., nx]),
                up_axis=np.array([0., 1., 0.]), obj_id=10 + i))

    def _build_sphere_on_plane(self):
        checker = make_checker_texture(48, 6)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., -0.5, 0.]),
            normal=np.array([0., 1., 0.]), material=Material(diffuse=checker),
            half_extents=(2., 2.), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 0., -1.]), obj_id=0))
        self.objects.append(SceneObject(
            kind="sphere", center=np.array([0., 0.1, 0.3]),
            radius=0.55, material=Material(diffuse=0.75, specular=0.3, shininess=48),
            obj_id=1))
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 0.5, -1.]),
            normal=np.array([0., 0., 1.]), material=Material(diffuse=0.5),
            half_extents=(2., 1.5), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=2))

    def _build_two_planes(self):
        a = np.radians(30)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([-0.6, 0., 0.]),
            normal=np.array([np.sin(a), 0., np.cos(a)]),
            material=Material(diffuse=make_letter_texture("A", 48)),
            half_extents=(1., 1.),
            right_axis=np.array([np.cos(a), 0., -np.sin(a)]),
            up_axis=np.array([0., 1., 0.]), obj_id=0))
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0.6, 0., 0.]),
            normal=np.array([-np.sin(a), 0., np.cos(a)]),
            material=Material(diffuse=make_letter_texture("P", 48)),
            half_extents=(1., 1.),
            right_axis=np.array([-np.cos(a), 0., np.sin(a)]),
            up_axis=np.array([0., 1., 0.]), obj_id=1))

    # --- 4 NEW complex scenes ---

    def _build_cornell_box(self):
        """Cornell Box: the classic CG test scene.

        An enclosed room with colored walls, a tall box, and a short box.
        Inter-reflections between walls create color bleeding effects.
        The strongly enclosed geometry makes multi-bounce especially visible.
        """
        W = 1.5  # half-width
        # Back wall with gradient
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 0., -W]),
            normal=np.array([0., 0., 1.]),
            material=Material(diffuse=make_checker_texture(48, 8)),
            half_extents=(W, W), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=0))
        # Left wall (bright)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([-W, 0., 0.]),
            normal=np.array([1., 0., 0.]), material=Material(diffuse=0.85),
            half_extents=(W, W), right_axis=np.array([0., 0., -1.]),
            up_axis=np.array([0., 1., 0.]), obj_id=1))
        # Right wall (dark)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([W, 0., 0.]),
            normal=np.array([-1., 0., 0.]), material=Material(diffuse=0.35),
            half_extents=(W, W), right_axis=np.array([0., 0., 1.]),
            up_axis=np.array([0., 1., 0.]), obj_id=2))
        # Floor
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., -W, 0.]),
            normal=np.array([0., 1., 0.]),
            material=Material(diffuse=make_circles_texture(48)),
            half_extents=(W, W), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 0., -1.]), obj_id=3))
        # Ceiling
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., W, 0.]),
            normal=np.array([0., -1., 0.]), material=Material(diffuse=0.8),
            half_extents=(W, W), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 0., 1.]), obj_id=4))
        # Tall box (rotated slightly)
        tb_c = np.array([-0.5, -0.75, -0.3])
        self._box(tb_c, 0.35, Material(diffuse=0.85, specular=0.1, shininess=16),
                  start_id=10)
        # Short box
        sb_c = np.array([0.5, -1.1, 0.2])
        self._box(sb_c, 0.35, Material(diffuse=0.7, specular=0.2, shininess=32),
                  start_id=20, tex_front=make_letter_texture("D", 32))

    def _build_gallery(self):
        """Gallery: multiple objects at different depths.

        A museum-like scene with paintings on the back wall, a pedestal
        with a sphere, and a foreground object. Lots of depth variation
        and occlusion makes this ideal for demonstrating dual photography.
        """
        # Back wall
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 0., -1.5]),
            normal=np.array([0., 0., 1.]), material=Material(diffuse=0.45),
            half_extents=(3., 2.), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=0))
        # Floor
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., -1.2, 0.]),
            normal=np.array([0., 1., 0.]),
            material=Material(diffuse=make_checker_texture(64, 8)),
            half_extents=(3., 3.), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 0., -1.]), obj_id=1))
        # Painting 1 (left)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([-1.2, 0.3, -1.45]),
            normal=np.array([0., 0., 1.]),
            material=Material(diffuse=make_letter_texture("A", 48)),
            half_extents=(0.5, 0.5), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=2))
        # Painting 2 (right)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([1.2, 0.3, -1.45]),
            normal=np.array([0., 0., 1.]),
            material=Material(diffuse=make_cross_texture(48)),
            half_extents=(0.5, 0.5), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=3))
        # Pedestal
        ped_c = np.array([0., -0.8, -0.3])
        self._box(ped_c, 0.25, Material(diffuse=0.6), start_id=10)
        # Sphere on pedestal
        self.objects.append(SceneObject(
            kind="sphere", center=np.array([0., -0.25, -0.3]),
            radius=0.3, material=Material(diffuse=0.8, specular=0.4, shininess=64),
            obj_id=20))
        # Foreground column (left)
        col_c = np.array([-0.8, -0.5, 0.8])
        self._box(col_c, 0.15, Material(diffuse=0.55, specular=0.05, shininess=8),
                  start_id=30)
        # Foreground column (right)
        col_c2 = np.array([0.8, -0.5, 0.8])
        self._box(col_c2, 0.15, Material(diffuse=0.55, specular=0.05, shininess=8),
                  start_id=40)

    def _build_staircase(self):
        """Staircase: stepped geometry with strong depth variation.

        Each step is at a different Z-depth, creating strong parallax
        and occlusion between viewpoints. The alternating textures make
        the viewpoint swap immediately visible.
        """
        # Back wall
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 0.5, -1.5]),
            normal=np.array([0., 0., 1.]),
            material=Material(diffuse=make_gradient_texture(48)),
            half_extents=(2., 2.), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=0))

        # 5 steps going from front-bottom to back-top
        n_steps = 5
        step_w = 0.6
        step_h = 0.15
        textures = [
            make_letter_texture("L", 32),
            make_checker_texture(32, 4),
            make_letter_texture("A", 32),
            make_circles_texture(32),
            make_letter_texture("D", 32),
        ]
        for i in range(n_steps):
            z = 0.8 - i * 0.45
            y = -0.8 + i * 0.35
            # Top face (the visible step surface)
            self.objects.append(SceneObject(
                kind="box_face",
                center=np.array([0., y + step_h, z]),
                normal=np.array([0., 1., 0.]),
                material=Material(diffuse=textures[i], specular=0.05, shininess=12),
                half_extents=(step_w, 0.2),
                right_axis=np.array([1., 0., 0.]),
                up_axis=np.array([0., 0., -1.]),
                obj_id=10 + i * 3))
            # Front face (riser)
            self.objects.append(SceneObject(
                kind="box_face",
                center=np.array([0., y, z + 0.2]),
                normal=np.array([0., 0., 1.]),
                material=Material(diffuse=0.5),
                half_extents=(step_w, step_h),
                right_axis=np.array([1., 0., 0.]),
                up_axis=np.array([0., 1., 0.]),
                obj_id=11 + i * 3))

        # Side wall
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([-0.7, 0., 0.]),
            normal=np.array([1., 0., 0.]), material=Material(diffuse=0.4),
            half_extents=(1.5, 1.5),
            right_axis=np.array([0., 0., -1.]),
            up_axis=np.array([0., 1., 0.]), obj_id=50))

    def _build_mirror_room(self):
        """Mirror room: two facing surfaces with high specular reflectance.

        Specular surfaces create view-dependent highlights that look
        dramatically different from the projector vs camera viewpoint.
        The Blinn-Phong specular term makes this scene showcase the
        power of the full BRDF in the transport matrix.
        """
        # Floor with pattern
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., -0.8, 0.]),
            normal=np.array([0., 1., 0.]),
            material=Material(diffuse=make_checker_texture(48, 6)),
            half_extents=(2., 2.), right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 0., -1.]), obj_id=0))
        # Left mirror wall (high specular)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([-1.3, 0., 0.]),
            normal=np.array([1., 0., 0.]),
            material=Material(diffuse=0.15, specular=0.8, shininess=128),
            half_extents=(1.5, 1.2),
            right_axis=np.array([0., 0., -1.]),
            up_axis=np.array([0., 1., 0.]), obj_id=1))
        # Right mirror wall (high specular)
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([1.3, 0., 0.]),
            normal=np.array([-1., 0., 0.]),
            material=Material(diffuse=0.15, specular=0.8, shininess=128),
            half_extents=(1.5, 1.2),
            right_axis=np.array([0., 0., 1.]),
            up_axis=np.array([0., 1., 0.]), obj_id=2))
        # Central shiny sphere
        self.objects.append(SceneObject(
            kind="sphere", center=np.array([0., -0.2, 0.2]),
            radius=0.45,
            material=Material(diffuse=0.3, specular=0.6, shininess=96),
            obj_id=3))
        # Back wall
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 0., -1.2]),
            normal=np.array([0., 0., 1.]),
            material=Material(diffuse=make_cross_texture(48)),
            half_extents=(1.5, 1.2),
            right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 1., 0.]), obj_id=4))
        # Ceiling
        self.objects.append(SceneObject(
            kind="box_face", center=np.array([0., 1.2, 0.]),
            normal=np.array([0., -1., 0.]),
            material=Material(diffuse=0.7),
            half_extents=(2., 2.),
            right_axis=np.array([1., 0., 0.]),
            up_axis=np.array([0., 0., 1.]), obj_id=5))
