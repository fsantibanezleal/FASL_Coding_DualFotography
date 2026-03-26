"""Physically-based BRDF models for light transport computation.

Implements energy-conserving reflectance models used in the transport
matrix computation. Each BRDF evaluates the reflected radiance given
incident and exitant directions and surface properties.

Models implemented:
- Lambertian: ideal diffuse reflector (rho/pi)
- Oren-Nayar: rough diffuse (view-dependent diffuse broadening)
- GGX/Cook-Torrance: microfacet specular with physically-based
  normal distribution, geometric shadowing, and Fresnel terms
- Combined: weighted sum of diffuse + specular

All models are vectorized for batch evaluation over N surface points.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INV_PI = 1.0 / np.pi
EPS = 1e-7


# ---------------------------------------------------------------------------
# Fresnel
# ---------------------------------------------------------------------------

def schlick_fresnel(cos_theta: np.ndarray, f0: float = 0.04) -> np.ndarray:
    """Schlick's approximation to the Fresnel reflectance.

    At grazing angles, all materials approach total reflection (F -> 1).
    f0 is the reflectance at normal incidence, determined by the
    index of refraction: f0 = ((n-1)/(n+1))^2.

    Args:
        cos_theta: Cosine of angle between view and half-vector. Shape (N,).
        f0: Reflectance at normal incidence. 0.04 for dielectrics, 0.7+ for metals.

    Returns:
        Fresnel reflectance, shape (N,).
    """
    return f0 + (1.0 - f0) * np.power(np.clip(1.0 - cos_theta, 0, 1), 5)


# ---------------------------------------------------------------------------
# GGX / Trowbridge-Reitz microfacet model
# ---------------------------------------------------------------------------

def ggx_ndf(n_dot_h: np.ndarray, alpha: float) -> np.ndarray:
    """GGX (Trowbridge-Reitz) Normal Distribution Function.

    Describes the statistical distribution of microfacet orientations.
    GGX has characteristic long tails that match real surfaces better
    than Blinn-Phong or Beckmann distributions.

    D(h) = alpha^2 / (pi * ((n.h)^2 * (alpha^2 - 1) + 1)^2)

    Args:
        n_dot_h: Cosine of angle between surface normal and half-vector. (N,)
        alpha: Roughness parameter (0 = mirror, 1 = fully rough).

    Returns:
        NDF values, shape (N,).
    """
    a2 = alpha * alpha
    denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 * INV_PI / (denom * denom + EPS)


def smith_ggx_g2(n_dot_l: np.ndarray, n_dot_v: np.ndarray, alpha: float) -> np.ndarray:
    """Smith masking-shadowing function for GGX microfacets.

    Accounts for the fact that microfacets can shadow each other
    (masking from the view direction, shadowing from the light direction).
    Uses the height-correlated Smith form for better accuracy.

    Args:
        n_dot_l: Cosine of angle between normal and light direction. (N,)
        n_dot_v: Cosine of angle between normal and view direction. (N,)
        alpha: Roughness parameter.

    Returns:
        Geometry term G2, shape (N,).
    """
    a2 = alpha * alpha
    ggx_v = n_dot_l * np.sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v + EPS)
    ggx_l = n_dot_v * np.sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l + EPS)
    return 2.0 * n_dot_l * n_dot_v / (ggx_v + ggx_l + EPS)


def cook_torrance_ggx(
    n_dot_l: np.ndarray,
    n_dot_v: np.ndarray,
    n_dot_h: np.ndarray,
    v_dot_h: np.ndarray,
    alpha: float,
    f0: float = 0.04,
) -> np.ndarray:
    """Cook-Torrance microfacet specular BRDF with GGX distribution.

    f_spec = D * G * F / (4 * |n.l| * |n.v|)

    This is the physically-based specular model used in modern rendering.
    It correctly handles energy conservation, Fresnel effect, and
    geometric self-shadowing of microfacets.

    Args:
        n_dot_l: cos(normal, light_dir). (N,)
        n_dot_v: cos(normal, view_dir). (N,)
        n_dot_h: cos(normal, half_vector). (N,)
        v_dot_h: cos(view_dir, half_vector). (N,)
        alpha: Surface roughness [0, 1].
        f0: Fresnel reflectance at normal incidence.

    Returns:
        Specular BRDF values, shape (N,).
    """
    D = ggx_ndf(n_dot_h, max(alpha, 0.01))
    G = smith_ggx_g2(n_dot_l, n_dot_v, max(alpha, 0.01))
    F = schlick_fresnel(v_dot_h, f0)
    return D * G * F / (4.0 * n_dot_l * n_dot_v + EPS)


# ---------------------------------------------------------------------------
# Oren-Nayar rough diffuse model
# ---------------------------------------------------------------------------

def oren_nayar(
    n_dot_l: np.ndarray,
    n_dot_v: np.ndarray,
    l_vec: np.ndarray,
    v_vec: np.ndarray,
    normal: np.ndarray,
    sigma: float,
    albedo: np.ndarray,
) -> np.ndarray:
    """Oren-Nayar diffuse BRDF for rough surfaces.

    Unlike Lambertian (constant rho/pi), Oren-Nayar accounts for
    microscale roughness causing view-dependent diffuse reflection.
    Rough surfaces appear brighter at grazing angles (retroreflection).

    Uses the simplified qualitative model (Oren & Nayar 1994).

    Args:
        n_dot_l: cos(normal, light). (N,)
        n_dot_v: cos(normal, view). (N,)
        l_vec: Light direction vectors. (N, 3) or (3,).
        v_vec: View direction vectors. (N, 3) or (3,).
        normal: Surface normals. (N, 3) or (3,).
        sigma: Surface roughness (standard deviation of microfacet slopes, radians).
        albedo: Surface albedo values. (N,).

    Returns:
        Diffuse BRDF values, shape (N,).
    """
    if sigma < 0.01:
        return albedo * INV_PI

    sigma2 = sigma * sigma
    A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33)
    B = 0.45 * sigma2 / (sigma2 + 0.09)

    theta_l = np.arccos(np.clip(n_dot_l, -1, 1))
    theta_v = np.arccos(np.clip(n_dot_v, -1, 1))
    alpha_on = np.maximum(theta_l, theta_v)
    beta_on = np.minimum(theta_l, theta_v)

    # Project light and view onto tangent plane, compute azimuth difference
    if l_vec.ndim == 1:
        l_proj = l_vec - n_dot_l * normal
        v_proj = v_vec - n_dot_v * normal
    else:
        l_proj = l_vec - n_dot_l[:, None] * normal
        v_proj = v_vec - n_dot_v[:, None] * normal

    l_len = np.linalg.norm(l_proj, axis=-1, keepdims=True) if l_proj.ndim > 1 else np.linalg.norm(l_proj)
    v_len = np.linalg.norm(v_proj, axis=-1, keepdims=True) if v_proj.ndim > 1 else np.linalg.norm(v_proj)

    if l_proj.ndim > 1:
        cos_phi = np.sum(l_proj * v_proj, axis=-1) / (
            np.squeeze(l_len) * np.squeeze(v_len) + EPS
        )
    else:
        cos_phi = np.dot(l_proj, v_proj) / (l_len * v_len + EPS)

    cos_phi = np.clip(cos_phi, -1, 1)

    return albedo * INV_PI * (A + B * np.maximum(0, cos_phi) * np.sin(alpha_on) * np.tan(beta_on))


# ---------------------------------------------------------------------------
# Combined BRDF evaluation (batch)
# ---------------------------------------------------------------------------

def evaluate_brdf_batch(
    n_dot_l: np.ndarray,
    n_dot_v: np.ndarray,
    normals: np.ndarray,
    light_dirs: np.ndarray,
    view_dirs: np.ndarray,
    albedo: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    f0: np.ndarray | float = 0.04,
) -> np.ndarray:
    """Evaluate combined diffuse + specular BRDF for N surface points.

    Uses GGX specular and optionally Oren-Nayar diffuse (falls back
    to Lambertian for roughness < 0.1). The metallic parameter
    interpolates between dielectric (metallic=0) and metallic (metallic=1)
    behavior, following the metallic workflow used in PBR rendering.

    For metals: no diffuse component, f0 = albedo (colored specular).
    For dielectrics: albedo goes to diffuse, f0 = 0.04.

    Args:
        n_dot_l: cos(n, light). (N,)
        n_dot_v: cos(n, view). (N,)
        normals: Surface normals. (N, 3).
        light_dirs: Directions from surface to light. (N, 3) or (3,).
        view_dirs: Directions from surface to camera. (N, 3) or (3,).
        albedo: Surface reflectance. (N,).
        roughness: Surface roughness [0, 1]. (N,) or scalar.
        metallic: Metallic factor [0, 1]. (N,) or scalar.
        f0: Base Fresnel reflectance. (N,) or scalar.

    Returns:
        Total BRDF value for each surface point, shape (N,).
    """
    N = n_dot_l.shape[0]

    # Ensure arrays
    roughness = np.broadcast_to(np.asarray(roughness, dtype=np.float64), (N,))
    metallic = np.broadcast_to(np.asarray(metallic, dtype=np.float64), (N,))

    # Half-vector
    if light_dirs.ndim == 1:
        half_vec = light_dirs + view_dirs
    else:
        if view_dirs.ndim == 1:
            half_vec = light_dirs + view_dirs[None, :]
        else:
            half_vec = light_dirs + view_dirs
    h_len = np.linalg.norm(half_vec, axis=-1, keepdims=True)
    half_vec = half_vec / (h_len + EPS)

    # Dot products for specular
    if normals.ndim == 1:
        n_dot_h = np.clip(np.dot(half_vec, normals), 0, 1)
    else:
        n_dot_h = np.clip(np.sum(half_vec * normals, axis=-1), 0, 1)

    if view_dirs.ndim == 1:
        v_dot_h = np.clip(np.sum(half_vec * view_dirs, axis=-1), 0, 1)
    else:
        v_dot_h = np.clip(np.sum(half_vec * view_dirs, axis=-1), 0, 1)

    # PBR metallic workflow: interpolate f0
    if isinstance(f0, (int, float)):
        f0_arr = np.full(N, f0)
    else:
        f0_arr = np.asarray(f0, dtype=np.float64)
    f0_effective = (1.0 - metallic) * f0_arr + metallic * albedo

    # Per-roughness-bin specular (use mean roughness for vectorized call)
    mean_alpha = float(np.mean(roughness))
    alpha = np.clip(mean_alpha, 0.01, 1.0)

    # GGX specular
    spec = cook_torrance_ggx(
        np.clip(n_dot_l, EPS, 1),
        np.clip(n_dot_v, EPS, 1),
        n_dot_h, v_dot_h,
        alpha=alpha,
        f0=float(np.mean(f0_effective)),
    )

    # Diffuse: Oren-Nayar for rough surfaces, Lambertian otherwise
    kS = schlick_fresnel(v_dot_h, float(np.mean(f0_effective)))
    kD = (1.0 - kS) * (1.0 - metallic)
    diff = kD * albedo * INV_PI

    # For high roughness, use Oren-Nayar adjustment
    high_rough = mean_alpha > 0.15
    if high_rough:
        on_factor = oren_nayar(
            n_dot_l, n_dot_v,
            light_dirs if light_dirs.ndim > 1 else np.broadcast_to(light_dirs, (N, 3)),
            view_dirs if view_dirs.ndim > 1 else np.broadcast_to(view_dirs, (N, 3)),
            normals if normals.ndim > 1 else np.broadcast_to(normals, (N, 3)),
            sigma=float(mean_alpha * 0.8),
            albedo=np.ones(N),
        )
        diff = kD * albedo * on_factor

    return diff + spec
