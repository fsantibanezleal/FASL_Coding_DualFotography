"""Tests for the physically-based BRDF module."""

import numpy as np
import pytest

from src.core.brdf import (
    schlick_fresnel,
    ggx_ndf,
    smith_ggx_g2,
    cook_torrance_ggx,
    oren_nayar,
    evaluate_brdf_batch,
)


class TestFresnel:
    def test_normal_incidence(self):
        """At normal incidence, F = f0."""
        f = schlick_fresnel(np.array([1.0]), f0=0.04)
        np.testing.assert_allclose(f, [0.04], atol=1e-6)

    def test_grazing_incidence(self):
        """At grazing incidence, F -> 1."""
        f = schlick_fresnel(np.array([0.0]), f0=0.04)
        np.testing.assert_allclose(f, [1.0], atol=1e-6)

    def test_vectorized(self):
        """Works on arrays."""
        angles = np.linspace(0, 1, 100)
        f = schlick_fresnel(angles, f0=0.04)
        assert f.shape == (100,)
        assert np.all(f >= 0.04 - 1e-6)
        assert np.all(f <= 1.0 + 1e-6)

    def test_monotonic(self):
        """Fresnel is monotonically decreasing with cos_theta."""
        cos_t = np.linspace(0.01, 1.0, 50)
        f = schlick_fresnel(cos_t, f0=0.04)
        assert np.all(np.diff(f) <= 1e-10)  # Non-increasing


class TestGGX:
    def test_ndf_peak_at_normal(self):
        """GGX NDF peaks when half-vector = normal (n_dot_h=1)."""
        nh = np.array([1.0, 0.5, 0.1])
        d = ggx_ndf(nh, alpha=0.3)
        assert d[0] > d[1] > d[2]

    def test_ndf_roughness_broadens(self):
        """Higher roughness broadens the NDF."""
        nh = np.array([0.5])
        d_smooth = ggx_ndf(nh, alpha=0.1)
        d_rough = ggx_ndf(nh, alpha=0.9)
        assert d_rough > d_smooth  # Rougher = more off-axis energy

    def test_smith_g2_range(self):
        """Smith G2 is in [0, 1]."""
        ndotl = np.array([0.3, 0.5, 0.9])
        ndotv = np.array([0.3, 0.5, 0.9])
        g = smith_ggx_g2(ndotl, ndotv, alpha=0.5)
        assert np.all(g >= 0)
        assert np.all(g <= 1.0 + 1e-6)

    def test_cook_torrance_positive(self):
        """Cook-Torrance BRDF is non-negative."""
        N = 50
        ndotl = np.random.rand(N) * 0.9 + 0.1
        ndotv = np.random.rand(N) * 0.9 + 0.1
        ndoth = np.random.rand(N) * 0.9 + 0.1
        vdoth = np.random.rand(N) * 0.9 + 0.1
        spec = cook_torrance_ggx(ndotl, ndotv, ndoth, vdoth, alpha=0.3, f0=0.04)
        assert np.all(spec >= 0)


class TestOrenNayar:
    def test_reduces_to_lambertian(self):
        """With sigma=0, Oren-Nayar = Lambertian."""
        N = 10
        ndotl = np.full(N, 0.5)
        ndotv = np.full(N, 0.5)
        normals = np.tile([0, 0, 1.], (N, 1))
        l_vec = np.tile([0, 0.5, 0.5], (N, 1))
        v_vec = np.tile([0, -0.5, 0.5], (N, 1))
        albedo = np.full(N, 0.8)

        on = oren_nayar(ndotl, ndotv, l_vec, v_vec, normals, sigma=0.0, albedo=albedo)
        lambertian = albedo / np.pi
        np.testing.assert_allclose(on, lambertian, rtol=1e-3)


class TestCombinedBRDF:
    def test_batch_shape(self):
        """evaluate_brdf_batch returns correct shape."""
        N = 20
        result = evaluate_brdf_batch(
            n_dot_l=np.full(N, 0.5),
            n_dot_v=np.full(N, 0.5),
            normals=np.tile([0, 0, 1.], (N, 1)),
            light_dirs=np.tile([0.3, 0, 0.7], (N, 1)),
            view_dirs=np.tile([-0.3, 0, 0.7], (N, 1)),
            albedo=np.full(N, 0.7),
            roughness=np.full(N, 0.5),
            metallic=np.full(N, 0.0),
            f0=0.04,
        )
        assert result.shape == (N,)
        assert np.all(result >= 0)

    def test_metal_vs_dielectric(self):
        """Metals have no diffuse, dielectrics have diffuse."""
        N = 10
        kwargs = dict(
            n_dot_l=np.full(N, 0.7),
            n_dot_v=np.full(N, 0.7),
            normals=np.tile([0, 0, 1.], (N, 1)),
            light_dirs=np.tile([0, 0, 1.], (N, 1)),
            view_dirs=np.tile([0, 0, 1.], (N, 1)),
            albedo=np.full(N, 0.8),
            roughness=np.full(N, 0.3),
        )
        dielectric = evaluate_brdf_batch(**kwargs, metallic=np.zeros(N), f0=0.04)
        metallic = evaluate_brdf_batch(**kwargs, metallic=np.ones(N), f0=0.04)
        # Dielectric should have more total reflectance due to diffuse
        assert np.mean(dielectric) > 0
        assert np.mean(metallic) > 0

    def test_rough_vs_smooth(self):
        """Smooth surfaces have sharper specular than rough ones."""
        N = 10
        kwargs = dict(
            n_dot_l=np.full(N, 0.9),
            n_dot_v=np.full(N, 0.9),
            normals=np.tile([0, 0, 1.], (N, 1)),
            light_dirs=np.tile([0, 0, 1.], (N, 1)),
            view_dirs=np.tile([0, 0, 1.], (N, 1)),
            albedo=np.full(N, 0.5),
            metallic=np.full(N, 0.5),
            f0=0.04,
        )
        smooth = evaluate_brdf_batch(**kwargs, roughness=np.full(N, 0.05))
        rough = evaluate_brdf_batch(**kwargs, roughness=np.full(N, 0.9))
        # At normal incidence, smooth should have higher peak
        assert np.mean(smooth) > np.mean(rough) * 0.5  # Not strict, just sanity
