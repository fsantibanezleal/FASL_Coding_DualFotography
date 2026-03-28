"""Tests for spectral transport matrix module.

Validates wavelength-dependent transport matrix creation, forward/dual
computation, and API endpoint integration.
"""

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.core.spectral import SpectralTransportMatrix, simulate_rgb_transport
from src.api.main import app, state


# ======================================================================
# Unit tests for SpectralTransportMatrix
# ======================================================================

class TestSpectralTransportMatrix:
    """Tests for the SpectralTransportMatrix class."""

    @pytest.fixture
    def spectral(self):
        """Create a simple spectral transport with known matrices."""
        stm = SpectralTransportMatrix(resolution=4)
        rng = np.random.default_rng(42)
        stm.add_channel('red', rng.random((16, 16)))
        stm.add_channel('green', rng.random((16, 16)))
        stm.add_channel('blue', rng.random((16, 16)))
        return stm

    def test_add_channel(self):
        """Adding a channel stores a copy of the matrix."""
        stm = SpectralTransportMatrix(resolution=2)
        T = np.eye(4)
        stm.add_channel('test', T)
        assert 'test' in stm.channels
        np.testing.assert_array_equal(stm.channels['test'], T)
        # Verify it's a copy
        T[0, 0] = 999
        assert stm.channels['test'][0, 0] == 1.0

    def test_get_state(self, spectral):
        """get_state should return resolution, channel names, and shapes."""
        state = spectral.get_state()
        assert state['resolution'] == 4
        assert set(state['channels']) == {'red', 'green', 'blue'}
        for name in ['red', 'green', 'blue']:
            assert state['shapes'][name] == [16, 16]

    def test_compute_rgb_from_scene(self, spectral):
        """Forward spectral computation should produce per-channel images."""
        n = 16
        pattern = np.ones(n)
        reflectance = {
            'red': np.ones(n) * 0.8,
            'green': np.ones(n) * 0.5,
            'blue': np.ones(n) * 0.3,
        }
        result = spectral.compute_rgb_from_scene(reflectance, pattern)
        assert set(result.keys()) == {'red', 'green', 'blue'}
        for name, img in result.items():
            assert img.shape == (n,)
            assert np.all(np.isfinite(img))

    def test_compute_rgb_without_reflectance(self, spectral):
        """If a channel has no reflectance entry, use T directly."""
        n = 16
        pattern = np.ones(n)
        # Only provide red reflectance
        reflectance = {'red': np.ones(n) * 0.5}
        result = spectral.compute_rgb_from_scene(reflectance, pattern)
        # green and blue should still work (using T @ pattern directly)
        assert 'green' in result
        assert 'blue' in result

    def test_dual_image_spectral(self, spectral):
        """Dual image for a channel should use T^T."""
        n = 16
        illumination = np.ones(n)
        dual = spectral.dual_image_spectral('red', illumination)
        assert dual.shape == (n,)
        # Verify it equals T^T @ illumination
        expected = spectral.channels['red'].T @ illumination
        np.testing.assert_allclose(dual, expected)

    def test_dual_image_unknown_channel(self, spectral):
        """Requesting unknown channel should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown channel"):
            spectral.dual_image_spectral('infrared', np.ones(16))

    def test_different_channels_produce_different_results(self, spectral):
        """Different channels should give different dual images."""
        illumination = np.ones(16)
        dual_r = spectral.dual_image_spectral('red', illumination)
        dual_g = spectral.dual_image_spectral('green', illumination)
        assert not np.allclose(dual_r, dual_g)


class TestSimulateRGBTransport:
    """Tests for the simulate_rgb_transport convenience function."""

    def test_creates_three_channels(self):
        """Should create red, green, blue channels."""
        stm = simulate_rgb_transport(resolution=4, seed=42)
        assert set(stm.channels.keys()) == {'red', 'green', 'blue'}

    def test_resolution_stored(self):
        """Resolution should be stored correctly."""
        stm = simulate_rgb_transport(resolution=8, seed=42)
        assert stm.resolution == 8

    def test_matrix_shapes(self):
        """Transport matrices should be (n*n, n*n)."""
        res = 4
        stm = simulate_rgb_transport(resolution=res, seed=42)
        n = res * res
        for name, T in stm.channels.items():
            assert T.shape == (n, n), f"Channel {name} has wrong shape"

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        stm1 = simulate_rgb_transport(resolution=4, seed=123)
        stm2 = simulate_rgb_transport(resolution=4, seed=123)
        for name in stm1.channels:
            np.testing.assert_array_equal(stm1.channels[name], stm2.channels[name])

    def test_channels_differ(self):
        """R, G, B channels should not be identical."""
        stm = simulate_rgb_transport(resolution=4, seed=42)
        assert not np.allclose(stm.channels['red'], stm.channels['green'])
        assert not np.allclose(stm.channels['green'], stm.channels['blue'])


# ======================================================================
# API endpoint tests
# ======================================================================

@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest_asyncio.fixture
async def client(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_spectral_simulate_endpoint(client):
    """POST /api/spectral-simulate should create spectral transport."""
    response = await client.post(
        "/api/spectral-simulate",
        json={"resolution": 4, "seed": 42},
    )
    assert response.status_code == 200
    data = response.json()
    assert data['resolution'] == 4
    assert set(data['channels']) == {'red', 'green', 'blue'}
    assert 'sample_images' in data
    assert set(data['sample_images'].keys()) == {'red', 'green', 'blue'}


@pytest.mark.asyncio
async def test_spectral_relight_endpoint(client):
    """POST /api/spectral-relight should work after spectral-simulate."""
    # First simulate
    await client.post(
        "/api/spectral-simulate",
        json={"resolution": 4, "seed": 42},
    )
    # Then relight
    response = await client.post(
        "/api/spectral-relight",
        json={"channel": "red", "pattern_type": "uniform"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data['channel'] == 'red'
    assert data['resolution'] == 4
    assert 'dual_image' in data


@pytest.mark.asyncio
async def test_spectral_relight_without_simulate(client):
    """POST /api/spectral-relight before simulate should return 400."""
    state.spectral = None
    response = await client.post(
        "/api/spectral-relight",
        json={"channel": "red", "pattern_type": "uniform"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_spectral_relight_invalid_channel(client):
    """POST /api/spectral-relight with invalid channel should return 400."""
    await client.post(
        "/api/spectral-simulate",
        json={"resolution": 4, "seed": 42},
    )
    response = await client.post(
        "/api/spectral-relight",
        json={"channel": "infrared", "pattern_type": "uniform"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_spectral_relight_all_channels(client):
    """All RGB channels should produce valid dual images."""
    await client.post(
        "/api/spectral-simulate",
        json={"resolution": 4, "seed": 42},
    )
    for ch in ['red', 'green', 'blue']:
        response = await client.post(
            "/api/spectral-relight",
            json={"channel": ch, "pattern_type": "spot"},
        )
        assert response.status_code == 200, f"Channel '{ch}' failed"
        data = response.json()
        assert len(data['dual_image']) == 16  # 4*4
