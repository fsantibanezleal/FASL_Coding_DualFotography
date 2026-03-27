"""Tests for the FastAPI backend endpoints.

Validates health, scene listing, simulation, relighting, and SVD endpoints.
"""

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.api.main import app, state


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest_asyncio.fixture
async def client(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """GET /api/health should return status ok."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"


@pytest.mark.asyncio
async def test_list_scenes(client):
    """GET /api/scenes should return available scene types."""
    response = await client.get("/api/scenes")
    assert response.status_code == 200
    data = response.json()
    assert "scenes" in data
    assert "cornell_box" in data["scenes"]
    assert len(data["scenes"]) >= 5


@pytest.mark.asyncio
async def test_simulate_creates_transport(client):
    """POST /api/simulate should compute transport matrix and return images."""
    response = await client.post(
        "/api/simulate",
        json={"scene_type": "flat_textured", "resolution": 8, "n_bounces": 0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["resolution"] == 8
    assert data["transport_shape"] == [64, 64]
    assert "camera_image" in data
    assert "dual_image" in data
    assert data["transport_norm"] > 0


@pytest.mark.asyncio
async def test_simulate_invalid_scene(client):
    """POST /api/simulate with invalid scene should return 400."""
    response = await client.post(
        "/api/simulate",
        json={"scene_type": "nonexistent_scene", "resolution": 8},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_relight_after_simulate(client):
    """POST /api/relight should work after simulation."""
    # First simulate
    await client.post(
        "/api/simulate",
        json={"scene_type": "flat_textured", "resolution": 8, "n_bounces": 0},
    )
    # Then relight
    response = await client.post(
        "/api/relight",
        json={"pattern_type": "horizontal_gradient"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["pattern_type"] == "horizontal_gradient"
    assert "relighted_image" in data


@pytest.mark.asyncio
async def test_relight_without_simulate(client):
    """POST /api/relight before simulate should return 400."""
    # Reset state
    state.photographer = None
    response = await client.post(
        "/api/relight",
        json={"pattern_type": "spot"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_svd_analysis(client):
    """GET /api/svd should return singular value decomposition."""
    # First simulate
    await client.post(
        "/api/simulate",
        json={"scene_type": "flat_textured", "resolution": 8, "n_bounces": 0},
    )
    response = await client.get("/api/svd", params={"rank": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["rank"] == 5
    assert len(data["singular_values"]) == 5
    assert data["effective_rank_90"] >= 1


@pytest.mark.asyncio
async def test_relight_all_patterns(client):
    """All built-in pattern types should produce valid relighting results."""
    await client.post(
        "/api/simulate",
        json={"scene_type": "flat_textured", "resolution": 8, "n_bounces": 0},
    )
    for pattern in ["horizontal_gradient", "vertical_gradient", "spot", "uniform", "checkerboard"]:
        response = await client.post(
            "/api/relight",
            json={"pattern_type": pattern},
        )
        assert response.status_code == 200, f"Pattern '{pattern}' failed"
