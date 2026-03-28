"""FastAPI backend for Dual Photography.

Provides REST API endpoints for:
    - Scene simulation and transport matrix computation
    - Dual image generation via Helmholtz reciprocity (T^T)
    - Virtual relighting with arbitrary projector patterns
    - SVD analysis of the transport matrix
    - Scene listing and health checks

The API wraps the core DualPhotographer pipeline, making it accessible
to web frontends and external tools without requiring direct Python imports.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.dual import DualPhotographer
from src.core.transport import TransportMatrix
from src.simulation.scene import SceneType, SyntheticScene


# -- Pydantic request/response models --

class SimulateRequest(BaseModel):
    """Request body for /api/simulate."""
    scene_type: str = Field("cornell_box", description="Scene type identifier")
    resolution: int = Field(32, ge=4, le=128, description="Grid resolution (NxN)")
    n_bounces: int = Field(0, ge=0, le=3, description="Number of indirect light bounces")
    compression_ratio: Optional[float] = Field(
        None, ge=0.05, le=1.0,
        description=(
            "If set, use compressed sensing with M = ratio * N random patterns "
            "instead of full canonical acquisition. Reduces acquisition time at "
            "the cost of reconstruction accuracy (Peers et al., 2009)."
        ),
    )


class RelightRequest(BaseModel):
    """Request body for /api/relight."""
    pattern_type: str = Field(
        "horizontal_gradient",
        description="Pattern name: horizontal_gradient, vertical_gradient, spot, uniform, checkerboard",
    )


class SvdRequest(BaseModel):
    """Request body for /api/svd."""
    rank: int = Field(10, ge=1, le=512, description="Number of singular values to retain")


# -- Application state --

class AppState:
    """Holds the current simulation state between requests."""

    def __init__(self):
        self.photographer: Optional[DualPhotographer] = None
        self.scene: Optional[SyntheticScene] = None
        self.resolution: int = 32


state = AppState()


# -- FastAPI app --

app = FastAPI(
    title="Dual Photography API",
    description=(
        "REST API for simulating light transport, generating dual images "
        "via Helmholtz reciprocity, and performing virtual relighting."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Helper functions --

def _scene_type_from_str(name: str) -> SceneType:
    """Convert a string scene name to SceneType enum."""
    name_lower = name.lower().replace(" ", "_")
    for st in SceneType:
        if st.value == name_lower:
            return st
    raise ValueError(f"Unknown scene type: {name}. Available: {[s.value for s in SceneType]}")


def _generate_pattern(pattern_type: str, shape: tuple[int, int]) -> np.ndarray:
    """Generate a named relighting pattern."""
    h, w = shape
    if pattern_type == "horizontal_gradient":
        return np.tile(np.linspace(0, 1, w), (h, 1))
    elif pattern_type == "vertical_gradient":
        return np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    elif pattern_type == "spot":
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return np.clip(1.0 - r / max(cy, cx), 0, 1)
    elif pattern_type == "uniform":
        return np.ones((h, w))
    elif pattern_type == "checkerboard":
        pattern = np.zeros((h, w))
        pattern[::2, ::2] = 1
        pattern[1::2, 1::2] = 1
        return pattern
    else:
        raise ValueError(
            f"Unknown pattern: {pattern_type}. "
            "Available: horizontal_gradient, vertical_gradient, spot, uniform, checkerboard"
        )


# -- Endpoints --

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/scenes")
async def list_scenes():
    """List all available scene types."""
    return {"scenes": [st.value for st in SceneType]}


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    """Run a full dual photography simulation.

    Builds a synthetic scene, computes the light transport matrix T,
    and generates both camera and dual images.
    """
    try:
        scene_type = _scene_type_from_str(req.scene_type)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    try:
        res = req.resolution
        scene = SyntheticScene(
            scene_type=scene_type,
            proj_shape=(res, res),
            cam_shape=(res, res),
            n_bounces=req.n_bounces,
        )
        T = scene.compute_transport_matrix()

        photographer = DualPhotographer(
            proj_shape=(res, res),
            cam_shape=(res, res),
        )

        if req.compression_ratio is not None:
            n_proj = res * res
            n_patterns = max(1, int(req.compression_ratio * n_proj))
            tm = TransportMatrix.from_compressed_sensing(
                scene_transport=T,
                n_patterns=n_patterns,
                proj_shape=(res, res),
                cam_shape=(res, res),
            )
            photographer.transport = tm
        else:
            photographer.compute_transport_from_simulation(T)

        # Store in state for subsequent relight/svd calls
        state.photographer = photographer
        state.scene = scene
        state.resolution = res

        # Generate default images
        camera_image = photographer.transport.forward(np.ones((res, res)))
        dual_image = photographer.dual_image()

        return {
            "scene_type": scene_type.value,
            "resolution": res,
            "transport_shape": list(T.shape),
            "camera_image": camera_image.tolist(),
            "dual_image": dual_image.tolist(),
            "transport_norm": float(np.linalg.norm(T)),
            "transport_sparsity": photographer.transport.sparsity(),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/relight")
async def relight(req: RelightRequest):
    """Apply virtual relighting with a new projector pattern."""
    if state.photographer is None:
        raise HTTPException(400, detail="No simulation loaded. Call /api/simulate first.")

    try:
        pattern = _generate_pattern(req.pattern_type, state.photographer.proj_shape)
        relighted = state.photographer.relight(pattern)
        return {
            "pattern_type": req.pattern_type,
            "relighted_image": relighted.tolist(),
            "pattern": pattern.tolist(),
        }
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/api/svd")
async def svd_analysis(rank: int = 10):
    """Get SVD decomposition analysis of the transport matrix."""
    if state.photographer is None:
        raise HTTPException(400, detail="No simulation loaded. Call /api/simulate first.")

    try:
        analysis = state.photographer.analyze_transport(rank=rank)
        return {
            "rank": rank,
            "singular_values": analysis["singular_values"][:rank].tolist(),
            "condition_number": float(analysis["condition_number"]),
            "energy_cumulative": analysis["energy_cumulative"][:rank].tolist(),
            "effective_rank_90": int(analysis["effective_rank_90"]),
            "effective_rank_99": int(analysis["effective_rank_99"]),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/api/transport")
async def get_transport(mode: str = "svd", rank: int = 10):
    """Get transport matrix in compact or full form.

    Args:
        mode: 'svd' returns truncated factors (U, S, Vt). 'full' returns dense matrix.
        rank: Number of singular values for SVD mode.
    """
    if state.photographer is None:
        raise HTTPException(400, detail="No simulation loaded. Call /api/simulate first.")

    try:
        T = state.photographer.transport.T

        if mode == 'svd':
            U, s, Vt = np.linalg.svd(T, full_matrices=False)
            k = min(rank, len(s))
            return {
                "mode": "svd",
                "rank": k,
                "U": U[:, :k].tolist(),
                "S": s[:k].tolist(),
                "Vt": Vt[:k, :].tolist(),
                "shape": list(T.shape),
            }
        else:
            return {
                "mode": "full",
                "matrix": T.tolist(),
                "shape": list(T.shape),
            }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/api/frequency")
async def frequency_analysis():
    """Get 2D Fourier frequency analysis of the transport matrix."""
    if state.photographer is None:
        raise HTTPException(400, detail="No simulation loaded. Call /api/simulate first.")

    try:
        freq = state.photographer.transport.frequency_analysis()
        return {
            "dc_fraction": freq["dc_fraction"],
            "high_freq_fraction": freq["high_freq_fraction"],
            "radial_profile": freq["radial_profile"].tolist(),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))
