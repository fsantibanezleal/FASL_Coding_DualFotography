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

from datetime import datetime
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.dual import DualPhotographer
from src.core.transport import TransportMatrix
from src.core.calibration import generate_gray_code_patterns, decode_gray_code, build_correspondence_map
from src.core.spectral import SpectralTransportMatrix, simulate_rgb_transport
from src.simulation.scene import SceneType, SyntheticScene


# -- Activity log --

_activity_log: List[str] = []
_MAX_LOG_ENTRIES = 200


def _log(message: str) -> None:
    """Append a timestamped message to the activity log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    _activity_log.append(f"[{timestamp}] {message}")
    # Keep bounded
    if len(_activity_log) > _MAX_LOG_ENTRIES:
        del _activity_log[: len(_activity_log) - _MAX_LOG_ENTRIES]


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


class CalibrateRequest(BaseModel):
    """Request body for /api/calibrate."""
    proj_width: int = Field(16, ge=2, le=256, description="Projector width in pixels")
    proj_height: int = Field(16, ge=2, le=256, description="Projector height in pixels")
    cam_width: int = Field(16, ge=2, le=256, description="Camera width in pixels")
    cam_height: int = Field(16, ge=2, le=256, description="Camera height in pixels")
    noise_level: float = Field(0.0, ge=0.0, le=1.0, description="Simulated capture noise (0-1)")


class SpectralSimulateRequest(BaseModel):
    """Request body for /api/spectral-simulate."""
    resolution: int = Field(8, ge=4, le=64, description="Grid resolution (NxN)")
    seed: int = Field(42, description="Random seed for reproducibility")


class SpectralRelightRequest(BaseModel):
    """Request body for /api/spectral-relight."""
    channel: str = Field("red", description="Wavelength channel: red, green, blue")
    pattern_type: str = Field(
        "uniform",
        description="Virtual illumination pattern: uniform, spot, horizontal_gradient",
    )


# -- Application state --

class AppState:
    """Holds the current simulation state between requests."""

    def __init__(self):
        self.photographer: Optional[DualPhotographer] = None
        self.scene: Optional[SyntheticScene] = None
        self.resolution: int = 32
        self.spectral: Optional[SpectralTransportMatrix] = None


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
    _log("Health check")
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/log")
async def get_log(limit: int = 50):
    """Return the last *limit* activity log entries (most recent last).

    Args:
        limit: Maximum number of entries to return (default 50).
    """
    entries = _activity_log[-limit:]
    return {"entries": entries, "total": len(_activity_log)}


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
    _log(f"Simulate: scene={req.scene_type}, res={req.resolution}, bounces={req.n_bounces}")

    try:
        scene_type = _scene_type_from_str(req.scene_type)
    except ValueError as e:
        _log(f"Simulate error: {e}")
        raise HTTPException(400, detail=str(e))

    try:
        res = req.resolution
        _log("Building scene and computing transport matrix...")
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

        _log(f"Simulate complete: T={T.shape}, norm={np.linalg.norm(T):.2f}")
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
        _log(f"Simulate failed: {e}")
        raise HTTPException(500, detail=str(e))


@app.post("/api/relight")
async def relight(req: RelightRequest):
    """Apply virtual relighting with a new projector pattern."""
    if state.photographer is None:
        _log("Relight failed: no simulation loaded")
        raise HTTPException(400, detail="No simulation loaded. Call /api/simulate first.")

    _log(f"Relight: pattern={req.pattern_type}")
    try:
        pattern = _generate_pattern(req.pattern_type, state.photographer.proj_shape)
        relighted = state.photographer.relight(pattern)
        _log(f"Relight complete: pattern={req.pattern_type}")
        return {
            "pattern_type": req.pattern_type,
            "relighted_image": relighted.tolist(),
            "pattern": pattern.tolist(),
        }
    except ValueError as e:
        _log(f"Relight error: {e}")
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        _log(f"Relight failed: {e}")
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


@app.post("/api/spectral-simulate")
async def spectral_simulate(req: SpectralSimulateRequest):
    """Simulate spectral (RGB) transport matrices.

    Creates wavelength-dependent transport matrices that capture how
    different wavelengths of light propagate through the scene.
    """
    _log(f"Spectral simulate: res={req.resolution}, seed={req.seed}")
    try:
        spectral = simulate_rgb_transport(resolution=req.resolution, seed=req.seed)
        state.spectral = spectral

        # Generate a sample forward image per channel with uniform illumination
        n = req.resolution * req.resolution
        uniform = np.ones(n)
        sample_images = {}
        for name, T in spectral.channels.items():
            img = T @ uniform
            sample_images[name] = np.clip(img, 0, None).tolist()

        _log(f"Spectral simulate complete: {len(spectral.channels)} channels")
        return {
            **spectral.get_state(),
            "sample_images": sample_images,
        }
    except Exception as e:
        _log(f"Spectral simulate failed: {e}")
        raise HTTPException(500, detail=str(e))


@app.post("/api/spectral-relight")
async def spectral_relight(req: SpectralRelightRequest):
    """Compute a spectral dual image (view from projector) for one channel.

    Requires a prior call to /api/spectral-simulate.
    """
    if state.spectral is None:
        _log("Spectral relight failed: no spectral simulation loaded")
        raise HTTPException(400, detail="No spectral simulation. Call /api/spectral-simulate first.")

    _log(f"Spectral relight: channel={req.channel}, pattern={req.pattern_type}")
    try:
        res = state.spectral.resolution
        n = res * res
        shape = (res, res)
        pattern = _generate_pattern(req.pattern_type, shape).flatten()

        dual_img = state.spectral.dual_image_spectral(req.channel, pattern)
        dual_img = np.clip(dual_img, 0, None)

        _log(f"Spectral relight complete: channel={req.channel}")
        return {
            "channel": req.channel,
            "pattern_type": req.pattern_type,
            "dual_image": dual_img.tolist(),
            "resolution": res,
        }
    except ValueError as e:
        _log(f"Spectral relight error: {e}")
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        _log(f"Spectral relight failed: {e}")
        raise HTTPException(500, detail=str(e))


@app.post("/api/calibrate")
async def calibrate(req: CalibrateRequest):
    """Simulate a projector-camera calibration using Gray code patterns.

    Generates Gray code patterns for the given projector resolution,
    simulates camera captures (with optional noise), decodes the
    patterns, and builds a correspondence map.
    """
    _log(f"Calibrate: proj={req.proj_width}x{req.proj_height}, "
         f"cam={req.cam_width}x{req.cam_height}, noise={req.noise_level}")

    try:
        # Generate patterns
        patterns = generate_gray_code_patterns(req.proj_width, req.proj_height)
        n_patterns = len(patterns)

        # Simulate captures: for simplicity, camera sees a 1:1 mapping
        # scaled to the camera resolution (identity correspondence)
        captures = []
        for p in patterns:
            # Resize pattern from projector resolution to camera resolution
            from numpy import ndarray
            # Simple nearest-neighbor resize
            y_idx = np.linspace(0, p.shape[0] - 1, req.cam_height).astype(int)
            x_idx = np.linspace(0, p.shape[1] - 1, req.cam_width).astype(int)
            resized = p[np.ix_(y_idx, x_idx)].astype(np.float64)

            # Add noise
            if req.noise_level > 0:
                noise = np.random.default_rng(42).normal(
                    0, req.noise_level * 255, resized.shape
                )
                resized = np.clip(resized + noise, 0, 255)

            captures.append(resized)

        captures_arr = np.array(captures)

        # Decode
        proj_x, proj_y = decode_gray_code(captures_arr, threshold=128.0)

        # Build correspondence map
        corr_map = build_correspondence_map(
            proj_x, proj_y, req.proj_width, req.proj_height
        )

        # Compute coverage (fraction of projector pixels with valid correspondence)
        valid = np.all(corr_map >= 0, axis=2)
        coverage = float(valid.sum()) / (req.proj_width * req.proj_height)

        _log(f"Calibrate complete: {n_patterns} patterns, coverage={coverage:.2%}")
        return {
            "n_patterns": n_patterns,
            "proj_shape": [req.proj_height, req.proj_width],
            "cam_shape": [req.cam_height, req.cam_width],
            "coverage": coverage,
            "proj_x_sample": proj_x[:4, :4].tolist(),
            "proj_y_sample": proj_y[:4, :4].tolist(),
        }
    except Exception as e:
        _log(f"Calibrate failed: {e}")
        raise HTTPException(500, detail=str(e))
