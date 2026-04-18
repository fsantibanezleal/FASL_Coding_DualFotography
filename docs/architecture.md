# Architecture — Dual Photography Lab

System design, technology stack justification, component responsibilities, data flow, and deployment model for the Dual Photography Lab application.

---

## 1. High-Level Overview

Dual Photography Lab is a **three-tier Python application** that makes the light transport matrix **T** the central data object of a small interactive lab. The three tiers mirror the README's *Frontend / API / Engine* architecture diagram (`docs/svg/architecture.svg`):

| Tier | Responsibility | Key Modules |
|------|----------------|-------------|
| **Presentation (Dash UI)** | Interactive controls, figure rendering, REST façade for experiments | `src/frontend/app.py`, `src/api/main.py` |
| **Domain Engine** | Transport matrix math, SVD, patterns, BRDF, spectral, calibration | `src/core/*` |
| **Scene Simulation** | Virtual ray-cast renderer, scene primitives, pattern-to-image forward model | `src/simulation/scene.py`, `src/simulation/renderer.py` |
| **Physical Capture (optional)** | Webcam + screen-as-projector acquisition with ambient subtraction | `src/capture/*` |

See `docs/svg/architecture.svg` for the visual component diagram and `docs/svg/pipeline.svg` for the end-to-end processing pipeline.

---

## 2. Technology Stack — Why Each Piece

| Layer | Choice | Why this, not something else |
|-------|--------|-----------------------------|
| Language | **Python 3.10+** | Scientific ecosystem (NumPy, SciPy), Felipe's team-wide standard, cross-platform without license fees (the original 2014 prototype was MATLAB-only). |
| Numerical core | **NumPy + SciPy** | Vectorized ray-scene intersection, dense SVD via LAPACK, pseudoinverse solves. |
| Web UI | **Dash + dash-bootstrap-components** | Pure-Python reactive UI — no separate JS frontend to maintain. Matches the "one repo, one stack" rule across Felipe's science portfolio. |
| Visualization | **Plotly** | Interactive singular-value spectrum, matrix heatmaps, relighting previews — all rendered client-side. |
| Imaging | **OpenCV + Pillow + scikit-image** | OpenCV for webcam capture / YUY2 conversion, Pillow for PNG encoding to base64 for the UI, scikit-image for auxiliary filters. |
| REST API | **FastAPI-compatible surface via `src/api/main.py`** | Allows scripted experiments and headless CI runs without touching the UI. |
| Tests | **pytest** (165 collected) | Fast, parametrized, integrates with ruff linting. |
| Packaging | **pyproject.toml + setuptools** | Single source of truth for dependencies, version, console scripts. |
| Desktop build | **PyInstaller** (`build.spec`, `Build_PyInstaller.ps1`) | One-file Windows executable for demos without Python install. |
| Web deploy | **cPanel Passenger WSGI** (`passenger_wsgi.py`) | Matches Felipe's shared-host deployment target. |

The stack deliberately avoids a framework split (e.g., React + FastAPI). Dash keeps the UI, callbacks, and scientific code in the same Python process, which removes a whole class of serialization / state-sync bugs while the lab is still evolving.

---

## 3. Component Responsibilities

### 3.1 `src/core/` — The Scientific Engine

- **`transport.py`** — `TransportMatrix`: holds `T` (dense NumPy array), provides `forward(p)`, `dual(c)`, `relight(p_new)`, `svd(rank)`, `save / load`. This is the single authoritative object representing a captured light transport.
- **`patterns.py`** — Canonical, Hadamard, Bernoulli, and Gray-code pattern generators. Each returns an `(N, H, W)` stack.
- **`dual.py`** — `DualPhotographer` orchestrator: wires patterns → capture/simulate → `TransportMatrix` → dual image.
- **`brdf.py`** — Lambertian and Blinn-Phong BRDFs used by the simulator.
- **`spectral.py`** — RGB (3-channel) transport matrices and per-channel relighting.
- **`calibration.py`** — Gray-code projector-camera pixel correspondence and geometric calibration.

### 3.2 `src/simulation/` — Virtual Lab

- **`scene.py`** — `SceneObject`, `SyntheticScene`: pinhole camera/projector, ray-casting against planes/rectangles/spheres, shadow-ray occlusion, texture mapping.
- **`renderer.py`** — `VirtualRenderer`: end-to-end `run_simulation(scene_type, …)` returning a `SimulationResult` (primal, dual, T, analysis metadata).

### 3.3 `src/capture/` — Physical Acquisition

- **`camera.py`** — `CameraCapture` (OpenCV webcam wrapper, YUY2 → RGB).
- **`projector.py`** — `ScreenProjector` (secondary-monitor fullscreen pattern display).
- **`acquisition.py`** — `AcquisitionPipeline` synchronizing projection and capture with ambient subtraction.

### 3.4 `src/frontend/` — Dash Application

- **`app.py`** — Controls (`dbc.Select` for scene type, resolution, SVD rank, camera/projector X, pattern type), figures, callbacks. Serves on port **8004**.
- **`assets/`** — CSS.

### 3.5 `src/api/` — REST Surface

- **`main.py`** — 11 endpoints: `/api/health`, `/api/scenes`, `/api/simulate`, `/api/relight`, `/api/svd`, `/api/transport`, `/api/frequency`, `/api/spectral-simulate`, `/api/spectral-relight`, `/api/calibrate`, `/api/log`. Used by tests and external scripts.

---

## 4. Data Flow — One Simulation Round-Trip

1. **UI event** — user picks a scene and clicks *Run Simulation* in the Dash frontend.
2. **Callback** — `src/frontend/app.py` packages parameters and calls `VirtualRenderer.run_simulation(...)`.
3. **Scene build** — `scene.py` instantiates primitives (box, wall, sphere, …) for the selected `SceneType`.
4. **Ray cast** — for every (projector pixel, camera pixel) pair, a ray is cast and tested for occlusion; `T[i, j]` is filled with `albedo * cos_in * cos_out / r^2` when the path is unblocked.
5. **TransportMatrix** — the dense `T` is wrapped in a `TransportMatrix` object.
6. **SVD** — `transport.svd(rank)` runs `numpy.linalg.svd`, keeps the top-k singular values, exposes `effective_rank_90`, `effective_rank_99`, `condition_number`.
7. **Primal & dual** — `c = T · 1` (uniform illumination) and `p_dual = T^T · 1` are computed.
8. **Plotly figures** — arrays are converted to Plotly heatmaps and base64 PNGs, returned to the browser.
9. **Relighting** — subsequent UI events reuse the cached `T` and compute `c_new = T · p_new` without re-rendering the scene.

The same flow is exercised headlessly by the REST API and by the 165 pytest tests.

---

## 5. Testing Strategy

| Suite | Scope |
|-------|-------|
| `tests/test_core.py` | Pattern generators, `TransportMatrix` ops, SVD, save/load |
| `tests/test_simulation.py` | Every scene type, non-negativity of `T`, off-diagonal structure, primal ≠ dual |
| `tests/test_brdf.py` | Lambertian and Blinn-Phong BRDF properties |
| `tests/test_spectral.py` | 3-channel (RGB) spectral transport |
| `tests/test_calibration.py` | Gray-code correspondence decoding |
| `tests/test_analysis.py` | Condition number, energy-fraction ranks |
| `tests/test_api.py` | Every REST endpoint returns a valid payload |
| `tests/test_frontend.py` | Every Dash callback fires cleanly for every scene |

Run: `pytest tests/ -v`. Target: **165 passing** on every commit to `main`.

---

## 6. Deployment Models

### 6.1 Local Development (default)

```bash
pip install -e ".[dev]"
python -m src.frontend.app
# http://127.0.0.1:8004
```

### 6.2 Standalone Windows Executable

```powershell
./Build_PyInstaller.ps1
# -> dist/dual-photography.exe
```

Uses `build.spec` with explicit hidden imports for Dash and OpenCV. Produces a single-file Windows binary suitable for demos on machines without Python.

### 6.3 cPanel Shared Hosting (Passenger WSGI)

- Entry point: `passenger_wsgi.py` (root of repo).
- Passenger loads the Dash `app.server` (Flask under the hood).
- Port is determined by the hosting layer (Passenger maps an incoming request to the WSGI callable — the hard-coded 8004 only applies to local dev runs).

### 6.4 Headless / CI

```bash
pytest tests/ -v
```

No display server required — the simulation, API, and SVD layers have zero GUI dependencies.

---

## 7. Design Principles

1. **One canonical object (`TransportMatrix`)** — every downstream operation (dual, relight, SVD, save/load, spectral) acts on this single class so that the physics stays in one file.
2. **Simulation first, hardware optional** — the virtual renderer is the default path; `src/capture/` is fully separable and never imported by the frontend.
3. **Deterministic tests** — scenes and patterns are seedable; `pytest` runs under 10 s end-to-end.
4. **Manager-first README, developer-first `docs/`** — the README answers "why should I care?", this file answers "how is it built?".

---

## 8. See Also

- [`docs/01_technical_reference.md`](01_technical_reference.md) — deep equations, ray-casting math, SVD interpretation
- [`docs/02_user_guide.md`](02_user_guide.md) — running the app, recommended experiments
- [`docs/03_history.md`](03_history.md) — version-by-version changelog
- [`docs/04_references.md`](04_references.md) — academic bibliography
- [`docs/svg/architecture.svg`](svg/architecture.svg) — visual component diagram
- [`docs/svg/pipeline.svg`](svg/pipeline.svg) — processing pipeline flow
