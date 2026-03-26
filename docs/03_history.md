# Development History — Dual Photography Lab

## Origins

The original codebase was a MATLAB implementation for dual photography acquisition, created around 2014 by Felipe Santibañez-Leal. It used MATLAB's Image Acquisition Toolbox with a webcam and secondary monitor to project patterns and capture responses. The acquisition loop generated random binary (Bernoulli) patterns, displayed them fullscreen via IrfanView, and captured synchronized camera images.

The repository also contained academic reference papers on compressive dual photography, inverse light transport, and radiosity.

## Rebuild (2026-03-26)

### Phase 1: Architecture and Core Engine

The project was rebuilt from scratch in Python, targeting a modern web-based architecture:

- **Decision**: Moved all legacy MATLAB code to `legacy/` folder, preserving it for reference
- **Stack chosen**: Python 3.12 + Dash (web framework) + NumPy/SciPy (math) + OpenCV (capture)
- **Core engine implemented**:
  - `TransportMatrix` class: computation from pattern-capture pairs, SVD decomposition, forward/dual/relight operations, save/load
  - `DualPhotographer` orchestrator: end-to-end pipeline management
  - Pattern generators: Canonical, Hadamard, Bernoulli, Gray code
- **Physical capture module**: `CameraCapture`, `ScreenProjector`, `AcquisitionPipeline` for real hardware workflows

### Phase 2: Initial Frontend

- Built Dash web application with interactive controls
- Initially used `dcc.Dropdown` and `dcc.Slider` components
- **Issue discovered**: Dash 4.x + dash-bootstrap-components 2.0 had CSS/z-index conflicts preventing dropdown interaction
- **Fix**: Replaced all `dcc.Dropdown` with `dbc.Select` (native HTML select elements), replaced sliders with dropdowns, ensured all values passed as strings

### Phase 3: Scene Simulation Rewrite

The initial simulation produced boring, nearly-identical primal and dual images. Root cause analysis identified four problems:

1. **Transport matrix was nearly diagonal**: projector pixel j mapped primarily to camera pixel j because both used the same sampling grid on a flat surface
2. **No perspective projection**: both devices sampled an identical grid on z=0, no ray-casting
3. **No occlusion model**: every surface point was assumed visible to both devices
4. **Insufficient camera-projector separation**: default positions were too close together

**Solution**: Complete rewrite of `scene.py` with:
- Proper pinhole camera model with perspective projection (field-of-view, look-at targeting)
- Ray-casting intersection against scene primitives (planes, bounded rectangles, spheres)
- Shadow ray occlusion testing for every light path
- 6 new scene types with 3D depth variation (Box+Wall, Sphere on Plane, Corner Room, Two Angled Planes, Cylinder, Flat Textured)
- Built-in test textures (letter bitmaps, checkerboard, arrow) rendered as pixel art
- New tests validating that T is far from diagonal and that primal/dual images are measurably different

---

## Changelog

### v1.1.0 (2026-03-26) — Ray-Casting Engine

- **BREAKING**: Scene types changed. Old types (`flat_wall`, `corner`, `v_groove`, `sphere`, `checkerboard`, `textured_wall`) replaced with new 3D scenes
- Rewrote `SyntheticScene` with proper perspective ray-casting
- Added `SceneObject` class with plane/sphere intersection and texture mapping
- Added occlusion (shadow ray) testing
- Built-in pixel-art textures: letters (F, D, P, L, A, R), checkerboard, arrow
- Default resolution increased to 32×32
- Default camera-projector separation increased to 3.0 units
- Added tests: `test_transport_not_diagonal`, `test_primal_dual_differ`
- 70 tests total, all passing

### v1.0.0 (2026-03-26) — Initial Release

- Complete Python rewrite from legacy MATLAB code
- Core engine: TransportMatrix, DualPhotographer, pattern generators
- Simulation module with analytical transport matrix computation
- Physical capture module: CameraCapture, ScreenProjector, AcquisitionPipeline
- Dash web frontend with interactive controls
- 64 tests, all passing
- Full documentation and README

### v0.x (2014-2024) — Legacy MATLAB

- MATLAB-based acquisition loop with webcam + IrfanView
- Random binary pattern generation (Bernoulli)
- WindowAPI MEX wrapper for fullscreen display
- YUY2 to RGB color conversion
- Academic bibliography collection
