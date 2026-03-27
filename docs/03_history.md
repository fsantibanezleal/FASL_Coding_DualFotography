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

## Key Mathematical Equations

The following equations underpin every module in the codebase:

### Light Transport Matrix

The transport matrix **T** encodes how each projector pixel illuminates each camera pixel:

```
T[j, i] = L_j(p_i)
```

where `p_i` is the i-th projector pixel and `L_j(p_i)` is the radiance arriving at camera pixel j when only projector pixel i is active.

### Forward and Dual Imaging

```
Primal (forward):   c = T * p           (camera image from projector pattern)
Dual (transpose):   I_dual = T^T * I    (projector-viewpoint image via Helmholtz reciprocity)
Relight:            c_new = T * p_new   (new illumination without re-capture)
```

### Helmholtz Reciprocity

Light transport between two points is symmetric — swapping source and detector yields the same transfer coefficient:

```
T_forward = T_backward^T
```

This is the physical principle that makes dual photography possible: transposing **T** is equivalent to swapping the roles of projector and camera.

### SVD Decomposition

Singular Value Decomposition provides rank-k approximation and spectral analysis:

```
T = U * Sigma * V^T
T_k = U_k * Sigma_k * V_k^T    (rank-k approximation, k << min(m,n))
```

The singular value spectrum reveals the effective dimensionality of the light transport and determines how many degrees of freedom the scene-illumination interaction has.

---

## Changelog

### v2.0.0 (2026-03-26) — Vectorized Engine, Specular BRDF, Multi-Bounce

- **Vectorized ray-casting**: All ray-scene intersections use NumPy batch operations. 32x32 scenes compute in ~0.4s (was ~4s with scalar loops).
- **Material system**: New `Material` class with diffuse, specular, shininess, and mirror properties. Replaces raw albedo floats.
- **Blinn-Phong BRDF**: Specular highlights that are view-dependent — dramatically different between primal and dual images. Mirror Room scene achieves primal-dual correlation of only 0.15.
- **Multi-bounce indirect illumination**: Iterative radiosity adds light bouncing between surfaces. Cornell Box with 3 bounces has 52% more total light energy than direct-only.
- **4 new complex scenes**: Cornell Box (enclosed room with boxes), Gallery (museum with paintings, pedestal, sphere, columns), Staircase (stepped geometry with strong parallax), Mirror Room (high-specular walls and shiny sphere).
- **New textures**: gradient, concentric circles, cross/plus sign patterns.
- **Frontend**: Light bounce control (0-3 bounces), 10 scene types in dropdown.
- **98 tests** covering materials, vectorized intersection, specular BRDF, multi-bounce transport, all scenes.

### v1.1.0 (2026-03-26) — Ray-Casting Engine

- **BREAKING**: Scene types changed from flat analytical to 3D ray-cast
- Rewrote `SyntheticScene` with proper perspective ray-casting
- Added `SceneObject` class with plane/sphere intersection and texture mapping
- Added occlusion (shadow ray) testing
- Built-in pixel-art textures: letters (F, D, P, L, A, R), checkerboard, arrow
- Default resolution increased to 32×32
- Default camera-projector separation increased to 3.0 units
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
