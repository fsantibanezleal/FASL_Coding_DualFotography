# Dual Photography Lab

Interactive application for **Dual Photography** — a computational imaging technique that reconstructs how a scene looks from a projector's viewpoint by exploiting **Helmholtz reciprocity** and the **light transport matrix**.

Based on: *Sen et al., "Dual Photography", ACM SIGGRAPH 2005*.

## What is Dual Photography?

When a projector illuminates a scene and a camera captures the result, the relationship is linear:

```
camera_image = T @ projector_pattern
```

where **T** is the *light transport matrix*. Each entry `T[i,j]` encodes how much light from projector pixel `j` reaches camera pixel `i` through all paths (direct, reflected, scattered).

By **Helmholtz reciprocity**, light paths are physically reversible. Transposing T swaps the roles of projector and camera:

```
dual_image = T^T @ virtual_illumination
```

This produces the scene as seen **from the projector's position**, illuminated from the camera — without any additional hardware or captures.

## Features

- **Synthetic Simulation**: Generate transport matrices for configurable 3D scenes (flat wall, corner, V-groove, sphere, checkerboard) without hardware
- **SVD Analysis**: Visualize singular value spectrum, effective rank, and condition number
- **Dual Image Generation**: Compute and display the dual photograph via T-transpose
- **Interactive Relighting**: Apply different illumination patterns and see the relighted scene in real-time
- **Physical Capture** (optional): Acquire transport matrices using a webcam and screen-as-projector
- **Inter-reflections**: Simulate multi-bounce light transport via iterative radiosity
- **Multiple Pattern Types**: Canonical, Hadamard, Bernoulli (compressed sensing), Gray code

## Quick Start

### 1. Install Dependencies

```bash
cd d:/_Repos/_SCIENCE/FASL_Coding_DualFotography
pip install -e ".[dev]"
```

### 2. Run the Web Application

```bash
python -m src.frontend.app
```

Open **http://127.0.0.1:8050** in your browser.

### 3. Run Tests

```bash
pytest tests/ -v
```

## Architecture

```
src/
├── core/              # Core dual photography engine
│   ├── patterns.py    # Illumination pattern generators (Hadamard, Bernoulli, etc.)
│   ├── transport.py   # Transport matrix: computation, SVD, forward/dual operations
│   └── dual.py        # High-level DualPhotographer orchestrator
├── simulation/        # Synthetic scene simulation
│   ├── scene.py       # 3D scene generator (geometry, normals, albedo)
│   └── renderer.py    # Virtual renderer pipeline
├── capture/           # Physical acquisition (optional hardware)
│   ├── camera.py      # Webcam capture via OpenCV
│   ├── projector.py   # Screen-based pattern projection
│   └── acquisition.py # Synchronized capture pipeline
└── frontend/          # Dash web application
    ├── app.py         # Main dashboard with interactive controls
    └── assets/        # CSS styles
```

## Mathematical Background

### Transport Matrix Acquisition

The transport matrix T can be measured by projecting known patterns and capturing the camera response:

| Method | Patterns Needed | Quality | Speed |
|--------|----------------|---------|-------|
| Canonical (one-at-a-time) | N = p*q | Exact | Slow |
| Hadamard (multiplexed) | N = p*q | Optimal SNR | Medium |
| Bernoulli (compressed sensing) | N << p*q | Good | Fast |

### SVD and Low-Rank Approximation

The transport matrix admits a truncated SVD:

```
T ≈ U_k @ diag(S_k) @ V_k^T
```

where keeping only the top-k singular values acts as denoising and compression. The application visualizes how reconstruction quality degrades with rank truncation.

### Inter-reflections

For scenes with concavities (corners, grooves), light bounces between surfaces. This is modeled via iterative radiosity:

```
T_total = sum_{b=0}^{B} (rho * F)^b @ T_direct
```

where F is the form factor matrix and rho is the surface albedo.

## References

1. Sen, P., Chen, B., Garg, G., Marschner, S., Horowitz, M., Levoy, M., & Lensch, H. P. A. (2005). **Dual Photography**. *ACM SIGGRAPH / Transactions on Graphics, 24*(3), 745-755.
2. Sen, P., & Darabi, S. (2009). **Compressive Dual Photography**. *Computer Graphics Forum (Eurographics), 28*(2), 609-618.
3. Hua, B. S., Sato, I., & Low, K. L. (2013). **Direct and Progressive Reconstruction of Dual Photography Images**. *IEEE ICIP*.

## License

MIT
