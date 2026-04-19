# User Guide — Dual Photography Lab

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip package manager

### Installation

```bash
cd d:/_Repos/_SCIENCE/FASL_Coding_DualFotography
pip install -e ".[dev]"
```

Or using the virtual environment:

```bash
.venv/Scripts/pip.exe install -e ".[dev]"
```

### Running the Application

```bash
python -m src.frontend.app
# or
.venv/Scripts/python.exe -m src.frontend.app
```

Open **http://127.0.0.1:8004** in your browser.

## Using the Web Interface

### Control Panel (Left Side)

| Control | Description |
|---------|-------------|
| **Scene Type** | Select the 3D scene geometry. "Box + Wall" is the most compelling demo. |
| **Resolution** | Image resolution (N×N pixels). Higher = more detail but slower. Start with 32×32. |
| **Surface Albedo** | Surface reflectance (0.1 = dark, 1.0 = white). Affects brightness. |
| **SVD Rank** | Truncation rank for the dual image. "Full" uses the complete transport matrix. Lower values show the effect of compression. |
| **Inter-reflections** | Reserved for future multi-bounce simulation. |
| **Projector/Camera X** | Horizontal position of each device. Larger separation = more parallax. |

### Results Panel (Right Side)

#### Primal vs Dual Images

- **Primal Image**: What the camera sees when the projector illuminates the scene uniformly. This is `c = T · 1`.
- **Dual Image**: What the projector "sees" when the camera position acts as a light source. This is `p_dual = T^T · 1`. The viewpoint has swapped — objects that were occluded from the camera may now be visible, and vice versa.

#### SVD Spectrum

The bar chart shows singular values in descending order. A steep drop-off means the transport matrix is effectively low-rank — a small number of components capture most of the light transport. The red line shows cumulative energy fraction.

#### Transport Matrix Analysis

- **Matrix size**: Dimensions of T (cam_pixels × proj_pixels)
- **Condition number**: Ratio of largest to smallest singular value. High values indicate that some transport paths carry far more energy than others.
- **Rank for 90%/99% energy**: How many singular values are needed to capture 90% or 99% of the total energy. This indicates how compressible the transport is.

#### Relighting

After running a simulation, you can relight the scene with different illumination patterns:

1. Select a pattern from the dropdown (e.g., "Left Half", "Center Spot")
2. Click "Relight"
3. The left image shows the illumination pattern; the right shows what the camera would see

This demonstrates that once T is known, you can synthesize camera images for arbitrary projector patterns without re-capturing.

## Recommended Experiments

### Experiment 1: Observe Viewpoint Swap

1. Set scene to "Box + Wall", resolution 32×32
2. Click "Run Simulation"
3. Compare primal vs dual: notice how the visible face of the box changes

### Experiment 2: Effect of Camera-Projector Separation

1. Set projector X to -0.5, camera X to 0.5
2. Run simulation — note the primal/dual images
3. Change projector X to -3.0, camera X to 3.0
4. Run again — the viewpoint difference should be much more dramatic

### Experiment 3: SVD Compression

1. Run a simulation at 32×32 with "Full" SVD rank
2. Note the dual image quality
3. Change SVD rank to 64, then 16, then 4
4. Observe how the dual image degrades — which features survive at low rank?

### Experiment 4: Relighting

1. Run a simulation
2. Try "Left Half" illumination — only the left side of the scene is lit
3. Try "Center Spot" — focused illumination on the center
4. Compare with "Uniform White" — the full-scene view

## Running Tests

```bash
pytest tests/ -v
```

The test suite includes:
- **test_core.py**: Pattern generation, transport matrix operations, SVD, save/load
- **test_simulation.py**: All scene types, non-negativity, off-diagonal structure, primal/dual differences
- **test_frontend.py**: All callbacks, helper functions, every scene type via the frontend

## Using the Code Programmatically

```python
from src.simulation.renderer import VirtualRenderer
from src.simulation.scene import SceneType

# Create renderer
renderer = VirtualRenderer(proj_shape=(32, 32), cam_shape=(32, 32))

# Run simulation
result = renderer.run_simulation(scene_type=SceneType.BOX_AND_WALL)

# Access results
print(result.primal_image.shape)   # (32, 32)
print(result.dual_image.shape)     # (32, 32)
print(result.analysis['effective_rank_90'])

# Save transport matrix for later use
result.transport.save("my_transport.npz")

# Relight with a custom pattern
import numpy as np
pattern = np.zeros((32, 32))
pattern[10:20, 10:20] = 1.0  # Square spotlight
relighted = result.transport.forward(pattern)
```
