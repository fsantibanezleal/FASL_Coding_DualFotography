# Technical Reference — Dual Photography Lab

## 1. Problem Statement

In computational photography, we often want to see a scene from a viewpoint where no camera exists. Dual photography solves this by exploiting a fundamental symmetry of light: **if you know how light travels from A to B, you also know how it travels from B to A**.

Given a projector and a camera observing a scene, we measure the complete light transport relationship between them. By mathematically transposing this relationship, we synthesize what the scene looks like from the projector's position — as if the projector were a camera and the camera were a light source.

## 2. Mathematical Foundation

### 2.1 The Transport Matrix

Light transport through a fixed scene is linear. If we vectorize the projector pattern into a column vector **p** of size (pq × 1), where p and q are the projector's vertical and horizontal pixel counts, and the camera image into **c** of size (mn × 1), then:

```
c = T · p
```

The matrix **T** has dimensions (mn × pq). Each entry T[i,j] represents the total fraction of light energy emitted by projector pixel j that ultimately arrives at camera pixel i. This includes all possible transport paths through the scene: direct illumination, diffuse reflection, specular bounces, subsurface scattering, and any other optical phenomenon.

### 2.2 Helmholtz Reciprocity

The Bidirectional Reflectance Distribution Function (BRDF) of most natural materials satisfies a symmetry property:

```
f_r(ω_in, ω_out) = f_r(ω_out, ω_in)
```

This means swapping the incident and exitant light directions yields the same reflectance value. The physical basis is the time-reversal symmetry of electromagnetic wave propagation: Maxwell's equations are symmetric under time reversal in non-absorbing media.

This symmetry propagates to the transport matrix. If we define a "dual" configuration where the camera emits light and the projector acts as a sensor:

```
p_dual = T^T · c_dual
```

The transpose of T encodes the reverse light transport. The resulting image **p_dual** shows the scene as it would appear from the projector's position, illuminated from the camera's position.

### 2.3 SVD and Low-Rank Approximation

The transport matrix can be decomposed via Singular Value Decomposition:

```
T = U · Σ · V^T
```

where U contains the left singular vectors (camera-space basis), Σ is a diagonal matrix of singular values in descending order, and V^T contains the right singular vectors (projector-space basis).

Keeping only the top k singular values gives the best possible rank-k approximation (by the Eckart-Young-Mirsky theorem):

```
T_k = U_k · Σ_k · V_k^T
```

This serves two purposes:
1. **Compression**: storing U_k, Σ_k, V_k requires far less memory than the full T
2. **Denoising**: small singular values typically correspond to noise; truncating them improves signal quality

### 2.4 Acquisition Methods

The transport matrix can be measured by projecting known patterns and capturing the camera response:

| Method | Patterns Required | Strengths | Weaknesses |
|--------|------------------|-----------|------------|
| **Canonical** (one pixel at a time) | N = pq | Exact T, simple | Very slow |
| **Hadamard** (multiplexed) | N = pq | Best SNR per measurement | Requires power-of-2 size |
| **Bernoulli** (random binary) | N << pq | Fast acquisition, compressed sensing | Requires sparse recovery solver |
| **Gray code** (structured light) | N = 2·ceil(log₂(max(p,q))) | Establishes pixel correspondence | Assumes one-to-one mapping |

For the pseudoinverse method, given N pattern-capture pairs:
```
P = [p_1 | p_2 | ... | p_N]    (pq × N)
C = [c_1 | c_2 | ... | c_N]    (mn × N)
T = C · pinv(P)
```

## 3. Ray-Casting Transport Matrix Computation

Our simulation computes T analytically by ray-casting, avoiding the need for physical hardware.

### 3.1 Perspective Projection Model

Both projector and camera are modeled as pinhole devices. Each pixel (row, col) in a device with resolution (H, W) and field-of-view θ maps to a ray direction:

```
u = 2·(col + 0.5)/W - 1        (normalized horizontal coordinate)
v = 1 - 2·(row + 0.5)/H        (normalized vertical, flipped)

ray_direction = forward + u·right·tan(θ/2)·aspect + v·up·tan(θ/2)
```

where `forward`, `right`, `up` form the device's orthonormal basis, and `aspect = W/H`.

### 3.2 Scene Intersection

Rays are tested against all scene primitives (planes, bounded rectangles, spheres). For each ray, we find the nearest intersection and extract:
- Hit position
- Surface normal
- Albedo (scalar or texture-mapped)

### 3.3 Transport Entry Computation

For projector pixel j and camera pixel i:

1. Cast a ray from the projector through pixel j → find hit point P_j on the scene
2. Check what camera pixel i sees → find hit point C_i on the scene
3. If C_i ≈ P_j (within one projector pixel footprint):
   - Compute cos_in = max(0, normal · direction_to_projector)
   - Compute cos_out = max(0, normal · direction_to_camera)
   - Test visibility: cast a shadow ray from P_j to camera, check for occlusion
   - T[i,j] = albedo · cos_in · cos_out / distance²

### 3.4 Occlusion Testing

For each (projector→surface→camera) light path, a shadow ray is cast from the surface point toward the camera. If this ray hits another object before reaching the camera, the path is blocked and T[i,j] = 0. This is what creates the visual differences between primal and dual images in scenes with 3D geometry.

## 4. Scene Types

| Scene | Description | Why It's Interesting |
|-------|-------------|---------------------|
| **Box + Wall** | A cube in front of a textured wall | Occlusion: each viewpoint sees a different face of the box and different wall areas |
| **Sphere on Plane** | Sphere on a checkerboard floor | Curved geometry creates smooth parallax differences |
| **Corner Room** | Two walls at 90° with textures | Each viewpoint sees a different wall predominantly |
| **Two Angled Planes** | V-shaped planes with letters | Angled surfaces favor different viewpoints |
| **Cylinder with Text** | Polygon-approximated cylinder | Text appears mirrored between viewpoints |
| **Flat Textured Wall** | Single wall with letter "F" | Baseline: minimal viewpoint difference |

## 5. Implementation Notes

### 5.1 Normalization

The transport matrix is normalized so that max(T) = 1.0. This ensures consistent visualization regardless of absolute light energy levels.

### 5.2 Pixel Footprint Matching

When checking if camera pixel i "sees" the same point illuminated by projector pixel j, we use an adaptive threshold based on the projector's pixel footprint at the hit distance:

```
footprint = distance_to_surface · tan(FOV / n_pixels)
threshold = 1.5 · footprint
```

This accounts for the fact that projector pixels illuminate finite areas, not infinitesimal points.

### 5.3 Performance Considerations

The transport matrix computation is O(n_cam × n_proj × n_objects). For a 32×32 resolution with 8 objects, this is ~8 million ray-object tests — fast enough for interactive use. At 64×64, it takes a few seconds. The computation is CPU-bound and could be accelerated with NumPy vectorization or GPU ray-tracing.
