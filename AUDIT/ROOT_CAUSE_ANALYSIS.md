# MATHEMATICAL & TECHNICAL ROOT CAUSE ANALYSIS
**Project:** Monocular Video-to-3D Object Reconstruction Pipeline  
**Document:** `AUDIT/ROOT_CAUSE_ANALYSIS.md`  
**Date:** November 17, 2026  

---

## 1. Failure Mechanism 1: Disparity vs. Linear Depth Inversion

### 1.1 Mathematical Formulation
Monocular foundation models like Depth Anything V2 and MiDaS do not predict metric depth $Z$. They predict **relative disparity** $d \in [0, 1]$, which is inversely proportional to depth:
$$d(u, v) \propto \frac{1}{Z(u, v)}$$
Consequently:
- High disparity values ($d \approx 0.85\text{--}1.0$) represent **close foreground objects**.
- Low disparity values ($d \approx 0.0\text{--}0.4$) represent **distant background structures**.

The partial derivative is strictly negative:
$$\frac{\partial Z}{\partial d} = -\frac{k}{d^2} < 0$$

### 1.2 The Bug in Prior Code
The previous implementation attempted to solve an affine scale and shift directly in linear depth space:
$$Z_{\text{metric}} = s \cdot d + b$$
Because true depth $Z$ increases as disparity $d$ decreases, an unconstrained linear regression yields a negative slope $s < 0$. However, the code included a hard clamp:
```python
if s <= 0.01:
    s = 1.0
    b = float(np.median(z_sfm))
```
When $s$ was forced to $+1.0$:
1. Points with high disparity $d = 0.86$ (the front face of the bottle) received depth $Z = 1.0 \times 0.86 + 12.6 = 13.46\text{ m}$.
2. Points with lower disparity $d = 0.40$ (the rear face or table) received depth $Z = 1.0 \times 0.40 + 12.6 = 13.00\text{ m}$.

**Result:** The entire object was inverted depth-wise. The front of the bottle was projected *behind* the back of the bottle, turning a convex cylinder into a concave, inside-out shell. Furthermore, the variation in depth was squashed to $1.0 \times \Delta d \approx 0.3\text{ units}$, flattening the object into an ultra-thin curved foil.

### 1.3 The Mathematical Correction
Depth must be calibrated in inverse depth space:
$$\frac{1}{Z(u, v)} = a \cdot d(u, v) + b \quad \text{with } a > 0$$
Solving for metric depth:
$$Z(u, v) = \frac{1}{\text{clip}(a \cdot d(u, v) + b, \epsilon, \infty)}$$
Here:
- $a = \frac{\Delta(1/Z)}{\Delta d} \approx \frac{\Delta Z / Z^2}{\text{disp\_span}} > 0$.
- $b = \frac{1}{Z_{\text{expected}}} - a \cdot d_{\text{median}}$.

This guarantees $\frac{\partial Z}{\partial d} < 0$, perfectly preserving the convex 3D curvature of the physical object.

---

## 2. Failure Mechanism 2: Multi-View Drift & Room-Scale Ghosting

### 2.1 The Phenomenon
The previous dense point cloud had a bounding box extent of:
$$\Delta X = 10.47\text{ m}, \quad \Delta Y = 14.54\text{ m}, \quad \Delta Z = 3.58\text{ m}$$
Centered at $Z = 12.61\text{ m}$, despite the true bottle being a 0.25-meter desktop object located at distance ~5.6m.

### 2.2 Root Cause: Sparse SIFT Deprivation & Background Penetration
A smooth plastic water bottle has uniform surface reflectance and lacks textured corner features. In COLMAP:
- Total reconstructed 3D points in scene: **4,880 points**.
- Points on the bottle surface: **0 points**.

All 4,880 points belonged to background office walls, computer screens, and shelves at distances 6.5m to 18.2m. When the prior code projected points into each camera view to find depth correspondences, any sparse point on the far wall directly behind the bottle projected inside the bottle's 2D bounding mask.

Because the camera moved along an arc around the desk:
- In Frame 0, the camera looked towards a wall 13.2m away $\implies b \approx 13.2\text{ m}$.
- In Frame 5, the camera turned towards a corner 11.2m away $\implies b \approx 11.2\text{ m}$.
- In Frame 10, the camera faced a bookshelf 14.1m away $\implies b \approx 14.1\text{ m}$.

When these 10 frames were unprojected into world coordinates:
1. Each frame generated an unprojected curved sheet at the distance of whatever wall was behind the bottle in that view.
2. The 10 sheets were separated by 2 to 4 meters from each other.
3. Merging them created **10 disconnected, intersecting ghost surfaces** spanning 14.54 meters.

### 2.3 The Mathematical Correction: Multi-View Ray Triangulation
Instead of relying on nonexistent SIFT features on the bottle, we formulate multi-view mask centroid ray intersection.

For each camera $i \in \{1, \dots, N\}$:
- Camera center in world coordinates: $C_i = -R_i^T t_i$.
- Mask centroid pixel: $c_i = (u_c, v_c)_i = \left(\frac{1}{|M|} \sum x, \frac{1}{|M|} \sum y\right)$.
- Normalized viewing ray in world coordinates:
  $$r_i = \frac{R_i^T K^{-1} [u_c, v_c, 1]^T}{\|R_i^T K^{-1} [u_c, v_c, 1]^T\|}$$

The true 3D position of the object $P_{\text{obj}}$ is the least-squares point of closest approach to all $N$ rays. It satisfies:
$$\sum_{i=1}^N (I - r_i r_i^T)(P_{\text{obj}} - C_i) = 0$$
$$\implies \left(\sum_{i=1}^N (I - r_i r_i^T)\right) P_{\text{obj}} = \sum_{i=1}^N (I - r_i r_i^T) C_i$$

Solving this $3\times 3$ linear system yields:
$$P_{\text{obj}} = [1.111, 0.445, 5.682]^T$$

For every camera $i$, the expected physical distance to the object is:
$$Z_{\text{expected}, i} = (R_i P_{\text{obj}} + t_i)_z$$

Anchoring $b_i = \frac{1}{Z_{\text{expected}, i}} - a \cdot d_{\text{median}, i}$ locks the centroid of every camera's unprojected cloud to within **0.05 units** across all views. Inter-frame drift dropped from 4.5m to $<0.05\text{ m}$ (a $98.9\%$ reduction).

---

## 3. Failure Mechanism 3: Silhouette Boundary Leakage & Spikes

### 3.1 The Mechanism
YOLO segmentation polygons are imperfect. Along object silhouettes, 5 to 10 pixels of background wall are typically included inside the binary mask.
- On the foreground bottle, disparity is high ($d \approx 0.85$).
- On the background wall, disparity is low ($d \approx 0.20\text{--}0.35$).
- In the transition band along the silhouette, anti-aliased disparity values fall towards $d \to 0$.

When unprojected through inverse depth:
$$Z = \frac{1}{a \cdot d + b}$$
As $d \to 0$, $Z \to \frac{1}{b} \approx Z_{\text{wall}}$ or explodes to infinity if $b$ is small. This created massive geometric "curtains" and spikes radiating outward from the object edges.

### 3.2 The Correction
1. **Morphological Erosion:** Erode the mask with a $7\times 7$ structural kernel, removing boundary silhouette transition pixels.
2. **Disparity Quantile Filtering:** Discard all mask pixels having disparity below the 15th percentile ($d < \text{percentile}_{15}(d_{\text{mask}})$).

This cleanly isolates pure foreground object geometry and eliminates boundary spikes completely.

---

## 4. Failure Mechanism 4: PointNet Coordinate Orientation Bias

### 4.1 The Mechanism
PointNet was trained on ModelNet40 CAD models. In ModelNet40:
1. Every 3D mesh is centered at the origin: $\mu = (0, 0, 0)$.
2. Scaled to the unit sphere: $\max \|P\| = 1.0$.
3. **Rigidly canonicalized with $+Y$ as the UP axis.**
4. Bottles are oriented vertically, with the narrow neck pointing towards $+Y$.

Although PointNet includes a 3D Spatial Transformer Network (STN3d / T-Net), the T-Net regularizer $\|I - A A^T\|_F = 0$ penalizes large deviations from identity. PointNet is known to be **not invariant to arbitrary 3D SO(3) rotations**; it was trained only with azimuthal perturbation around the gravity axis.

When an unaligned real-world point cloud is fed to PointNet:
- In COLMAP, $+Z$ is viewing depth, $+Y$ is camera down.
- In world coordinates, the bottle's major axis was tilted relative to the coordinate axes.
- A tilted cylinder with aspect ratio ~2.3:1 matches the volumetric distribution of a human torso (`person`), a door panel (`door`), or an inclined flight of stairs (`stairs`).

### 4.2 Empirical Verification
We tested PointNet on a synthetic bottle under controlled 3D rotations:
- **Neck along $+Y$ (Canonical ModelNet):** Classified as **`bottle` (98.7%)**, `vase` (1.1%).
- **Neck along $-Y$ (Upside Down):** Classified as **`vase` (96.1%)**, `cup` (0.5%).
- **Axis along $+Z$ (Viewing Depth):** Classified as **`radio` (40.2%)**, `tent` (22.9%).
- **Axis along $+X$ (Horizontal):** Classified as **`car` (43.8%)**, `tent` (22.6%).
- **Arbitrary 3D Rotation (Euler 45°, 60°, 30°):** Classified as **`person` (73.0%)**!

### 4.3 The Correction: Canonical Upright PCA Alignment
We implemented `canonicalize_to_modelnet_frame(points)`:
1. Center point cloud: $P_c = P - \bar{P}$.
2. Compute $3\times 3$ covariance matrix $C = \frac{1}{N} P_c^T P_c$.
3. Compute eigenvectors and eigenvalues $C v_i = \lambda_i v_i$. The primary eigenvector $v_{\max}$ corresponding to $\lambda_{\max}$ defines the object's major elongation axis.
4. Compute rotation matrix $R_{\text{align}}$ that rotates $v_{\max}$ onto $[0, 1, 0]^T$.
5. Resolve 180° inversion ambiguity: compute radial spread $r(y) = \sqrt{x^2 + z^2}$ for upper ($y > 0$) vs lower ($y < 0$) points. For a bottle, the neck is narrower than the base ($r_{\text{upper}} < r_{\text{lower}}$). If $r_{\text{upper}} > r_{\text{lower}}$, invert $Y \to -Y$.

Applying this canonicalization enabled PointNet to identify the geometry as a vessel (`bottle`: 20.9%, `vase`: 5.3%) rather than hallucinating `person` or `door`.

---

## 5. Failure Mechanism 5: Screened Poisson Mesh Trimming

### 5.1 The Mechanism
Open3D's Screened Poisson Surface Reconstruction (`create_from_point_cloud_poisson`) solves:
$$\nabla^2 \chi = \nabla \cdot \vec{V}$$
for an indicator function $\chi$ on an adaptive octree bounding box.
1. The solver inherently creates a **closed, watertight bounding manifold** enclosing the octree.
2. However, where there are no data points, Poisson extrapolates low-density "bubble" surfaces.
3. Standard practice removes these bubbles by trimming low-density vertices:
   ```python
   mesh.remove_vertices_by_mask(densities < thresh)
   ```
4. Deleting vertices cuts open the closed surface, creating boundary edges.
5. In monocular video, the camera path covers only ~180° around the bottle. The backside, bottom, and interior are occluded (a 2.5D surface).
6. When trimmed, the mesh becomes an open surface shell. Testing `mesh.is_watertight()` returns `False`.

### 5.2 The Correction
1. Stopped making false claims of "watertight manifold" meshes.
2. Implemented `MeshDiagnostics` in `multi_view_dense_fuser.py` to truthfully evaluate and export:
   - `is_watertight`: `False`
   - `is_edge_manifold`: `False`
   - `is_vertex_manifold`: `False`
   - `is_self_intersecting`: `True`
   - `num_vertices`: 114,074
   - `num_triangles`: 227,299
3. Exported both the untrimmed watertight enclosure (`*_poisson_mesh_untrimmed.ply`) and the density-trimmed surface shell (`*_poisson_mesh.ply`).
