# Phase 2: Feature Detection, Matching, Geometric Verification, and Two-View Structure from Motion

This document provides a comprehensive technical breakdown of the geometric computer vision algorithms implemented in Phase 2 of the **3D Scene and Object Reconstruction** system.

---

## 1. What is a Feature Detector?

A **feature detector** is an algorithm that scans an image to identify salient, repeatable 2D interest points (termed **keypoints**). A good keypoint possesses:
* **Repeatability**: Can be detected across varying viewpoints, illumination, scale, and sensor noise.
* **Distinctiveness**: Sits on local image regions with high spatial gradient variance in multiple directions (corners, junctions, blobs).
* **Locality**: Occupies a compact pixel neighborhood, making it robust to partial occlusion.

In contrast to flat surfaces (e.g., plain white walls) where image gradients are zero ($\nabla I \approx 0$), interest points occur where the local autocorrelation matrix (second-moment matrix) has two large eigenvalues:

$$M = \sum_{(x, y) \in W} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

---

## 2. What is a Descriptor?

A **descriptor** is a compact numerical vector (e.g., a 128-dimensional floating-point vector in SIFT or a 256-bit binary string in ORB) that summarizes the local photometric appearance around a keypoint.

| Property | Keypoint | Descriptor |
| :--- | :--- | :--- |
| **Data Representation** | Coordinate tuple: $(x, y, \sigma, \theta, \text{response})$ | Multi-dimensional vector $\mathbf{d} \in \mathbb{R}^{128}$ or $\mathbf{d} \in \{0, 1\}^{256}$ |
| **Role** | Specifies **where** in the image to look | Specifies **what** the local patch looks like |
| **Invariance** | Scale and location detection | Contrast, rotation, and illumination invariance |

---

## 3. How SIFT Works (High-Level Overview)

David Lowe's **Scale-Invariant Feature Transform (SIFT)** achieves scale, rotation, and affine illumination invariance through four computational stages:

```
  Input Image
       │
       ▼
[1. Scale-Space Extrema]   ──> Convolve image across octaves with Gaussian kernels G(x, y, σ)
       │                       Compute Difference-of-Gaussians: DoG = D(x, y, σ) = (G(kσ) - G(σ)) * I
       ▼
[2. Keypoint Localization] ──> Subpixel Taylor series expansion; eliminate low-contrast extrema
       │                       Reject unstable edge responses via Hessian matrix eigenvalue ratio
       ▼
[3. Orientation Assignment]──> Compute gradient magnitudes & orientations in neighborhood
       │                       Assign dominant orientation peak θ to achieve 2D rotation invariance
       ▼
[4. Descriptor Generation] ──> Sample 16x16 patch rotated by θ; divide into 4x4 sub-grids
                               Form 8-bin orientation histogram per sub-grid: 4x4 x 8 = 128-D vector
                               L2 normalize, clamp values at 0.2, and re-normalize
```

---

## 4. How ORB Differs from SIFT

**Oriented FAST and Rotated BRIEF (ORB)** is an open-source, computationally lightweight alternative engineered for real-time mobile and embedded systems:

| Architectural Metric | SIFT (Scale-Invariant Feature Transform) | ORB (Oriented FAST and Rotated BRIEF) |
| :--- | :--- | :--- |
| **Detection Backbone** | Difference-of-Gaussians (DoG) pyramid | Multi-scale FAST corner detection with Harris score |
| **Orientation** | Gradient orientation histogram | Intensity centroid vector: $\mathbf{C} = \left(\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}\right)$ |
| **Descriptor Type** | 128-dimensional float32 vector | 256-bit (32-byte) binary string |
| **Distance Metric** | Euclidean ($L_2$ norm via FLANN KD-Trees) | Hamming Distance (bitwise XOR via POPCNT CPU instruction) |
| **Compute Latency** | Moderate to High (~30–80 ms per frame) | Extremely Fast (~5–12 ms per frame) |
| **Viewpoint Invariance** | Excellent across wide baselines & affine changes | Moderate; sensitive to large out-of-plane rotations |

---

## 5. Why We Need Feature Matching

Individual images provide disconnected 2D pixel coordinates. To establish 3D geometry, the system must know which point in Image 1 corresponds to the exact same physical scene element in Image 2.

### Why Appearance Matching Alone is Insufficient:
Descriptors measure only local appearance. In real-world environments, repetitive textures (keyboards, window grids, floor tiles, carpet patterns) produce near-identical descriptor vectors at different physical coordinates. Pure nearest-neighbor matching yields significant **false correspondences (outliers)**. We require:
1. **Lowe's Ratio Test** (discriminative distance gating).
2. **Epipolar Geometry & RANSAC** (rigid geometric consistency gating).

---

## 6. What is Lowe's Ratio Test?

For a query descriptor $\mathbf{d}_1$ from Image 1, we locate its closest neighbor $\mathbf{d}_{2, \text{best}}$ (distance $D_1$) and second-closest neighbor $\mathbf{d}_{2, \text{second}}$ (distance $D_2$) in Image 2:

$$\text{Ratio} = \frac{D_1}{D_2} < \tau \quad (\text{typically } \tau = 0.75\text{--}0.80)$$

```
      Query Feature (Image 1)
               │
       ┌───────┴───────┐
       ▼               ▼
1st Nearest Neighbor   2nd Nearest Neighbor
(Distance D1)          (Distance D2)
```

* **Distinct, True Match**: $D_1$ is small, while $D_2$ is significantly larger $\implies D_1 / D_2 \ll 1$ (Retained).
* **Ambiguous / Repetitive Match**: The feature matches multiple identical patterns in Image 2 $\implies D_1 \approx D_2 \implies D_1 / D_2 \approx 1$ (Rejected).

---

## 7. The Fundamental and Essential Matrices

```
        Camera 1 (Pinhole)                             Camera 2 (Pinhole)
            Center C₁                                      Center C₂
               ○                                              ○
               │  ╲                                        ╱  │
               │    ╲                                    ╱    │
               │   x₁ ╲                                ╱ x₂   │
        ┌──────┼───────●──────────────────────────────●───────┼──────┐
        │      │        ╲                            ╱        │      │
        │Image 1│         ╲                        ╱         │Image 2│
        │      │           ╲                      ╱           │      │
        └──────┴────────────╲────────────────────╱────────────┴──────┘
                              ╲                ╱
                                ╲            ╱
                                  ╲        ╱
                                    ▼    ▼
                               3D Point X (Scene)
```

### 7.1 The Fundamental Matrix ($F$)
For uncalibrated pixel coordinates $x_1 = [u_1, v_1, 1]^T$ and $x_2 = [u_2, v_2, 1]^T$:

$$x_2^T F x_1 = 0$$

* $F$ is a $3 \times 3$, rank-2 matrix with 7 degrees of freedom.
* The vector $l_2 = F x_1$ represents the **epipolar line** in Image 2 along which the observation $x_2$ must lie.

### 7.2 The Essential Matrix ($E$)
When camera intrinsic matrix $K$ is known, pixel coordinates are transformed to normalized ray directions:

$$x_{n1} = K^{-1} x_1, \quad x_{n2} = K^{-1} x_2$$

The Essential Matrix $E$ relates normalized coordinates:

$$x_{n2}^T E x_{n1} = 0 \quad \text{where} \quad E = K^T F K = [t]_\times R$$

* $R \in \mathrm{SO}(3)$ represents relative camera rotation.
* $[t]_\times$ is the skew-symmetric matrix of the relative translation direction vector:

$$[t]_\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}$$

---

## 8. What is RANSAC?

**RANdom SAmple Consensus (RANSAC)** is an iterative estimator designed to solve model parameters in the presence of high outlier ratios (often $30\%\text{--}60\%$ in feature matching).

```
[Iterate N times]:
  1. Randomly sample minimal subset (8 point pairs for Fundamental Matrix).
  2. Compute candidate matrix F_sample.
  3. Evaluate symmetric epipolar distance d_sym(x1, x2; F_sample) across all correspondences.
  4. Count inliers where d_sym < threshold (e.g. 2.0 pixels).
[Termination]:
  Select candidate model maximizing inlier count; re-estimate F using all consensus inliers.
```

---

## 9. How We Recover Camera Rotation and Translation

Singular Value Decomposition (SVD) of the Essential Matrix $E = U \Sigma V^T$ enforces $\Sigma = \text{diag}(1, 1, 0)$ and produces four possible mathematical solutions for relative motion $[R \mid t]$:

$$W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

$$\text{Candidate Rotations: } R_A = U W V^T, \quad R_B = U W^T V^T$$
$$\text{Candidate Translations: } t = \pm u_3 \quad (\text{last column of } U)$$

### Cheirality Condition
Only **one** of the four configurations places triangulated 3D points in front of both camera sensors (positive depth $Z_1 > 0$ and $Z_2 > 0$). `cv2.recoverPose` tests Cheirality across all inliers and selects the true physical pose.

---

## 10. What is Triangulation?

**Triangulation** determines the 3D position $X$ of a physical point given its 2D projections $x_1$ and $x_2$ and camera projection matrices $P_1 = K [I \mid 0]$ and $P_2 = K [R \mid t]$.

Because camera rays never intersect perfectly due to pixel discretization and noise, **Direct Linear Transformation (DLT)** solves for the point minimizing algebraic error:

$$\begin{bmatrix}
u_1 P_1^{(3)} - P_1^{(1)} \\
v_1 P_1^{(3)} - P_1^{(2)} \\
u_2 P_2^{(3)} - P_2^{(1)} \\
v_2 P_2^{(3)} - P_2^{(2)}
\end{bmatrix} X = 0$$

where $P^{(i)}$ denotes row $i$ of the projection matrix. SVD yields the homogeneous solution $[X_1, X_2, X_3, W]^T$, which is dehomogenized to $[X_1/W, X_2/W, X_3/W]^T$.

---

## 11. What is Reprojection Error?

**Reprojection error** is the geometric Euclidean distance (in pixels) between an observed 2D feature point $x_i$ and the forward projection $\hat{x}_i = \pi(P_i X)$ of the reconstructed 3D point:

$$e_i = \| x_i - \pi(P_i X) \|_2 = \sqrt{(u_i - \hat{u}_i)^2 + (v_i - \hat{v}_i)^2}$$

$$\text{Mean Two-View Reprojection Error} = \frac{1}{2} (e_1 + e_2)$$

* High reprojection error indicates poor triangulation, small baseline degeneracy, or an incorrect feature match.
* In our pipeline, points with mean reprojection error $> 4.0\text{ pixels}$ are rejected.

---

## 12. Why Does Monocular Reconstruction Have an Unknown Scale?

In a single monocular camera, multiplying all 3D scene coordinates $X$ by a scalar $\lambda > 0$ and simultaneously scaling the camera translation $t$ by $\lambda$ produces **identical image projections**:

$$\hat{x} \sim K [R \mid \lambda t] (\lambda X) = \lambda K [R \mid t] X \sim K [R \mid t] X$$

A small toy chair close to the camera projects to the identical pixels as a life-sized chair placed further away. Monocular reconstruction recovers camera translation direction as a unit vector ($\|t\| = 1.0$) up to an arbitrary global scale factor $\lambda$. Real-world metric scale requires an external reference (known object dimension, IMU baseline, or calibration target).

