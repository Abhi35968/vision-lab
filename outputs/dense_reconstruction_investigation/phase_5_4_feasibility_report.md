# Phase 5.4 Feasibility Report: Dense Object Reconstruction & Feature-Density Investigation
*(Updated with Phase 5.4.1 Evidence Audit & Technical Corrections)*

## Executive Summary

This investigation rigorously evaluates whether the existing 47-keyframe handheld smartphone recording (`VID_20261102_015654418.mp4`, 9.60 seconds, 572 frames at 59.6 fps) contains sufficient visual, photometric, and multi-view geometric information to support dense 3D reconstruction of the target tabletop water bottle.

**Primary Finding:** Standard multi-view dense reconstruction (e.g. PatchMatch MVS, COLMAP dense, OpenMVS) is **physically and geometrically ill-posed** on the current dataset due to three compounding structural factors:
1. **Severe Physical Feature Sparsity & Spatial Asymmetry:** Across 47 keyframes, SIFT detected an accumulated total of only **1,087 keypoints inside the bottle masks (3.29%)** vs. **31,990 in the background scene (96.71%)**. The average spatial feature density inside the mask is **0.47 keypoints / 10,000 px²** vs. **4.72** in the background (a 10.1-fold density deficit). Furthermore, **57.5% of all bottle keypoint detections are concentrated exclusively on the opaque grooved cap (top 20% height)**; the smooth middle cylindrical body (45% of total height) is virtually devoid of repeatable native texture.
2. **Narrow Registered Camera Orbit Arc:** The 32 cameras successfully registered by incremental Structure from Motion span an unwrapped horizontal angular arc of **57.0° (15.8% of a full 360° circle)** around the bottle centroid. The camera motion is an oscillating arc (pendulum sweep) rather than a complete orbit. The remaining 303.0° of the circumference is unconstrained by camera poses.
3. **Violation of the Lambertian Surface Assumption:** The clear PET plastic container filled with water ($n \approx 1.333$) violates the photometric brightness constancy assumption via viewpoint-dependent specular reflections and refractive transmission of background tabletop textures.

**Engineering Verdict:** **Recommended: Option C (Re-capture with controlled conditions and 360° orbit) combined with Option D (Silhouette-based Visual Hull / Parametric Prior Modeling)**. Continuing with classical feature-based reconstruction (Option A) or running conventional MVS (Option B) cannot reconstruct a complete 3D object from 29 sparse points on an untextured, refractive cylinder.

---

## 1. Feature Density & Silhouette Spatial Coverage Analysis

### 1.1 Per-Frame Feature Partitioning
SIFT feature extraction ($n=4,000$, $\text{contrast}=0.04$, $\text{edge}=10.0$, $\sigma=1.6$) was executed across all 47 keyframes ($1080 \times 1920$ resolution) and partitioned using YOLO segmentation masks:

| Metric | Target Object (Bottle Mask) | Background Scene (Room / Table) | Disparity Factor |
| :--- | :--- | :--- | :--- |
| **Accumulated Keypoint Detections** | **1,087** (3.29%) | **31,990** (96.71%) | 29.4x fewer raw detections |
| **Mean Spatial Feature Density** | **0.47 kps / 10k px²** | **4.72 kps / 10k px²** | **10.1x lower density** |
| **Frame Area Occupancy (Mean)** | 28.32% (in 40 masked frames) | 71.68% (in 40 masked frames) | - |
| **Keyframes with Mask Dropouts** | 7 frames ([18, 19, 20, 21, 35, 37, 39]) | 0 frames | - |

> [!NOTE]
> **Keypoint Counting Distinction**: The 1,087 figure represents accumulated per-frame 2D SIFT detections across the video sequence, **not unique physical 3D surface points or feature tracks**. Multiple 2D detections across frames correspond to the same physical points tracked over time.

### 1.2 Vertical Silhouette Concentration
Categorization of all 1,087 bottle keypoint detections across normalized bounding box vertical height:
- **Cap Zone ($y \in [0.00, 0.20)$)**: **625 keypoints (57.5%)**
- **Neck Zone ($y \in [0.20, 0.35)$)**: **130 keypoints (12.0%)**
- **Body / Label Zone ($y \in [0.35, 0.80)$)**: **147 keypoints (13.5%)**
- **Base Zone ($y \in [0.80, 1.00]$)**: **185 keypoints (17.0%)**
- **Sum**: $625 + 130 + 147 + 185 = 1,087$ (100.0%)

> [!WARNING]
> The bottle body—which constitutes 45% of the vertical height and over 60% of the surface area—contains only 13.5% of detected features. The smooth transparent plastic surface lacks high-frequency spatial gradients required for feature-based triangulation. Note that this vertical breakdown is an image-space 2D bounding-box partition, serving as an empirical proxy rather than a 3D CAD sectioning.

---

## 2. Multi-View Track Longevity & Survival Analysis

Evaluation of all 2,077 triangulated 3D feature tracks from the Phase 5.2 incremental SfM reconstruction:

| Track Category | Track Count | Mean Length | Survival Rate ($\ge 3$ views) | Max Track Length |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Background Tracks** | **2,027 (97.6%)** | **3.41 views** | **57.0%** | 17 views |
| **Pure Object Tracks** | **38 (1.8%)** | **3.29 views** | **57.9%** | 8 views |
| **Mixed Object-Background Tracks** | **12 (0.6%)** | **3.58 views** | **75.0%** | 7 views |

### Track Longevity Observations
- **Pure Object Tracks (38 tracks)**: Track length distribution `{2: 16, 3: 11, 4: 3, 5: 3, 6: 3, 7: 1, 8: 1}`. 42.1% survive for only 2 views; only 2 tracks survive $\ge 7$ views (both located on the high-contrast cap ridges).
- **Mixed Tracks (12 tracks)**: Features observed inside the mask in some views and outside in others, arising from refracted tabletop textures seen through the liquid volume and mask boundary segmentation tolerances.
- **Pure Background Tracks (2,027 tracks)**: Dominate the reconstruction (97.6%) and survive up to 17 views due to rich, static Lambertian textures in the room.

---

## 3. Pairwise Descriptor Matching & Epipolar Verification

SIFT descriptor matching across adjacent registered camera pairs using FLANN KD-Tree and Lowe's ratio test (threshold = 0.75):

| Region | Mean Raw Matches / Pair | Lowe Ratio Pass % | Epipolar Inlier Rate % | Median Epipolar Error |
| :--- | :--- | :--- | :--- | :--- |
| **Inside Mask (Bottle)** | 14.8 matches | 54.3% | 24.2% | 0.42 px (when $\ge 8$ matches) |
| **Outside Mask (Scene)** | 812.4 matches | 30.5% | 80.6% | 0.58 px |

> [!CRITICAL]
> In over 70% of consecutive frame pairs, the number of Lowe-ratio passed matches inside the bottle mask was **strictly less than 8** (frequently 2 to 5 matches). It is mathematically impossible to estimate an independent Fundamental or Essential matrix solely from features on the bottle. The SfM camera poses were solved entirely by the background scene.

---

## 4. Dataset Limitations: Orbit Geometry & Optical Physics

### 4.1 Camera Orbit Angular Coverage
- **Refined Object Centroid**: `[-1.482, -2.604, 6.218]` (derived from the 14 high-confidence points).
- **Registered Camera Arc**: Azimuth angle spans from **130.5°** to **187.5°**.
- **Continuous Unwrapped Arc**: **57.0°** (15.8% of full circle).
- **Camera Distance Stability**: Mean distance = 7.48 ± 0.64 arbitrary SfM units.
- **Trajectory Character**: Oscillating pendulum arc:
  - Cam 0–16: Sweeps left-to-right ($169.9^\circ \to 130.5^\circ$, span $39.4^\circ$).
  - Cam 17–28: Reverses and sweeps right-to-left ($130.5^\circ \to 187.5^\circ$, span $57.0^\circ$).
  - Cam 29–31: Swings back towards center ($187.5^\circ \to 174.4^\circ$).
- **Registered Reconstruction Aperture**: The registered cameras observe only a 57.0° frontal sector. Keyframes 32–46 (the final ~3 seconds of the video) failed to register in incremental SfM (Camera 32 was rejected during PnP). The remaining 303.0° of the circumference is unconstrained by camera poses.

### 4.2 Audit of Dropout Keyframes
The 7 dropout keyframes where no mask was produced are: [18, 19, 20, 21, 35, 37, 39].
- **Hypothesis Audit**: In the initial report draft, dropouts were attributed to "rapid camera panning and motion blur".
- **Empirical Re-evaluation**:
  - Full-frame Laplacian variance in frame 20 (19.2) and frame 21 (21.0) is higher than in frame 16 (14.2) and frame 17 (8.1) where detection succeeded.
  - Frame 37 (16.3) is sharper than frame 36 (11.5) and frame 40 (11.7) where detection succeeded.
  - Raw YOLO predictions reveal that in frames 18–21 and 35, 37, 39, the transparent bottle was misclassified by the COCO pretrained model as `chair`, `couch`, or `vase` with confidence below the 0.15 acceptance threshold.
  - **Correction**: The dropouts were primarily caused by **semantic misclassification instability and confidence thresholding** on transparent objects, not physical motion blur.

### 4.3 Optical Non-Lambertian Physics
1. **Specular Highlights**: Handheld camera movement causes environment light glints to travel across the curved bottle wall. Keypoints attached to moving highlights exhibit non-rigid apparent motion, violating epipolar constraints.
2. **Refractive Transmission**: Water ($n \approx 1.333$) bends light rays traversing the bottle. Background features seen through the water appear warped and shift non-linearly with viewpoint change.

---

## 5. Technical Comparison of Dense Reconstruction Paradigms

Distinguishing general algorithmic principles, hardware/software constraints, dataset-specific limits, and tested vs estimated metrics:

| Method | Algorithmic Principle | Software & Hardware Profile | Dataset-Specific Feasibility | Testing Status |
| :--- | :--- | :--- | :--- | :--- |
| **Classical PatchMatch MVS (PMVS2)** | Multi-view photometric patch correlation (NCC / SAD) | CPU-compatible (x86-64); 2–6 GB RAM at 1080p | **INFEASIBLE**: Fails on transparent plastic due to moving specularities and refracted background | Conceptually validated via SIFT patch matching failure (< 8 matches) |
| **COLMAP Dense Reconstruction** | Dense PatchMatch stereo + geometric consistency fusion + Poisson meshing | Standard Windows release **requires NVIDIA CUDA GPU**; 8–14 GB RAM | **INFEASIBLE**: Blocked by CUDA requirement; geometric filter purges refractive rays as outliers | Documented architectural constraint; dense executable requires CUDA |
| **OpenMVS** | Multi-view dense point fusion + Delaunay tetrahedral meshing | CPU supported; memory-intensive on 16 GB RAM without downsampling | **INFEASIBLE**: Assumes opaque Lambertian surfaces; produces hollow voids or table surface | Technical evaluation based on OpenMVS specifications |
| **Shape-from-Silhouette (Visual Hull)** | Volumetric intersection of back-projected 2D visual cones | Pure CPU; Python/NumPy native; < 1 GB RAM | **DEGRADED**: Invariant to texture/refraction, but 57.0° orbit causes infinite cone extrusion along $Z$ | Mathematically verified; requires $\ge 180^\circ$ orbit |
| **Neural Radiance Fields (NeRF / 3DGS)** | Differentiable volumetric rendering / 3D Gaussian splatting | Requires NVIDIA CUDA GPU (>= 8 GB VRAM); CPU training is impractical | **INFEASIBLE**: Blocked by GPU constraint; standard NeRF cannot model refraction without Ref-NeRF | Literature-based evaluation; no neural training conducted |
| **Parametric Revolution Prior** | Fits rotational surface to silhouette rays & 3D cap inliers | Lightweight numerical optimization (< 2s, CPU); < 500 MB RAM | **RESEARCH HYPOTHESIS**: Feasible only with strong geometric symmetry assumptions | Theoretical concept; requires empirical validation |

---

## 6. Transparent-Object Reconstruction: Engineering Assessment

### 6.1 Analysis of Shape-from-Silhouette & Visual Hulls
- **Why it bypasses Lambertian reflectance**: Visual hulls rely strictly on the 2D projected boundary of the object. The interior color is ignored, making the formulation theoretically invariant to transparency, surface texturelessness, and refractive light paths.
- **Why transparent segmentation is challenging**: Refraction causes background textures to bleed through the edge of the plastic, blurring boundary contrast. Environmental reflections and specular glare frequently bleed across the silhouette contour, causing mask edge jitter.
- **Sensitivity to calibration and coverage**: Visual cone intersection ($\mathcal{V} = \bigcap_k \mathcal{C}_k$) is strictly dependent on accurate camera poses $(R_k, t_k)$ and intrinsics $K$. Minor calibration errors erode the reconstructed volume. Furthermore, a 57.0° viewing wedge cannot bound the lateral or posterior boundaries of the object.
- **Inability to recover concavities**: Visual hulls can only reconstruct the *visual hull* (the minimal bounding volume). Internal concavities (such as the indented bottle neck groove beneath the cap, ergonomic grip indentations, or the concave punt under the base) never appear on the silhouette and cannot be carved.
- **Insufficiency of planar 360° orbit**: A single horizontal orbit at constant elevation views the top of the bottle cap and the base at near-grazing angles ($\sim 90^\circ$ to normal), leaving the top and bottom surfaces unconstrained. Full surface coverage requires varying camera elevations (oblique high and low angles).

### 6.2 Evaluation of Figure-of-Revolution Prior
- **Working Hypothesis**: Because water bottles exhibit rotational symmetry about a longitudinal axis, fitting a parametric cylinder/spline profile from silhouette rays has been proposed.
- **Critical Evaluation**:
  - The 14 high-confidence 3D points from Phase 5.3.3 are **heavily clustered on the bottle cap (top 20% height)**.
  - 14 points clustered within a narrow height span ($< 2$ cm equivalent) do **not** provide an adequate longitudinal lever arm to reliably fit a 3D central axis in world space without severe angular lever-arm error.
  - Furthermore, real plastic bottles often exhibit non-axisymmetric features (seams, grip indentations, rectangular base corners).
  - Therefore, figure-of-revolution modeling is a **preliminary research hypothesis requiring validation**, not an established reconstruction solution.

---

## 7. Engineering Recommendations: Options A, B, C, D

### Option A: Continue Classical Feature-Based Reconstruction — **REJECTED**
- **Reason:** Only 29 geometrically validated 3D sparse points exist (14 high-confidence, 15 moderate-confidence), with over 57% clustered on the cap. Interpolating a continuous 3D surface mesh from 29 points on a smooth, textureless cylinder is mathematically degenerate.

### Option B: Conventional MVS Pipeline — **REJECTED**
- **Reason:** Incompatible with optical physics (refraction of water and specular reflection violates photometric consistency) and system constraints (COLMAP dense requires CUDA; OpenMVS fails on non-Lambertian surfaces and risks OOM).

### Option C: Capture a New Video with Controlled Conditions — **RECOMMENDED AS PRIMARY PREREQUISITE**
- **Required Capture Protocols**:
  1. **Continuous 360° Closed Orbit with Elevation Variation**: Move smoothly in a continuous multi-elevation loop around the object to cover all 360° of azimuth and view the cap/base obliquely.
  2. **Texture Treatment / Contrast Enhancement**:
     - Apply removable matte spray (e.g. 3D scanning chalk spray, dry shampoo, or fine powder) to convert the transparent surface into an opaque, high-contrast Lambertian surface.
     - Alternatively, place the object on a high-contrast printed turntable.
  3. **Diffused Lighting & Stable Exposure**: Eliminate specular spot reflections and ensure constant lighting.

### Option D: Different Representation for Transparent Objects — **RECOMMENDED ALGORITHMIC APPROACH**
- If re-capturing without physical spray is required:
  1. **Multi-View Silhouette Carving (Visual Hull)**: Uses binary segmentation masks (requires Option C's 360° orbit to bound all sides).
  2. **Prior-Guided Optimization**: Fit CAD or symmetry priors constrained jointly by silhouette rays and multi-view cap inliers.

---

## 8. Phase 5.4.1 — Numerical Validation, Limitations and Corrections

### 8.1 Reconciled Point & Track Hierarchy
The numerical relationships across all phases of the project are explicitly reconciled below:

```
Total Triangulated SfM Tracks: 2,077
│
├── Pure Background Tracks: 2,027 (97.6%)  [Zero observations inside bottle mask]
│
└── Tracks Intersecting Bottle Mask (>= 1 view): 50 (2.4%)
    ├── Insufficient Mask Evidence in Phase 5.3.2: 3 tracks (Tracks 1767, 1771, 1772)
    │   └── Points 986, 987, 988 had only 1 observation on an available mask (< 2 required)
    │
    └── Phase 5.3.2 Object Candidates: 47 tracks / points
        ├── Strong Object Candidates: 22
        ├── Weak Object Candidates:   20
        └── Ambiguous Candidates:      5
            │
            └── Ingested into Phase 5.3.3 Geometric Refinement: 47 points
                ├── High Geometric Confidence:     14 points  ┐
                ├── Moderate Geometric Confidence: 15 points  ┴─> 29 Geometrically Validated Points
                ├── Low Geometric Confidence:      13 points
                └── Geometric Outliers:             5 points
```

In Phase 5.4 mask-occupancy classification:
- **38 Pure Object Tracks**: All available masked observations are inside the mask.
- **12 Mixed Tracks**: Observations span both inside and outside the mask.
- Total mask-intersecting tracks = $38 + 12 = 50$ tracks.

### 8.2 Audit Summary

#### Verified Findings (Confirmed Accurate)
1. **Feature Density Disparity**: SIFT feature density inside the bottle mask is **0.47 kps / 10k px²** vs. **4.72 kps / 10k px²** in the background (a confirmed 10.1-fold density deficit).
2. **Cap Feature Concentration**: **57.5% of all bottle keypoints (625 / 1,087)** are concentrated on the top 20% vertical height (cap/grooves).
3. **Background Track Dominance**: **97.6% (2,027 / 2,077)** of all triangulated 3D tracks belong to the background scene.
4. **Epipolar Matching Deficiency**: Over 70% of consecutive registered frame pairs have $< 8$ ratio-passed SIFT matches inside the bottle mask, confirming that camera poses were solved entirely by the background scene.
5. **Registered Angular Arc**: The 32 registered cameras cover an unwrapped horizontal arc of **57.0°** ($[130.5^\circ, 187.5^\circ]$).

#### Corrected Findings (Technical Adjustments)
1. **Track Count Reconciliation**: Corrected the report text from 31 pure / 19 mixed tracks to the exact figures from `mask_region_track_statistics.csv`: **38 pure object tracks and 12 mixed tracks** (total 50 tracks intersecting masks).
2. **Keypoint Accumulation vs Unique Points**: Clarified that 1,087 keypoints represents accumulated per-frame 2D detections, not unique 3D surface points.
3. **Dropout Cause Correction**: Refuted the claim that dropouts in frames [18, 19, 20, 21, 35, 37, 39] were caused by motion blur. Empirical Laplacian variance and raw YOLO outputs prove the failure was caused by **semantic misclassification and confidence thresholding** (< 0.15) of the pretrained YOLO model on transparent containers.
4. **Camera Coverage Interpretation**: Replaced the claim of a "303° confirmed physical blind spot" with the precise statement that **the registered SfM reconstruction covers 57.0°**, leaving 303.0° unconstrained by camera poses because frames 32–46 failed to register in incremental SfM.
5. **Figure-of-Revolution Status**: Reclassified figure-of-revolution modeling from a suggested solution to an **unvalidated hypothesis**, demonstrating that 14 cap-clustered points cannot reliably constrain a 3D longitudinal cylinder axis without external priors.

#### Unresolved Uncertainties
1. **Physical Motion in Unregistered Frames (32–46)**: Whether the smartphone operator remained stationary or began orbiting during the final 3 seconds cannot be proven from camera poses, as incremental SfM failed to register those frames.
2. **True Internal Liquid Geometry**: The exact refractive impact of the water meniscus on internal light paths cannot be modeled without metric calibration and volumetric refraction ray-tracing.

#### Claims That Should NOT Be Used in Project Presentations
1. *"The video exhibits severe motion blur that caused bottle detection failure."* (False: frames 20 and 21 have high sharpness; failure was YOLO semantic misclassification).
2. *"A 360° orbit was captured but 303° was a blind spot."* (Inaccurate: the video was a 9.6s frontal sweep; only 57.0° was registered by SfM).
3. *"The water bottle has 1,087 surface features."* (False: 1,087 is the sum of per-frame 2D SIFT detections across 47 frames; only 38 pure multi-view tracks exist).
4. *"A figure-of-revolution cylinder can be fit to the 14 high-confidence points."* (Misleading: the 14 points are clustered on the cap and do not span the cylinder height).
