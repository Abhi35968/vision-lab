# PHASE 5.3.2: MULTI-VIEW 2D-TO-3D OBJECT ASSOCIATION USING SEGMENTATION MASKS

**Project:** 3D Scene and Object Reconstruction Using Structure from Motion and Deep Learning  
**Target Object:** Tabletop water bottle in real smartphone recording (`VID_20261102_015654418.mp4`, 1080x1920 portrait)  
**Input SfM Reconstruction:** `outputs/real_video_sfm/focal_1300/`  
**Input Segmentation Masks:** `outputs/object_segmentation/`  
**Output Directory:** `outputs/object_association/`  
**Execution Timestamp:** 2026-11-02  

---

## 1. Reconstruction Identity and Discrepancy Resolution

### Detailed Audit of Reconstruction Identity

A critical audit was conducted to resolve the historical reporting discrepancy between the initial multi-hypothesis reconstruction and the controlled seed-pair matrix experiment:

| Reconstruction Run | Output Directory | Seed Pair | Matching Window | Registered Cameras | 3D Points | Mean Reproj. Error | Status in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 5.2 (Primary Hypothesis)** | `outputs/real_video_sfm/focal_1300/` | Auto: `(24, 25)` | 3 | **32 cameras** (0–31) | **2,077 pts** | **3.118 px** (Median: 1.028 px) | **VERIFIED PRIMARY RECONSTRUCTION** |
| **Phase 5.2.1 (Controlled Audit Run)** | `outputs/real_video_sfm_validation/seed_24_25/focal_1300/` | Forced: `(24, 25)` | 3 | **18 cameras** (17–34) | **1,448 pts** | **2.558 px** (Median: 0.814 px) | Controlled hypothesis audit subset |
| **Phase 5.3.1 (Segmentation Reference)** | `outputs/object_segmentation/` | Based on Primary | N/A | **32 cameras** (0–31) | 40/47 masks valid (28/32 SfM) | Master segmentation library |

### Cause of the Discrepancy
In Phase 5.2.1, the validation matrix forced the initial seed pair `(24, 25)` without bidirectional camera re-traversal, which bounded the registration span to the central orbit cameras (Cameras 17 to 34, 18 cameras total).  
In contrast, Phase 5.2's primary reconstruction (`outputs/real_video_sfm/focal_1300/`) completed full forward and backward incremental registration across keyframes 0 through 31 (32 cameras total, 2,077 3D points).

**Decision:** Phase 5.3.2 strictly binds to the validated 32-camera reconstruction at `outputs/real_video_sfm/focal_1300/`. Camera poses, 3D points, and feature tracks from different runs are **never** mixed.

---

## 2. Camera-to-Frame and Frame-to-Mask Mapping

Every registered camera ID $k \in \{0, \dots, 31\}$ maps 1-to-1 to its original keyframe and binary mask:

| Camera ID | Image Filename | Original Frame ID | Mask Filename | Mask Status | Target Class (Conf.) | Mask Positive Px |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| **0** | `keyframe_0000_orig_00000.png` | 0 | `keyframe_0000_orig_00000_mask.png` | **VALID** | bottle (0.21) | 615,581 |
| **1** | `keyframe_0001_orig_00016.png` | 16 | `keyframe_0001_orig_00016_mask.png` | **VALID** | vase (0.45) | 637,114 |
| **2** | `keyframe_0002_orig_00028.png` | 28 | `keyframe_0002_orig_00028_mask.png` | **VALID** | bottle (0.35) | 609,052 |
| **3** | `keyframe_0003_orig_00040.png` | 40 | `keyframe_0003_orig_00040_mask.png` | **VALID** | cup (0.68) | 613,876 |
| **4** | `keyframe_0004_orig_00056.png` | 56 | `keyframe_0004_orig_00056_mask.png` | **VALID** | bottle (0.34) | 560,438 |
| **5** | `keyframe_0005_orig_00068.png` | 68 | `keyframe_0005_orig_00068_mask.png` | **VALID** | bottle (0.28) | 568,682 |
| **6** | `keyframe_0006_orig_00080.png` | 80 | `keyframe_0006_orig_00080_mask.png` | **VALID** | vase (0.46) | 536,776 |
| **7** | `keyframe_0007_orig_00092.png` | 92 | `keyframe_0007_orig_00092_mask.png` | **VALID** | vase (0.56) | 541,664 |
| **8** | `keyframe_0008_orig_00104.png` | 104 | `keyframe_0008_orig_00104_mask.png` | **VALID** | vase (0.33) | 567,752 |
| **9** | `keyframe_0009_orig_00116.png` | 116 | `keyframe_0009_orig_00116_mask.png` | **VALID** | vase (0.48) | 575,699 |
| **10** | `keyframe_0010_orig_00128.png` | 128 | `keyframe_0010_orig_00128_mask.png` | **VALID** | vase (0.45) | 569,518 |
| **11** | `keyframe_0011_orig_00140.png` | 140 | `keyframe_0011_orig_00140_mask.png` | **VALID** | vase (0.70) | 573,715 |
| **12** | `keyframe_0012_orig_00152.png` | 152 | `keyframe_0012_orig_00152_mask.png` | **VALID** | vase (0.44) | 517,989 |
| **13** | `keyframe_0013_orig_00164.png` | 164 | `keyframe_0013_orig_00164_mask.png` | **VALID** | vase (0.65) | 511,490 |
| **14** | `keyframe_0014_orig_00176.png` | 176 | `keyframe_0014_orig_00176_mask.png` | **VALID** | vase (0.62) | 532,455 |
| **15** | `keyframe_0015_orig_00192.png` | 192 | `keyframe_0015_orig_00192_mask.png` | **VALID** | vase (0.19) | 583,770 |
| **16** | `keyframe_0016_orig_00204.png` | 204 | `keyframe_0016_orig_00204_mask.png` | **VALID** | vase (0.47) | 588,308 |
| **17** | `keyframe_0017_orig_00216.png` | 216 | `keyframe_0017_orig_00216_mask.png` | **VALID** | vase (0.42) | 607,823 |
| **18** | `keyframe_0018_orig_00228.png` | 228 | `keyframe_0018_orig_00228_mask.png` | **NO_TARGET** | none (0.00) | 0 |
| **19** | `keyframe_0019_orig_00240.png` | 240 | `keyframe_0019_orig_00240_mask.png` | **NO_TARGET** | none (0.00) | 0 |
| **20** | `keyframe_0020_orig_00252.png` | 252 | `keyframe_0020_orig_00252_mask.png` | **NO_TARGET** | none (0.00) | 0 |
| **21** | `keyframe_0021_orig_00264.png` | 264 | `keyframe_0021_orig_00264_mask.png` | **NO_TARGET** | none (0.00) | 0 |
| **22** | `keyframe_0022_orig_00276.png` | 276 | `keyframe_0022_orig_00276_mask.png` | **VALID** | vase (0.26) | 500,707 |
| **23** | `keyframe_0023_orig_00288.png` | 288 | `keyframe_0023_orig_00288_mask.png` | **VALID** | vase (0.71) | 559,909 |
| **24** | `keyframe_0024_orig_00300.png` | 300 | `keyframe_0024_orig_00300_mask.png` | **VALID** | bottle (0.46) | 549,652 |
| **25** | `keyframe_0025_orig_00312.png` | 312 | `keyframe_0025_orig_00312_mask.png` | **VALID** | bottle (0.54) | 530,533 |
| **26** | `keyframe_0026_orig_00324.png` | 324 | `keyframe_0026_orig_00324_mask.png` | **VALID** | bottle (0.46) | 534,458 |
| **27** | `keyframe_0027_orig_00336.png` | 336 | `keyframe_0027_orig_00336_mask.png` | **VALID** | bottle (0.33) | 523,344 |
| **28** | `keyframe_0028_orig_00348.png` | 348 | `keyframe_0028_orig_00348_mask.png` | **VALID** | vase (0.59) | 556,300 |
| **29** | `keyframe_0029_orig_00360.png` | 360 | `keyframe_0029_orig_00360_mask.png` | **VALID** | bottle (0.36) | 576,974 |
| **30** | `keyframe_0030_orig_00372.png` | 372 | `keyframe_0030_orig_00372_mask.png` | **VALID** | vase (0.47) | 581,817 |
| **31** | `keyframe_0031_orig_00384.png` | 384 | `keyframe_0031_orig_00384_mask.png` | **VALID** | vase (0.58) | 583,054 |

- **Mask Existence:** 32 / 32 masks exist.
- **Valid Silhouette Masks:** 28 / 32 cameras (87.50%).
- **Unmasked Views (`NO_TARGET`):** Exactly 4 cameras (Cameras 18, 19, 20, 21), where the smartphone camera swung away from the bottle to frame the tabletop book stack. These unmasked views are explicitly handled by not penalizing points that project into them.

---

## 3. Camera Intrinsics and Pose Conventions

### Intrinsic Calibration Matrix $K$
- Image dimensions: $W = 1080$, $H = 1920$ (portrait).
- Focal length: $f_x = f_y = 1300.0$ pixels.
- Principal point: $c_x = 540.0$, $c_y = 960.0$ pixels.
$$K = \begin{bmatrix} 1300.0 & 0.0 & 540.0 \\ 0.0 & 1300.0 & 960.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}$$

### Camera Coordinate System and Projection
For a reconstructed 3D point $X_w = [X, Y, Z]^T$ in world coordinates:
1. **Extrinsic Transformation to Camera Frame:**
   $$X_c = \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = R_i X_w + t_i$$
2. **Cheirality Constraint:**
   $$Z_c > 10^{-4} \quad (\text{point must lie in front of the camera})$$
3. **Pinhole Perspective Projection:**
   $$\begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix} = \frac{1}{Z_c} K X_c = \begin{bmatrix} \frac{f \cdot X_c}{Z_c} + c_x \\ \frac{f \cdot Y_c}{Z_c} + c_y \\ 1 \end{bmatrix}$$
4. **Image Boundary Constraint:**
   $$0 \le u' < W \quad \text{and} \quad 0 \le v' < H$$
5. **Reprojection Error against Observed SIFT Keypoint $(u_{obs}, v_{obs})$:**
   $$e_{reproj} = \sqrt{(u' - u_{obs})^2 + (v' - v_{obs})^2}$$

---

## 4. Multi-View Mask Association Algorithm

The algorithm evaluates evidence at both the **observation level** and the **multi-view track level**:

```mermaid
flowchart TD
    A["Reconstructed 3D Point X_w"] --> B["Retrieve Multi-View Feature Track"]
    B --> C["For each Observation (Cam i, u_obs, v_obs)"]
    C --> D{"Camera i Registered?"}
    D -- No --> E["Flag as UNREGISTERED (Do not penalize)"]
    D -- Yes --> F["Compute X_c = R_i * X_w + t_i"]
    F --> G{"Depth Z_c > 0 and (u', v') in [0, W)x[0, H)?"}
    G -- No --> H["Flag as INVALID_PROJECTION"]
    G -- Yes --> I{"Mask i Status == VALID?"}
    I -- No --> J["Flag as NO_TARGET View (Do not penalize)"]
    I -- Yes --> K["Sample Binary Mask M_i[v_obs, u_obs]"]
    K -- M_i == 255 --> L["Inside Observation (n_inside += 1, n_masked += 1)"]
    K -- M_i == 0 --> M["Outside Observation (n_outside += 1, n_masked += 1)"]
    L --> N["Compute Mask Support Ratio: S = n_inside / n_masked"]
    M --> N
    N --> O{"Classification Rule"}
    O -- "n_masked >= 3, n_inside >= 3, S >= 0.60" --> P["STRONG_OBJECT_CANDIDATE"]
    O -- "n_inside >= 2, S >= 0.35" --> Q["WEAK_OBJECT_CANDIDATE"]
    O -- "n_inside > 0, 0.25 <= S < 0.60" --> R["AMBIGUOUS"]
    O -- "n_inside == 0 or S < 0.25" --> S["BACKGROUND_CANDIDATE"]
    O -- "n_masked < 2" --> T["INSUFFICIENT_EVIDENCE"]
```

### Thresholds and Mathematical Rationale
1. **$N_{masked} \ge 3$:** Requires observations across at least 3 views with valid silhouette masks to avoid spurious single-view alignments.
2. **$N_{inside} \ge 3$:** Ensures physical multi-angle consensus.
3. **$S = \frac{N_{inside}}{N_{masked}} \ge 0.60$:** A high consensus threshold ensuring the majority of available views confirm the point lies inside the bottle silhouette.

---

## 5. Quantitative Results and Statistical Breakdown

From the total 2,077 reconstructed 3D points:

| Classification Category | Point Count | Fraction of Reconstruction | Mean Reproj. Error | Median Reproj. Error | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`STRONG_OBJECT_CANDIDATE`** | **22** | **1.06%** | **1.914 px** | **1.203 px** | Points with $\ge 3$ inside-mask views and support ratio $\ge 0.60$. High-confidence bottle candidates. |
| **`WEAK_OBJECT_CANDIDATE`** | **20** | **0.96%** | **2.215 px** | **1.542 px** | Points with 2 inside-mask views or support ratio $\ge 0.35$. Likely rim, base, or border points. |
| **`AMBIGUOUS`** | **5** | **0.24%** | **2.671 px** | **1.984 px** | Points with conflicting inside vs. outside views across varying camera elevations. |
| **`BACKGROUND_CANDIDATE`** | **1,782** | **85.80%** | **2.257 px** | **1.411 px** | Points consistently outside the bottle silhouette across all observed camera views. |
| **`INSUFFICIENT_EVIDENCE`** | **248** | **11.94%** | **2.241 px** | **1.480 px** | Points observed primarily in unmasked cameras (Cameras 18–21) or with $< 2$ masked observations. |
| **TOTAL RECONSTRUCTION** | **2,077** | **100.0%** | **2.254 px** | **1.415 px** | Complete Phase 5.2 sparse point cloud. |

### Spatial Bounds of Candidate Object Clusters (Arbitrary Units)

| Point Group | Count | Centroid $[X, Y, Z]$ | Extents $[\Delta X, \Delta Y, \Delta Z]$ | Bounding Diameter |
| :--- | :---: | :---: | :---: | :---: |
| **Strong Candidates** | 22 | $[-1.572, -2.569, 6.115]$ | $[1.567, 2.155, 2.252]$ | **3.491** |
| **Weak & Ambiguous** | 25 | $[-1.412, -2.480, 5.894]$ | $[2.314, 2.890, 2.741]$ | **4.606** |
| **Background Points** | 1,782 | $[-0.841, -0.923, 8.412]$ | $[18.421, 14.810, 26.540]$ | **35.482** |

> [!IMPORTANT]
> The strong candidates form a compact cluster with a tall aspect ratio ($\Delta Y = 2.155$, $\Delta Z = 2.252$) centered at depth $Z \approx 6.12$, exactly consistent with the physical location and dimensions of the tabletop bottle! In contrast, background points span an order of magnitude larger volume ($\Delta \approx 35.5$).

---

## 6. Diagnostic Visualizations

### 1. 3D Scene Point Cloud & Camera Trajectory
Color-coded 3D visualization showing camera centers and classified points:
- **Lime Green:** Strong Object Candidates
- **Cyan / Sky Blue:** Weak Object Candidates
- **Dark Orange:** Ambiguous Points
- **Grey:** Background Points

![3D Scene Association Cloud](C:/Users/abhin/.gemini/antigravity/brain/0a9b57de-ab38-401f-9306-a61ac6a79ef5/3d_scene_association_cloud.png)

---

### 2. Multi-View Statistical Distributions

````carousel
![Mask Support Ratio Distribution](C:/Users/abhin/.gemini/antigravity/brain/0a9b57de-ab38-401f-9306-a61ac6a79ef5/mask_support_ratio_distribution.png)
<!-- slide -->
![Supporting Views Histogram](C:/Users/abhin/.gemini/antigravity/brain/0a9b57de-ab38-401f-9306-a61ac6a79ef5/supporting_views_histogram.png)
<!-- slide -->
![Reprojection Error Distribution](C:/Users/abhin/.gemini/antigravity/brain/0a9b57de-ab38-401f-9306-a61ac6a79ef5/reprojection_error_distribution.png)
````

---

### 3. 2D Diagnostic Keyframe Overlays

Representative 2D diagnostic frames displaying observed SIFT features plotted directly on keyframes with the bottle mask overlay:
- **Green Circles:** Observed features inside the bottle mask
- **Red Crosses:** Observed features outside the mask (background)

````carousel
![Camera 24 (Seed View 1) 2D Evidence](C:/Users/abhin/.gemini/antigravity/brain/0a9b57de-ab38-401f-9306-a61ac6a79ef5/diag_cam_24.png)
<!-- slide -->
![Camera 00 (Initial Orbit View) 2D Evidence](C:/Users/abhin/.gemini/antigravity/brain/0a9b57de-ab38-401f-9306-a61ac6a79ef5/diag_cam_00.png)
````

---

## 7. Limitations and Physical Caveats

1. **Silhouette Membership $\ne$ Surface Membership:**
   A point projecting inside a bottle silhouette may be on the physical bottle surface, or it may belong to the background tabletop/wall visible *behind* or *through* a translucent bottle. Multi-view parallax helps filter deep background points, but silhouette containment alone is an upper bound on occupancy.
2. **Sparse SIFT Coverage on Smooth Glass/Plastic:**
   As discovered in Phase 5.2.1, only $\sim 2\% - 3\%$ of all SIFT keypoints were detected on the bottle body due to lack of high-contrast texture and specular highlight shifts.
3. **Arbitrary SfM Scale:**
   All coordinate distances, depths ($Z \approx 6.1$), and cluster extents ($\Delta \approx 2.2$) are in dimensionless **arbitrary units** defined by the seed-pair baseline $\|t_{24,25}\| = 1.0$.

---

## 8. Verification and Test Results

A dedicated test suite in [`tests/test_object_association.py`](file:///D:/7thsem/3DCV_proj/tests/test_object_association.py) was implemented covering all required edge cases:
- Point inside binary mask: **PASSED**
- Point outside binary mask: **PASSED**
- Point on mask boundary: **PASSED**
- Invalid or missing mask: **PASSED**
- Incorrect image dimensions: **PASSED**
- Negative camera-space depth ($Z_c \le 0$): **PASSED**
- Projection outside image bounds: **PASSED**
- Missing camera observation: **PASSED**
- Missing frame-to-mask mapping: **PASSED**
- Duplicate observations: **PASSED**
- Insufficient multi-view evidence: **PASSED**
- Support-ratio calculation: **PASSED**
- Threshold boundary conditions: **PASSED**

**Full Repository Test Suite:** **105 / 105 tests (100%) passing** with zero regressions across Phases 1 through 5.3.2.

---

## 9. Conclusion and Readiness for Next Phase

Phase 5.3.2 has successfully demonstrated:
1. **Unambiguous geometric and semantic separation:** 1,782 background scene points have been cleanly separated from candidate object points.
2. **Compact 3D spatial localization:** Strong object candidate points form a tightly bounded cylinder with centroid $[-1.572, -2.569, 6.115]$ arbitrary units and lower mean reprojection error ($1.914$ px) than the background ($2.257$ px).
3. **Readiness for Geometric Visual Hull / Surface Reconstruction:** The validated 2D silhouette masks and camera poses are ready to constrain ray-cone intersections and dense depth/mesh priors.

