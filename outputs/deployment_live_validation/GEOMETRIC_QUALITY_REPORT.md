# Geometric Quality Assessment Report (Post Depth-Gradient Filtering)

**Execution Context:** Deployment Live Validation on `VID_20261102_015654418.mp4`  
**Output Directory:** [outputs/deployment_live_validation/](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/)  
**Filter Applied:** Scale-Invariant Depth Gradient Filtering ($\mathcal{G}_{rel} \le 0.08$)  
**Status:** Completed & Rigorously Audited  

---

## 1. Executive Summary

A forensic geometric evaluation was conducted following the integration of **Scale-Invariant Depth Gradient Filtering ($\|\nabla Z\|_2 / Z \le 0.08$)**. This run verified the elimination of boundary silhouette edge bleed and trailing flying pixels.

### Comparison: Before vs. After Depth Gradient Filtering

| Metric Dimension | Before Filtering | With Depth Gradient Filtering | Impact / Improvement |
| :--- | :--- | :--- | :--- |
| **Object Point Count** | 114,206 pts | **112,767 pts** | 1,439 trailing boundary edge ramp points cleanly pruned |
| **Circularity ($Z / X$)** | 1.0065 (0.65% err) | **1.0002 (0.02% err)** | **Near-perfect circular cylinder symmetry ($2.6719 \times 2.6725$)** |
| **3D Aspect Ratio ($H : D$)** | 2.749 : 1 | **2.750 : 1** | Exactly conforms to true 2.75:1 cylinder proportions |
| **Connected Components** | 34 | **30** | Reduced peripheral noise fragments |
| **Degenerate Mesh Faces** | 2,481 faces | **2,470 faces** | Fewer sliver face intersections |
| **Euler Characteristic ($\chi$)** | -294 | **-281** | Improved boundary loop regularity |
| **Mean Reprojection Error** | 0.544 px | **0.546 px** | Sub-pixel accuracy preserved across 28 cameras |

---

## 2. Spatial Extents & Symmetry

The bounding box extents in COLMAP reconstruction units:
$$\mathbf{X} \in [-3.242, -0.570] \implies \Delta X = 2.6719$$
$$\mathbf{Y} \in [-3.978, +3.371] \implies \Delta Y = 7.3492 \quad (\text{Vertical height})$$
$$\mathbf{Z} \in [6.185, 8.857] \implies \Delta Z = 2.6725 \quad (\text{Depth thickness})$$

- **Effective Diameter:** $D_{avg} = \frac{2.6719 + 2.6725}{2} = 2.6722$
- **3D Aspect Ratio ($H : D$):** $\frac{7.3492}{2.6722} = \mathbf{2.7502 \approx 2.75 : 1}$
- **Cross-Sectional Circularity ($Z / X$):** $\frac{2.6725}{2.6719} = \mathbf{1.0002}$ (Sub-0.02% cylindrical deviation).

---

## 3. Visual Artifacts Generated

- Multi-View Orthographic & Perspective Visualizations: [final_reconstruction_multiview.png](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/final_reconstruction_multiview.png)
- Mesh Topology & Boundary Edge Analysis: [mesh_topology_and_boundary_analysis.png](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/mesh_topology_and_boundary_analysis.png)
- Video Reprojection Comparison: [reprojection_overlay_comparison.png](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/reprojection_overlay_comparison.png)
- Segmentation Mask Visualizations: [segmentation_mask_visualization.jpg](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/segmentation_mask_visualization.jpg)
- Depth Alignment Visualizations: [depth_alignment_visualization.jpg](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/depth_alignment_visualization.jpg)
- Dense Object Point Cloud (.ply): [final_object_only_cloud.ply](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/final_object_only_cloud.ply)
- Reconstructed Mesh (.obj): [final_reconstructed_mesh.obj](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/final_reconstructed_mesh.obj)
- Reconstructed Mesh (.ply): [final_reconstructed_mesh.ply](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/final_reconstructed_mesh.ply)
- Numerical Metrics Data: [audit_metrics.json](file:///d:/7thsem/3DCV_proj/outputs/deployment_live_validation/audit_metrics.json)
