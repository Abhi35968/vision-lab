"""
Reconstruction Visualizer Module (Phase 5.5).
Generates publication-quality diagnostic visualizations for Object-Specific 3D Reconstruction:
1. 3D bottle mesh multi-angle rendering.
2. 3D mesh with sparse geometric inlier points overlaid.
3. 3D scene showing registered camera poses and reconstructed bottle model.
4. Contact sheet comparing observed masks vs projected model silhouettes (best, median, worst).
5. 2D axial bottle profile cross-section with annotated anatomical zones.
"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from src.object_reconstruction.bottle_models import (
    BottleParameters,
    BottleProfileSample,
    Mesh3D,
    SilhouetteViewMetric,
)


class ReconstructionVisualizer:
    """Generates visual figures for Phase 5.5 reconstruction."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_bottle_mesh_3d(
        self,
        mesh: Mesh3D,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Renders shaded 3D mesh from isometric, front, and top viewpoints."""
        if save_path is None:
            save_path = self.output_dir / "bottle_mesh_3d.png"

        fig = plt.figure(figsize=(15, 6))

        views = [
            ("Isometric View (30°, 45°)", 30, 45),
            ("Front View (0°, 90°)", 0, 90),
            ("Top View (90°, -90°)", 85, -90),
        ]

        # Subsample faces slightly for fast, clean rendering
        step = max(1, mesh.num_faces // 1200)
        sub_faces = mesh.faces[::step]
        triangles = mesh.vertices[sub_faces]

        for idx, (title, elev, azim) in enumerate(views, 1):
            ax = fig.add_subplot(1, 3, idx, projection="3d")
            poly = Poly3DCollection(triangles, alpha=0.75, edgecolor="#2b5c8f", linewidths=0.3)
            poly.set_facecolor("#6baed6")
            ax.add_collection3d(poly)

            # Set bounds
            v = mesh.vertices
            ax.set_xlim(v[:, 0].min() - 0.5, v[:, 0].max() + 0.5)
            ax.set_ylim(v[:, 1].min() - 0.5, v[:, 1].max() + 0.5)
            ax.set_zlim(v[:, 2].min() - 0.5, v[:, 2].max() + 0.5)

            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("X (SfM units)", fontsize=8)
            ax.set_ylabel("Y (SfM units)", fontsize=8)
            ax.set_zlabel("Z (SfM units)", fontsize=8)

        plt.suptitle(
            f"Parametric Surface-of-Revolution Bottle Mesh (Vertices: {mesh.num_vertices:,}, Faces: {mesh.num_faces:,})",
            fontsize=13, fontweight="bold", y=0.98
        )
        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_mesh_with_sparse_points(
        self,
        mesh: Mesh3D,
        high_conf_points: np.ndarray,
        mod_conf_points: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Renders 3D bottle mesh overlaid with high- and moderate-confidence sparse points."""
        if save_path is None:
            save_path = self.output_dir / "mesh_with_sparse_points.png"

        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Subsampled mesh wireframe/surface
        step = max(1, mesh.num_faces // 1000)
        sub_faces = mesh.faces[::step]
        triangles = mesh.vertices[sub_faces]
        poly = Poly3DCollection(triangles, alpha=0.35, edgecolor="#4a7bb0", linewidths=0.2)
        poly.set_facecolor("#c6dbef")
        ax.add_collection3d(poly)

        # Plot High Confidence Points (Red)
        if len(high_conf_points) > 0:
            ax.scatter(
                high_conf_points[:, 0], high_conf_points[:, 1], high_conf_points[:, 2],
                c="#d95f02", s=60, marker="o", edgecolors="black", linewidths=0.8,
                label=f"High-Confidence Points ({len(high_conf_points)} pts, Cap Clustered)",
                zorder=10,
            )

        # Plot Moderate Confidence Points (Blue)
        if mod_conf_points is not None and len(mod_conf_points) > 0:
            ax.scatter(
                mod_conf_points[:, 0], mod_conf_points[:, 1], mod_conf_points[:, 2],
                c="#2b5c8f", s=40, marker="^", edgecolors="black", linewidths=0.6,
                label=f"Moderate-Confidence Points ({len(mod_conf_points)} pts)",
                zorder=8,
            )

        v = mesh.vertices
        ax.set_xlim(v[:, 0].min() - 0.8, v[:, 0].max() + 0.8)
        ax.set_ylim(v[:, 1].min() - 0.8, v[:, 1].max() + 0.8)
        ax.set_zlim(v[:, 2].min() - 0.8, v[:, 2].max() + 0.8)

        ax.view_init(elev=20, azim=45)
        ax.set_xlabel("X (SfM units)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y (SfM units)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Z (SfM units)", fontsize=10, fontweight="bold")
        ax.set_title(
            "Reconstructed Bottle Mesh with Sparse SfM Inlier Points Overlaid\n"
            "Demonstrates Spatial Concentration of Measured Geometry on Bottle Cap",
            fontsize=12, fontweight="bold", pad=15
        )
        ax.legend(loc="upper left", fontsize=10, frameon=True)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_camera_poses_and_bottle(
        self,
        mesh: Mesh3D,
        cameras: Dict[str, dict],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Renders complete 3D scene with camera centers, viewing directions, and bottle model."""
        if save_path is None:
            save_path = self.output_dir / "camera_poses_and_bottle.png"

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Bottle Model
        step = max(1, mesh.num_faces // 800)
        triangles = mesh.vertices[mesh.faces[::step]]
        poly = Poly3DCollection(triangles, alpha=0.60, edgecolor="#1f78b4", linewidths=0.3)
        poly.set_facecolor("#a6cee3")
        ax.add_collection3d(poly)

        # 2. Camera Centers and Trajectory
        reg_cams = [
            (int(cid), np.array(c["camera_center"]))
            for cid, c in cameras.items()
            if c.get("is_registered", False)
        ]
        reg_cams = sorted(reg_cams, key=lambda x: x[0])
        cam_centers = np.array([c[1] for c in reg_cams])

        # Plot camera trajectory path
        ax.plot(
            cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
            color="#33a02c", linestyle="-", linewidth=2.0, alpha=0.8, label="Camera Trajectory"
        )
        # Plot camera centers
        ax.scatter(
            cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
            c=np.arange(len(cam_centers)), cmap="plasma", s=50, edgecolors="black",
            label=f"Registered Cameras ({len(reg_cams)} views)"
        )

        # Plot camera optical viewing vectors
        for cid, c_pos in reg_cams[::3]:  # Every 3rd camera for clean visual
            c_data = cameras[str(cid)]
            R = np.array(c_data["rotation_matrix"])
            look_dir = R.T @ np.array([0.0, 0.0, 1.0])  # Optical axis in world
            look_dir /= np.linalg.norm(look_dir)
            ax.quiver(
                c_pos[0], c_pos[1], c_pos[2],
                look_dir[0] * 1.2, look_dir[1] * 1.2, look_dir[2] * 1.2,
                color="#e31a1c", alpha=0.7, arrow_length_ratio=0.25
            )

        ax.view_init(elev=25, azim=135)
        ax.set_xlabel("X (SfM units)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y (SfM units)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Z (SfM units)", fontsize=10, fontweight="bold")
        ax.set_title(
            f"3D Reconstruction Scene: Multi-View Camera Trajectory & Parametric Bottle Model\n"
            f"Illustrates Camera Orbit Convergence on Tabletop Object Centroid",
            fontsize=12, fontweight="bold", pad=15
        )
        ax.legend(loc="upper right", fontsize=10, frameon=True)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_silhouette_contact_sheet(
        self,
        representative_comparisons: List[dict],
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Renders a 3-row contact sheet (Best IoU, Median IoU, Worst IoU) comparing:
        Column 1: Observed Segmentation Mask (Green)
        Column 2: Rendered Model Silhouette (Magenta)
        Column 3: Color-Coded Overlay (White=Overlap, Red=FP, Blue=FN)
        """
        if save_path is None:
            save_path = self.output_dir / "silhouette_overlay_contact_sheet.png"

        n_rows = len(representative_comparisons)
        fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        for row_idx, item in enumerate(representative_comparisons):
            obs = item["observed_mask"]
            rend = item["rendered_mask"]
            title_prefix = item["label"]  # "Best IoU", "Median IoU", "Worst IoU"
            iou = item["iou"]
            cid = item["camera_id"]

            # Resize for clean display if large
            h_disp, w_disp = 480, 270
            obs_s = cv2.resize(obs, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST)
            rend_s = cv2.resize(rend, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST)

            # Col 1: Observed mask
            axes[row_idx, 0].imshow(obs_s, cmap="Greens")
            axes[row_idx, 0].set_title(f"{title_prefix} (Cam {cid}): Observed YOLO Mask", fontsize=10, fontweight="bold")
            axes[row_idx, 0].axis("off")

            # Col 2: Rendered mask
            axes[row_idx, 1].imshow(rend_s, cmap="Purples")
            axes[row_idx, 1].set_title(f"Projected Model Silhouette", fontsize=10, fontweight="bold")
            axes[row_idx, 1].axis("off")

            # Col 3: Overlap RGB composite
            # White = Intersection, Red = False Positive (Rendered only), Blue = False Negative (Observed only)
            comp = np.zeros((h_disp, w_disp, 3), dtype=np.uint8)
            inter = (rend_s > 0) & (obs_s > 0)
            fp = (rend_s > 0) & (obs_s == 0)
            fn = (rend_s == 0) & (obs_s > 0)

            comp[inter] = [240, 240, 240]  # White
            comp[fp] = [230, 40, 40]       # Red (FP)
            comp[fn] = [40, 100, 230]      # Blue (FN)

            axes[row_idx, 2].imshow(comp)
            axes[row_idx, 2].set_title(f"Overlay Error Map (IoU = {iou*100:.1f}%)", fontsize=10, fontweight="bold")
            axes[row_idx, 2].axis("off")

        plt.suptitle(
            "Silhouette Verification: Multi-View Observed Masks vs. Rendered Model Projections\n"
            "Overlay Key: White = Intersection (True Positive), Red = Model False Positive, Blue = Uncovered Mask (False Negative)",
            fontsize=12, fontweight="bold", y=0.99
        )
        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_bottle_profile_cross_section(
        self,
        profile_samples: List[BottleProfileSample],
        params: BottleParameters,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plots 2D axial profile curve r(z) with annotated anatomical zones."""
        if save_path is None:
            save_path = self.output_dir / "bottle_profile_cross_section.png"

        zs = np.array([p.z_axial for p in profile_samples])
        rs = np.array([p.radius for p in profile_samples])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.4, 1]})

        # Panel 1: Symmetrical Full Bottle Silhouette
        # Left boundary: -r(z), Right boundary: +r(z)
        ax1.plot(-rs, zs, color="#2b5c8f", linewidth=2.2)
        ax1.plot(rs, zs, color="#2b5c8f", linewidth=2.2)
        ax1.plot([-rs[0], rs[0]], [zs[0], zs[0]], color="#2b5c8f", linewidth=2.2)
        ax1.plot([-rs[-1], rs[-1]], [zs[-1], zs[-1]], color="#2b5c8f", linewidth=2.2)

        # Color-code regions
        H = params.total_height
        z_base = H * params.base_chamfer_z_ratio
        z_shoulder = H * params.shoulder_z_ratio
        z_neck = H * params.neck_z_ratio
        z_cap = H * params.cap_z_ratio

        # Fill regions symmetrically
        mask_base = zs <= z_base
        mask_body = (zs > z_base) & (zs <= z_shoulder)
        mask_shoulder = (zs > z_shoulder) & (zs <= z_neck)
        mask_neck = (zs > z_neck) & (zs <= z_cap)
        mask_cap = zs > z_cap

        ax1.fill_betweenx(zs[mask_base], -rs[mask_base], rs[mask_base], color="#ffff99", alpha=0.6, label="Base Zone")
        ax1.fill_betweenx(zs[mask_body], -rs[mask_body], rs[mask_body], color="#fdc086", alpha=0.6, label="Main Body (R_body)")
        ax1.fill_betweenx(zs[mask_shoulder], -rs[mask_shoulder], rs[mask_shoulder], color="#386cb0", alpha=0.5, label="Shoulder S-Curve")
        ax1.fill_betweenx(zs[mask_neck], -rs[mask_neck], rs[mask_neck], color="#7fc97f", alpha=0.6, label="Neck (R_neck)")
        ax1.fill_betweenx(zs[mask_cap], -rs[mask_cap], rs[mask_cap], color="#7570b3", alpha=0.6, label="Cap Ridge (R_cap)")

        # Center line
        ax1.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        ax1.set_xlabel("Radial Distance from Axis (Arbitrary SfM Units)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Axial Height z (Arbitrary SfM Units)", fontsize=11, fontweight="bold")
        ax1.set_title("Parametric Surface-of-Revolution Cross-Section", fontsize=12, fontweight="bold")
        ax1.legend(loc="lower left", fontsize=9, frameon=True)
        ax1.set_xlim(-params.body_radius * 1.5, params.body_radius * 1.5)

        # Panel 2: Radial Function Curve r(z)
        ax2.plot(zs, rs, color="#e7298a", linewidth=2.5, marker="o", markersize=3, label="Radial Profile r(z)")
        ax2.axvline(z_base, color="#ffff99", linestyle="--", linewidth=1.5, label="z_base")
        ax2.axvline(z_shoulder, color="#fdc086", linestyle="--", linewidth=1.5, label="z_shoulder")
        ax2.axvline(z_neck, color="#386cb0", linestyle="--", linewidth=1.5, label="z_neck")
        ax2.axvline(z_cap, color="#7570b3", linestyle="--", linewidth=1.5, label="z_cap")

        ax2.set_xlabel("Axial Coordinate z (0 = Base, H = Cap)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Radius r(z)", fontsize=11, fontweight="bold")
        ax2.set_title(f"Profile Function r(z) [Height={H:.2f}, R_body={params.body_radius:.2f}]", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=9, frameon=True)

        plt.suptitle("Phase 5.5 Parametric Bottle Geometric Profile Specification", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path
