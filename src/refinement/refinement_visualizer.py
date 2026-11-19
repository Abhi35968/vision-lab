"""
Refinement Visualizer Module (Phase 5.3.3)
Generates high-resolution diagnostic figures visualising 3D scene geometry,
camera trajectories, reprojection error percentiles, triangulation parallax, and spatial clustering.
All 3D visualisations maintain unified coordinate axes and consistent world scale.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.refinement.refinement_models import PointGeometricProfile


class RefinementVisualizer:
    """Renders comprehensive multi-panel 3D and statistical diagnostic visualizations."""

    COLOR_MAP = {
        "HIGH_GEOMETRIC_CONFIDENCE": "#32CD32",      # Lime Green
        "MODERATE_GEOMETRIC_CONFIDENCE": "#00D2FF",  # Cyan
        "LOW_GEOMETRIC_CONFIDENCE": "#FFC800",       # Gold / Yellow
        "GEOMETRIC_OUTLIER": "#DC143C",              # Crimson Red
        "INSUFFICIENT_GEOMETRIC_EVIDENCE": "#A020F0",# Purple
    }

    ORIG_COLOR_MAP = {
        "STRONG_OBJECT_CANDIDATE": "#2E7D32",        # Dark Green
        "WEAK_OBJECT_CANDIDATE": "#0288D1",          # Blue
        "AMBIGUOUS": "#FF8F00",                      # Amber Orange
    }

    @classmethod
    def plot_3d_scene_comparison(
        cls,
        all_sfm_points: np.ndarray,
        camera_centers: Dict[int, np.ndarray],
        profiles: List[PointGeometricProfile],
        output_path: Union[str, Path],
    ):
        """
        Renders side-by-side 3D views comparing the original reconstruction with refined candidates.
        """
        fig = plt.figure(figsize=(18, 8), dpi=150)

        # ---------------- Left Panel: Original Baseline ----------------
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.set_title("Original 3D SfM Scene & Candidate Points (Phase 5.3.2)", fontsize=13, fontweight="bold")

        # Full SfM background points
        if len(all_sfm_points) > 0:
            ax1.scatter(
                all_sfm_points[:, 0], all_sfm_points[:, 1], all_sfm_points[:, 2],
                c="#B0BEC5", s=1.5, alpha=0.25, label="Background Points (2,077)", rasterized=True
            )

        # Camera trajectory
        if camera_centers:
            cam_pts = np.array(list(camera_centers.values()))
            sorted_cams = np.array([camera_centers[cid] for cid in sorted(camera_centers.keys())])
            ax1.plot(sorted_cams[:, 0], sorted_cams[:, 1], sorted_cams[:, 2], color="#1565C0", linestyle="--", linewidth=1.2, alpha=0.7)
            ax1.scatter(cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2], c="#1E88E5", s=25, marker="^", label="Camera Poses (32)")

        # Original candidates
        for orig_cat, color in cls.ORIG_COLOR_MAP.items():
            cat_pts = np.array([p.point_3d for p in profiles if p.original_classification == orig_cat])
            if len(cat_pts) > 0:
                ax1.scatter(cat_pts[:, 0], cat_pts[:, 1], cat_pts[:, 2], c=color, s=28, edgecolors="k", linewidth=0.4, label=f"Original {orig_cat.replace('_', ' ').title()} ({len(cat_pts)})")

        ax1.set_xlabel("X (arbitrary units)", fontsize=10)
        ax1.set_ylabel("Y (arbitrary units)", fontsize=10)
        ax1.set_zlabel("Z (arbitrary units)", fontsize=10)
        ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax1.view_init(elev=22, azim=-65)

        # ---------------- Right Panel: Refined Geometric Tiers ----------------
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.set_title("Refined Object Points by Confidence Tier (Phase 5.3.3)", fontsize=13, fontweight="bold")

        # Camera trajectory on right panel
        if camera_centers:
            ax2.plot(sorted_cams[:, 0], sorted_cams[:, 1], sorted_cams[:, 2], color="#1565C0", linestyle="--", linewidth=1.2, alpha=0.7)
            ax2.scatter(cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2], c="#1E88E5", s=25, marker="^", label="Camera Poses")

        # Plot each refined tier
        tier_names = [
            "HIGH_GEOMETRIC_CONFIDENCE",
            "MODERATE_GEOMETRIC_CONFIDENCE",
            "LOW_GEOMETRIC_CONFIDENCE",
            "GEOMETRIC_OUTLIER",
        ]

        for tier in tier_names:
            pts_tier = np.array([p.point_3d for p in profiles if p.refined_status == tier])
            color = cls.COLOR_MAP.get(tier, "#9E9E9E")
            label_text = f"{tier.replace('_', ' ').title()} ({len(pts_tier)})"
            if len(pts_tier) > 0:
                marker = "o" if tier != "GEOMETRIC_OUTLIER" else "x"
                ax2.scatter(
                    pts_tier[:, 0], pts_tier[:, 1], pts_tier[:, 2],
                    c=color, s=35, marker=marker, edgecolors="k" if marker == "o" else None, linewidth=0.5, label=label_text
                )

        ax2.set_xlabel("X (arbitrary units)", fontsize=10)
        ax2.set_ylabel("Y (arbitrary units)", fontsize=10)
        ax2.set_zlabel("Z (arbitrary units)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax2.view_init(elev=22, azim=-65)

        # Synchronize axis limits for identical coordinate frame and scale
        if len(all_sfm_points) > 0:
            x_min, x_max = np.percentile(all_sfm_points[:, 0], [1, 99])
            y_min, y_max = np.percentile(all_sfm_points[:, 1], [1, 99])
            z_min, z_max = np.percentile(all_sfm_points[:, 2], [1, 99])
            for ax in [ax1, ax2]:
                ax.set_xlim(x_min - 1.0, x_max + 1.0)
                ax.set_ylim(y_min - 1.0, y_max + 1.0)
                ax.set_zlim(z_min - 1.0, z_max + 1.0)

        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)
        print(f"[INFO] 3D scene comparison plot saved to: {out_p}")

    @classmethod
    def plot_reprojection_error_distributions(
        cls,
        profiles: List[PointGeometricProfile],
        output_path: Union[str, Path],
    ):
        """
        Renders boxplots and cumulative distributions of reprojection errors across confidence tiers.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        # Panel A: Boxplot across refined tiers
        tier_order = ["HIGH_GEOMETRIC_CONFIDENCE", "MODERATE_GEOMETRIC_CONFIDENCE", "LOW_GEOMETRIC_CONFIDENCE", "GEOMETRIC_OUTLIER"]
        box_data = []
        labels = []
        colors = []

        for tier in tier_order:
            errs = [p.median_reproj_error_px for p in profiles if p.refined_status == tier]
            if errs:
                box_data.append(errs)
                labels.append(f"{tier.split('_')[0]}\n(n={len(errs)})")
                colors.append(cls.COLOR_MAP.get(tier, "#9E9E9E"))

        ax1 = axes[0]
        if box_data:
            bp = ax1.boxplot(box_data, labels=labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
            for patch, col in zip(bp['boxes'], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)
            ax1.axhline(2.0, color="green", linestyle="--", linewidth=1, label="High Threshold (2.0 px)")
            ax1.axhline(4.0, color="orange", linestyle="--", linewidth=1, label="Diagnostic Cutoff (4.0 px)")
            ax1.set_ylabel("Median Reprojection Error (pixels)", fontsize=10)
            ax1.set_title("Median Reprojection Error by Tier", fontsize=11, fontweight="bold")
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(True, linestyle=":", alpha=0.6)

        # Panel B: CDF of errors
        ax2 = axes[1]
        all_errs = sorted([p.median_reproj_error_px for p in profiles])
        if all_errs:
            y_vals = np.linspace(0, 100, len(all_errs))
            ax2.plot(all_errs, y_vals, color="#1565C0", linewidth=2, label="Cumulative Error Distribution")
            ax2.axvline(2.0, color="green", linestyle="--", linewidth=1.2, label="2.0 px Target")
            ax2.axvline(4.0, color="orange", linestyle="--", linewidth=1.2, label="4.0 px Target")
            ax2.set_xlabel("Median Reprojection Error (pixels)", fontsize=10)
            ax2.set_ylabel("Cumulative Percentage (%)", fontsize=10)
            ax2.set_title("Cumulative Reprojection Error Distribution", fontsize=11, fontweight="bold")
            ax2.set_xlim(0, max(6.0, min(20.0, np.percentile(all_errs, 95) + 2.0)))
            ax2.set_ylim(0, 105)
            ax2.legend(loc="lower right", fontsize=8)
            ax2.grid(True, linestyle=":", alpha=0.6)

        # Panel C: Median vs Max Error Scatter
        ax3 = axes[2]
        for tier in tier_order:
            pts = [p for p in profiles if p.refined_status == tier]
            if pts:
                meds = [p.median_reproj_error_px for p in pts]
                maxs = [min(30.0, p.max_reproj_error_px) for p in pts]
                ax3.scatter(meds, maxs, c=cls.COLOR_MAP.get(tier, "#9E9E9E"), label=tier.split('_')[0], s=35, edgecolors="k", linewidth=0.5, alpha=0.85)

        ax3.plot([0, 15], [0, 15], color="grey", linestyle=":", label="1:1 Equality")
        ax3.set_xlabel("Median Reprojection Error (px)", fontsize=10)
        ax3.set_ylabel("Max Reprojection Error (px, capped at 30)", fontsize=10)
        ax3.set_title("Median vs. Max Reprojection Error", fontsize=11, fontweight="bold")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Reprojection error distribution plot saved to: {out_p}")

    @classmethod
    def plot_triangulation_and_depth(
        cls,
        profiles: List[PointGeometricProfile],
        output_path: Union[str, Path],
    ):
        """
        Renders parallax angle distributions and baseline-to-depth relationships.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Panel A: Max Angle vs Mask Support Ratio
        ax1 = axes[0]
        for tier in ["HIGH_GEOMETRIC_CONFIDENCE", "MODERATE_GEOMETRIC_CONFIDENCE", "LOW_GEOMETRIC_CONFIDENCE", "GEOMETRIC_OUTLIER"]:
            pts = [p for p in profiles if p.refined_status == tier]
            if pts:
                angles = [p.max_triangulation_angle_deg for p in pts]
                ratios = [p.original_mask_support_ratio for p in pts]
                ax1.scatter(angles, ratios, c=cls.COLOR_MAP.get(tier, "#9E9E9E"), label=tier.split('_')[0], s=40, edgecolors="k", linewidth=0.5, alpha=0.85)

        ax1.axvline(3.0, color="green", linestyle="--", linewidth=1.2, label="High Conf Parallax (3.0°)")
        ax1.axvline(1.8, color="cyan", linestyle="--", linewidth=1.2, label="Mod Conf Parallax (1.8°)")
        ax1.axhline(0.60, color="grey", linestyle=":", linewidth=1.0, label="Min Support Ratio (0.60)")
        ax1.set_xlabel("Maximum Triangulation Angle (degrees)", fontsize=10)
        ax1.set_ylabel("Mask Support Ratio", fontsize=10)
        ax1.set_title("Triangulation Parallax vs. Mask Support", fontsize=11, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Panel B: Parallax Angle Histogram
        ax2 = axes[1]
        all_angles = [p.max_triangulation_angle_deg for p in profiles]
        if all_angles:
            counts, bins, _ = ax2.hist(all_angles, bins=12, color="#0288D1", edgecolor="black", alpha=0.7)
            med_ang = float(np.median(all_angles))
            ax2.axvline(med_ang, color="red", linestyle="-", linewidth=1.8, label=f"Median Parallax ({med_ang:.1f}°)")
            ax2.axvline(3.0, color="green", linestyle="--", linewidth=1.2, label="High Threshold (3.0°)")
            ax2.set_xlabel("Maximum Triangulation Angle (degrees)", fontsize=10)
            ax2.set_ylabel("Point Count", fontsize=10)
            ax2.set_title("Triangulation Parallax Distribution", fontsize=11, fontweight="bold")
            ax2.legend(loc="upper right", fontsize=8)
            ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Triangulation angle distribution plot saved to: {out_p}")

    @classmethod
    def plot_spatial_clustering_analysis(
        cls,
        profiles: List[PointGeometricProfile],
        sensitivity_results: List[Dict[str, Any]],
        knn_stats: Dict[str, Any],
        output_path: Union[str, Path],
    ):
        """
        Renders spatial clustering elbow curves and parameter sensitivity charts.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        # Panel A: 2D Projection (X vs Y) of Spatial Clusters
        ax1 = axes[0]
        pts_x = [p.point_3d[0] for p in profiles]
        pts_y = [p.point_3d[1] for p in profiles]
        is_out = [p.is_spatial_outlier for p in profiles]

        for i, p in enumerate(profiles):
            color = "#DC143C" if p.is_spatial_outlier else "#00BCD4"
            marker = "x" if p.is_spatial_outlier else "o"
            ax1.scatter(pts_x[i], pts_y[i], c=color, s=40, marker=marker, edgecolors="k" if marker == "o" else None, linewidth=0.5)

        ax1.scatter([], [], c="#00BCD4", marker="o", edgecolors="k", label="Main Bottle Cluster")
        ax1.scatter([], [], c="#DC143C", marker="x", label="Spatial Outliers")
        ax1.set_xlabel("X (arbitrary units)", fontsize=10)
        ax1.set_ylabel("Y (arbitrary units)", fontsize=10)
        ax1.set_title("Top-Down Spatial Distribution (X vs Y)", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Panel B: Sorted k-NN distance curve
        ax2 = axes[1]
        sorted_d3 = sorted([p.nn_distance_3 for p in profiles])
        if sorted_d3:
            ax2.plot(range(len(sorted_d3)), sorted_d3, color="#7B1FA2", linewidth=2, label="3-NN Distance (k=3)")
            p75 = knn_stats.get("p75", 0.8)
            ax2.axhline(p75, color="orange", linestyle="--", linewidth=1.2, label=f"75th percentile ({p75:.2f})")
            ax2.axhline(2.0, color="red", linestyle=":", linewidth=1.2, label="Outlier Threshold (2.0)")
            ax2.set_xlabel("Point Index (Sorted by Distance)", fontsize=10)
            ax2.set_ylabel("Distance to 3rd Nearest Neighbor", fontsize=10)
            ax2.set_title("k-NN Distance Distribution (Elbow Curve)", fontsize=11, fontweight="bold")
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(True, linestyle=":", alpha=0.6)

        # Panel C: DBSCAN Sensitivity
        ax3 = axes[2]
        if sensitivity_results:
            eps_vals = [r["eps"] for r in sensitivity_results]
            n_clusters = [r["num_clusters"] for r in sensitivity_results]
            n_noise = [r["num_noise"] for r in sensitivity_results]

            x_idx = np.arange(len(eps_vals))
            width = 0.35
            ax3.bar(x_idx - width/2, n_clusters, width, label="Clusters Found", color="#1976D2")
            ax3.bar(x_idx + width/2, n_noise, width, label="Noise Points", color="#E53935")
            ax3.set_xticks(x_idx)
            ax3.set_xticklabels([f"ε={e:.2f}" for e in eps_vals], fontsize=8, rotation=30)
            ax3.set_ylabel("Count", fontsize=10)
            ax3.set_title("DBSCAN Sensitivity Analysis Across ε", fontsize=11, fontweight="bold")
            ax3.legend(loc="upper right", fontsize=8)
            ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Spatial clustering analysis plot saved to: {out_p}")

