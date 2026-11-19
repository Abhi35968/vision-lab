"""
Feasibility Visualizer Module
Generates diagnostic plots for Phase 5.4 Dense Object Reconstruction Feasibility:
1. Feature density across frames (inside vs outside mask).
2. Silhouette vertical feature distribution (cap, neck, body, base).
3. Pairwise matching quality & epipolar inlier rates (object vs background).
4. Camera orbit polar angular coverage and blind-spot visualization.
5. MVS method feasibility comparison scorecard.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.feasibility.feature_density_analyzer import FrameFeatureDensityRecord
from src.feasibility.matching_quality_analyzer import PairwiseMatchingRecord
from src.feasibility.dataset_limitations_analyzer import CameraOrbitRecord, DatasetLimitationsSummary
from src.feasibility.mvs_comparator import ReconstructionMethodEvaluation


class FeasibilityVisualizer:
    """Renders informative multi-panel visual diagnostics for Phase 5.4."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_feature_density_across_frames(
        self,
        records: List[FrameFeatureDensityRecord],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plots keypoint counts and area-normalized densities across all keyframes."""
        if save_path is None:
            save_path = self.output_dir / "feature_density_across_frames.png"

        frames = [r.frame_idx for r in records]
        kps_in = [r.inside_mask_keypoints for r in records]
        kps_out = [r.outside_mask_keypoints for r in records]
        d_in = [r.density_inside_per_10k_px2 for r in records]
        d_out = [r.density_outside_per_10k_px2 for r in records]
        registered = [r.is_registered for r in records]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        # Subplot 1: Raw Keypoint Counts
        ax1.plot(frames, kps_out, color="#2b5c8f", marker="o", markersize=4, label="Outside Mask (Background Scene)", linewidth=1.8)
        ax1.plot(frames, kps_in, color="#d95f02", marker="s", markersize=5, label="Inside Mask (Target Bottle)", linewidth=2.2)
        ax1.fill_between(frames, 0, kps_in, color="#d95f02", alpha=0.25)
        ax1.set_ylabel("Detected SIFT Keypoints", fontsize=11, fontweight="bold")
        ax1.set_title("Per-Frame SIFT Feature Distribution: Object Mask vs. Background", fontsize=13, fontweight="bold", pad=10)
        ax1.legend(loc="upper right", frameon=True, fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Mark unregistered frames
        for f, reg in zip(frames, registered):
            if not reg:
                ax1.axvspan(f - 0.4, f + 0.4, color="gray", alpha=0.3)
                ax2.axvspan(f - 0.4, f + 0.4, color="gray", alpha=0.3)

        # Subplot 2: Area-Normalized Feature Density
        ax2.plot(frames, d_out, color="#1b9e77", marker="^", markersize=4, label="Background Density (kps / 10k px²)", linewidth=1.8)
        ax2.plot(frames, d_in, color="#e7298a", marker="d", markersize=5, label="Bottle Density (kps / 10k px²)", linewidth=2.2)
        ax2.fill_between(frames, 0, d_in, color="#e7298a", alpha=0.25)
        ax2.set_xlabel("Keyframe Index", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Density (Keypoints / 10k px²)", fontsize=11, fontweight="bold")
        ax2.set_title("Spatial Feature Density (Area-Normalized): Severe Object Sparsity", fontsize=12, fontweight="bold", pad=8)
        ax2.legend(loc="upper right", frameon=True, fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_silhouette_feature_distribution(
        self,
        records: List[FrameFeatureDensityRecord],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plots the vertical breakdown of features across Cap, Neck, Body, and Base."""
        if save_path is None:
            save_path = self.output_dir / "silhouette_feature_distribution.png"

        frames = [r.frame_idx for r in records if r.mask_present]
        cap = [r.cap_keypoints for r in records if r.mask_present]
        neck = [r.neck_keypoints for r in records if r.mask_present]
        body = [r.body_keypoints for r in records if r.mask_present]
        base = [r.base_keypoints for r in records if r.mask_present]

        tot_cap = sum(cap)
        tot_neck = sum(neck)
        tot_body = sum(body)
        tot_base = sum(base)
        grand_total = tot_cap + tot_neck + tot_body + tot_base

        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1.2])

        # Panel 1: Stacked Bar Chart across frames
        ax1 = fig.add_subplot(gs[0])
        ax1.bar(frames, cap, label="Cap (Top 20% Height)", color="#7570b3", alpha=0.85)
        ax1.bar(frames, neck, bottom=cap, label="Neck (20-35% Height)", color="#386cb0", alpha=0.85)
        ax1.bar(frames, body, bottom=np.array(cap)+np.array(neck), label="Body/Label (35-80% Height)", color="#fdc086", alpha=0.85)
        ax1.bar(frames, base, bottom=np.array(cap)+np.array(neck)+np.array(body), label="Base (80-100% Height)", color="#ffff99", alpha=0.85)
        ax1.set_xlabel("Keyframe Index", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Keypoints Inside Mask", fontsize=11, fontweight="bold")
        ax1.set_title("Vertical Feature Distribution Across Silhouette", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9, frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Aggregate Pie Chart
        ax2 = fig.add_subplot(gs[1])
        colors = ["#7570b3", "#386cb0", "#fdc086", "#ffff99"]
        labels = [
            f"Cap ({tot_cap/grand_total*100:.1f}%)",
            f"Neck ({tot_neck/grand_total*100:.1f}%)",
            f"Body ({tot_body/grand_total*100:.1f}%)",
            f"Base ({tot_base/grand_total*100:.1f}%)",
        ] if grand_total > 0 else ["Cap", "Neck", "Body", "Base"]
        ax2.pie([tot_cap, tot_neck, tot_body, tot_base], labels=labels, colors=colors, startangle=140, autopct="%1.0f%%")
        ax2.set_title(f"Aggregate Silhouette\nTotal Mask KPs: {grand_total}", fontsize=11, fontweight="bold")

        # Panel 3: Schematic Bottle Silhouette Vertical Profile
        ax3 = fig.add_subplot(gs[2])
        # Draw schematic bottle outline
        t = np.linspace(0, 1, 200)
        # Width profile of a bottle
        w = np.where(t < 0.20, 0.25, np.where(t < 0.35, 0.20 + (t-0.20)*2.0, np.where(t < 0.85, 0.50, 0.48 - (t-0.85)*0.5)))
        ax3.plot(-w, t, color="black", linewidth=2)
        ax3.plot(w, t, color="black", linewidth=2)
        ax3.plot([-0.25, 0.25], [0, 0], color="black", linewidth=2)
        ax3.plot([-0.40, 0.40], [1, 1], color="black", linewidth=2)

        # Shading regions
        ax3.fill_betweenx([0.0, 0.20], [-0.25, -0.25], [0.25, 0.25], color="#7570b3", alpha=0.35, label=f"Cap ({tot_cap} kps)")
        ax3.fill_betweenx([0.20, 0.35], [-0.30, -0.45], [0.30, 0.45], color="#386cb0", alpha=0.35, label=f"Neck ({tot_neck} kps)")
        ax3.fill_betweenx([0.35, 0.80], [-0.50, -0.50], [0.50, 0.50], color="#fdc086", alpha=0.35, label=f"Body ({tot_body} kps)")
        ax3.fill_betweenx([0.80, 1.00], [-0.48, -0.40], [0.48, 0.40], color="#ffff99", alpha=0.35, label=f"Base ({tot_base} kps)")

        ax3.invert_yaxis()  # 0 at top, 1 at bottom
        ax3.set_xlim(-0.8, 0.8)
        ax3.set_ylim(1.05, -0.05)
        ax3.set_ylabel("Normalized Height (Top -> Base)", fontsize=10, fontweight="bold")
        ax3.set_title("Vertical Concentration", fontsize=11, fontweight="bold")
        ax3.legend(loc="center", fontsize=9, frameon=True)
        ax3.grid(False)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_matching_quality_comparison(
        self,
        records: List[PairwiseMatchingRecord],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Compares Lowe's ratio test pass rates and epipolar inlier rates for Object vs Background."""
        if save_path is None:
            save_path = self.output_dir / "matching_quality_comparison.png"

        valid = [r for r in records if r.outside_raw_matches > 0]
        pair_labels = [f"({r.frame_idx1},{r.frame_idx2})" for r in valid]
        x = np.arange(len(valid))

        in_ratio_pct = [r.inside_ratio_pass_pct for r in valid]
        out_ratio_pct = [r.outside_ratio_pass_pct for r in valid]

        in_inl_pct = [r.inside_epipolar_inlier_pct for r in valid]
        out_inl_pct = [r.outside_epipolar_inlier_pct for r in valid]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        width = 0.38
        # Plot 1: Lowe Ratio Pass Percentage
        ax1.bar(x - width/2, out_ratio_pct, width=width, label="Outside Mask (Background)", color="#2b5c8f", alpha=0.85)
        ax1.bar(x + width/2, in_ratio_pct, width=width, label="Inside Mask (Target Bottle)", color="#d95f02", alpha=0.85)
        ax1.set_ylabel("Lowe's Ratio Pass (%)", fontsize=11, fontweight="bold")
        ax1.set_title("Lowe's Ratio Test (Threshold = 0.75): Descriptor Distinctiveness", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=10, frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Plot 2: Epipolar RANSAC Inlier Rate
        ax2.bar(x - width/2, out_inl_pct, width=width, label="Outside Epipolar Inlier Rate (%)", color="#1b9e77", alpha=0.85)
        ax2.bar(x + width/2, in_inl_pct, width=width, label="Inside Epipolar Inlier Rate (%)", color="#e7298a", alpha=0.85)
        ax2.set_xlabel("Adjacent Registered Camera Pair", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Epipolar Inliers (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Epipolar Verification (Fundamental Matrix RANSAC, 3.0 px)", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=9)
        ax2.legend(loc="upper right", fontsize=10, frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_camera_orbit_angular_coverage(
        self,
        records: List[CameraOrbitRecord],
        summary: DatasetLimitationsSummary,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Renders polar plot illustrating camera orbit angular span vs. unobserved blind spot."""
        if save_path is None:
            save_path = self.output_dir / "camera_orbit_angular_coverage.png"

        az_deg = [r.azimuth_deg_raw for r in records]
        dists = [r.distance_to_object for r in records]

        # Convert to polar coordinates (radians)
        theta = np.radians(az_deg)
        r = np.array(dists)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="polar")

        # Plot camera positions
        scatter = ax.scatter(theta, r, c=np.arange(len(records)), cmap="plasma", s=80, edgecolors="black", zorder=5, label="Registered Cameras")

        # Connect camera trajectory in temporal sequence
        ax.plot(theta, r, color="#4daf4a", linestyle="-", linewidth=2, alpha=0.7, zorder=4, label="Camera Flight Path")

        # Shade the observed arc
        min_rad = np.radians(summary.orbit_min_azimuth_deg)
        max_rad = np.radians(summary.orbit_max_azimuth_deg)
        # Highlight observed sector
        # Unwrapped azimuths can be drawn as a wedge
        theta_grid = np.linspace(min_rad, max_rad, 100)
        ax.fill_between(theta_grid, 0, max(r) * 1.15, color="#2b5c8f", alpha=0.15, label=f"Observed Wedge ({summary.orbit_angular_span_deg:.1f}° / {summary.orbit_coverage_pct_of_360:.1f}%)")

        # Mark object centroid at center
        ax.plot(0, 0, marker="*", markersize=18, color="red", label="Target Bottle Centroid", zorder=10)

        # Add text box for blind spot
        ax.set_title(
            f"Camera Orbit Angular Coverage Around Target Object\n"
            f"Observed Arc: {summary.orbit_angular_span_deg:.1f}° | Unobserved Blind Spot: {summary.unobserved_angular_blindspot_deg:.1f}° ({100-summary.orbit_coverage_pct_of_360:.1f}%)",
            fontsize=12, fontweight="bold", pad=20
        )
        ax.legend(loc="lower left", bbox_to_anchor=(0.85, -0.05), fontsize=9, frameon=True)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_mvs_feasibility_matrix(
        self,
        methods: List[ReconstructionMethodEvaluation],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Renders visual scorecard heatmap comparing the 6 reconstruction paradigms."""
        if save_path is None:
            save_path = self.output_dir / "mvs_feasibility_matrix.png"

        # Define evaluation dimensions
        criteria = [
            "Win 11 Native",
            "CPU Feasible",
            "RAM <= 16GB",
            "Handles Transp.",
            "SfM Pose Ready",
            "Dataset Viable",
        ]

        method_names = [m.method_name for m in methods]

        # Score mapping: 2 = Green (Full), 1 = Yellow (Degraded/Conditional), 0 = Red (Infeasible)
        scores = []
        for m in methods:
            row = []
            # Win 11
            row.append(2 if "Supported" in m.windows_compatibility or "100%" in m.windows_compatibility else 1)
            # CPU
            row.append(2 if "CPU-only" in m.hardware_requirement or "CPU-compatible" in m.hardware_requirement else 0)
            # RAM
            row.append(2 if "< 1 GB" in m.max_ram_footprint or "4 - 8" in m.max_ram_footprint or "1 - 3" in m.max_ram_footprint else (1 if "8 - 14" in m.max_ram_footprint else 0))
            # Transparency
            row.append(2 if m.handles_transparency_and_refraction else 0)
            # SfM pose
            row.append(2 if "Directly" in m.sfm_pose_compatibility or "Native" in m.sfm_pose_compatibility else 1)
            # Dataset viable
            row.append(2 if "FEASIBLE" in m.dataset_feasibility_verdict else (1 if "DEGRADED" in m.dataset_feasibility_verdict else 0))
            scores.append(row)

        score_arr = np.array(scores)

        fig, ax = plt.subplots(figsize=(13, 7))
        cmap = matplotlib.colors.ListedColormap(["#d73027", "#fee08b", "#1a9850"])
        cax = ax.imshow(score_arr, cmap=cmap, vmin=0, vmax=2, aspect="auto")

        ax.set_xticks(np.arange(len(criteria)))
        ax.set_yticks(np.arange(len(method_names)))
        ax.set_xticklabels(criteria, fontsize=11, fontweight="bold")
        ax.set_yticklabels(method_names, fontsize=11, fontweight="bold")

        # Annotate text
        text_labels = {0: "FAIL / NO", 1: "PARTIAL", 2: "PASS / YES"}
        for i in range(len(method_names)):
            for j in range(len(criteria)):
                val = score_arr[i, j]
                ax.text(j, i, text_labels[val], ha="center", va="center", color="black", fontweight="bold", fontsize=10)

        ax.set_title("Phase 5.4 3D Reconstruction Paradigm Feasibility Scorecard", fontsize=13, fontweight="bold", pad=15)

        # Legend
        cbar = fig.colorbar(cax, ticks=[0.33, 1.0, 1.66], fraction=0.03, pad=0.04)
        cbar.ax.set_yticklabels(["Infeasible", "Degraded / Heavy", "Compatible"], fontsize=10, fontweight="bold")

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

