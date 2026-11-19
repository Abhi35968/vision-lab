"""
Object Association Visualizer Module
Produces publication-grade visual diagnostics for Phase 5.3.2:
1. 3D point cloud & camera trajectory visualization with color-coded classification.
2. Multi-view statistical diagnostic plots (Mask Support Ratio, Supporting Views, Reprojection Errors).
3. 2D keyframe diagnostic overlays showing observed feature points inside vs outside masks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from src.association.multiview_mask_associator import PointMaskAssociation


class ObjectAssociationVisualizer:
    """Renderer for 3D spatial associations and 2D multi-view evidence diagnostics."""

    @classmethod
    def plot_3d_scene(
        cls,
        associations: List[PointMaskAssociation],
        camera_centers: Dict[int, np.ndarray],
        output_path: Union[str, Path],
    ):
        """
        Generates a 3D scatter plot of camera poses and classified point clouds.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Partition points
        strong_pts = [a.point_3d for a in associations if a.classification == "STRONG_OBJECT_CANDIDATE"]
        weak_pts = [a.point_3d for a in associations if a.classification == "WEAK_OBJECT_CANDIDATE"]
        ambig_pts = [a.point_3d for a in associations if a.classification == "AMBIGUOUS"]
        bg_pts = [a.point_3d for a in associations if a.classification == "BACKGROUND_CANDIDATE"]

        # Plot background points (subsample if large to keep visual clear)
        if bg_pts:
            bg_arr = np.array(bg_pts)
            ax.scatter(
                bg_arr[:, 0], bg_arr[:, 2], -bg_arr[:, 1],
                c="lightgrey", s=10, alpha=0.35, label=f"Background ({len(bg_pts)} pts)"
            )

        # Plot ambiguous points
        if ambig_pts:
            amb_arr = np.array(ambig_pts)
            ax.scatter(
                amb_arr[:, 0], amb_arr[:, 2], -amb_arr[:, 1],
                c="orange", s=25, alpha=0.7, marker="^", label=f"Ambiguous ({len(ambig_pts)} pts)"
            )

        # Plot weak candidates
        if weak_pts:
            wk_arr = np.array(weak_pts)
            ax.scatter(
                wk_arr[:, 0], wk_arr[:, 2], -wk_arr[:, 1],
                c="deepskyblue", s=35, alpha=0.8, marker="s", label=f"Weak Object Candidate ({len(weak_pts)} pts)"
            )

        # Plot strong candidates
        if strong_pts:
            st_arr = np.array(strong_pts)
            ax.scatter(
                st_arr[:, 0], st_arr[:, 2], -st_arr[:, 1],
                c="limegreen", s=55, alpha=0.95, edgecolors="darkgreen",
                marker="o", label=f"Strong Object Candidate ({len(strong_pts)} pts)"
            )

        # Plot camera trajectory
        if camera_centers:
            sorted_cams = sorted(camera_centers.items(), key=lambda x: x[0])
            cam_coords = np.array([c[1] for c in sorted_cams])
            ax.plot(
                cam_coords[:, 0], cam_coords[:, 2], -cam_coords[:, 1],
                color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Camera Trajectory"
            )
            ax.scatter(
                cam_coords[:, 0], cam_coords[:, 2], -cam_coords[:, 1],
                color="crimson", s=30, marker="o"
            )
            # Label start and seed cams
            for cid, c_center in sorted_cams:
                if cid in [0, 24, 25, sorted_cams[-1][0]]:
                    ax.text(
                        c_center[0], c_center[2], -c_center[1],
                        f" Cam {cid}", color="darkred", fontsize=8, fontweight="bold"
                    )

        ax.set_xlabel("X (arbitrary units)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Z / Depth (arbitrary units)", fontsize=11, fontweight="bold")
        ax.set_zlabel("-Y / Elevation (arbitrary units)", fontsize=11, fontweight="bold")
        ax.set_title("3D Scene & Object Association Point Cloud", fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="upper right", framealpha=0.9)
        plt.tight_layout()

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), dpi=300)
        plt.close()
        print(f"[INFO] 3D scene point cloud plot saved to: {out_p}")

    @classmethod
    def plot_diagnostic_distributions(
        cls,
        associations: List[PointMaskAssociation],
        output_dir: Union[str, Path],
    ):
        """
        Generates 3 statistical distribution diagnostic plots:
        1. mask_support_ratio_distribution.png
        2. supporting_views_histogram.png
        3. reprojection_error_distribution.png
        """
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Plot 1: Mask Support Ratio Distribution
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        ratios = [a.mask_support_ratio for a in associations]
        non_zero_ratios = [r for r in ratios if r > 0]

        bins = np.linspace(0.0, 1.0, 21)
        ax.hist(ratios, bins=bins, color="steelblue", edgecolor="black", alpha=0.7, label=f"All Points (N={len(ratios)})")
        
        # Overlay vertical threshold lines
        ax.axvline(0.60, color="green", linestyle="--", linewidth=2, label="Strong Threshold (ratio >= 0.60)")
        ax.axvline(0.35, color="orange", linestyle=":", linewidth=2, label="Weak Threshold (ratio >= 0.35)")

        ax.set_xlabel("Mask Support Ratio (inside / masked_observations)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of 3D Points (Log Scale)", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_title("Distribution of Mask-Support Ratios Across Reconstructed 3D Points", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        p1_path = plots_dir / "mask_support_ratio_distribution.png"
        plt.savefig(str(p1_path), dpi=300)
        plt.close()
        print(f"[INFO] Saved: {p1_path}")

        # -------------------------------------------------------------
        # Plot 2: Supporting Views Histogram
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        view_counts = [len(a.supporting_camera_ids) for a in associations]
        max_views = max(view_counts) if view_counts else 0

        # Bar plot
        bins_v = np.arange(-0.5, max_views + 1.5, 1.0)
        counts, edges = np.histogram(view_counts, bins=bins_v)
        centers = 0.5 * (edges[:-1] + edges[1:])

        bars = ax.bar(centers, counts, width=0.8, color="teal", edgecolor="black", alpha=0.8)
        # Highlight >= 3 views
        for i, bar in enumerate(bars):
            if int(centers[i]) >= 3:
                bar.set_color("forestgreen")
            elif int(centers[i]) >= 1:
                bar.set_color("darkorange")
            else:
                bar.set_color("grey")

        ax.set_xlabel("Number of Supporting Camera Views (Observed Inside Bottle Mask)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Point Count (Log Scale)", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xticks(range(0, max_views + 1))
        ax.set_title("Supporting Views per Reconstructed 3D Point", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="forestgreen", edgecolor="black", label=">= 3 Views (Strong Object Criteria)"),
            Patch(facecolor="darkorange", edgecolor="black", label="1-2 Views (Weak / Edge Candidates)"),
            Patch(facecolor="grey", edgecolor="black", label="0 Views (Strict Background)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        plt.tight_layout()

        p2_path = plots_dir / "supporting_views_histogram.png"
        plt.savefig(str(p2_path), dpi=300)
        plt.close()
        print(f"[INFO] Saved: {p2_path}")

        # -------------------------------------------------------------
        # Plot 3: Reprojection Error Distribution (Candidate vs Background)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        cand_errs = [a.mean_reprojection_error_px for a in associations if a.classification in ["STRONG_OBJECT_CANDIDATE", "WEAK_OBJECT_CANDIDATE"]]
        bg_errs = [a.mean_reprojection_error_px for a in associations if a.classification == "BACKGROUND_CANDIDATE"]

        # Clip for visualization (exclude extreme outliers for clear histogram)
        c_clipped = np.clip(cand_errs, 0.0, 10.0) if cand_errs else []
        b_clipped = np.clip(bg_errs, 0.0, 10.0) if bg_errs else []

        bins_e = np.linspace(0.0, 10.0, 25)
        ax.hist(b_clipped, bins=bins_e, color="grey", alpha=0.5, density=True, label=f"Background Points (N={len(bg_errs)})")
        if len(c_clipped) > 0:
            ax.hist(c_clipped, bins=bins_e, color="forestgreen", alpha=0.75, density=True, label=f"Object Candidates (N={len(cand_errs)})")

        ax.axvline(4.0, color="red", linestyle="--", linewidth=1.5, label="SfM Outlier Threshold (4.0 px)")
        ax.set_xlabel("Mean Reprojection Error (pixels)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized Probability Density", fontsize=12, fontweight="bold")
        ax.set_title("Reprojection Error: Object Candidates vs. Background Points", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        p3_path = plots_dir / "reprojection_error_distribution.png"
        plt.savefig(str(p3_path), dpi=300)
        plt.close()
        print(f"[INFO] Saved: {p3_path}")

    @classmethod
    def render_2d_diagnostic_keyframes(
        cls,
        associations: List[PointMaskAssociation],
        keyframes_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        output_dir: Union[str, Path],
        target_camera_ids: List[int] = [0, 12, 24, 25],
    ):
        """
        Renders 2D keyframe diagnostic images showing the bottle mask and observed feature points.
        """
        diag_dir = Path(output_dir) / "2d_diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        kf_path = Path(keyframes_dir)
        m_path = Path(masks_dir)

        # Build lookup table of observations per camera
        cams_obs: Dict[int, List[Tuple[float, float, bool, float]]] = {}
        for a in associations:
            is_strong = (a.classification == "STRONG_OBJECT_CANDIDATE")
            for obs in a.observations:
                cid = obs.camera_id
                if obs.is_valid_projection:
                    cams_obs.setdefault(cid, []).append((
                        obs.obs_coords[0], obs.obs_coords[1],
                        obs.is_obs_inside_mask,
                        obs.reprojection_error_px or 0.0
                    ))

        for cid in target_camera_ids:
            # Find image
            matches = list(kf_path.glob(f"keyframe_{cid:04d}_*.png"))
            if not matches:
                continue
            img_file = matches[0]
            mask_file = m_path / f"{img_file.stem}_mask.png"

            img = cv2.imread(str(img_file))
            if img is None:
                continue

            h, w = img.shape[:2]
            canvas = img.copy()

            # Overlay mask
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    if np.any(mask > 0):
                        overlay = canvas.copy()
                        overlay[mask > 0] = (255, 180, 0)
                        canvas = cv2.addWeighted(canvas, 0.65, overlay, 0.35, 0)
                        contours, _ = cv2.findContours(
                            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(canvas, contours, -1, (255, 120, 0), thickness=3)

            # Plot feature observations
            obs_list = cams_obs.get(cid, [])
            n_in = sum(1 for o in obs_list if o[2])
            n_out = len(obs_list) - n_in

            for u, v, is_inside, err in obs_list:
                px = int(round(u))
                py = int(round(v))
                if not (0 <= px < w and 0 <= py < h):
                    continue

                if is_inside:
                    # Green circle
                    cv2.circle(canvas, (px, py), 6, (0, 230, 0), thickness=-1)
                    cv2.circle(canvas, (px, py), 8, (0, 100, 0), thickness=2)
                else:
                    # Red cross
                    cv2.drawMarker(canvas, (px, py), (0, 0, 220), markerType=cv2.MARKER_TILTED_CROSS, markerSize=7, thickness=2)

            # Header Banner
            cv2.rectangle(canvas, (20, 20), (700, 90), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                canvas, f"Cam {cid:02d}: {img_file.name}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )
            stats_str = f"Inside Mask: {n_in} pts (Green) | Outside Mask: {n_out} pts (Red)"
            cv2.putText(
                canvas, stats_str, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2, cv2.LINE_AA
            )

            out_save = diag_dir / f"diag_cam_{cid:02d}_{img_file.stem}.png"
            cv2.imwrite(str(out_save), canvas)
            print(f"[INFO] 2D diagnostic image saved: {out_save}")
