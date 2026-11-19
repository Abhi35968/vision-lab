"""
Structure from Motion Visualization Module
Generates production-quality plots for camera trajectories, reprojection error distributions,
and 3D point cloud outlier classifications using Matplotlib.
All spatial dimensions and depths are strictly represented in 'arbitrary_units'.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Use headless backend to avoid GUI window popups and ensure headless compatibility
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.sfm.camera_registration import CameraPose
from src.sfm.feature_tracks import FeatureTrackSystem


class SfMVisualizer:
    """
    Renders and exports diagnostic plots for incremental multi-view reconstruction:
    1. Camera Trajectory and 3D Scene Geometry (arbitrary units)
    2. Reprojection Error Histogram and Per-Camera Residuals (pixels)
    3. 3D Point Cloud Classified by Reprojection Outlier Status
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/sfm/multi_view",
        secondary_output_dir: Optional[Union[str, Path]] = "outputs/visualizations",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.secondary_dir = Path(secondary_output_dir) if secondary_output_dir else None
        if self.secondary_dir:
            self.secondary_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, fig: plt.Figure, filename: str) -> List[Path]:
        """Saves a figure to the primary output directory and secondary directory if configured."""
        saved_paths = []
        primary_path = self.output_dir / filename
        fig.savefig(str(primary_path), dpi=150, bbox_inches="tight")
        saved_paths.append(primary_path)

        if self.secondary_dir:
            sec_path = self.secondary_dir / filename
            fig.savefig(str(sec_path), dpi=150, bbox_inches="tight")
            saved_paths.append(sec_path)

        plt.close(fig)
        return saved_paths

    def plot_camera_trajectory(
        self,
        registered_cameras: Dict[int, CameraPose],
        track_system: FeatureTrackSystem,
        filename: str = "camera_trajectory.png",
    ) -> List[Path]:
        """
        Plots 3D camera trajectory, viewing direction vectors, and reconstructed sparse point cloud.
        Units are explicitly arbitrary_units.
        """
        reg_cams = [cam for cam in sorted(registered_cameras.values(), key=lambda c: c.image_id) if cam.is_registered]
        tri_tracks = [t for t in track_system.tracks.values() if t.is_triangulated and t.point_3d is not None]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Plot sparse 3D point cloud
        if tri_tracks:
            pts = np.array([t.point_3d for t in tri_tracks])
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c="steelblue", s=8, alpha=0.4, label=f"3D Points ({len(pts)})"
            )

        # 2. Plot camera trajectory and optical viewing axes
        if reg_cams:
            cam_centers = np.array([c.camera_center for c in reg_cams])
            
            # Trajectory path line
            ax.plot(
                cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
                color="crimson", linestyle="--", linewidth=1.5, alpha=0.8, label="Camera Path"
            )

            # Camera centers
            ax.scatter(
                cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
                c="red", marker="^", s=70, edgecolors="black", label=f"Cameras ({len(reg_cams)})"
            )

            # Viewing direction arrows (principal optical axis d = R.T @ [0, 0, 1])
            arrow_scale = 0.25
            if len(cam_centers) > 1:
                # Scale arrows relative to trajectory spread
                spread = np.max(np.ptp(cam_centers, axis=0))
                if spread > 1e-3:
                    arrow_scale = spread * 0.15

            for cam in reg_cams:
                C = cam.camera_center
                # Optical forward direction in world coordinates: third row of R or R.T @ [0, 0, 1]
                R = cam.rotation_matrix
                view_dir = R.T @ np.array([0.0, 0.0, 1.0])
                view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8) * arrow_scale

                ax.quiver(
                    C[0], C[1], C[2],
                    view_dir[0], view_dir[1], view_dir[2],
                    color="darkorange", length=1.0, normalize=False, linewidth=1.2, arrow_length_ratio=0.3
                )
                ax.text(
                    C[0], C[1], C[2] + arrow_scale * 0.2,
                    f"C{cam.image_id}",
                    fontsize=8, fontweight="bold", color="darkred"
                )

        ax.set_title("Incremental SfM Camera Trajectory & Scene (arbitrary units)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("X (arbitrary_units)", labelpad=8)
        ax.set_ylabel("Y (arbitrary_units)", labelpad=8)
        ax.set_zlabel("Z (arbitrary_units)", labelpad=8)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

        return self._save_figure(fig, filename)

    def plot_reprojection_error_distribution(
        self,
        master_report: dict,
        camera_summaries: List[dict],
        filename: str = "reprojection_error_distribution.png",
    ) -> List[Path]:
        """
        Plots a dual-panel diagnostic visualization:
        1. Reprojection error histogram with thresholds at 1px, 2px, 4px, 10px.
        2. Per-camera mean and max reprojection error bar chart.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel 1: Error Distribution Percentages / Counts
        counts = master_report.get("error_distribution_counts", {})
        pcts = master_report.get("error_distribution_percentages", {})
        total_obs = master_report.get("total_observations", 0)

        categories = ["< 1 px\n(Sub-pixel)", "1 - 2 px\n(Good)", "2 - 4 px\n(Acceptable)", "> 4 px\n(Outliers)", "> 10 px\n(Severe)"]
        c1 = counts.get("below_1px", 0)
        c2 = max(0, counts.get("below_2px", 0) - c1)
        c4 = max(0, counts.get("below_4px", 0) - counts.get("below_2px", 0))
        c_gt4 = counts.get("above_4px", 0)
        c_gt10 = counts.get("above_10px", 0)

        vals = [c1, c2, c4, c_gt4, c_gt10]
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#8b0000"]

        bars = ax1.bar(categories, vals, color=colors, edgecolor="black", alpha=0.85, width=0.55)
        ax1.set_title(f"Reprojection Error Breakdown (Total Obs: {total_obs})", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Number of Observations", fontsize=10)
        ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

        # Annotate percentages above bars
        for bar, val in zip(bars, vals):
            pct = (val / max(1, total_obs)) * 100.0
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(1, total_obs * 0.015),
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

        ax1.set_ylim(0, max(vals + [10]) * 1.25)

        # Panel 2: Per-Camera Reprojection Error Bar Chart
        if camera_summaries:
            cam_labels = [f"C{c['camera_id']}\n{Path(c['image_name']).stem}" for c in camera_summaries]
            means = [c["mean_error_px"] for c in camera_summaries]
            maxs = [c["max_error_px"] for c in camera_summaries]
            x = np.arange(len(cam_labels))
            w = 0.35

            ax2.bar(x - w / 2, means, width=w, label="Mean Error (px)", color="#1f77b4", edgecolor="black", alpha=0.85)
            ax2.bar(x + w / 2, maxs, width=w, label="Max Error (px)", color="#ff7f0e", edgecolor="black", alpha=0.85)

            ax2.axhline(1.0, color="green", linestyle="--", linewidth=1.0, label="1 px (Sub-pixel)")
            ax2.axhline(4.0, color="red", linestyle="--", linewidth=1.2, label="4 px Outlier Gate")

            ax2.set_xticks(x)
            ax2.set_xticklabels(cam_labels, fontsize=8)
            ax2.set_ylabel("Reprojection Error (pixels)", fontsize=10)
            ax2.set_title("Per-Camera Error Statistics (pixels)", fontsize=11, fontweight="bold")
            ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
            ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
        else:
            ax2.text(0.5, 0.5, "No Registered Cameras", ha="center", va="center")

        plt.suptitle("Multi-View SfM Reprojection Error Diagnostics", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def plot_point_cloud_outliers(
        self,
        registered_cameras: Dict[int, CameraPose],
        track_system: FeatureTrackSystem,
        camera_matrix_k: np.ndarray,
        threshold_px: float = 4.0,
        filename: str = "point_cloud_outliers.png",
    ) -> List[Path]:
        """
        Plots 3D point cloud classified into inliers (reproj error <= 4px) and outliers (reproj error > 4px).
        All spatial coordinates are in arbitrary_units.
        """
        fx, fy = camera_matrix_k[0, 0], camera_matrix_k[1, 1]
        cx, cy = camera_matrix_k[0, 2], camera_matrix_k[1, 2]

        tri_tracks = [t for t in track_system.tracks.values() if t.is_triangulated and t.point_3d is not None]
        reg_cams = {cid: cam for cid, cam in registered_cameras.items() if cam.is_registered}

        inlier_pts = []
        outlier_pts = []

        for track in tri_tracks:
            X = track.point_3d
            errs = []
            for cid, obs in track.observations.items():
                if cid in reg_cams:
                    cam = reg_cams[cid]
                    X_c = cam.rotation_matrix @ X + cam.translation_vector.ravel()
                    z = X_c[2]
                    if z > 1e-4:
                        u = fx * (X_c[0] / z) + cx
                        v = fy * (X_c[1] / z) + cy
                        errs.append(np.sqrt((u - obs.coords[0]) ** 2 + (v - obs.coords[1]) ** 2))
            
            med_err = np.median(errs) if errs else 0.0
            if med_err <= threshold_px:
                inlier_pts.append(X)
            else:
                outlier_pts.append(X)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if inlier_pts:
            in_arr = np.array(inlier_pts)
            ax.scatter(
                in_arr[:, 0], in_arr[:, 1], in_arr[:, 2],
                c="mediumseagreen", s=10, alpha=0.6, label=f"Inliers <= {threshold_px}px ({len(in_arr)})"
            )

        if outlier_pts:
            out_arr = np.array(outlier_pts)
            ax.scatter(
                out_arr[:, 0], out_arr[:, 1], out_arr[:, 2],
                c="crimson", s=18, marker="x", alpha=0.8, label=f"Outliers > {threshold_px}px ({len(out_arr)})"
            )

        if reg_cams:
            cam_centers = np.array([c.camera_center for c in reg_cams.values()])
            ax.scatter(
                cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
                c="darkblue", marker="^", s=60, edgecolors="white", label=f"Cameras ({len(reg_cams)})"
            )

        ax.set_title(f"3D Point Cloud Inliers vs Reprojection Outliers (arbitrary units)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("X (arbitrary_units)", labelpad=8)
        ax.set_ylabel("Y (arbitrary_units)", labelpad=8)
        ax.set_zlabel("Z (arbitrary_units)", labelpad=8)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

        return self._save_figure(fig, filename)

    def plot_2d_reprojection_overlays(
        self,
        registered_cameras: Dict[int, CameraPose],
        track_system: FeatureTrackSystem,
        camera_matrix_k: np.ndarray,
        keyframe_dir: Optional[Union[str, Path]] = None,
        max_cameras_to_plot: int = 6,
        subsample_points: int = 150,
    ) -> List[Path]:
        """
        Renders 2D reprojection overlays on selected keyframe images.
        Shows observed 2D keypoints (green circles), reprojected 3D points (red crosses),
        and residual error lines (yellow).
        """
        import cv2

        reg_cams = [c for c in sorted(registered_cameras.values(), key=lambda c: c.image_id) if c.is_registered]
        if not reg_cams:
            return []

        # Sample up to max_cameras_to_plot evenly across the registered sequence
        step = max(1, len(reg_cams) // max_cameras_to_plot)
        sampled_cams = reg_cams[::step][:max_cameras_to_plot]

        fx, fy = camera_matrix_k[0, 0], camera_matrix_k[1, 1]
        cx, cy = camera_matrix_k[0, 2], camera_matrix_k[1, 2]

        overlay_dir = self.output_dir / "reprojection_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        kf_base = Path(keyframe_dir) if keyframe_dir else Path("outputs/real_video_validation/keyframes")
        if not kf_base.exists():
            kf_base = Path("data/extracted_frames")

        for cam in sampled_cams:
            img_path = kf_base / cam.image_name
            if not img_path.exists():
                matches = list(kf_base.glob(f"*{cam.image_name}*"))
                if matches:
                    img_path = matches[0]
                else:
                    continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            h, w = bgr.shape[:2]
            overlay = bgr.copy()

            # Gather visible 3D points for this camera
            obs_proj_pairs = []
            for track in track_system.tracks.values():
                if not track.is_triangulated or track.point_3d is None:
                    continue
                if cam.image_id in track.observations:
                    obs = track.observations[cam.image_id]
                    X = track.point_3d
                    X_c = cam.rotation_matrix @ X + cam.translation_vector.ravel()
                    z = X_c[2]
                    if z > 1e-4:
                        u = fx * (X_c[0] / z) + cx
                        v = fy * (X_c[1] / z) + cy
                        if 0 <= u < w and 0 <= v < h:
                            err = np.sqrt((u - obs.coords[0]) ** 2 + (v - obs.coords[1]) ** 2)
                            obs_proj_pairs.append((obs.coords, (u, v), err))

            if not obs_proj_pairs:
                continue

            # Subsample if too many
            if len(obs_proj_pairs) > subsample_points:
                sub_step = max(1, len(obs_proj_pairs) // subsample_points)
                obs_proj_pairs = obs_proj_pairs[::sub_step]

            # Draw residuals
            for (u_obs, v_obs), (u_proj, v_proj), err in obs_proj_pairs:
                pt_obs = (int(round(u_obs)), int(round(v_obs)))
                pt_proj = (int(round(u_proj)), int(round(v_proj)))

                # Yellow line for residual
                cv2.line(overlay, pt_obs, pt_proj, (0, 255, 255), 1, cv2.LINE_AA)
                # Green circle for observed keypoint
                cv2.circle(overlay, pt_obs, 4, (0, 255, 0), -1, cv2.LINE_AA)
                # Red cross for reprojected 3D point
                sz = 4
                cv2.line(overlay, (pt_proj[0] - sz, pt_proj[1] - sz), (pt_proj[0] + sz, pt_proj[1] + sz), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(overlay, (pt_proj[0] - sz, pt_proj[1] + sz), (pt_proj[0] + sz, pt_proj[1] - sz), (0, 0, 255), 1, cv2.LINE_AA)

            # Header info overlay
            mean_e = cam.mean_reprojection_error_px or 0.0
            header_text = f"Cam {cam.image_id:02d} ({cam.image_name}) | Mean Reproj: {mean_e:.2f}px | PnP Inliers: {cam.num_pnp_inliers}"
            cv2.rectangle(overlay, (10, 10), (w - 10, 50), (0, 0, 0), -1)
            cv2.putText(overlay, header_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            out_file = overlay_dir / f"reprojection_overlay_cam_{cam.image_id:02d}.png"
            cv2.imwrite(str(out_file), overlay)
            saved_paths.append(out_file)

        return saved_paths

    def generate_all(
        self,
        registered_cameras: Dict[int, CameraPose],
        track_system: FeatureTrackSystem,
        camera_matrix_k: np.ndarray,
        master_report: dict,
        camera_summaries: List[dict],
        keyframe_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[Path]]:
        """Convenience method to execute and export all diagnostic visualizer routines."""
        return {
            "camera_trajectory": self.plot_camera_trajectory(registered_cameras, track_system),
            "reprojection_error_distribution": self.plot_reprojection_error_distribution(master_report, camera_summaries),
            "point_cloud_outliers": self.plot_point_cloud_outliers(registered_cameras, track_system, camera_matrix_k),
            "reprojection_overlays": self.plot_2d_reprojection_overlays(registered_cameras, track_system, camera_matrix_k, keyframe_dir),
        }

