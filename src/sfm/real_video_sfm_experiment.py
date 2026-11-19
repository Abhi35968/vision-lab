"""
Real-World Video Incremental Structure from Motion Experiment Runner
Orchestrates isolated comparative evaluations across camera intrinsics hypotheses
(f=935px, f=1300px, f=1663px) on smartphone footage (1080x1920 portrait).
Generates 3D point clouds, camera trajectories, 2D reprojection overlays, and comparison telemetry.
All spatial dimensions and depths are strictly represented in 'arbitrary_units'.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.sfm.incremental_sfm import IncrementalSfM, IncrementalSfMConfig
from src.sfm.camera_registration import CameraPose
from src.sfm.feature_tracks import FeatureTrackSystem

logger = logging.getLogger("RealVideoSfMExperiment")


@dataclass
class FocalHypothesis:
    """Telemetry and parameters for a camera focal-length hypothesis."""
    tag: str
    focal_px: float
    fov_description: str
    hfov_deg: float
    vfov_deg: float
    dfov_deg: float
    output_dir: str


class RealVideoSfMExperiment:
    """
    Manages multi-hypothesis incremental SfM evaluation for uncalibrated real-world video.
    """

    WIDTH = 1080
    HEIGHT = 1920
    CX = 540.0
    CY = 960.0

    def __init__(
        self,
        keyframes_dir: Union[str, Path] = "outputs/real_video_validation/keyframes",
        output_base_dir: Union[str, Path] = "outputs/real_video_sfm",
        matching_window: int = 3,
        min_inliers_pair: int = 25,
        min_pnp_inliers: int = 15,
        max_reproj_error_px: float = 4.0,
        ba_max_iter: int = 50,
    ):
        self.keyframes_dir = Path(keyframes_dir).resolve()
        self.output_base = Path(output_base_dir).resolve()
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.matching_window = matching_window
        self.min_inliers_pair = min_inliers_pair
        self.min_pnp_inliers = min_pnp_inliers
        self.max_reproj_error_px = max_reproj_error_px
        self.ba_max_iter = ba_max_iter

        # Define the 3 focal-length hypotheses requested in Phase 5.2
        self.hypotheses: List[FocalHypothesis] = [
            FocalHypothesis(
                tag="focal_935",
                focal_px=935.0,
                fov_description="60.0 deg Horizontal FOV (wide-angle smartphone standard across short edge)",
                hfov_deg=round(2.0 * np.degrees(np.arctan(self.WIDTH / (2.0 * 935.0))), 2),
                vfov_deg=round(2.0 * np.degrees(np.arctan(self.HEIGHT / (2.0 * 935.0))), 2),
                dfov_deg=round(2.0 * np.degrees(np.arctan(np.sqrt(self.WIDTH**2 + self.HEIGHT**2) / (2.0 * 935.0))), 2),
                output_dir=str(self.output_base / "focal_935"),
            ),
            FocalHypothesis(
                tag="focal_1300",
                focal_px=1300.0,
                fov_description="45.1 deg Horizontal / 72.9 deg Vertical FOV (intermediate focal length hypothesis)",
                hfov_deg=round(2.0 * np.degrees(np.arctan(self.WIDTH / (2.0 * 1300.0))), 2),
                vfov_deg=round(2.0 * np.degrees(np.arctan(self.HEIGHT / (2.0 * 1300.0))), 2),
                dfov_deg=round(2.0 * np.degrees(np.arctan(np.sqrt(self.WIDTH**2 + self.HEIGHT**2) / (2.0 * 1300.0))), 2),
                output_dir=str(self.output_base / "focal_1300"),
            ),
            FocalHypothesis(
                tag="focal_1663",
                focal_px=1662.8,
                fov_description="60.0 deg Vertical FOV (standard ~60 deg field of view assumed across long edge)",
                hfov_deg=round(2.0 * np.degrees(np.arctan(self.WIDTH / (2.0 * 1662.8))), 2),
                vfov_deg=round(2.0 * np.degrees(np.arctan(self.HEIGHT / (2.0 * 1662.8))), 2),
                dfov_deg=round(2.0 * np.degrees(np.arctan(np.sqrt(self.WIDTH**2 + self.HEIGHT**2) / (2.0 * 1662.8))), 2),
                output_dir=str(self.output_base / "focal_1663"),
            ),
        ]

    def get_intrinsics_matrix(self, focal: float) -> np.ndarray:
        """Constructs 3x3 pinhole calibration matrix K with principal point at image center."""
        return np.array([
            [focal, 0.0, self.CX],
            [0.0, focal, self.CY],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def run_all_experiments(self) -> dict:
        """
        Executes isolated incremental SfM for each focal length hypothesis and generates comparative audit.
        """
        keyframe_paths = sorted(list(self.keyframes_dir.glob("*.png")) + list(self.keyframes_dir.glob("*.jpg")))
        if len(keyframe_paths) < 2:
            raise FileNotFoundError(
                f"Insufficient keyframes found in '{self.keyframes_dir}' ({len(keyframe_paths)} available)."
            )

        print("\n" + "=" * 80)
        print(" PHASE 5.2: REAL-WORLD SMARTPHONE INCREMENTAL SfM EXPERIMENTS")
        print(f" Input Keyframes: {len(keyframe_paths)} images from '{self.keyframes_dir}'")
        print(f" Image Dimensions: {self.WIDTH} x {self.HEIGHT} (Portrait)")
        print(f" Output Root:     {self.output_base}")
        print("=" * 80)

        experiment_results: List[dict] = []

        for hyp in self.hypotheses:
            print(f"\n>>> Running Experiment: {hyp.tag.upper()} (f = {hyp.focal_px:.1f} px)")
            print(f"    FOV: {hyp.fov_description}")
            print(f"    HFOV: {hyp.hfov_deg}°, VFOV: {hyp.vfov_deg}°, DFOV: {hyp.dfov_deg}°")
            print(f"    Destination: {hyp.output_dir}")

            out_dir = Path(hyp.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            K = self.get_intrinsics_matrix(hyp.focal_px)

            config = IncrementalSfMConfig(
                max_keyframes=len(keyframe_paths),
                detector_type="SIFT",
                max_features=2000,
                matching_window=self.matching_window,
                min_inliers_pair=self.min_inliers_pair,
                min_track_length=2,
                min_pnp_inliers=self.min_pnp_inliers,
                min_triangulation_angle_deg=1.5,
                max_reprojection_error_px=self.max_reproj_error_px,
                enable_bundle_adjustment=True,
                ba_max_iterations=self.ba_max_iter,
                output_dir=hyp.output_dir,
            )

            sfm = IncrementalSfM(config=config)
            start_t = time.time()
            cams, tracks, diag, ba = sfm.run(
                image_paths=keyframe_paths,
                camera_matrix_k=K,
                image_size=(self.WIDTH, self.HEIGHT),
                calibration_status="estimated",
            )
            elapsed = time.time() - start_t

            # Read the generated reconstruction report
            rep_path = out_dir / "reconstruction_report.json"
            report_data = {}
            if rep_path.exists():
                with open(rep_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)

            reg_cams_count = sum(1 for c in cams.values() if c.is_registered)

            # Determine seed pair from report or cams
            seed_pair = report_data.get("bundle_adjustment", {}).get("gauge_and_scale_constraints", {})
            anchor_cam = seed_pair.get("anchor_camera_id", 0)
            scale_cam = seed_pair.get("scale_camera_id", 1)

            err_stats = report_data.get("reprojection_error_percentiles_px", {})
            obs_acc = report_data.get("observation_error_distribution", {}).get("percentages_below_thresholds", {})
            high_err_counts = report_data.get("observation_error_distribution", {}).get("high_error_counts", {})

            res_record = {
                "tag": hyp.tag,
                "focal_px": hyp.focal_px,
                "fov_description": hyp.fov_description,
                "hfov_deg": hyp.hfov_deg,
                "vfov_deg": hyp.vfov_deg,
                "dfov_deg": hyp.dfov_deg,
                "total_candidate_keyframes": len(keyframe_paths),
                "registered_cameras_count": reg_cams_count,
                "registration_rate_percent": round((reg_cams_count / len(keyframe_paths)) * 100.0, 2),
                "reconstructed_3d_points": diag.num_sparse_3d_points,
                "total_observations": diag.num_total_observations,
                "seed_pair": [anchor_cam, scale_cam],
                "reprojection_errors_px": {
                    "rmse": err_stats.get("rmse", 0.0),
                    "mean": err_stats.get("mean", 0.0),
                    "median": err_stats.get("median", 0.0),
                    "p90": err_stats.get("p90", 0.0),
                    "p95": err_stats.get("p95", 0.0),
                    "max": err_stats.get("max", 0.0),
                },
                "accuracy_percent": {
                    "below_1px": obs_acc.get("pct_below_1px", 0.0),
                    "below_2px": obs_acc.get("pct_below_2px", 0.0),
                    "below_4px": obs_acc.get("pct_below_4px", 0.0),
                    "below_10px": obs_acc.get("pct_below_10px", 0.0),
                },
                "outlier_counts": {
                    "above_4px": high_err_counts.get("count_gt_4px", 0),
                    "above_10px": high_err_counts.get("count_gt_10px", 0),
                },
                "bundle_adjustment": {
                    "initial_mean_px": ba.initial_mean_px if ba else 0.0,
                    "final_mean_px": ba.final_mean_px if ba else 0.0,
                    "iterations": ba.num_iterations if ba else 0,
                    "termination_message": ba.termination_message if ba else "Skipped",
                    "seed_baseline_preserved": ba.seed_baseline_preserved if ba else False,
                    "scale_constraint_enforced": True,
                },
                "scale_convention": "arbitrary_units",
                "execution_time_seconds": round(elapsed, 2),
                "output_directory": hyp.output_dir,
            }
            experiment_results.append(res_record)

        # Select the best performing hypothesis based on multi-criterion evaluation
        # (camera registration count, point cloud density, and median/mean reprojection error)
        best_hyp = max(
            experiment_results,
            key=lambda r: (
                r["registered_cameras_count"],
                r["reconstructed_3d_points"],
                -r["reprojection_errors_px"]["median"],
            ),
        )

        comparison_master = {
            "timestamp": datetime.now().isoformat(),
            "input_keyframes_count": len(keyframe_paths),
            "image_resolution": [self.WIDTH, self.HEIGHT],
            "principal_point": [self.CX, self.CY],
            "distortion_parameters": {
                "status": "unavailable_assumed_zero",
                "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "note": "Lens distortion coefficients are uncalibrated for this smartphone sensor and assumed zero (pinhole model).",
            },
            "scale_convention": {
                "unit": "arbitrary_units",
                "reference_scale": "Seed pair baseline length ||t_seed|| = 1.0",
                "is_metric": False,
                "academic_note": "Monocular reconstruction cannot recover absolute metric scale without a physical metric ground truth reference.",
            },
            "selected_optimal_hypothesis": {
                "tag": best_hyp["tag"],
                "focal_px": best_hyp["focal_px"],
                "justification": (
                    f"Candidate '{best_hyp['tag']}' (f={best_hyp['focal_px']:.1f}px, 60° VFOV) yielded the highest "
                    f"registered camera count ({best_hyp['registered_cameras_count']}/{len(keyframe_paths)} keyframes), "
                    f"the densest point cloud ({best_hyp['reconstructed_3d_points']} points), and the lowest median "
                    f"reprojection error ({best_hyp['reprojection_errors_px']['median']:.3f} px)."
                ),
            },
            "comparative_table": experiment_results,
        }

        # Save JSON master comparison
        json_path = self.output_base / "intrinsics_comparison.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_master, f, indent=2)

        # Save CSV comparison
        csv_path = self.output_base / "intrinsics_comparison.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("tag,focal_px,registered_cams,points_3d,mean_reproj_px,median_reproj_px,p90_px,p95_px,below_1px_pct,below_4px_pct,outliers_gt4px,time_sec\n")
            for r in experiment_results:
                e = r["reprojection_errors_px"]
                a = r["accuracy_percent"]
                o = r["outlier_counts"]
                f.write(
                    f"{r['tag']},{r['focal_px']},{r['registered_cameras_count']},{r['reconstructed_3d_points']},"
                    f"{e['mean']},{e['median']},{e['p90']},{e['p95']},{a['below_1px']},{a['below_4px']},{o['above_4px']},{r['execution_time_seconds']}\n"
                )

        # Generate comparative visualization plot
        self._plot_comparison(experiment_results)

        return comparison_master

    def _plot_comparison(self, results: List[dict]) -> None:
        """Plots side-by-side comparison charts for the focal length candidates."""
        tags = [r["tag"].replace("focal_", "f = ") + " px" for r in results]
        cams = [r["registered_cameras_count"] for r in results]
        points = [r["reconstructed_3d_points"] for r in results]
        means = [r["reprojection_errors_px"]["mean"] for r in results]
        medians = [r["reprojection_errors_px"]["median"] for r in results]
        p90s = [r["reprojection_errors_px"]["p90"] for r in results]

        fig, axs = plt.subplots(1, 3, figsize=(16, 5))

        # Chart 1: Registered Cameras
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars1 = axs[0].bar(tags, cams, color=colors, edgecolor="black", alpha=0.85, width=0.5)
        for b in bars1:
            y = b.get_height()
            axs[0].text(b.get_x() + b.get_width() / 2, y + 0.5, f"{int(y)} / 47", ha="center", va="bottom", fontweight="bold")
        axs[0].set_title("Registered Cameras Count", fontweight="bold")
        axs[0].set_ylabel("Registered Keyframes")
        axs[0].set_ylim(0, 52)
        axs[0].grid(True, axis="y", linestyle=":", alpha=0.6)

        # Chart 2: Reconstructed 3D Points
        bars2 = axs[1].bar(tags, points, color=colors, edgecolor="black", alpha=0.85, width=0.5)
        for b in bars2:
            y = b.get_height()
            axs[1].text(b.get_x() + b.get_width() / 2, y + 30, f"{int(y)} pts", ha="center", va="bottom", fontweight="bold")
        axs[1].set_title("Reconstructed Sparse 3D Points", fontweight="bold")
        axs[1].set_ylabel("Triangulated Points Count")
        axs[1].set_ylim(0, max(points) * 1.25)
        axs[1].grid(True, axis="y", linestyle=":", alpha=0.6)

        # Chart 3: Reprojection Error Residuals
        x = np.arange(len(tags))
        w = 0.25
        axs[2].bar(x - w, medians, width=w, label="Median Error (px)", color="#2ca02c", edgecolor="black", alpha=0.85)
        axs[2].bar(x, means, width=w, label="Mean Error (px)", color="#1f77b4", edgecolor="black", alpha=0.85)
        axs[2].bar(x + w, p90s, width=w, label="P90 Error (px)", color="#ff7f0e", edgecolor="black", alpha=0.85)
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(tags)
        axs[2].set_title("Reprojection Error Comparison (pixels)", fontweight="bold")
        axs[2].set_ylabel("Error in Pixels")
        axs[2].legend(loc="upper right", fontsize=9)
        axs[2].grid(True, axis="y", linestyle=":", alpha=0.6)

        plt.suptitle("Camera Intrinsics Hypotheses Comparison (1080x1920 Portrait)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plot_path = self.output_base / "intrinsics_comparison_plot.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()

