"""
Reconstruction Metrics and Diagnostics Module
Evaluates reconstruction quality, computes error percentiles, and exports Open3D point clouds and camera trajectories.
Ensures rigorous scale metadata tracking (arbitrary units) and observation consistency.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import open3d as o3d

from src.sfm.camera_registration import CameraPose
from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem
from src.sfm.bundle_adjustment import BundleAdjustmentResult


@dataclass
class ReconstructionDiagnostics:
    """Consolidated numerical quality metrics, percentiles, and scale metadata for multi-view SfM reconstruction."""
    total_candidate_images: int
    registered_cameras_count: int
    registration_success_rate_percent: float
    total_3d_points_count: int
    valid_3d_points_count: int
    num_sparse_3d_points: int
    num_total_observations: int
    mean_track_length: float
    median_track_length: float
    max_track_length: int
    track_length_distribution: Dict[str, int]
    rmse_reprojection_error_px: float
    mean_reprojection_error_px: float
    median_reprojection_error_px: float
    p90_reprojection_error_px: float
    p95_reprojection_error_px: float
    max_reprojection_error_px: float
    error_percentages_below_thresholds: Dict[str, float]
    high_error_observation_counts: Dict[str, int]
    min_point_depth: float
    max_point_depth: float
    median_point_depth: float
    scale_metadata: Dict[str, Union[str, float, bool]] = field(default_factory=dict)
    track_integrity_summary: Dict[str, Union[int, bool, dict, list]] = field(default_factory=dict)
    bundle_adjustment_summary: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "total_candidate_images": self.total_candidate_images,
            "registered_cameras_count": self.registered_cameras_count,
            "registration_success_rate_percent": round(self.registration_success_rate_percent, 2),
            "points_count": {
                "total_reconstructed": self.total_3d_points_count,
                "valid_points": self.valid_3d_points_count,
            },
            "num_sparse_3d_points": self.num_sparse_3d_points,
            "num_total_observations": self.num_total_observations,
            "scale_metadata": self.scale_metadata,
            "track_statistics": {
                "mean_track_length": round(self.mean_track_length, 2),
                "median_track_length": round(self.median_track_length, 2),
                "max_track_length": self.max_track_length,
                "distribution": self.track_length_distribution,
            },
            "track_integrity": self.track_integrity_summary,
            "reprojection_error_percentiles_px": {
                "rmse": round(self.rmse_reprojection_error_px, 3),
                "mean": round(self.mean_reprojection_error_px, 3),
                "median": round(self.median_reprojection_error_px, 3),
                "p90": round(self.p90_reprojection_error_px, 3),
                "p95": round(self.p95_reprojection_error_px, 3),
                "max": round(self.max_reprojection_error_px, 3),
                "unit": "pixels",
            },
            "observation_error_distribution": {
                "percentages_below_thresholds": {
                    "pct_below_1px": round(self.error_percentages_below_thresholds.get("pct_below_1px", 0.0), 2),
                    "pct_below_2px": round(self.error_percentages_below_thresholds.get("pct_below_2px", 0.0), 2),
                    "pct_below_4px": round(self.error_percentages_below_thresholds.get("pct_below_4px", 0.0), 2),
                    "pct_below_10px": round(self.error_percentages_below_thresholds.get("pct_below_10px", 0.0), 2),
                },
                "high_error_counts": self.high_error_observation_counts,
            },
            "scene_depth_bounds": {
                "unit": "arbitrary_units",
                "min_depth": round(self.min_point_depth, 3),
                "max_depth": round(self.max_point_depth, 3),
                "median_depth": round(self.median_point_depth, 3),
                "note": "Depth is expressed in arbitrary units defined by the initial seed baseline. Do not interpret as physical meters.",
            },
            "bundle_adjustment": self.bundle_adjustment_summary,
        }


class ReconstructionMetrics:
    """
    Computes quality benchmarks and exports 3D scene artifacts (PLY, NPY, JSON).
    """

    @staticmethod
    def compute_diagnostics(
        total_candidates: int,
        registered_cameras: Dict[int, CameraPose],
        track_system: FeatureTrackSystem,
        camera_matrix_k: np.ndarray,
        ba_result: Optional[BundleAdjustmentResult] = None,
    ) -> ReconstructionDiagnostics:
        """Computes statistical metrics across all registered views and reconstructed 3D points."""
        reg_cams = [cam for cam in registered_cameras.values() if cam.is_registered]
        reg_count = len(reg_cams)
        success_rate = (reg_count / max(1, total_candidates)) * 100.0

        all_tri_tracks = [t for t in track_system.tracks.values() if t.is_triangulated]
        valid_tri_tracks = [t for t in all_tri_tracks if t.point_3d is not None and np.all(np.isfinite(t.point_3d))]
        num_points = len(valid_tri_tracks)

        # Track lengths
        track_lengths = [t.length for t in valid_tri_tracks]
        mean_len = float(np.mean(track_lengths)) if track_lengths else 0.0
        med_len = float(np.median(track_lengths)) if track_lengths else 0.0
        max_len = int(np.max(track_lengths)) if track_lengths else 0

        len_dist: Dict[str, int] = {}
        for l in track_lengths:
            b = str(l) if l < 6 else "6+"
            len_dist[b] = len_dist.get(b, 0) + 1

        # Track observation integrity audit
        integrity_report = track_system.validate_track_integrity()

        # Reprojection errors and depths across all valid observations
        fx, fy = camera_matrix_k[0, 0], camera_matrix_k[1, 1]
        cx, cy = camera_matrix_k[0, 2], camera_matrix_k[1, 2]

        obs_errors = []
        depths = []
        total_obs = 0

        for t in valid_tri_tracks:
            X = t.point_3d
            for img_id, obs in t.observations.items():
                if img_id in registered_cameras and registered_cameras[img_id].is_registered:
                    total_obs += 1
                    cam = registered_cameras[img_id]
                    X_cam = cam.rotation_matrix @ X + cam.translation_vector.ravel()
                    z = X_cam[2]
                    if np.isfinite(z):
                        depths.append(z)
                    if z > 1e-4:
                        u = fx * (X_cam[0] / z) + cx
                        v = fy * (X_cam[1] / z) + cy
                        if np.isfinite(u) and np.isfinite(v):
                            err = float(np.sqrt((u - obs.coords[0]) ** 2 + (v - obs.coords[1]) ** 2))
                            obs_errors.append(err)

        if obs_errors:
            errs_arr = np.array(obs_errors, dtype=np.float64)
            rmse_err = float(np.sqrt(np.mean(errs_arr ** 2)))
            mean_err = float(np.mean(errs_arr))
            med_err = float(np.median(errs_arr))
            p90_err = float(np.percentile(errs_arr, 90))
            p95_err = float(np.percentile(errs_arr, 95))
            max_err = float(np.max(errs_arr))

            n_obs = len(errs_arr)
            c_lt_1 = int(np.sum(errs_arr < 1.0))
            c_lt_2 = int(np.sum(errs_arr < 2.0))
            c_lt_4 = int(np.sum(errs_arr <= 4.0))
            c_lt_10 = int(np.sum(errs_arr <= 10.0))
            c_gt_4 = int(np.sum(errs_arr > 4.0))
            c_gt_10 = int(np.sum(errs_arr > 10.0))

            err_pcts = {
                "pct_below_1px": float((c_lt_1 / n_obs) * 100.0),
                "pct_below_2px": float((c_lt_2 / n_obs) * 100.0),
                "pct_below_4px": float((c_lt_4 / n_obs) * 100.0),
                "pct_below_10px": float((c_lt_10 / n_obs) * 100.0),
            }
            high_err_counts = {
                "count_gt_4px": c_gt_4,
                "count_gt_10px": c_gt_10,
            }
        else:
            rmse_err = mean_err = med_err = p90_err = p95_err = max_err = 0.0
            err_pcts = {
                "pct_below_1px": 0.0,
                "pct_below_2px": 0.0,
                "pct_below_4px": 0.0,
                "pct_below_10px": 0.0,
            }
            high_err_counts = {
                "count_gt_4px": 0,
                "count_gt_10px": 0,
            }

        if depths:
            min_d = float(np.min(depths))
            max_d = float(np.max(depths))
            med_d = float(np.median(depths))
        else:
            min_d = max_d = med_d = 0.0

        # Construct scale metadata
        seed_baseline = ba_result.seed_baseline_after if ba_result is not None else 1.0
        scale_metadata = {
            "scale_status": "unscaled_arbitrary_units",
            "distance_unit": "arbitrary_units",
            "seed_pair_baseline_length": round(seed_baseline, 6),
            "is_metric": False,
            "metric_reference_note": (
                "Monocular Structure from Motion recovers geometry up to an arbitrary scale factor defined by ||t_seed|| = 1.0. "
                "All coordinate distances and depths are in arbitrary units. Absolute metric dimensions require ground truth physical reference."
            ),
        }

        return ReconstructionDiagnostics(
            total_candidate_images=total_candidates,
            registered_cameras_count=reg_count,
            registration_success_rate_percent=success_rate,
            total_3d_points_count=len(all_tri_tracks),
            valid_3d_points_count=num_points,
            num_sparse_3d_points=num_points,
            num_total_observations=total_obs,
            mean_track_length=mean_len,
            median_track_length=med_len,
            max_track_length=max_len,
            track_length_distribution=dict(sorted(len_dist.items(), key=lambda x: str(x[0]))),
            rmse_reprojection_error_px=rmse_err,
            mean_reprojection_error_px=mean_err,
            median_reprojection_error_px=med_err,
            p90_reprojection_error_px=p90_err,
            p95_reprojection_error_px=p95_err,
            max_reprojection_error_px=max_err,
            error_percentages_below_thresholds=err_pcts,
            high_error_observation_counts=high_err_counts,
            min_point_depth=min_d,
            max_point_depth=max_d,
            median_point_depth=med_d,
            scale_metadata=scale_metadata,
            track_integrity_summary=integrity_report,
            bundle_adjustment_summary=ba_result.to_dict() if ba_result else None,
        )

    @staticmethod
    def export_artifacts(
        output_dir: Union[str, Path],
        registered_cameras: Dict[int, CameraPose],
        track_system: FeatureTrackSystem,
        diagnostics: ReconstructionDiagnostics,
        camera_matrix_k: np.ndarray,
        ba_result: Optional[BundleAdjustmentResult] = None,
    ):
        """Exports PLY point cloud, NPY arrays, and JSON telemetry reports."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        tri_tracks = [t for t in track_system.tracks.values() if t.is_triangulated and t.point_3d is not None]

        # 1. Export 3D Points to PLY and NPY
        if tri_tracks:
            pts_array = np.array([t.point_3d for t in tri_tracks], dtype=np.float64)
            colors_array = np.tile([0.3, 0.6, 0.9], (len(pts_array), 1))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_array)
            pcd.colors = o3d.utility.Vector3dVector(colors_array)
            o3d.io.write_point_cloud(str(out_path / "sparse_points.ply"), pcd)

            np.save(str(out_path / "sparse_points.npy"), pts_array)
        else:
            np.save(str(out_path / "sparse_points.npy"), np.empty((0, 3)))

        # 2. Export Camera Poses to NPY and JSON
        poses_dict = {}
        for cid, cam in registered_cameras.items():
            poses_dict[cid] = cam.to_dict()

        with open(out_path / "cameras.json", "w", encoding="utf-8") as f:
            json.dump(poses_dict, f, indent=2)

        np.save(str(out_path / "camera_poses.npy"), poses_dict, allow_pickle=True)

        # 3. Export Feature Tracks JSON
        track_system.export_json(out_path / "feature_tracks.json", include_unregistered=False)

        # 4. Export Bundle Adjustment Report JSON
        if ba_result is not None:
            with open(out_path / "bundle_adjustment_report.json", "w", encoding="utf-8") as f:
                json.dump(ba_result.to_dict(), f, indent=2)

        # 5. Export Master Reconstruction Report JSON
        with open(out_path / "reconstruction_report.json", "w", encoding="utf-8") as f:
            json.dump(diagnostics.to_dict(), f, indent=2)
