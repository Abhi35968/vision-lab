"""
Outlier Diagnostics Module
Calculates per-observation, per-point, and per-camera reprojection errors and exports detailed CSV/JSON telemetry.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.sfm.camera_registration import CameraPose
from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem


@dataclass
class ObservationErrorRecord:
    """Detailed telemetry for a single 2D-3D projection correspondence."""
    observation_id: int
    camera_id: int
    image_name: str
    point_id: int
    track_id: int
    obs_u: float
    obs_v: float
    proj_u: float
    proj_v: float
    error_u: float
    error_v: float
    reprojection_error_px: float
    is_outlier_4px: bool
    is_outlier_10px: bool

    def to_csv_row(self) -> str:
        return (
            f"{self.observation_id},{self.camera_id},{self.image_name},{self.point_id},"
            f"{self.track_id},{self.obs_u:.3f},{self.obs_v:.3f},{self.proj_u:.3f},"
            f"{self.proj_v:.3f},{self.error_u:.3f},{self.error_v:.3f},"
            f"{self.reprojection_error_px:.3f},{self.is_outlier_4px},{self.is_outlier_10px}\n"
        )


class OutlierDiagnostics:
    """
    Computes fine-grained reprojection residual statistics and identifies outlier observations,
    points, and camera views without destructive deletion.
    """

    def __init__(
        self,
        threshold_outlier_px: float = 4.0,
        threshold_severe_px: float = 10.0,
    ):
        self.threshold_outlier = threshold_outlier_px
        self.threshold_severe = threshold_severe_px

    def analyze(
        self,
        registered_cameras: Dict[int, CameraPose],
        track_system: FeatureTrackSystem,
        camera_matrix_k: np.ndarray,
    ) -> Tuple[List[ObservationErrorRecord], List[dict], List[dict], dict]:
        """
        Evaluates 2D Euclidean reprojection errors for all observations across all registered views.
        
        Returns:
            (observation_records, point_summaries, camera_summaries, master_report)
        """
        fx, fy = camera_matrix_k[0, 0], camera_matrix_k[1, 1]
        cx, cy = camera_matrix_k[0, 2], camera_matrix_k[1, 2]

        tri_tracks = [t for t in track_system.tracks.values() if t.is_triangulated and t.point_3d is not None]
        sorted_cams = {cid: cam for cid, cam in sorted(registered_cameras.items()) if cam.is_registered}

        observation_records: List[ObservationErrorRecord] = []
        point_summaries: List[dict] = []
        obs_by_camera: Dict[int, List[float]] = {cid: [] for cid in sorted_cams.keys()}
        obs_gt_4px_by_camera: Dict[int, int] = {cid: 0 for cid in sorted_cams.keys()}
        obs_gt_10px_by_camera: Dict[int, int] = {cid: 0 for cid in sorted_cams.keys()}

        obs_counter = 0

        for p_idx, track in enumerate(tri_tracks):
            X = track.point_3d
            track_errors: List[float] = []

            for cid, obs in track.observations.items():
                if cid in sorted_cams:
                    cam = sorted_cams[cid]
                    X_cam = cam.rotation_matrix @ X + cam.translation_vector.ravel()
                    z = X_cam[2]

                    if z > 1e-4:
                        u_proj = fx * (X_cam[0] / z) + cx
                        v_proj = fy * (X_cam[1] / z) + cy
                    else:
                        u_proj = 1e4
                        v_proj = 1e4

                    u_obs, v_obs = obs.coords
                    du = float(u_proj - u_obs)
                    dv = float(v_proj - v_obs)
                    err = float(np.sqrt(du ** 2 + dv ** 2))

                    track_errors.append(err)
                    obs_by_camera[cid].append(err)

                    is_out_4 = err > self.threshold_outlier
                    is_out_10 = err > self.threshold_severe

                    if is_out_4:
                        obs_gt_4px_by_camera[cid] += 1
                    if is_out_10:
                        obs_gt_10px_by_camera[cid] += 1

                    rec = ObservationErrorRecord(
                        observation_id=obs_counter,
                        camera_id=cid,
                        image_name=cam.image_name,
                        point_id=p_idx,
                        track_id=track.track_id,
                        obs_u=float(u_obs),
                        obs_v=float(v_obs),
                        proj_u=float(u_proj),
                        proj_v=float(v_proj),
                        error_u=du,
                        error_v=dv,
                        reprojection_error_px=err,
                        is_outlier_4px=is_out_4,
                        is_outlier_10px=is_out_10,
                    )
                    observation_records.append(rec)
                    obs_counter += 1

            if track_errors:
                mean_e = float(np.mean(track_errors))
                med_e = float(np.median(track_errors))
                max_e = float(np.max(track_errors))
                min_e = float(np.min(track_errors))
                gt4 = sum(1 for e in track_errors if e > self.threshold_outlier)
                gt10 = sum(1 for e in track_errors if e > self.threshold_severe)
                is_high = (med_e > self.threshold_outlier) or (max_e > self.threshold_severe)
            else:
                mean_e = med_e = max_e = min_e = 0.0
                gt4 = gt10 = 0
                is_high = False

            point_summaries.append({
                "point_id": p_idx,
                "track_id": track.track_id,
                "num_observations": len(track_errors),
                "mean_error_px": round(mean_e, 3),
                "median_error_px": round(med_e, 3),
                "max_error_px": round(max_e, 3),
                "min_error_px": round(min_e, 3),
                "x_world": round(float(X[0]), 4),
                "y_world": round(float(X[1]), 4),
                "z_world": round(float(X[2]), 4),
                "unit": "arbitrary_units",
                "num_obs_gt_4px": gt4,
                "num_obs_gt_10px": gt10,
                "is_high_error_track": is_high,
            })

        # Camera Error Summaries
        camera_summaries: List[dict] = []
        for cid, cam in sorted_cams.items():
            errs = obs_by_camera[cid]
            n_obs = len(errs)
            if n_obs > 0:
                c_mean = float(np.mean(errs))
                c_med = float(np.median(errs))
                c_p90 = float(np.percentile(errs, 90))
                c_max = float(np.max(errs))
                c_gt4 = obs_gt_4px_by_camera[cid]
                c_pct4 = (c_gt4 / n_obs) * 100.0
                c_gt10 = obs_gt_10px_by_camera[cid]
                c_pct10 = (c_gt10 / n_obs) * 100.0
            else:
                c_mean = c_med = c_p90 = c_max = 0.0
                c_gt4 = c_pct4 = c_gt10 = c_pct10 = 0.0

            camera_summaries.append({
                "camera_id": cid,
                "image_name": cam.image_name,
                "num_observations": n_obs,
                "mean_error_px": round(c_mean, 3),
                "median_error_px": round(c_med, 3),
                "p90_error_px": round(c_p90, 3),
                "max_error_px": round(c_max, 3),
                "count_gt_4px": c_gt4,
                "pct_gt_4px": round(c_pct4, 2),
                "count_gt_10px": c_gt10,
                "pct_gt_10px": round(c_pct10, 2),
                "center_x": round(float(cam.camera_center[0]), 4),
                "center_y": round(float(cam.camera_center[1]), 4),
                "center_z": round(float(cam.camera_center[2]), 4),
                "unit": "arbitrary_units",
                "is_high_error_camera": c_mean > self.threshold_outlier,
            })

        # Master Outlier Report
        all_errors = [rec.reprojection_error_px for rec in observation_records]
        total_obs = len(all_errors)

        c_lt_1 = sum(1 for e in all_errors if e < 1.0)
        c_lt_2 = sum(1 for e in all_errors if e < 2.0)
        c_lt_4 = sum(1 for e in all_errors if e <= self.threshold_outlier)
        c_lt_10 = sum(1 for e in all_errors if e <= self.threshold_severe)
        c_gt_4 = sum(1 for e in all_errors if e > self.threshold_outlier)
        c_gt_10 = sum(1 for e in all_errors if e > self.threshold_severe)

        # High error entities
        high_error_tracks = [p for p in point_summaries if p["is_high_error_track"]]
        high_error_cameras = [c for c in camera_summaries if c["is_high_error_camera"]]

        # Top 25 outliers
        sorted_recs = sorted(observation_records, key=lambda r: r.reprojection_error_px, reverse=True)
        top_outliers = [
            {
                "observation_id": r.observation_id,
                "camera_id": r.camera_id,
                "image_name": r.image_name,
                "point_id": r.point_id,
                "track_id": r.track_id,
                "reprojection_error_px": round(r.reprojection_error_px, 3),
                "obs_coords": [round(r.obs_u, 2), round(r.obs_v, 2)],
                "proj_coords": [round(r.proj_u, 2), round(r.proj_v, 2)],
            }
            for r in sorted_recs[:25]
        ]

        master_report = {
            "total_observations": total_obs,
            "total_reconstructed_points": len(tri_tracks),
            "total_registered_cameras": len(sorted_cams),
            "error_thresholds_px": {
                "subpixel": 1.0,
                "good": 2.0,
                "outlier": self.threshold_outlier,
                "severe_outlier": self.threshold_severe,
            },
            "error_distribution_counts": {
                "below_1px": c_lt_1,
                "below_2px": c_lt_2,
                "below_4px": c_lt_4,
                "below_10px": c_lt_10,
                "above_4px": c_gt_4,
                "above_10px": c_gt_10,
            },
            "error_distribution_percentages": {
                "pct_below_1px": round((c_lt_1 / max(1, total_obs)) * 100.0, 2),
                "pct_below_2px": round((c_lt_2 / max(1, total_obs)) * 100.0, 2),
                "pct_below_4px": round((c_lt_4 / max(1, total_obs)) * 100.0, 2),
                "pct_below_10px": round((c_lt_10 / max(1, total_obs)) * 100.0, 2),
                "pct_above_4px": round((c_gt_4 / max(1, total_obs)) * 100.0, 2),
                "pct_above_10px": round((c_gt_10 / max(1, total_obs)) * 100.0, 2),
            },
            "high_error_summary": {
                "total_high_error_tracks": len(high_error_tracks),
                "total_high_error_cameras": len(high_error_cameras),
                "high_error_cameras_list": [c["camera_id"] for c in high_error_cameras],
            },
            "top_25_outliers": top_outliers,
            "policy_note": "Outlier observations are diagnosed and exported for inspection; they are NOT automatically deleted to preserve traceability.",
        }

        return observation_records, point_summaries, camera_summaries, master_report

    def export_all(
        self,
        output_dir: Union[str, Path],
        observation_records: List[ObservationErrorRecord],
        point_summaries: List[dict],
        camera_summaries: List[dict],
        master_report: dict,
    ):
        """Saves observation_errors.csv, point_errors.csv, camera_errors.csv, and outlier_report.json."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. observation_errors.csv
        obs_csv_path = out_path / "observation_errors.csv"
        with open(obs_csv_path, "w", encoding="utf-8") as f:
            f.write(
                "observation_id,camera_id,image_name,point_id,track_id,obs_u,obs_v,"
                "proj_u,proj_v,error_u,error_v,reprojection_error_px,is_outlier_4px,is_outlier_10px\n"
            )
            for rec in observation_records:
                f.write(rec.to_csv_row())

        # 2. point_errors.csv
        pt_csv_path = out_path / "point_errors.csv"
        with open(pt_csv_path, "w", encoding="utf-8") as f:
            f.write(
                "point_id,track_id,num_observations,mean_error_px,median_error_px,"
                "max_error_px,min_error_px,x_world,y_world,z_world,unit,num_obs_gt_4px,"
                "num_obs_gt_10px,is_high_error_track\n"
            )
            for p in point_summaries:
                f.write(
                    f"{p['point_id']},{p['track_id']},{p['num_observations']},{p['mean_error_px']},"
                    f"{p['median_error_px']},{p['max_error_px']},{p['min_error_px']},{p['x_world']},"
                    f"{p['y_world']},{p['z_world']},{p['unit']},{p['num_obs_gt_4px']},{p['num_obs_gt_10px']},"
                    f"{p['is_high_error_track']}\n"
                )

        # 3. camera_errors.csv
        cam_csv_path = out_path / "camera_errors.csv"
        with open(cam_csv_path, "w", encoding="utf-8") as f:
            f.write(
                "camera_id,image_name,num_observations,mean_error_px,median_error_px,"
                "p90_error_px,max_error_px,count_gt_4px,pct_gt_4px,count_gt_10px,"
                "pct_gt_10px,center_x,center_y,center_z,unit,is_high_error_camera\n"
            )
            for c in camera_summaries:
                f.write(
                    f"{c['camera_id']},{c['image_name']},{c['num_observations']},{c['mean_error_px']},"
                    f"{c['median_error_px']},{c['p90_error_px']},{c['max_error_px']},{c['count_gt_4px']},"
                    f"{c['pct_gt_4px']},{c['count_gt_10px']},{c['pct_gt_10px']},{c['center_x']},"
                    f"{c['center_y']},{c['center_z']},{c['unit']},{c['is_high_error_camera']}\n"
                )

        # 4. outlier_report.json
        outlier_json_path = out_path / "outlier_report.json"
        with open(outlier_json_path, "w", encoding="utf-8") as f:
            json.dump(master_report, f, indent=2)

