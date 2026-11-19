"""
Reprojection Error Filter and Outlier Observation Detector (Phase 5.3.3)
Provides robust reprojection error statistics (mean, median, std, percentiles, max)
and identifies individual observation outliers without blindly discarding entire 3D points.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.association.mask_projection import CameraProjector
from src.refinement.refinement_models import ObservationReprojectionInfo, RefinementConfig


class ReprojectionFilter:
    """Evaluates robust reprojection statistics and identifies observation outliers."""

    def __init__(self, config: Optional[RefinementConfig] = None):
        self.config = config or RefinementConfig()

    @staticmethod
    def compute_robust_statistics(errors: List[float]) -> Dict[str, Any]:
        """
        Computes robust distributional metrics for a collection of reprojection errors.
        """
        if not errors:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "max": 0.0,
                "count_gt_2px": 0,
                "count_gt_4px": 0,
                "count_gt_10px": 0,
            }

        arr = np.asarray(errors, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "count_gt_2px": int(np.sum(arr > 2.0)),
            "count_gt_4px": int(np.sum(arr > 4.0)),
            "count_gt_10px": int(np.sum(arr > 10.0)),
        }

    def evaluate_point_reprojections(
        self,
        point_3d: Union[np.ndarray, Tuple[float, float, float], list],
        observations: List[Dict[str, Any]],
        registered_cameras: Dict[int, Any],
        camera_intrinsics_k: np.ndarray,
        image_size: Tuple[int, int] = (1080, 1920),
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[List[ObservationReprojectionInfo], Dict[str, Any]]:
        """
        Evaluates projections across all observing cameras, computes per-view errors,
        detects individual observation outliers, and recomputes filtered statistics.
        """
        X_w = np.asarray(point_3d, dtype=np.float64).reshape(3, 1)

        obs_infos: List[ObservationReprojectionInfo] = []
        raw_errors: List[float] = []

        for obs in observations:
            cid = obs.get("image_id", obs.get("camera_id"))
            img_name = obs.get("image_name", f"frame_{cid:04d}.png")
            coords = tuple(obs.get("coords", (0.0, 0.0)))

            cam_info = registered_cameras.get(cid)
            if cam_info is None:
                continue

            if isinstance(cam_info, dict):
                R = np.asarray(cam_info["rotation_matrix"], dtype=np.float64).reshape(3, 3)
                t = np.asarray(cam_info["translation_vector"], dtype=np.float64).reshape(3, 1)
                C = np.asarray(cam_info.get("camera_center", -R.T @ t), dtype=np.float64).reshape(3, 1)
            else:
                R = np.asarray(cam_info.rotation_matrix, dtype=np.float64).reshape(3, 3)
                t = np.asarray(cam_info.translation_vector, dtype=np.float64).reshape(3, 1)
                C = np.asarray(getattr(cam_info, "camera_center", -R.T @ t), dtype=np.float64).reshape(3, 1)

            # Camera-to-point Euclidean distance in world units
            cam_dist = float(np.linalg.norm(X_w - C))

            # Project
            proj_res = CameraProjector.project_point(
                point_3d_world=X_w.flatten(),
                rotation_matrix=R,
                translation_vector=t,
                camera_matrix_k=camera_intrinsics_k,
                image_size=image_size,
            )

            if not proj_res.is_valid:
                # Store with large dummy error or skip
                obs_info = ObservationReprojectionInfo(
                    camera_id=cid,
                    image_name=img_name,
                    obs_coords=coords,
                    proj_coords=proj_res.coords if proj_res.coords != (0, 0) else None,
                    error_px=999.0,
                    depth=proj_res.depth,
                    camera_distance=cam_dist,
                    is_inside_mask=False,
                    is_outlier=True,
                    outlier_reason=proj_res.rejection_reason,
                )
                obs_infos.append(obs_info)
                continue

            err = CameraProjector.compute_reprojection_error(proj_res.coords, coords)
            raw_errors.append(err)

            # Check mask occupancy if masks provided
            is_inside_mask = False
            if masks is not None:
                from pathlib import Path
                m_name = Path(img_name).stem + "_mask.png"
                mask = masks.get(m_name, masks.get(img_name))
                if mask is not None:
                    samp = CameraProjector.sample_mask(coords, mask)
                    is_inside_mask = samp.is_inside

            obs_info = ObservationReprojectionInfo(
                camera_id=cid,
                image_name=img_name,
                obs_coords=coords,
                proj_coords=proj_res.coords,
                error_px=err,
                depth=proj_res.depth,
                camera_distance=cam_dist,
                is_inside_mask=is_inside_mask,
                is_outlier=False,
                outlier_reason=None,
            )
            obs_infos.append(obs_info)

        # Compute raw statistics
        raw_stats = self.compute_robust_statistics(raw_errors)
        med_err = raw_stats["median"]

        # Outlier observation identification
        # An observation is an outlier if error > 12.0 px, or (if >=3 views) error > max(4.0, 3.0 * median)
        filtered_errors: List[float] = []
        num_outliers = 0

        for info in obs_infos:
            if info.error_px >= 999.0:
                num_outliers += 1
                continue

            is_outlier = False
            reason = None

            if info.error_px > self.config.outlier_single_obs_thresh_px:
                is_outlier = True
                reason = f"Error {info.error_px:.2f}px exceeds absolute outlier threshold ({self.config.outlier_single_obs_thresh_px}px)"
            elif len(raw_errors) >= 3:
                relative_thresh = max(4.0, 3.0 * med_err)
                if info.error_px > relative_thresh:
                    is_outlier = True
                    reason = f"Error {info.error_px:.2f}px exceeds 3x median threshold ({relative_thresh:.2f}px)"

            if is_outlier:
                info.is_outlier = True
                info.outlier_reason = reason
                num_outliers += 1
            else:
                filtered_errors.append(info.error_px)

        # If all observations were flagged as outliers, retain raw to preserve signal
        if not filtered_errors and raw_errors:
            filtered_errors = list(raw_errors)

        filtered_stats = self.compute_robust_statistics(filtered_errors)

        stats_summary = {
            "raw": raw_stats,
            "filtered": filtered_stats,
            "num_observations": len(obs_infos),
            "num_valid_projections": len(raw_errors),
            "num_outliers": num_outliers,
        }

        return obs_infos, stats_summary
