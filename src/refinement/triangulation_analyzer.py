"""
Triangulation Geometry and Multi-View Depth Consistency Analyzer (Phase 5.3.3)
Evaluates pairwise ray triangulation angles, baseline geometry,
positive depth validation, and distance consistency across observing camera views.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.refinement.refinement_models import RefinementConfig


class TriangulationAnalyzer:
    """Analyzes multi-view ray intersections, triangulation angles, and depth geometry."""

    def __init__(self, config: Optional[RefinementConfig] = None):
        self.config = config or RefinementConfig()

    @staticmethod
    def get_camera_center(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        """Computes camera optical center C = -R^T * t in world coordinates."""
        R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        t = np.asarray(translation_vector, dtype=np.float64).reshape(3, 1)
        return (-R.T @ t).flatten()

    def analyze_triangulation(
        self,
        point_3d: Union[np.ndarray, Tuple[float, float, float], list],
        camera_ids: List[int],
        registered_cameras: Dict[int, Any],
    ) -> Dict[str, Any]:
        """
        Computes pairwise triangulation angles, baseline geometry, and camera-to-point distances.

        Args:
            point_3d: 3D point (X, Y, Z) in world space.
            camera_ids: List of camera IDs that observed this 3D point.
            registered_cameras: Dict mapping camera_id -> camera info.

        Returns:
            Dictionary with triangulation angles, baseline, depth stats, and degeneracy flags.
        """
        X_w = np.asarray(point_3d, dtype=np.float64).flatten()

        if not np.all(np.isfinite(X_w)) or len(X_w) != 3:
            return {
                "num_distinct_cameras": 0,
                "max_angle_deg": 0.0,
                "median_angle_deg": 0.0,
                "min_angle_deg": 0.0,
                "max_camera_baseline": 0.0,
                "mean_camera_distance": 0.0,
                "relative_distance_std": 0.0,
                "baseline_to_depth_ratio": 0.0,
                "has_positive_depth_all_views": False,
                "is_degenerate": True,
                "degeneracy_reason": "3D point coordinates contain non-finite values (NaN/Inf)",
            }

        # Deduplicate camera IDs while preserving order
        unique_cams = []
        seen = set()
        for cid in camera_ids:
            if cid in registered_cameras and cid not in seen:
                unique_cams.append(cid)
                seen.add(cid)

        n_cams = len(unique_cams)
        if n_cams < 2:
            return {
                "num_distinct_cameras": n_cams,
                "max_angle_deg": 0.0,
                "median_angle_deg": 0.0,
                "min_angle_deg": 0.0,
                "max_camera_baseline": 0.0,
                "mean_camera_distance": 0.0,
                "relative_distance_std": 0.0,
                "baseline_to_depth_ratio": 0.0,
                "has_positive_depth_all_views": False,
                "is_degenerate": True,
                "degeneracy_reason": f"Insufficient distinct cameras ({n_cams} < 2) for multi-view triangulation",
            }

        camera_centers = []
        ray_vectors = []
        camera_distances = []
        positive_depth_flags = []

        for cid in unique_cams:
            cinfo = registered_cameras[cid]
            if isinstance(cinfo, dict):
                R = np.asarray(cinfo["rotation_matrix"], dtype=np.float64).reshape(3, 3)
                t = np.asarray(cinfo["translation_vector"], dtype=np.float64).reshape(3, 1)
            else:
                R = np.asarray(cinfo.rotation_matrix, dtype=np.float64).reshape(3, 3)
                t = np.asarray(cinfo.translation_vector, dtype=np.float64).reshape(3, 1)

            # Camera center
            C = (-R.T @ t).flatten()
            camera_centers.append(C)

            # Ray from camera center to 3D point
            diff = X_w - C
            dist = float(np.linalg.norm(diff))
            camera_distances.append(dist)

            if dist > 1e-7:
                ray_vectors.append(diff / dist)
            else:
                ray_vectors.append(np.array([0.0, 0.0, 1.0]))

            # Positive depth check in camera frame: Z_c = (R * X_w + t)[2]
            X_c = R @ X_w.reshape(3, 1) + t
            positive_depth_flags.append(float(X_c[2, 0]) >= 1e-4)

        # Pairwise triangulation angles
        angles_deg = []
        baselines = []

        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                # Ray angle: dot product of normalized rays
                v_i = ray_vectors[i]
                v_j = ray_vectors[j]
                cos_theta = np.clip(float(np.dot(v_i, v_j)), -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cos_theta)))
                angles_deg.append(angle_deg)

                # Baseline distance
                b_dist = float(np.linalg.norm(camera_centers[i] - camera_centers[j]))
                baselines.append(b_dist)

        max_ang = float(np.max(angles_deg)) if angles_deg else 0.0
        med_ang = float(np.median(angles_deg)) if angles_deg else 0.0
        min_ang = float(np.min(angles_deg)) if angles_deg else 0.0

        max_b = float(np.max(baselines)) if baselines else 0.0
        mean_dist = float(np.mean(camera_distances)) if camera_distances else 0.0
        std_dist = float(np.std(camera_distances)) if camera_distances else 0.0
        rel_std = (std_dist / mean_dist) if mean_dist > 1e-7 else 0.0
        b_to_d = (max_b / mean_dist) if mean_dist > 1e-7 else 0.0

        all_positive_depth = bool(all(positive_depth_flags))

        is_degenerate = False
        deg_reason = None

        if max_ang < self.config.min_triangulation_angle_degenerate_deg:
            is_degenerate = True
            deg_reason = (
                f"Degenerate triangulation geometry: maximum ray angle {max_ang:.2f}° is below "
                f"minimum threshold ({self.config.min_triangulation_angle_degenerate_deg}°)"
            )
        elif not all_positive_depth:
            is_degenerate = True
            deg_reason = "Point is behind camera plane in at least one observing view"

        return {
            "num_distinct_cameras": n_cams,
            "max_angle_deg": round(max_ang, 2),
            "median_angle_deg": round(med_ang, 2),
            "min_angle_deg": round(min_ang, 2),
            "max_camera_baseline": round(max_b, 4),
            "mean_camera_distance": round(mean_dist, 4),
            "min_camera_distance": round(float(np.min(camera_distances)), 4) if camera_distances else 0.0,
            "max_camera_distance": round(float(np.max(camera_distances)), 4) if camera_distances else 0.0,
            "relative_distance_std": round(rel_std, 4),
            "baseline_to_depth_ratio": round(b_to_d, 4),
            "has_positive_depth_all_views": all_positive_depth,
            "is_degenerate": is_degenerate,
            "degeneracy_reason": deg_reason,
        }

