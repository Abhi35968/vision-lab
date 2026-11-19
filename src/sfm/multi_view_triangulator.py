"""
Multi-View Triangulator Module
Reconstructs 3D points in arbitrary_units from N-view feature tracks using Linear DLT with multi-view geometric validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from src.sfm.camera_registration import CameraPose
from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem


@dataclass
class MultiViewTriangulationResult:
    """Telemetry and diagnostics from multi-view incremental triangulation."""
    newly_triangulated_count: int
    total_candidates_evaluated: int
    filter_reasons: Dict[str, int] = field(default_factory=dict)
    mean_reprojection_error_px: float = 0.0
    mean_parallax_deg: float = 0.0

    def to_dict(self) -> dict:
        return {
            "newly_triangulated_count": self.newly_triangulated_count,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "yield_percent": round(
                (self.newly_triangulated_count / max(1, self.total_candidates_evaluated) * 100), 2
            ),
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 3),
            "mean_parallax_deg": round(self.mean_parallax_deg, 2),
            "filter_reasons": self.filter_reasons,
        }


class MultiViewTriangulator:
    """
    Solves linear N-view DLT triangulation for multi-view feature tracks.
    
    Mathematical Formulation:
    --------------------------
    For a physical 3D scene point X = [X, Y, Z, 1]^T observed across M >= 2 registered cameras:
    Each observing camera i has projection matrix P_i = K [R_i | t_i].
    The 2D observation x_i = [u_i, v_i]^T satisfies the cross-product constraint:
        x_i × (P_i X) = 0
    
    This yields two independent linear equations per observation:
        u_i P_i^{(3)} - P_i^{(1)} = 0
        v_i P_i^{(3)} - P_i^{(2)} = 0
    
    Stacking across all M cameras produces a (2M x 4) linear system:
        A X = 0
    
    Solved via Singular Value Decomposition (SVD): A = U Σ V^T.
    The solution X is the right singular vector in V corresponding to the smallest singular value.
    The Euclidean coordinate is obtained by dehomogenization: X_euclidean = [X_0/X_3, X_1/X_3, X_2/X_3]^T.
    
    Geometric Validation Filters:
    ------------------------------
    1. Finite Coordinate Filter: |X_3| >= 1e-7 (eliminates points at infinity).
    2. Strict Cheirality Filter: Z_i = (R_i X + t_i)_3 > 0 for ALL observing cameras.
    3. Parallax Filter: Max baseline angle between viewing rays >= min_parallax_deg.
    4. Reprojection Error Filter: Mean error across all observing views <= max_reproj_error.
    """

    def __init__(
        self,
        max_reprojection_error_px: float = 4.0,
        min_parallax_angle_deg: float = 1.5,
    ):
        self.max_reprojection_error = max_reprojection_error_px
        self.min_parallax_deg = min_parallax_angle_deg

    @staticmethod
    def linear_dlt_triangulate_nviews(
        projection_matrices: List[np.ndarray],
        pixel_coords: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """
        Solves A X = 0 for M >= 2 camera views using SVD.
        
        Args:
            projection_matrices: List of (3, 4) P_i = K [R_i | t_i]
            pixel_coords: List of (u_i, v_i) pixel observations
            
        Returns:
            (3,) Euclidean 3D coordinate in world frame, or None if degenerate.
        """
        M = len(projection_matrices)
        if M < 2:
            return None

        A = np.zeros((2 * M, 4), dtype=np.float64)
        for i in range(M):
            P = projection_matrices[i]
            u, v = pixel_coords[i]
            A[2 * i]     = u * P[2, :] - P[0, :]
            A[2 * i + 1] = v * P[2, :] - P[1, :]

        # SVD solution
        _, _, Vt = np.linalg.svd(A)
        X_hom = Vt[-1]

        # Check dehomogenization divisor
        if abs(X_hom[3]) < 1e-7:
            return None

        return X_hom[:3] / X_hom[3]

    def triangulate_track(
        self,
        track: FeatureTrack,
        registered_cameras: Dict[int, CameraPose],
        K: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], float, float, str]:
        """
        Attempts to triangulate a single multi-view feature track across registered cameras.
        
        Returns:
            (is_valid, point_3d, mean_reprojection_error, max_parallax_deg, filter_reason)
        """
        # Find which registered cameras observe this track
        obs_cams: List[CameraPose] = []
        obs_pixels: List[Tuple[float, float]] = []
        obs_proj_matrices: List[np.ndarray] = []

        for img_id, obs in track.observations.items():
            if img_id in registered_cameras and registered_cameras[img_id].is_registered:
                cam = registered_cameras[img_id]
                obs_cams.append(cam)
                obs_pixels.append(obs.coords)
                obs_proj_matrices.append(cam.projection_matrix(K))

        if len(obs_cams) < 2:
            return False, None, 0.0, 0.0, "insufficient_registered_views"

        # 1. Linear DLT solve
        X = self.linear_dlt_triangulate_nviews(obs_proj_matrices, obs_pixels)
        if X is None:
            return False, None, 0.0, 0.0, "dlt_dehomogenization_failure"

        # 2. Cheirality check (positive depth in ALL observing cameras)
        for cam in obs_cams:
            Z = cam.rotation_matrix[2, :] @ X + cam.translation_vector[2, 0]
            if Z <= 0.0:
                return False, None, 0.0, 0.0, "cheirality_violation"

        # 3. Parallax angle check
        max_parallax = 0.0
        M = len(obs_cams)
        rays = []
        for cam in obs_cams:
            ray = X - cam.camera_center
            norm = np.linalg.norm(ray)
            if norm < 1e-7:
                return False, None, 0.0, 0.0, "point_at_optical_center"
            rays.append(ray / norm)

        for i in range(M):
            for j in range(i + 1, M):
                dot = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(dot)))
                if angle > max_parallax:
                    max_parallax = angle

        if max_parallax < self.min_parallax_deg:
            return False, None, 0.0, max_parallax, "insufficient_parallax"

        # 4. Multi-view reprojection error check
        errors = []
        for i in range(M):
            P = obs_proj_matrices[i]
            hom = np.append(X, 1.0)
            proj = P @ hom
            if abs(proj[2]) < 1e-7:
                return False, None, 0.0, max_parallax, "reprojection_singularity"
            px_est = proj[:2] / proj[2]
            px_obs = np.array(obs_pixels[i])
            err = float(np.linalg.norm(px_est - px_obs))
            errors.append(err)

        mean_err = float(np.mean(errors))
        if mean_err > self.max_reprojection_error:
            return False, None, mean_err, max_parallax, "reprojection_error_exceeded"

        return True, X, mean_err, max_parallax, "success"

    def triangulate_unreconstructed_tracks(
        self,
        track_system: FeatureTrackSystem,
        registered_cameras: Dict[int, CameraPose],
        camera_matrix_k: np.ndarray,
    ) -> MultiViewTriangulationResult:
        """
        Iterates over all un-triangulated feature tracks and reconstructs valid 3D points.
        """
        new_count = 0
        total_eval = 0
        reasons: Dict[str, int] = {}
        all_errors = []
        all_parallaxes = []

        registered_ids = {img_id for img_id, cam in registered_cameras.items() if cam.is_registered}

        for track in track_system.tracks.values():
            if track.is_triangulated:
                continue

            # Check if observed by at least 2 registered cameras
            common = track.get_image_ids().intersection(registered_ids)
            if len(common) < 2:
                continue

            total_eval += 1
            valid, X, mean_err, parallax, reason = self.triangulate_track(
                track, registered_cameras, camera_matrix_k
            )
            reasons[reason] = reasons.get(reason, 0) + 1

            if valid and X is not None:
                track.point_3d = X
                track.is_triangulated = True
                track.reprojection_error = mean_err
                new_count += 1
                all_errors.append(mean_err)
                all_parallaxes.append(parallax)

        return MultiViewTriangulationResult(
            newly_triangulated_count=new_count,
            total_candidates_evaluated=total_eval,
            filter_reasons=reasons,
            mean_reprojection_error_px=float(np.mean(all_errors)) if all_errors else 0.0,
            mean_parallax_deg=float(np.mean(all_parallaxes)) if all_parallaxes else 0.0,
        )

