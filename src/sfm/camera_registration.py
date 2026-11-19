"""
Camera Registration Module
Progressively registers candidate keyframes into the 3D scene using Perspective-n-Point (PnP) and RANSAC.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem


@dataclass
class CameraPose:
    """Represents a registered camera's extrinsic parameters in the world coordinate system."""
    image_id: int
    image_name: str
    rotation_matrix: np.ndarray         # (3, 3) float64
    translation_vector: np.ndarray      # (3, 1) float64
    camera_center: np.ndarray           # (3,) float64: C = -R^T * t
    rvec: np.ndarray                    # (3, 1) Rodrigues rotation vector
    tvec: np.ndarray                    # (3, 1) translation vector
    num_2d_3d_matches: int
    num_pnp_inliers: int
    inlier_ratio: float
    mean_reprojection_error_px: float
    is_registered: bool
    rejection_reason: Optional[str] = None

    def projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """Returns 3x4 camera projection matrix P = K [R | t]."""
        Rt = np.hstack([self.rotation_matrix, self.translation_vector.reshape(3, 1)])
        return K @ Rt

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "rotation_matrix": self.rotation_matrix.tolist() if self.rotation_matrix is not None else None,
            "translation_vector": self.translation_vector.ravel().tolist() if self.translation_vector is not None else None,
            "camera_center": self.camera_center.tolist() if self.camera_center is not None else None,
            "num_2d_3d_matches": self.num_2d_3d_matches,
            "num_pnp_inliers": self.num_pnp_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 3),
            "is_registered": self.is_registered,
            "rejection_reason": self.rejection_reason,
        }


class CameraRegistrar:
    """
    Registers new camera views using Perspective-n-Point (PnP) with RANSAC outlier rejection.
    
    Why PnP is Needed:
    ------------------
    Two-view Epipolar Geometry (Essential Matrix) only gives relative motion between TWO cameras
    at an arbitrary relative scale. As we add a 3rd, 4th, or Nth camera, we cannot independently
    triangulate each pair, or every pair would have a completely incompatible scale and origin!
    
    Instead, PnP directly localizes a new 2D image against the already reconstructed 3D points
    in the world coordinate system, maintaining exact gauge consistency and scale across all views.
    
    Workflow:
    ---------
    1. Identify all 3D scene points observed by the candidate camera (via Feature Tracks).
    2. Construct 2D-3D correspondences: X_j (3D world) <-> x_ij (2D pixel).
    3. Solve PnP via RANSAC (cv2.solvePnPRansac) to eliminate false 2D-3D matches.
    4. Refine pose nonlinearly via Levenberg-Marquardt (cv2.solvePnPRefineLM).
    5. Validate inlier count and reprojection RMSE against strict acceptance criteria.
    6. Rejection handling: If criteria are not met, camera is rejected without guessing.
    """

    def __init__(
        self,
        min_pnp_inliers: int = 15,
        max_reprojection_error_px: float = 4.0,
        ransac_confidence: float = 0.999,
        ransac_iterations: int = 2000,
    ):
        self.min_pnp_inliers = min_pnp_inliers
        self.max_reprojection_error = max_reprojection_error_px
        self.ransac_confidence = ransac_confidence
        self.ransac_iterations = ransac_iterations

    @staticmethod
    def select_next_best_view(
        candidate_image_ids: List[int],
        track_system: FeatureTrackSystem,
    ) -> Tuple[Optional[int], int]:
        """
        Next-Best-View (NBV) selection:
        Finds the unregistered camera that observes the maximum number of already triangulated 3D points.
        """
        best_img_id = None
        max_points = -1

        for img_id in candidate_image_ids:
            tri_tracks = track_system.get_triangulated_tracks_for_image(img_id)
            count = len(tri_tracks)
            if count > max_points:
                max_points = count
                best_img_id = img_id

        return best_img_id, max_points

    def register_camera(
        self,
        image_id: int,
        image_name: str,
        camera_matrix_k: np.ndarray,
        track_system: FeatureTrackSystem,
    ) -> CameraPose:
        """
        Attempts to register camera `image_id` into the current reconstruction.
        """
        tri_tracks = track_system.get_triangulated_tracks_for_image(image_id)
        num_matches = len(tri_tracks)

        # Early check: minimum correspondence threshold
        if num_matches < self.min_pnp_inliers:
            return CameraPose(
                image_id=image_id,
                image_name=image_name,
                rotation_matrix=np.empty((0, 0)),
                translation_vector=np.empty((0, 0)),
                camera_center=np.empty((0,)),
                rvec=np.empty((0,)),
                tvec=np.empty((0,)),
                num_2d_3d_matches=num_matches,
                num_pnp_inliers=0,
                inlier_ratio=0.0,
                mean_reprojection_error_px=999.0,
                is_registered=False,
                rejection_reason=f"Insufficient 2D-3D correspondences ({num_matches} < {self.min_pnp_inliers} required).",
            )

        # Construct 2D-3D correspondence matrices
        pts_3d = []
        pts_2d = []
        track_ids = []

        for track in tri_tracks:
            obs = track.observations[image_id]
            pts_3d.append(track.point_3d)
            pts_2d.append(obs.coords)
            track_ids.append(track.track_id)

        obj_pts = np.asarray(pts_3d, dtype=np.float64).reshape(-1, 3)
        img_pts = np.asarray(pts_2d, dtype=np.float64).reshape(-1, 2)

        # 1. PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=camera_matrix_k,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.max_reprojection_error,
            confidence=self.ransac_confidence,
            iterationsCount=self.ransac_iterations,
        )

        if not success or inliers is None or len(inliers) < self.min_pnp_inliers:
            inlier_count = len(inliers) if inliers is not None else 0
            return CameraPose(
                image_id=image_id,
                image_name=image_name,
                rotation_matrix=np.empty((0, 0)),
                translation_vector=np.empty((0, 0)),
                camera_center=np.empty((0,)),
                rvec=np.empty((0,)),
                tvec=np.empty((0,)),
                num_2d_3d_matches=num_matches,
                num_pnp_inliers=inlier_count,
                inlier_ratio=float(inlier_count / max(1, num_matches)),
                mean_reprojection_error_px=999.0,
                is_registered=False,
                rejection_reason=f"PnP RANSAC failed or inliers too low ({inlier_count} < {self.min_pnp_inliers}).",
            )

        inlier_indices = inliers.ravel()
        num_inliers = len(inlier_indices)
        inlier_ratio = float(num_inliers / num_matches)

        # 2. Nonlinear Levenberg-Marquardt refinement on inlier set
        try:
            rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                objectPoints=obj_pts[inlier_indices],
                imagePoints=img_pts[inlier_indices],
                cameraMatrix=camera_matrix_k,
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
            )
            rvec, tvec = rvec_ref, tvec_ref
        except Exception:
            # Fall back to RANSAC rvec/tvec if refineLM encounters ill-conditioning
            pass

        # 3. Compute rotation matrix and optical center
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        C = (-R.T @ t).ravel()

        # 4. Inlier reprojection error evaluation
        proj_pts, _ = cv2.projectPoints(obj_pts[inlier_indices], rvec, tvec, camera_matrix_k, None)
        proj_pts = proj_pts.reshape(-1, 2)
        errors = np.linalg.norm(img_pts[inlier_indices] - proj_pts, axis=1)
        mean_reproj_error = float(np.mean(errors))

        # Check acceptance threshold
        if mean_reproj_error > self.max_reprojection_error:
            return CameraPose(
                image_id=image_id,
                image_name=image_name,
                rotation_matrix=R,
                translation_vector=t,
                camera_center=C,
                rvec=rvec,
                tvec=tvec,
                num_2d_3d_matches=num_matches,
                num_pnp_inliers=num_inliers,
                inlier_ratio=inlier_ratio,
                mean_reprojection_error_px=mean_reproj_error,
                is_registered=False,
                rejection_reason=f"Reprojection error exceeded threshold ({mean_reproj_error:.2f} > {self.max_reprojection_error} px).",
            )

        return CameraPose(
            image_id=image_id,
            image_name=image_name,
            rotation_matrix=R,
            translation_vector=t,
            camera_center=C,
            rvec=rvec,
            tvec=tvec,
            num_2d_3d_matches=num_matches,
            num_pnp_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            mean_reprojection_error_px=mean_reproj_error,
            is_registered=True,
        )

