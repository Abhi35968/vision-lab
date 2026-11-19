"""
Pose Estimator Module
Recovers relative camera rotation R and translation direction t from the Essential Matrix.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass
class PoseEstimationResult:
    """Relative camera motion parameters and Cheirality verification telemetry."""
    rotation_matrix: Optional[np.ndarray]       # (3, 3) float64 in SO(3)
    translation_vector: Optional[np.ndarray]    # (3, 1) float64 unit direction in S^2
    cheirality_inliers_count: int              # Number of points lying in front of both cameras
    cheirality_ratio: float                    # Fraction of inliers with positive depth
    camera1_center: np.ndarray                 # World position of Cam 1: [0, 0, 0]^T
    camera2_center: Optional[np.ndarray]       # World position of Cam 2: -R^T * t
    rotation_angles_deg: Tuple[float, float, float] # (roll, pitch, yaw) Euler angles
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "rotation_matrix": self.rotation_matrix.tolist() if self.rotation_matrix is not None else None,
            "translation_direction": self.translation_vector.ravel().tolist() if self.translation_vector is not None else None,
            "camera1_center": self.camera1_center.tolist(),
            "camera2_center": self.camera2_center.tolist() if self.camera2_center is not None else None,
            "cheirality_inliers_count": self.cheirality_inliers_count,
            "cheirality_ratio": round(self.cheirality_ratio, 4),
            "rotation_euler_deg": {
                "roll": round(self.rotation_angles_deg[0], 2),
                "pitch": round(self.rotation_angles_deg[1], 2),
                "yaw": round(self.rotation_angles_deg[2], 2),
            },
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
            "scale_note": "Monocular reconstruction recovers translation direction up to an arbitrary scale factor (unit norm). Absolute metric dimensions require ground truth scale.",
        }


class PoseEstimator:
    """
    Decomposes the Essential Matrix E to recover relative camera pose [R | t].
    
    Coordinate System Convention:
    -----------------------------
    1. Reference Camera (Camera 1):
       The world coordinate system is fixed to Camera 1's optical center:
           R₁ = I₃ₓ₃,   t₁ = [0, 0, 0]ᵀ
           P₁ = K [I | 0]
    
    2. Target Camera (Camera 2):
       A 3D scene point X₁ defined in Camera 1's coordinate frame is transformed into
       Camera 2's frame according to:
           X₂ = R X₁ + t
           P₂ = K [R | t]
       The optical center of Camera 2 in world coordinates (Camera 1 frame) is:
           C₂ = -Rᵀ t
    
    Cheirality Condition:
    ---------------------
    Decomposing E via SVD yields 4 mathematically valid combinations of (R, t):
        (R_A, +t), (R_A, -t), (R_B, +t), (R_B, -t)
    Only ONE physical solution places the triangulated 3D points strictly in front of
    both camera sensors (positive depth Z₁ > 0 and Z₂ > 0). cv2.recoverPose evaluates
    this Cheirality constraint and selects the physically true pose.
    
    Scale Ambiguity:
    ----------------
    Monocular Structure from Motion is inherently scale-ambiguous. The recovered translation
    vector t has unit Euclidean norm (||t|| = 1.0) and represents purely a direction of travel,
    NOT a physical distance in meters.
    """

    def __init__(self, min_cheirality_ratio: float = 0.70):
        self.min_cheirality_ratio = min_cheirality_ratio

    @staticmethod
    def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
        """Converts 3x3 rotation matrix to Roll, Pitch, Yaw angles in degrees."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0

        return (
            float(np.degrees(roll)),
            float(np.degrees(pitch)),
            float(np.degrees(yaw)),
        )

    def estimate_pose(
        self,
        essential_matrix: np.ndarray,
        inlier_pts1: np.ndarray,
        inlier_pts2: np.ndarray,
        camera_matrix_k: np.ndarray,
    ) -> PoseEstimationResult:
        """
        Decomposes Essential Matrix into valid (R, t) passing Cheirality checks.
        """
        """
        Decomposes Essential Matrix into valid (R, t) passing Cheirality checks.
        """
        total_inliers = len(inlier_pts1)
        if total_inliers < 5 or essential_matrix is None:
            return PoseEstimationResult(
                rotation_matrix=None,
                translation_vector=None,
                cheirality_inliers_count=0,
                cheirality_ratio=0.0,
                camera1_center=np.zeros(3, dtype=np.float64),
                camera2_center=None,
                rotation_angles_deg=(0.0, 0.0, 0.0),
                is_valid=False,
                rejection_reason="Insufficient inlier points (< 5) or missing Essential Matrix.",
            )

        # Decompose E and perform Cheirality check via OpenCV recoverPose
        num_valid, R, t, cheirality_mask = cv2.recoverPose(
            essential_matrix,
            inlier_pts1,
            inlier_pts2,
            camera_matrix_k,
        )

        cheirality_ratio = num_valid / float(total_inliers) if total_inliers > 0 else 0.0
        is_valid = cheirality_ratio >= self.min_cheirality_ratio

        rejection_reason = None
        if not is_valid:
            rejection_reason = (
                f"Cheirality failure: Only {num_valid}/{total_inliers} ({cheirality_ratio:.1%}) "
                f"points have positive depth (threshold: {self.min_cheirality_ratio:.1%})."
            )

        # Normalize translation to unit length
        t_norm = np.linalg.norm(t)
        t_unit = t / t_norm if t_norm > 1e-9 else t

        # Camera 2 optical center in Camera 1 frame: C2 = -R^T * t
        cam2_center = (-R.T @ t_unit).ravel()
        euler_deg = self.rotation_matrix_to_euler_angles(R)

        return PoseEstimationResult(
            rotation_matrix=R,
            translation_vector=t_unit,
            cheirality_inliers_count=num_valid,
            cheirality_ratio=cheirality_ratio,
            camera1_center=np.zeros(3, dtype=np.float64),
            camera2_center=cam2_center,
            rotation_angles_deg=euler_deg,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        )

    # Alias for compatibility
    recover_pose = estimate_pose

