"""
Triangulator Module
Reconstructs 3D point clouds in arbitrary_units from verified 2D correspondences using Linear DLT.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
try:
    import open3d as o3d
except Exception:
    o3d = None


@dataclass
class TriangulationResult:
    """3D point cloud, reprojection statistics, and 2D-to-3D correspondence map."""
    points_3d: np.ndarray                      # (N, 3) float64 Euclidean 3D coordinates
    point_colors: Optional[np.ndarray]        # (N, 3) float64 RGB colors in [0, 1]
    reprojection_errors: np.ndarray           # (N,) float64 mean error in pixels
    mean_reprojection_error: float            # Global RMSE/mean error
    median_reprojection_error: float
    total_candidates: int
    points_reconstructed: int
    filter_reasons: Dict[str, int]
    is_valid: bool
    correspondence_tracks: List[dict]         # List of {point_id, pt1_px, pt2_px, error_px}

    def to_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "points_reconstructed": self.points_reconstructed,
            "reconstruction_yield_percent": round(
                (self.points_reconstructed / self.total_candidates * 100), 2
            ) if self.total_candidates > 0 else 0.0,
            "mean_reprojection_error_px": round(self.mean_reprojection_error, 3),
            "median_reprojection_error_px": round(self.median_reprojection_error, 3),
            "filter_reasons": self.filter_reasons,
            "is_valid": self.is_valid,
        }


class Triangulator:
    """
    Computes 3D world coordinates from two-view pixel correspondences.
    
    Mathematical Foundations:
    -------------------------
    1. Linear Triangulation via DLT:
       Given projection matrices P₁ = K [I | 0] and P₂ = K [R | t], homogeneous 3D point
       X = [X, Y, Z, W]ᵀ satisfies x₁ × (P₁ X) = 0 and x₂ × (P₂ X) = 0.
       cv2.triangulatePoints solves the resulting 4x4 linear algebraic system via SVD:
           A X = 0
    
    2. Dehomogenization:
       Convert homogeneous coordinate X = [X₁, X₂, X₃, W]ᵀ to Euclidean R³:
           P_euclidean = [X₁/W, X₂/W, X₃/W]ᵀ
    
    3. Geometric Point Validation:
       A triangulated point is rejected if:
       - |W| < 1e-6 (point at infinity or near-parallel rays).
       - Behind Camera 1: Z₁ <= 0.
       - Behind Camera 2: Z₂ = (R P_euclidean + t)₃ <= 0.
       - Parallax ray angle < min_angle_deg (ill-conditioned baseline).
       - Mean reprojection error > max_reprojection_error_px.
    """

    def __init__(
        self,
        max_reprojection_error_px: float = 3.0,
        min_parallax_angle_deg: float = 1.0,
        min_points_threshold: int = 10,
    ):
        self.max_reprojection_error = max_reprojection_error_px
        self.min_parallax_deg = min_parallax_angle_deg
        self.min_points_threshold = min_points_threshold

    @staticmethod
    def project_point(P: np.ndarray, point_3d: np.ndarray) -> np.ndarray:
        """Projects Euclidean 3D point (3,) into 2D pixel coordinates (2,) using P = K [R | t]."""
        hom_3d = np.append(point_3d, 1.0)
        proj = P @ hom_3d
        z = proj[2]
        if abs(z) < 1e-9:
            return np.array([np.nan, np.nan], dtype=np.float64)
        return proj[:2] / z

    def triangulate(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        camera_matrix_k: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        image1_bgr: Optional[np.ndarray] = None,
    ) -> TriangulationResult:
        """
        Triangulates matched 2D inliers and filters invalid 3D points.
        """
        N = len(pts1)
        if N == 0:
            return TriangulationResult(
                points_3d=np.empty((0, 3), dtype=np.float64),
                point_colors=None,
                reprojection_errors=np.empty((0,), dtype=np.float64),
                mean_reprojection_error=0.0,
                median_reprojection_error=0.0,
                total_candidates=0,
                points_reconstructed=0,
                filter_reasons={},
                is_valid=False,
                correspondence_tracks=[],
            )

        # 1. Construct Camera Projection Matrices
        # Camera 1 (Reference): P1 = K [I | 0]
        # Camera 2 (Target):    P2 = K [R | t]
        P1 = camera_matrix_k @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
        P2 = camera_matrix_k @ np.hstack([R, t.reshape(3, 1)])

        # 2. Linear DLT Triangulation
        # Input to cv2.triangulatePoints must be (2, N) float32
        pts1_t = np.ascontiguousarray(pts1.T, dtype=np.float32)
        pts2_t = np.ascontiguousarray(pts2.T, dtype=np.float32)
        pts4d_hom = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)  # (4, N)

        # 3. Geometric Filtering and Point Verification
        valid_points_3d: List[np.ndarray] = []
        valid_colors: List[np.ndarray] = []
        valid_errors: List[float] = []
        valid_tracks: List[dict] = []

        filter_counts = {
            "w_near_zero": 0,
            "behind_cam1": 0,
            "behind_cam2": 0,
            "insufficient_parallax": 0,
            "high_reprojection_error": 0,
        }

        # Camera centers in world frame (Camera 1 frame)
        c1 = np.zeros(3, dtype=np.float64)
        c2 = (-R.T @ t).ravel()

        h_img, w_img = image1_bgr.shape[:2] if image1_bgr is not None else (0, 0)

        for i in range(N):
            W = pts4d_hom[3, i]
            if abs(W) < 1e-6:
                filter_counts["w_near_zero"] += 1
                continue

            # Dehomogenize
            X1 = pts4d_hom[:3, i] / W

            # Cheirality check in Camera 1: Z1 > 0
            if X1[2] <= 0:
                filter_counts["behind_cam1"] += 1
                continue

            # Cheirality check in Camera 2: Z2 = (R X1 + t)_3 > 0
            X2 = R @ X1 + t.ravel()
            if X2[2] <= 0:
                filter_counts["behind_cam2"] += 1
                continue

            # Parallax ray angle check
            ray1 = X1 - c1
            ray2 = X1 - c2
            norm1 = np.linalg.norm(ray1)
            norm2 = np.linalg.norm(ray2)
            if norm1 > 1e-9 and norm2 > 1e-9:
                cos_theta = np.dot(ray1, ray2) / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                parallax_deg = np.degrees(np.arccos(cos_theta))
                if parallax_deg < self.min_parallax_deg:
                    filter_counts["insufficient_parallax"] += 1
                    continue

            # Reprojection error computation
            proj1 = self.project_point(P1, X1)
            proj2 = self.project_point(P2, X1)

            err1 = np.linalg.norm(pts1[i] - proj1)
            err2 = np.linalg.norm(pts2[i] - proj2)
            mean_err = (err1 + err2) / 2.0

            if mean_err > self.max_reprojection_error or np.isnan(mean_err):
                filter_counts["high_reprojection_error"] += 1
                continue

            # Point passed all geometric gates
            valid_points_3d.append(X1)
            valid_errors.append(mean_err)

            # Sample RGB color from Image 1 pixel
            if image1_bgr is not None:
                px = int(round(np.clip(pts1[i, 0], 0, w_img - 1)))
                py = int(round(np.clip(pts1[i, 1], 0, h_img - 1)))
                bgr = image1_bgr[py, px]
                rgb = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float64) / 255.0
                valid_colors.append(rgb)

            # Record 2D-to-3D traceability
            valid_tracks.append({
                "point_id": len(valid_points_3d) - 1,
                "candidate_idx": i,
                "point_3d": X1.tolist(),
                "pt1_pixel": [round(float(pts1[i, 0]), 2), round(float(pts1[i, 1]), 2)],
                "pt2_pixel": [round(float(pts2[i, 0]), 2), round(float(pts2[i, 1]), 2)],
                "reproj_error_px": round(mean_err, 3),
            })

        reconstructed_count = len(valid_points_3d)
        is_valid = reconstructed_count >= self.min_points_threshold

        pts3d_arr = np.array(valid_points_3d, dtype=np.float64) if reconstructed_count > 0 else np.empty((0, 3), dtype=np.float64)
        colors_arr = np.array(valid_colors, dtype=np.float64) if valid_colors else None
        errors_arr = np.array(valid_errors, dtype=np.float64) if valid_errors else np.empty((0,), dtype=np.float64)

        mean_rmse = float(np.mean(errors_arr)) if len(errors_arr) > 0 else 0.0
        median_rmse = float(np.median(errors_arr)) if len(errors_arr) > 0 else 0.0

        return TriangulationResult(
            points_3d=pts3d_arr,
            point_colors=colors_arr,
            reprojection_errors=errors_arr,
            mean_reprojection_error=mean_rmse,
            median_reprojection_error=median_rmse,
            total_candidates=N,
            points_reconstructed=reconstructed_count,
            filter_reasons=filter_counts,
            is_valid=is_valid,
            correspondence_tracks=valid_tracks,
        )

    @staticmethod
    def save_point_cloud(
        points_3d: np.ndarray,
        colors: Optional[np.ndarray],
        output_ply_path: Union[str, Path],
        output_npy_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Saves sparse 3D point cloud to .ply (via Open3D) and numpy .npy."""
        if len(points_3d) == 0:
            return False

        ply_p = Path(output_ply_path).resolve()
        ply_p.parent.mkdir(parents=True, exist_ok=True)

        # Write PLY file
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            if colors is not None and len(colors) == len(points_3d):
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(ply_p), pcd)
        else:
            try:
                import trimesh
                pc = trimesh.PointCloud(vertices=points_3d, colors=colors)
                pc.export(str(ply_p))
            except Exception:
                pass

        # Write NumPy array file
        if output_npy_path is not None:
            npy_p = Path(output_npy_path).resolve()
            np.save(str(npy_p), points_3d)

        return True

