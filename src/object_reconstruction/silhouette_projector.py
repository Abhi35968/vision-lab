"""
Silhouette Projector Module (Phase 5.5).
Projects 3D mesh geometry into calibrated camera views and computes silhouette IoU
against ground-truth binary segmentation masks.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.object_reconstruction.bottle_models import Mesh3D, SilhouetteViewMetric


class SilhouetteProjector:
    """
    Renders 2D binary silhouettes from a 3D mesh and evaluates silhouette IoU.
    """

    def __init__(
        self,
        intrinsic_matrix: np.ndarray,
        image_shape: Tuple[int, int] = (1920, 1080),  # (height, width)
    ):
        self.K = np.asarray(intrinsic_matrix, dtype=np.float64)
        self.image_shape = image_shape

    def project_and_rasterize(
        self,
        mesh: Mesh3D,
        R: np.ndarray,
        t: np.ndarray,
        downsample_factor: int = 1,
    ) -> Tuple[np.ndarray, bool]:
        """
        Projects 3D mesh into camera (R, t) and rasterizes front-facing triangles into binary mask.

        Args:
            mesh: 3D mesh in world coordinates.
            R: (3, 3) rotation matrix from world to camera.
            t: (3, 1) translation vector from world to camera.
            downsample_factor: Downsampling factor for fast optimization search (1 = full res).

        Returns:
            Tuple of (binary_mask (H, W) uint8 in {0, 255}, is_valid_projection).
        """
        H_orig, W_orig = self.image_shape
        H = H_orig // downsample_factor
        W = W_orig // downsample_factor

        # Scale intrinsic matrix if downsampling
        if downsample_factor > 1:
            K_scaled = self.K.copy()
            K_scaled[0, 0] /= downsample_factor
            K_scaled[1, 1] /= downsample_factor
            K_scaled[0, 2] /= downsample_factor
            K_scaled[1, 2] /= downsample_factor
        else:
            K_scaled = self.K

        V = mesh.vertices  # (N, 3)
        t_vec = np.asarray(t, dtype=np.float64).reshape(3, 1)
        R_mat = np.asarray(R, dtype=np.float64).reshape(3, 3)

        # Transform to camera frame: V_cam = R @ V.T + t
        V_cam = (R_mat @ V.T + t_vec).T  # (N, 3)

        # Cheirality check: if most vertices are behind camera, projection is invalid
        z_vals = V_cam[:, 2]
        if np.sum(z_vals > 0.1) < 0.8 * len(V):
            return np.zeros((H, W), dtype=np.uint8), False

        # Perspective projection
        z_safe = np.maximum(z_vals, 0.05)
        u = (K_scaled[0, 0] * V_cam[:, 0] + K_scaled[0, 2] * z_safe) / z_safe
        v = (K_scaled[1, 1] * V_cam[:, 1] + K_scaled[1, 2] * z_safe) / z_safe
        pts2d = np.column_stack([u, v])

        # 2D Screen-space back-face culling to prevent even-odd winding overlap cancellation
        v0 = pts2d[mesh.faces[:, 0]]
        v1 = pts2d[mesh.faces[:, 1]]
        v2 = pts2d[mesh.faces[:, 2]]
        # 2D signed cross product
        area2d = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])

        # Keep front-facing triangles with positive screen area
        front_faces = mesh.faces[area2d > 0.0]
        if len(front_faces) == 0:
            return np.zeros((H, W), dtype=np.uint8), False

        front_triangles = pts2d[front_faces].astype(np.int32)

        canvas = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(canvas, list(front_triangles), 255)

        return canvas, True

    def evaluate_view(
        self,
        camera_id: int,
        image_name: str,
        is_train_split: bool,
        mesh: Mesh3D,
        R: np.ndarray,
        t: np.ndarray,
        observed_mask: Optional[np.ndarray],
    ) -> Optional[SilhouetteViewMetric]:
        """
        Computes silhouette overlap statistics between projected model and observed mask.
        Returns None if observed mask is missing or projection is behind camera.
        """
        if observed_mask is None or np.count_nonzero(observed_mask > 0) == 0:
            return None

        rendered_mask, is_valid = self.project_and_rasterize(
            mesh=mesh,
            R=R,
            t=t,
            downsample_factor=1,
        )
        if not is_valid:
            return None

        # Ensure shapes match
        if rendered_mask.shape != observed_mask.shape:
            rendered_mask = cv2.resize(
                rendered_mask,
                (observed_mask.shape[1], observed_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        rend_bin = rendered_mask > 0
        obs_bin = observed_mask > 0

        inter_px = int(np.count_nonzero(rend_bin & obs_bin))
        union_px = int(np.count_nonzero(rend_bin | obs_bin))
        rend_px = int(np.count_nonzero(rend_bin))
        obs_px = int(np.count_nonzero(obs_bin))

        iou = (inter_px / union_px) if union_px > 0 else 0.0
        precision = (inter_px / rend_px) if rend_px > 0 else 0.0
        recall = (inter_px / obs_px) if obs_px > 0 else 0.0

        return SilhouetteViewMetric(
            camera_id=camera_id,
            image_name=image_name,
            is_train_split=is_train_split,
            intersection_px=inter_px,
            union_px=union_px,
            iou=round(float(iou), 4),
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            observed_mask_px=obs_px,
            rendered_mask_px=rend_px,
        )

