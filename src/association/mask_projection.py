"""
Camera Projection and Mask Sampling Module
Provides rigorous geometric projection of 3D world points into camera views,
validates depths, image bounds, and finite numerical constraints,
and evaluates binary segmentation mask membership with boundary distance metrics.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class ProjectionResult:
    """Represents the validated geometric projection of a 3D point into a camera view."""
    is_valid: bool
    depth: float = 0.0
    proj_u: float = 0.0
    proj_v: float = 0.0
    is_inside_image: bool = False
    rejection_reason: Optional[str] = None

    @property
    def coords(self) -> Tuple[float, float]:
        """Returns projected (u, v) pixel coordinates."""
        return (self.proj_u, self.proj_v)


@dataclass
class MaskSampleResult:
    """Represents mask occupancy and boundary proximity for a 2D pixel coordinate."""
    is_valid: bool
    is_inside: bool = False
    mask_value: int = 0
    distance_to_boundary_px: float = 0.0
    rejection_reason: Optional[str] = None


class CameraProjector:
    """Handles pinhole camera projection and binary mask occupancy sampling."""

    @staticmethod
    def project_point(
        point_3d_world: Union[np.ndarray, Tuple[float, float, float], list],
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
        camera_matrix_k: np.ndarray,
        image_size: Tuple[int, int],  # (width, height)
        min_depth: float = 1e-4,
    ) -> ProjectionResult:
        """
        Projects a 3D world point into a calibrated camera image plane.

        Args:
            point_3d_world: (X, Y, Z) in world coordinates.
            rotation_matrix: 3x3 camera rotation matrix R.
            translation_vector: 3x1 or 3-element translation vector t.
            camera_matrix_k: 3x3 pinhole intrinsic matrix K.
            image_size: (width, height) of the image canvas in pixels.
            min_depth: Minimum positive depth threshold along optical axis.

        Returns:
            ProjectionResult containing depth, projected pixel coordinates, and validity flags.
        """
        X_w = np.asarray(point_3d_world, dtype=np.float64).reshape(3, 1)

        # Check finite coordinates
        if not np.all(np.isfinite(X_w)):
            return ProjectionResult(
                is_valid=False,
                rejection_reason="Point coordinates contain non-finite values (NaN or Inf)"
            )

        R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        t = np.asarray(translation_vector, dtype=np.float64).reshape(3, 1)
        K = np.asarray(camera_matrix_k, dtype=np.float64).reshape(3, 3)

        # Camera-space transformation: X_c = R * X_w + t
        X_c = R @ X_w + t
        depth = float(X_c[2, 0])

        # Positive depth / cheirality check
        if depth < min_depth:
            return ProjectionResult(
                is_valid=False,
                depth=depth,
                rejection_reason=f"Point is behind or on the camera plane (depth {depth:.4f} < {min_depth})"
            )

        # Pinhole projection: x_hom = K * X_c
        p_hom = K @ X_c
        u_proj = float(p_hom[0, 0] / depth)
        v_proj = float(p_hom[1, 0] / depth)

        if not (np.isfinite(u_proj) and np.isfinite(v_proj)):
            return ProjectionResult(
                is_valid=False,
                depth=depth,
                rejection_reason="Projected coordinates resulted in non-finite values"
            )

        width, height = image_size
        inside_image = (0.0 <= u_proj < float(width)) and (0.0 <= v_proj < float(height))

        if not inside_image:
            return ProjectionResult(
                is_valid=False,
                depth=depth,
                proj_u=round(u_proj, 3),
                proj_v=round(v_proj, 3),
                is_inside_image=False,
                rejection_reason=f"Projected point ({u_proj:.1f}, {v_proj:.1f}) lies outside image boundaries ({width}x{height})"
            )

        return ProjectionResult(
            is_valid=True,
            depth=round(depth, 4),
            proj_u=round(u_proj, 3),
            proj_v=round(v_proj, 3),
            is_inside_image=True,
            rejection_reason=None,
        )

    @staticmethod
    def compute_reprojection_error(
        proj_coords: Tuple[float, float],
        obs_coords: Tuple[float, float],
    ) -> float:
        """Computes Euclidean reprojection error in pixels."""
        du = proj_coords[0] - obs_coords[0]
        dv = proj_coords[1] - obs_coords[1]
        return round(float(np.sqrt(du * du + dv * dv)), 4)

    @staticmethod
    def sample_mask(
        coords: Tuple[float, float],
        mask: Optional[np.ndarray],
        dist_transform: Optional[np.ndarray] = None,
    ) -> MaskSampleResult:
        """
        Samples binary mask value at specified (u, v) coordinates.

        Args:
            coords: (u, v) pixel coordinates (u=column, v=row).
            mask: 2D uint8 binary mask array of shape (height, width).
            dist_transform: Optional precomputed signed distance transform.

        Returns:
            MaskSampleResult with occupancy boolean and distance to mask boundary.
        """
        if mask is None:
            return MaskSampleResult(
                is_valid=False,
                rejection_reason="Mask array is None"
            )

        if len(mask.shape) != 2:
            return MaskSampleResult(
                is_valid=False,
                rejection_reason=f"Expected 2D mask, but got shape {mask.shape}"
            )

        h, w = mask.shape
        u, v = coords

        if not (np.isfinite(u) and np.isfinite(v)):
            return MaskSampleResult(
                is_valid=False,
                rejection_reason="Sampling coordinates contain NaN or Inf"
            )

        col = int(round(u))
        row = int(round(v))

        # Check bounds
        if not (0 <= col < w and 0 <= row < h):
            return MaskSampleResult(
                is_valid=False,
                rejection_reason=f"Coordinate ({col}, {row}) out of mask bounds ({w}x{h})"
            )

        val = int(mask[row, col])
        is_inside = (val > 0)

        # Distance to boundary calculation
        dist = 0.0
        if dist_transform is not None:
            if dist_transform.shape == mask.shape:
                dist = float(dist_transform[row, col])
        else:
            # Quick 8-neighborhood boundary check if distance transform is not provided
            if is_inside:
                # Check if any neighbor is 0 (boundary pixel)
                r_min, r_max = max(0, row - 1), min(h, row + 2)
                c_min, c_max = max(0, col - 1), min(w, col + 2)
                if np.any(mask[r_min:r_max, c_min:c_max] == 0):
                    dist = 0.5  # On mask contour boundary
                else:
                    dist = 1.0  # Interior
            else:
                dist = -1.0  # Exterior

        return MaskSampleResult(
            is_valid=True,
            is_inside=is_inside,
            mask_value=val,
            distance_to_boundary_px=round(dist, 2),
            rejection_reason=None,
        )

