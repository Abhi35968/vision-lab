"""
Camera Calibration Module
Implements classical Zhang checkerboard calibration using OpenCV and estimates intrinsic matrix K.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class CalibrationResult:
    """Complete camera calibration parameters and error telemetry."""
    status: str                         # "calibrated" or "uncalibrated"
    camera_matrix_k: List[List[float]]  # 3x3 Intrinsic matrix
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_coefficients: List[float] # [k1, k2, p1, p2, k3]
    distortion_model: str               # "radtan"
    image_width: int
    image_height: int
    checkerboard_rows: int              # Internal corner rows
    checkerboard_cols: int              # Internal corner columns
    square_size_mm: float
    unit: str
    reprojection_error_rmse: float
    total_images_provided: int
    images_used: int
    calibrated_at: str
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


class CameraCalibrator:
    """
    Performs camera calibration using multi-view planar checkerboard captures.
    
    Mathematical Foundations:
    1. Perspective Pinhole Camera Matrix (K):
           K = [ fx   0   cx ]
               [  0  fy   cy ]
               [  0   0    1 ]
       where fx, fy are focal lengths in pixel units, and (cx, cy) is the principal point.
    
    2. Radial & Tangential Lens Distortion:
       Real smartphone lenses introduce non-linear optical aberrations:
           x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶) + [2*p1*x*y + p2*(r² + 2*x²)]
           y_distorted = y(1 + k1*r² + k2*r⁴ + k3*r⁶) + [p1*(r² + 2*y²) + 2*p2*x*y]
       where r² = x² + y². Estimating [k1, k2, p1, p2, k3] allows rectifying camera rays.
    
    3. Essential Matrix & Euclidean Metric Reconstruction:
       Without K, two views are related by the Fundamental Matrix F = K'⁻ᵀ [t]ₓ R K⁻¹,
       which only allows projective reconstruction (up to an arbitrary 15-parameter
       projective ambiguity where angles and straight lines are not preserved).
       With K known, the Essential Matrix E = K'ᵀ F K = [t]ₓ R resolves Euclidean geometry.
    """

    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),  # (internal corners across width, height)
        square_size_mm: float = 25.0,                 # Size of one checkerboard square in mm
        subpixel_window: Tuple[int, int] = (11, 11),
        min_successful_images: int = 5,
    ):
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.subpixel_window = subpixel_window
        self.min_successful_images = min_successful_images

        # Prepare 3D object points: (0,0,0), (1,0,0), ..., (cols-1, rows-1, 0) * square_size
        cols, rows = self.checkerboard_size
        self._objp = np.zeros((rows * cols, 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size_mm

        # Subpixel refinement termination criteria: 30 iterations or epsilon 0.001
        self._term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def calibrate_from_images(
        self,
        image_paths: List[Union[str, Path]],
        output_json_path: Optional[Union[str, Path]] = None,
    ) -> CalibrationResult:
        """
        Detects checkerboard corners across provided images and optimizes K and distortion parameters.
        
        If fewer than min_successful_images contain valid checkerboard corners, raises ValueError.
        """
        valid_paths = [Path(p) for p in image_paths if Path(p).exists()]
        if not valid_paths:
            raise FileNotFoundError(
                f"No calibration images found among {len(image_paths)} paths provided."
            )

        obj_points: List[np.ndarray] = []  # 3D points in real world space
        img_points: List[np.ndarray] = []  # 2D points in image plane
        image_size: Optional[Tuple[int, int]] = None  # (width, height)
        successful_images = 0

        for p in valid_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])  # (width, height)

            # Find the internal checkerboard corners
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)

            if ret and corners is not None:
                # Refine corner locations to subpixel precision
                refined_corners = cv2.cornerSubPix(
                    gray, corners, self.subpixel_window, (-1, -1), self._term_crit
                )
                obj_points.append(self._objp)
                img_points.append(refined_corners)
                successful_images += 1

        if successful_images < self.min_successful_images:
            raise ValueError(
                f"Calibration failed: Only {successful_images} of {len(valid_paths)} images "
                f"yielded valid checkerboard corners (minimum required: {self.min_successful_images}). "
                "Ensure the entire pattern is visible, well-lit, and in sharp focus."
            )

        assert image_size is not None

        # Execute OpenCV Camera Calibration
        ret_error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )

        # Compute precise Root Mean Square Reprojection Error
        total_error = 0.0
        total_points = 0
        for i in range(len(obj_points)):
            projected_pts, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            obs_pts = img_points[i].reshape(-1, 2).astype(np.float64)
            proj_pts = projected_pts.reshape(-1, 2).astype(np.float64)
            sq_err = np.sum((obs_pts - proj_pts) ** 2)
            total_error += float(sq_err)
            total_points += len(obs_pts)
        rmse = float(np.sqrt(total_error / total_points)) if total_points > 0 else 0.0

        fx = float(mtx[0, 0])
        fy = float(mtx[1, 1])
        cx = float(mtx[0, 2])
        cy = float(mtx[1, 2])
        dist_coeffs = [float(x) for x in dist.ravel()]

        result = CalibrationResult(
            status="calibrated",
            camera_matrix_k=[[float(c) for c in row] for row in mtx],
            fx=round(fx, 4),
            fy=round(fy, 4),
            cx=round(cx, 4),
            cy=round(cy, 4),
            distortion_coefficients=[round(c, 6) for c in dist_coeffs],
            distortion_model="radtan",
            image_width=image_size[0],
            image_height=image_size[1],
            checkerboard_rows=self.checkerboard_size[1],
            checkerboard_cols=self.checkerboard_size[0],
            square_size_mm=self.square_size_mm,
            unit="mm",
            reprojection_error_rmse=round(rmse, 4),
            total_images_provided=len(valid_paths),
            images_used=successful_images,
            calibrated_at=datetime.now().isoformat(),
            note="Calibrated using OpenCV planar checkerboard calibration.",
        )

        if output_json_path is not None:
            out_p = Path(output_json_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

        return result

    @staticmethod
    def get_uncalibrated_placeholder(
        width: int = 1920,
        height: int = 1080,
        note: str = "Calibration pending. Using estimated pinhole placeholder.",
    ) -> CalibrationResult:
        """Generates an explicit uncalibrated placeholder result without fake numbers."""
        return CalibrationResult(
            status="uncalibrated",
            camera_matrix_k=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            fx=0.0,
            fy=0.0,
            cx=0.0,
            cy=0.0,
            distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
            distortion_model="radtan",
            image_width=width,
            image_height=height,
            checkerboard_rows=6,
            checkerboard_cols=9,
            square_size_mm=25.0,
            unit="mm",
            reprojection_error_rmse=0.0,
            total_images_provided=0,
            images_used=0,
            calibrated_at=datetime.now().isoformat(),
            note=note,
        )
