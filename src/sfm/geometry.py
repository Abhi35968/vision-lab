"""
Geometric Verification Module
Estimates Fundamental and Essential matrices using RANSAC and evaluates epipolar constraints.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class GeometricVerificationResult:
    """Telemetry and matrices from two-view geometric verification."""
    fundamental_matrix: Optional[np.ndarray]   # (3, 3) float64
    essential_matrix: Optional[np.ndarray]     # (3, 3) float64
    inliers_mask: np.ndarray                   # Boolean array (M,)
    num_inliers: int
    num_outliers: int
    inlier_ratio: float
    inlier_pts1: np.ndarray                    # (Inliers, 2)
    inlier_pts2: np.ndarray                    # (Inliers, 2)
    mean_epipolar_error_px: float              # Symmetric epipolar distance (pixels)
    median_epipolar_error_px: float
    is_valid: bool
    calibration_status: str                    # "calibrated", "estimated", "uncalibrated"
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "num_inliers": self.num_inliers,
            "num_outliers": self.num_outliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "mean_epipolar_error_px": round(self.mean_epipolar_error_px, 3),
            "median_epipolar_error_px": round(self.median_epipolar_error_px, 3),
            "is_valid": self.is_valid,
            "calibration_status": self.calibration_status,
            "rejection_reason": self.rejection_reason,
            "fundamental_matrix": self.fundamental_matrix.tolist() if self.fundamental_matrix is not None else None,
            "essential_matrix": self.essential_matrix.tolist() if self.essential_matrix is not None else None,
        }


class GeometricVerifier:
    """
    Applies RANSAC to verify feature correspondences against rigid two-view epipolar geometry.
    
    Mathematical Foundations:
    -------------------------
    1. Epipolar Constraint:
       Two pinhole camera views observing a static 3D point X satisfy the coplanarity condition:
           x₂ᵀ F x₁ = 0
       where x₁, x₂ are homogeneous image pixel coordinates [u, v, 1]ᵀ, and F is the 3x3 rank-2
       Fundamental Matrix.
    
    2. Essential Matrix (Calibrated Views):
       When the camera calibration matrix K is available, normalized image coordinates are:
           x_n₁ = K⁻¹ x₁,   x_n₂ = K⁻¹ x₂
       The Essential Matrix relates normalized coordinates:
           x_n₂ᵀ E x_n₁ = 0   where   E = Kᵀ F K = [t]ₓ R
       An algebraically valid Essential Matrix has two equal non-zero singular values:
           Σ = diag(σ, σ, 0).
    
    3. Epipolar Distance Metric (Symmetric Point-to-Line Error):
       The epipolar line in image 2 is l₂ = F x₁ = [a, b, c]ᵀ. The perpendicular distance from
       x₂ to l₂ in pixels is:
           d(x₂, l₂) = |x₂ᵀ F x₁| / sqrt(a² + b²)
       Symmetric epipolar distance evaluates geometric deviation across both image planes:
           d_sym = d(x₂, F x₁) + d(x₁, Fᵀ x₂)
    """

    def __init__(
        self,
        ransac_reproj_threshold: float = 1.0,   # Pixel threshold for RANSAC inlier classification
        ransac_confidence: float = 0.999,        # RANSAC probability confidence
        min_inliers: int = 15,                   # Minimum required inliers for valid geometry
        min_inlier_ratio: float = 0.25,          # Minimum inlier percentage
    ):
        self.ransac_threshold = ransac_reproj_threshold
        self.confidence = ransac_confidence
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio

    @staticmethod
    def enforce_essential_constraints(E_raw: np.ndarray) -> np.ndarray:
        """Projects a raw 3x3 matrix onto the Essential Matrix manifold: diag(1, 1, 0)."""
        U, S, Vt = np.linalg.svd(E_raw)
        sigma = (S[0] + S[1]) / 2.0
        E_enforced = U @ np.diag([sigma, sigma, 0.0]) @ Vt
        return E_enforced

    def compute_symmetric_epipolar_errors(
        self, pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray
    ) -> np.ndarray:
        """Computes geometric symmetric epipolar distance (in pixels) for all point pairs."""
        N = len(pts1)
        if N == 0:
            return np.empty((0,), dtype=np.float64)

        # Convert to homogeneous coordinates (N, 3)
        h1 = np.hstack([pts1, np.ones((N, 1), dtype=np.float64)])
        h2 = np.hstack([pts2, np.ones((N, 1), dtype=np.float64)])

        # Epipolar lines: l2 = F * x1, l1 = F^T * x2
        l2 = (F @ h1.T).T   # (N, 3)
        l1 = (F.T @ h2.T).T # (N, 3)

        # Numerator: |x2^T F x1|
        num = np.abs(np.sum(h2 * (F @ h1.T).T, axis=1))

        # Denominators: sqrt(a^2 + b^2)
        denom2 = np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2) + 1e-9
        denom1 = np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2) + 1e-9

        # Symmetric point-to-line error in pixel units
        errors = (num / denom2) + (num / denom1)
        return errors

    def verify(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        camera_matrix_k: Optional[np.ndarray] = None,
        calibration_status: str = "uncalibrated",
    ) -> GeometricVerificationResult:
        """
        Estimates Fundamental / Essential matrix using RANSAC and isolates inlier correspondences.
        """
        total_pts = len(pts1)
        if total_pts < 8:
            return GeometricVerificationResult(
                fundamental_matrix=None,
                essential_matrix=None,
                inliers_mask=np.zeros((total_pts,), dtype=bool),
                num_inliers=0,
                num_outliers=total_pts,
                inlier_ratio=0.0,
                inlier_pts1=np.empty((0, 2), dtype=np.float32),
                inlier_pts2=np.empty((0, 2), dtype=np.float32),
                mean_epipolar_error_px=0.0,
                median_epipolar_error_px=0.0,
                is_valid=False,
                calibration_status=calibration_status,
                rejection_reason=f"Insufficient correspondences ({total_pts} < 8 required for 8-point RANSAC).",
            )

        # 1. Fundamental Matrix Estimation via RANSAC
        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence,
            maxIters=2000,
        )

        if F is None or mask is None:
            return GeometricVerificationResult(
                fundamental_matrix=None,
                essential_matrix=None,
                inliers_mask=np.zeros((total_pts,), dtype=bool),
                num_inliers=0,
                num_outliers=total_pts,
                inlier_ratio=0.0,
                inlier_pts1=np.empty((0, 2), dtype=np.float32),
                inlier_pts2=np.empty((0, 2), dtype=np.float32),
                mean_epipolar_error_px=0.0,
                median_epipolar_error_px=0.0,
                is_valid=False,
                calibration_status=calibration_status,
                rejection_reason="RANSAC failed to estimate a stable Fundamental Matrix.",
            )

        inlier_mask = mask.ravel().astype(bool)
        num_inliers = int(np.sum(inlier_mask))
        num_outliers = total_pts - num_inliers
        inlier_ratio = num_inliers / float(total_pts)

        # 2. Epipolar Error Calculation on Inliers
        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]

        if num_inliers > 0:
            errors = self.compute_symmetric_epipolar_errors(inlier_pts1, inlier_pts2, F)
            mean_err = float(np.mean(errors))
            median_err = float(np.median(errors))
        else:
            mean_err, median_err = 0.0, 0.0

        # Check inlier quality thresholds
        is_valid = (num_inliers >= self.min_inliers) and (inlier_ratio >= self.min_inlier_ratio)
        rejection_reason = None
        if not is_valid:
            if num_inliers < self.min_inliers:
                rejection_reason = f"Inliers ({num_inliers}) < minimum required ({self.min_inliers})."
            elif inlier_ratio < self.min_inlier_ratio:
                rejection_reason = f"Inlier ratio ({inlier_ratio:.2%}) < minimum threshold ({self.min_inlier_ratio:.2%})."

        # 3. Essential Matrix Estimation (if K is available)
        E: Optional[np.ndarray] = None
        if camera_matrix_k is not None and is_valid:
            # E = K^T * F * K
            E_raw = camera_matrix_k.T @ F @ camera_matrix_k
            E = self.enforce_essential_constraints(E_raw)

        return GeometricVerificationResult(
            fundamental_matrix=F,
            essential_matrix=E,
            inliers_mask=inlier_mask,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            inlier_ratio=inlier_ratio,
            inlier_pts1=inlier_pts1,
            inlier_pts2=inlier_pts2,
            mean_epipolar_error_px=mean_err,
            median_epipolar_error_px=median_err,
            is_valid=is_valid,
            calibration_status=calibration_status,
            rejection_reason=rejection_reason,
        )

    @staticmethod
    def visualize_epipolar_lines(
        img1: np.ndarray,
        img2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        F: np.ndarray,
        inlier_mask: Optional[np.ndarray] = None,
        num_lines: int = 25,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draws epipolar lines and matching points across both images.
        
        Lines on img2 correspond to points from img1 (l2 = F * x1).
        Lines on img1 correspond to points from img2 (l1 = F^T * x2).
        """
        if inlier_mask is not None:
            pts1 = pts1[inlier_mask]
            pts2 = pts2[inlier_mask]

        vis1 = img1.copy()
        vis2 = img2.copy()
        h, w = img1.shape[:2]

        total = len(pts1)
        if total == 0 or F is None:
            return vis1, vis2

        indices = np.linspace(0, total - 1, min(total, num_lines), dtype=int)

        h1 = np.hstack([pts1[indices], np.ones((len(indices), 1), dtype=np.float64)])
        h2 = np.hstack([pts2[indices], np.ones((len(indices), 1), dtype=np.float64)])

        lines2 = (F @ h1.T).T   # Epipolar lines in Image 2
        lines1 = (F.T @ h2.T).T # Epipolar lines in Image 1

        colors = plt_colors = [
            (int(c[0]), int(c[1]), int(c[2]))
            for c in np.random.RandomState(42).randint(0, 255, size=(len(indices), 3))
        ]

        # Draw lines on Image 2 and points on Image 1
        for i, (l2, p1, p2, col) in enumerate(zip(lines2, pts1[indices], pts2[indices], colors)):
            # Line equation ax + by + c = 0
            a, b, c = l2
            if abs(b) > 1e-6:
                x0, y0 = 0, int(round(-c / b))
                x1, y1 = w, int(round(-(c + a * w) / b))
                cv2.line(vis2, (x0, y0), (x1, y1), col, 1, cv2.LINE_AA)
            cv2.circle(vis1, (int(round(p1[0])), int(round(p1[1]))), 4, col, -1)
            cv2.circle(vis2, (int(round(p2[0])), int(round(p2[1]))), 4, col, -1)

        # Draw lines on Image 1
        for l1, col in zip(lines1, colors):
            a, b, c = l1
            if abs(b) > 1e-6:
                x0, y0 = 0, int(round(-c / b))
                x1, y1 = w, int(round(-(c + a * w) / b))
                cv2.line(vis1, (x0, y0), (x1, y1), col, 1, cv2.LINE_AA)

        cv2.putText(vis1, "Image 1: Points & Epipolar Lines (F^T x2)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis2, "Image 2: Points & Epipolar Lines (F x1)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return vis1, vis2

    # Alias for convenience
    verify_matches = verify


