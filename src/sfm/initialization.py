"""
Initialization Module
Selects the optimal initial camera pair based on multi-criteria geometric scoring and seeds the 3D reconstruction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.sfm.multi_view_matcher import VerifiedPairMatch
from src.sfm.pose_estimator import PoseEstimator, PoseEstimationResult
from src.sfm.triangulator import Triangulator, TriangulationResult
from src.sfm.feature_tracks import FeatureTrackSystem


@dataclass
class InitialPairScore:
    """Multi-criteria evaluation telemetry for a candidate initialization pair."""
    img1_idx: int
    img2_idx: int
    num_inliers: int
    inlier_ratio: float
    median_parallax_deg: float
    spatial_coverage_ratio: float
    triangulation_yield_percent: float
    mean_reprojection_error_px: float
    composite_score: float
    is_acceptable: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "img1_idx": self.img1_idx,
            "img2_idx": self.img2_idx,
            "num_inliers": self.num_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "median_parallax_deg": round(self.median_parallax_deg, 2),
            "spatial_coverage_ratio": round(self.spatial_coverage_ratio, 3),
            "triangulation_yield_percent": round(self.triangulation_yield_percent, 2),
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 3),
            "composite_score": round(self.composite_score, 2),
            "is_acceptable": self.is_acceptable,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class InitializationResult:
    """Outputs from reconstruction seeding: initial poses and triangulated 3D points."""
    pair: Tuple[int, int]
    pair_names: Tuple[str, str]
    camera1_pose: dict                  # R1, t1, C1
    camera2_pose: dict                  # R2, t2, C2
    initial_points_3d: np.ndarray       # (N, 3) float64
    initial_points_colors: Optional[np.ndarray]
    mean_reprojection_error_px: float
    num_initial_points: int
    candidate_evaluations: List[InitialPairScore]
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "initial_pair": list(self.pair),
            "initial_pair_names": list(self.pair_names),
            "camera1_pose": self.camera1_pose,
            "camera2_pose": self.camera2_pose,
            "num_initial_points": self.num_initial_points,
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 3),
            "candidate_evaluations": [e.to_dict() for e in self.candidate_evaluations],
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
        }


class InitialPairSelector:
    """
    Evaluates candidate image pairs to select the optimal initial baseline for incremental SfM.
    
    Why Highest Inlier Count Alone Fails:
    ------------------------------------
    In video recordings, two frames separated by 1 or 2 time steps exhibit extreme visual overlap
    and huge inlier counts (~1000+ matches), but the camera translation is nearly zero.
    Under near-zero baseline, rays are parallel, rendering depth triangulation numerically degenerate
    (the rays intersect at infinity or produce massive depth variance along the optical axis).
    
    Optimal Initialization Requirements:
    -------------------------------------
    1. Sufficient inliers (to support robust initial geometry: >= 30).
    2. Sufficient triangulation parallax (median baseline angle >= 2.5 degrees).
    3. Wide spatial distribution (keypoints dispersed across the image frame, not clustered in a corner).
    4. Low reprojection RMSE (< 3.0 px) and high positive depth Cheirality yield.
    """

    def __init__(
        self,
        min_inliers: int = 30,
        min_parallax_deg: float = 1.5,
        max_parallax_deg: float = 50.0,
        min_spatial_coverage: float = 0.10,
        max_reprojection_error_px: float = 4.0,
    ):
        self.min_inliers = min_inliers
        self.min_parallax_deg = min_parallax_deg
        self.max_parallax_deg = max_parallax_deg
        self.min_spatial_coverage = min_spatial_coverage
        self.max_reprojection_error = max_reprojection_error_px
        self.pose_estimator = PoseEstimator(min_cheirality_ratio=0.70)
        self.triangulator = Triangulator(max_reprojection_error_px=max_reprojection_error_px)

    @staticmethod
    def compute_spatial_coverage(pts: np.ndarray, img_size: Tuple[int, int]) -> float:
        """Computes normalized bounding box area of 2D correspondences."""
        if len(pts) < 3:
            return 0.0
        width, height = img_size
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        bbox_area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
        total_area = float(width * height)
        return float(np.clip(bbox_area / total_area, 0.0, 1.0))

    @staticmethod
    def compute_ray_parallax_angles(
        pts1: np.ndarray,
        pts2: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the angular disparity (parallax angle in degrees) between viewing rays.
        """
        K_inv = np.linalg.inv(K)
        N = len(pts1)
        hom1 = np.hstack([pts1, np.ones((N, 1), dtype=np.float64)])
        hom2 = np.hstack([pts2, np.ones((N, 1), dtype=np.float64)])

        # Viewing rays in respective camera coordinates
        rays1 = (K_inv @ hom1.T).T
        rays2 = (K_inv @ hom2.T).T

        # Normalize rays
        rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)
        rays2 /= np.linalg.norm(rays2, axis=1, keepdims=True)

        # Rotate ray 2 into camera 1's frame: v2_cam1 = R^T * ray2
        rays2_in_1 = (R.T @ rays2.T).T

        # Dot product between rays
        dots = np.sum(rays1 * rays2_in_1, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles_rad = np.arccos(dots)
        return np.degrees(angles_rad)

    def select_and_initialize(
        self,
        verified_pairs: List[VerifiedPairMatch],
        camera_matrix_k: np.ndarray,
        image_size: Tuple[int, int],
        track_system: FeatureTrackSystem,
        forced_seed_pair: Optional[Tuple[int, int]] = None,
    ) -> InitializationResult:
        """
        Scores all verified pairs and initializes the reconstruction with the highest-scoring candidate,
        or forces a specific seed pair if forced_seed_pair is provided.
        """
        evaluations: List[InitialPairScore] = []
        best_pair_match: Optional[VerifiedPairMatch] = None
        best_pose_res: Optional[PoseEstimationResult] = None
        best_tri_res: Optional[TriangulationResult] = None
        highest_score = -1.0

        is_forced = forced_seed_pair is not None
        forced_norm = (min(forced_seed_pair), max(forced_seed_pair)) if is_forced else None
        forced_candidate: Optional[Tuple[VerifiedPairMatch, PoseEstimationResult, TriangulationResult]] = None

        for pair in verified_pairs:
            if not pair.is_valid or pair.num_inliers < self.min_inliers or pair.essential_matrix is None:
                evaluations.append(
                    InitialPairScore(
                        img1_idx=pair.img1_idx,
                        img2_idx=pair.img2_idx,
                        num_inliers=pair.num_inliers,
                        inlier_ratio=pair.inlier_ratio,
                        median_parallax_deg=0.0,
                        spatial_coverage_ratio=0.0,
                        triangulation_yield_percent=0.0,
                        mean_reprojection_error_px=999.0,
                        composite_score=0.0,
                        is_acceptable=False,
                        rejection_reason=f"Insufficient inliers ({pair.num_inliers} < {self.min_inliers}) or missing E.",
                    )
                )
                continue

            # 1. Pose recovery
            pose_res = self.pose_estimator.estimate_pose(
                pair.essential_matrix,
                pair.inlier_pts1,
                pair.inlier_pts2,
                camera_matrix_k,
            )
            if not pose_res.is_valid:
                evaluations.append(
                    InitialPairScore(
                        img1_idx=pair.img1_idx,
                        img2_idx=pair.img2_idx,
                        num_inliers=pair.num_inliers,
                        inlier_ratio=pair.inlier_ratio,
                        median_parallax_deg=0.0,
                        spatial_coverage_ratio=0.0,
                        triangulation_yield_percent=0.0,
                        mean_reprojection_error_px=999.0,
                        composite_score=0.0,
                        is_acceptable=False,
                        rejection_reason=f"Cheirality recovery failed: {pose_res.rejection_reason}",
                    )
                )
                continue

            # 2. Parallax analysis
            parallax_angles = self.compute_ray_parallax_angles(
                pair.inlier_pts1,
                pair.inlier_pts2,
                camera_matrix_k,
                pose_res.rotation_matrix,
            )
            median_parallax = float(np.median(parallax_angles))

            # 3. Spatial coverage
            cov1 = self.compute_spatial_coverage(pair.inlier_pts1, image_size)
            cov2 = self.compute_spatial_coverage(pair.inlier_pts2, image_size)
            mean_coverage = (cov1 + cov2) / 2.0

            # 4. Preliminary triangulation
            tri_res = self.triangulator.triangulate(
                pair.inlier_pts1,
                pair.inlier_pts2,
                camera_matrix_k,
                pose_res.rotation_matrix,
                pose_res.translation_vector,
            )

            # Acceptability checks
            acceptable = True
            rejection_reason = None
            if median_parallax < self.min_parallax_deg:
                acceptable = False
                rejection_reason = f"Baseline too small: median parallax {median_parallax:.2f}° < {self.min_parallax_deg}°."
            elif median_parallax > self.max_parallax_deg:
                acceptable = False
                rejection_reason = f"Baseline too wide / distorted: median parallax {median_parallax:.2f}° > {self.max_parallax_deg}°."
            elif mean_coverage < self.min_spatial_coverage:
                acceptable = False
                rejection_reason = f"Poor spatial distribution: coverage {mean_coverage:.2f} < {self.min_spatial_coverage}."
            elif not tri_res.is_valid or tri_res.points_reconstructed < self.min_inliers // 2:
                acceptable = False
                rejection_reason = f"Triangulation yield too low: {tri_res.points_reconstructed} points."

            # Multi-criteria scoring function (range ~0 - 100)
            inlier_score = min(40.0, (pair.num_inliers / 200.0) * 40.0)
            parallax_score = min(30.0, (median_parallax / 15.0) * 30.0)
            coverage_score = min(20.0, (mean_coverage / 0.6) * 20.0)
            reproj_penalty = max(0.0, min(10.0, (tri_res.mean_reprojection_error / self.max_reprojection_error) * 10.0))
            composite_score = inlier_score + parallax_score + coverage_score - reproj_penalty if acceptable else 0.0

            score_record = InitialPairScore(
                img1_idx=pair.img1_idx,
                img2_idx=pair.img2_idx,
                num_inliers=pair.num_inliers,
                inlier_ratio=pair.inlier_ratio,
                median_parallax_deg=median_parallax,
                spatial_coverage_ratio=mean_coverage,
                triangulation_yield_percent=float(tri_res.points_reconstructed / max(1, tri_res.total_candidates) * 100),
                mean_reprojection_error_px=tri_res.mean_reprojection_error,
                composite_score=composite_score,
                is_acceptable=acceptable,
                rejection_reason=rejection_reason,
            )
            evaluations.append(score_record)

            pair_norm = (min(pair.img1_idx, pair.img2_idx), max(pair.img1_idx, pair.img2_idx))
            if is_forced and pair_norm == forced_norm:
                if pose_res.is_valid and tri_res.is_valid and tri_res.points_reconstructed > 0:
                    forced_candidate = (pair, pose_res, tri_res)

            if not is_forced and acceptable and composite_score > highest_score:
                highest_score = composite_score
                best_pair_match = pair
                best_pose_res = pose_res
                best_tri_res = tri_res

        if is_forced:
            if forced_candidate is not None:
                best_pair_match, best_pose_res, best_tri_res = forced_candidate
            else:
                return InitializationResult(
                    pair=(-1, -1),
                    pair_names=("", ""),
                    camera1_pose={},
                    camera2_pose={},
                    initial_points_3d=np.empty((0, 3)),
                    initial_points_colors=None,
                    mean_reprojection_error_px=0.0,
                    num_initial_points=0,
                    candidate_evaluations=evaluations,
                    is_valid=False,
                    rejection_reason=f"Forced seed pair {forced_seed_pair} was not found or failed geometric verification/triangulation.",
                )

        if best_pair_match is None or best_pose_res is None or best_tri_res is None:
            return InitializationResult(
                pair=(-1, -1),
                pair_names=("", ""),
                camera1_pose={},
                camera2_pose={},
                initial_points_3d=np.empty((0, 3)),
                initial_points_colors=None,
                mean_reprojection_error_px=0.0,
                num_initial_points=0,
                candidate_evaluations=evaluations,
                is_valid=False,
                rejection_reason="No candidate image pair satisfied all initialization constraints.",
            )

        # Coordinate Convention for Reconstruction:
        # World Frame fixed to Camera 1:
        # R1 = Eye(3), t1 = [0, 0, 0]^T
        # Camera 2: R2 = pose_res.R, t2 = pose_res.t
        R1 = np.eye(3, dtype=np.float64)
        t1 = np.zeros((3, 1), dtype=np.float64)
        C1 = np.zeros(3, dtype=np.float64)

        R2 = best_pose_res.rotation_matrix
        t2 = best_pose_res.translation_vector
        C2 = -R2.T @ t2.ravel()

        cam1_pose = {
            "rotation_matrix": R1.tolist(),
            "translation_vector": t1.ravel().tolist(),
            "camera_center": C1.tolist(),
        }
        cam2_pose = {
            "rotation_matrix": R2.tolist(),
            "translation_vector": t2.ravel().tolist(),
            "camera_center": C2.tolist(),
        }

        # Associate triangulated 3D points with feature tracks
        # For each correspondence track in best_tri_res, find matching FeatureTrack and update
        for item in best_tri_res.correspondence_tracks:
            p_id = item["point_id"]
            cand_idx = item.get("candidate_idx", p_id)
            if cand_idx < len(best_pair_match.inlier_kp_indices1):
                kp1_idx = best_pair_match.inlier_kp_indices1[cand_idx]
                track = track_system.get_track_for_keypoint(best_pair_match.img1_idx, kp1_idx)
                if track is not None:
                    track.point_3d = best_tri_res.points_3d[p_id]
                    track.is_triangulated = True
                    track.reprojection_error = float(item.get("reproj_error_px", item.get("reprojection_error_px", 0.0)))

        return InitializationResult(
            pair=(best_pair_match.img1_idx, best_pair_match.img2_idx),
            pair_names=(best_pair_match.img1_name, best_pair_match.img2_name),
            camera1_pose=cam1_pose,
            camera2_pose=cam2_pose,
            initial_points_3d=best_tri_res.points_3d,
            initial_points_colors=best_tri_res.point_colors,
            mean_reprojection_error_px=best_tri_res.mean_reprojection_error,
            num_initial_points=best_tri_res.points_reconstructed,
            candidate_evaluations=evaluations,
            is_valid=True,
        )
