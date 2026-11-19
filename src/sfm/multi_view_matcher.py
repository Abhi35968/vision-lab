"""
Multi-View Feature Matcher Module
Coordinates feature extraction, windowed pairwise matching, and geometric verification across multiple keyframes.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.sfm.feature_extractor import ExtractedFeatures, FeatureExtractor
from src.sfm.feature_matcher import FeatureMatcher, MatchResult
from src.sfm.geometry import GeometricVerificationResult, GeometricVerifier
from src.sfm.feature_tracks import FeatureTrackSystem


@dataclass
class VerifiedPairMatch:
    """Pairwise matching and geometric verification results for an image pair."""
    img1_idx: int
    img2_idx: int
    img1_name: str
    img2_name: str
    raw_matches_count: int
    num_inliers: int
    inlier_ratio: float
    mean_epipolar_error_px: float
    median_epipolar_error_px: float
    inlier_pts1: np.ndarray             # (K, 2) float32
    inlier_pts2: np.ndarray             # (K, 2) float32
    inlier_kp_indices1: List[int]       # Indices in keypoints1
    inlier_kp_indices2: List[int]       # Indices in keypoints2
    essential_matrix: Optional[np.ndarray]
    fundamental_matrix: Optional[np.ndarray]
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "img1_idx": self.img1_idx,
            "img2_idx": self.img2_idx,
            "img1_name": self.img1_name,
            "img2_name": self.img2_name,
            "raw_matches_count": self.raw_matches_count,
            "num_inliers": self.num_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "mean_epipolar_error_px": round(self.mean_epipolar_error_px, 3),
            "median_epipolar_error_px": round(self.median_epipolar_error_px, 3),
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
            "has_essential_matrix": self.essential_matrix is not None,
        }


class MultiViewMatcher:
    """
    Executes windowed multi-view feature matching and builds a connected observation graph.
    
    Trade-off: Complete Graph vs. Windowed Matching:
    -------------------------------------------------
    1. Complete Pairwise Matching (All Pairs):
       - Pair count: N * (N - 1) / 2 = O(N²)
       - In sequential video recording, frames far apart in time share zero camera overlap.
         Evaluating all pairs on a CPU introduces extreme redundant compute.
    
    2. Consecutive Matching (Window W = 1):
       - Pair count: N - 1 = O(N)
       - Linear chain offers minimal baseline diversity and suffers from severe track drift
         and early fragmentation if an intermediate frame loses tracking.
    
    3. Windowed Matching (Window W = 2 or 3):
       - Pair count: ~ W * N = O(W · N)
       - Evaluates pairs (i, i+1), (i, i+2), ..., (i, i+W).
       - Provides redundant triangulation baselines and multi-view tracks (spanning 3 to 4+ views)
         while keeping CPU execution linear with video length.
    """

    def __init__(
        self,
        extractor: Optional[FeatureExtractor] = None,
        matcher: Optional[FeatureMatcher] = None,
        verifier: Optional[GeometricVerifier] = None,
        matching_window: int = 2,
        min_inliers_threshold: int = 20,
    ):
        self.extractor = extractor or FeatureExtractor(detector="SIFT", n_features=2000)
        self.matcher = matcher or FeatureMatcher(matcher_type="FLANN", lowe_ratio=0.75)
        self.verifier = verifier or GeometricVerifier(ransac_reproj_threshold=2.0)
        self.matching_window = max(1, matching_window)
        self.min_inliers_threshold = min_inliers_threshold

    def extract_all_features(
        self,
        image_paths: List[Union[str, Path]],
    ) -> List[ExtractedFeatures]:
        """Extracts and caches local invariant features for all input keyframe images."""
        features_list: List[ExtractedFeatures] = []
        for path in image_paths:
            feat = self.extractor.extract(path)
            features_list.append(feat)
        return features_list

    def match_keyframes(
        self,
        features_list: List[ExtractedFeatures],
        camera_matrix_k: Optional[np.ndarray] = None,
        calibration_status: str = "uncalibrated",
    ) -> Tuple[List[VerifiedPairMatch], FeatureTrackSystem]:
        """
        Performs windowed pairwise matching across all keyframes and constructs the track system.
        """
        N = len(features_list)
        verified_pairs: List[VerifiedPairMatch] = []
        track_system = FeatureTrackSystem(min_track_length=2)

        for i in range(N):
            feat1 = features_list[i]
            img1_name = Path(feat1.image_path).name if feat1.image_path else f"frame_{i:04d}.png"

            # Match within sliding window W
            max_j = min(N, i + self.matching_window + 1)
            for j in range(i + 1, max_j):
                feat2 = features_list[j]
                img2_name = Path(feat2.image_path).name if feat2.image_path else f"frame_{j:04d}.png"

                # 1. Feature Matching (KNN + Ratio test + Mutual check)
                match_res = self.matcher.match(feat1, feat2)

                if match_res.final_matches_count < 8:
                    verified_pairs.append(
                        VerifiedPairMatch(
                            img1_idx=i,
                            img2_idx=j,
                            img1_name=img1_name,
                            img2_name=img2_name,
                            raw_matches_count=match_res.raw_matches_count,
                            num_inliers=0,
                            inlier_ratio=0.0,
                            mean_epipolar_error_px=0.0,
                            median_epipolar_error_px=0.0,
                            inlier_pts1=np.empty((0, 2), dtype=np.float32),
                            inlier_pts2=np.empty((0, 2), dtype=np.float32),
                            inlier_kp_indices1=[],
                            inlier_kp_indices2=[],
                            essential_matrix=None,
                            fundamental_matrix=None,
                            is_valid=False,
                            rejection_reason="Insufficient raw correspondences for epipolar geometry (< 8).",
                        )
                    )
                    continue

                # 2. Geometric Verification via Epipolar RANSAC
                geo_res = self.verifier.verify(
                    match_res.matched_pts1,
                    match_res.matched_pts2,
                    camera_matrix_k=camera_matrix_k,
                    calibration_status=calibration_status,
                )

                if not geo_res.is_valid or geo_res.num_inliers < self.min_inliers_threshold:
                    verified_pairs.append(
                        VerifiedPairMatch(
                            img1_idx=i,
                            img2_idx=j,
                            img1_name=img1_name,
                            img2_name=img2_name,
                            raw_matches_count=match_res.raw_matches_count,
                            num_inliers=geo_res.num_inliers,
                            inlier_ratio=geo_res.inlier_ratio,
                            mean_epipolar_error_px=geo_res.mean_epipolar_error_px,
                            median_epipolar_error_px=geo_res.median_epipolar_error_px,
                            inlier_pts1=geo_res.inlier_pts1,
                            inlier_pts2=geo_res.inlier_pts2,
                            inlier_kp_indices1=[],
                            inlier_kp_indices2=[],
                            essential_matrix=geo_res.essential_matrix,
                            fundamental_matrix=geo_res.fundamental_matrix,
                            is_valid=False,
                            rejection_reason=f"Geometric verification rejected (inliers {geo_res.num_inliers} < {self.min_inliers_threshold}).",
                        )
                    )
                    continue

                # Map inliers back to original keypoint indices
                inlier_kp_idx1: List[int] = []
                inlier_kp_idx2: List[int] = []
                for k, is_inlier in enumerate(geo_res.inliers_mask):
                    if is_inlier:
                        dmatch = match_res.final_matches[k]
                        inlier_kp_idx1.append(dmatch.queryIdx)
                        inlier_kp_idx2.append(dmatch.trainIdx)

                verified_pair = VerifiedPairMatch(
                    img1_idx=i,
                    img2_idx=j,
                    img1_name=img1_name,
                    img2_name=img2_name,
                    raw_matches_count=match_res.raw_matches_count,
                    num_inliers=geo_res.num_inliers,
                    inlier_ratio=geo_res.inlier_ratio,
                    mean_epipolar_error_px=geo_res.mean_epipolar_error_px,
                    median_epipolar_error_px=geo_res.median_epipolar_error_px,
                    inlier_pts1=geo_res.inlier_pts1,
                    inlier_pts2=geo_res.inlier_pts2,
                    inlier_kp_indices1=inlier_kp_idx1,
                    inlier_kp_indices2=inlier_kp_idx2,
                    essential_matrix=geo_res.essential_matrix,
                    fundamental_matrix=geo_res.fundamental_matrix,
                    is_valid=True,
                )
                verified_pairs.append(verified_pair)

                # 3. Add to FeatureTrackSystem
                for m_idx in range(len(inlier_kp_idx1)):
                    kp1_idx = inlier_kp_idx1[m_idx]
                    kp2_idx = inlier_kp_idx2[m_idx]
                    pt1 = (float(geo_res.inlier_pts1[m_idx, 0]), float(geo_res.inlier_pts1[m_idx, 1]))
                    pt2 = (float(geo_res.inlier_pts2[m_idx, 0]), float(geo_res.inlier_pts2[m_idx, 1]))

                    desc1 = feat1.descriptors[kp1_idx] if feat1.descriptors is not None else None
                    desc2 = feat2.descriptors[kp2_idx] if feat2.descriptors is not None else None

                    track_system.add_match(
                        img1_id=i,
                        img1_name=img1_name,
                        kp1_idx=kp1_idx,
                        pt1=pt1,
                        img2_id=j,
                        img2_name=img2_name,
                        kp2_idx=kp2_idx,
                        pt2=pt2,
                        desc1=desc1,
                        desc2=desc2,
                    )

        return verified_pairs, track_system
