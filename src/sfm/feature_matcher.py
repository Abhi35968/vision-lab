"""
Feature Matcher Module
Matches keypoint descriptors across views using KNN, Lowe's ratio test, and mutual consistency.
"""

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.sfm.feature_extractor import ExtractedFeatures


@dataclass
class MatchResult:
    """Structured container for descriptor correspondences between two images."""
    raw_matches_count: int
    ratio_matches_count: int
    final_matches_count: int
    final_matches: List[cv2.DMatch]
    matched_pts1: np.ndarray              # (M, 2) float32 coordinates in Image 1
    matched_pts2: np.ndarray              # (M, 2) float32 coordinates in Image 2
    matcher_type: str                     # "FLANN" or "BF"
    matching_time_ms: float

    def to_dict(self) -> dict:
        return {
            "raw_matches_count": self.raw_matches_count,
            "ratio_matches_count": self.ratio_matches_count,
            "final_matches_count": self.final_matches_count,
            "matcher_type": self.matcher_type,
            "matching_time_ms": round(self.matching_time_ms, 2),
        }


class FeatureMatcher:
    """
    Pairs corresponding local feature descriptors across two image views.
    
    Why Descriptor Similarity Alone is Insufficient for 3D Reconstruction:
    ---------------------------------------------------------------------
    Local descriptors (e.g., SIFT, ORB) measure photometric appearance within a small patch.
    In real scenes, repetitive textures (such as brick walls, keyboards, windows, carpet weave)
    produce mathematically identical descriptor vectors at multiple physically distant 3D locations.
    
    Descriptor matching operates in high-dimensional feature space with zero spatial awareness:
    it cannot verify whether a 2D match conforms to rigid 3D camera motion. Consequently, raw
    descriptor matches contain significant outliers (false correspondences). Reliable 3D
    reconstruction requires downstream geometric verification via Epipolar Geometry and RANSAC
    to reject appearance matches that violate physical multi-view geometry.
    """

    def __init__(
        self,
        matcher_type: str = "FLANN",        # "FLANN" or "BF"
        lowe_ratio: float = 0.75,           # Lowe's distance ratio threshold (tighter = fewer but higher-quality matches)
        cross_check: bool = True,           # Enforce mutual consistency (reduces false matches)
    ):
        self.matcher_type = matcher_type.upper()
        self.lowe_ratio = lowe_ratio
        self.cross_check = cross_check

    def _create_matcher(self, descriptor_dtype: str, detector_name: str) -> cv2.DescriptorMatcher:
        """Instantiates appropriate matcher (FLANN for SIFT float32, BF-Hamming for ORB uint8)."""
        if detector_name == "ORB" or descriptor_dtype == "uint8":
            # Binary descriptors require Hamming distance metric
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Floating-point descriptors (SIFT)
        if self.matcher_type == "FLANN":
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match(self, feat1: ExtractedFeatures, feat2: ExtractedFeatures) -> MatchResult:
        """
        Computes 2D-to-2D correspondences between two extracted feature sets.
        
        Applies KNN (k=2) matching followed by Lowe's ratio test and optional cross-check.
        """
        # Edge case: either image has no descriptors
        if (
            feat1.descriptors is None
            or feat2.descriptors is None
            or len(feat1.descriptors) < 2
            or len(feat2.descriptors) < 2
        ):
            return MatchResult(
                raw_matches_count=0,
                ratio_matches_count=0,
                final_matches_count=0,
                final_matches=[],
                matched_pts1=np.empty((0, 2), dtype=np.float32),
                matched_pts2=np.empty((0, 2), dtype=np.float32),
                matcher_type=self.matcher_type,
                matching_time_ms=0.0,
            )

        matcher = self._create_matcher(feat1.descriptor_dtype, feat1.detector_name)

        t0 = time.perf_counter()

        # 1. Forward KNN Matching (k=2)
        raw_knn = matcher.knnMatch(feat1.descriptors, feat2.descriptors, k=2)
        raw_count = len(raw_knn)

        # 2. Lowe's Ratio Test
        # Retain match only if distance to nearest neighbor is significantly smaller
        # than distance to second nearest neighbor (discriminative match).
        ratio_matches: List[cv2.DMatch] = []
        for pair in raw_knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.lowe_ratio * n.distance:
                    ratio_matches.append(m)

        ratio_count = len(ratio_matches)

        # 3. Optional Mutual Consistency (Cross-Check)
        final_matches: List[cv2.DMatch] = []
        if self.cross_check and ratio_count > 0:
            # Backward KNN match: feat2 -> feat1
            backward_knn = matcher.knnMatch(feat2.descriptors, feat1.descriptors, k=2)
            backward_dict = {}
            for pair in backward_knn:
                if len(pair) == 2 and pair[0].distance < self.lowe_ratio * pair[1].distance:
                    # queryIdx in backward is feat2 index, trainIdx is feat1 index
                    backward_dict[pair[0].queryIdx] = pair[0].trainIdx

            for m in ratio_matches:
                # m.queryIdx in forward is feat1, m.trainIdx is feat2
                if backward_dict.get(m.trainIdx) == m.queryIdx:
                    final_matches.append(m)
        else:
            final_matches = ratio_matches

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Extract aligned 2D pixel coordinates for verified matches
        if len(final_matches) > 0:
            pts1 = np.float32([feat1.keypoints[m.queryIdx].pt for m in final_matches])
            pts2 = np.float32([feat2.keypoints[m.trainIdx].pt for m in final_matches])
        else:
            pts1 = np.empty((0, 2), dtype=np.float32)
            pts2 = np.empty((0, 2), dtype=np.float32)

        return MatchResult(
            raw_matches_count=raw_count,
            ratio_matches_count=ratio_count,
            final_matches_count=len(final_matches),
            final_matches=final_matches,
            matched_pts1=pts1,
            matched_pts2=pts2,
            matcher_type=self.matcher_type,
            matching_time_ms=elapsed_ms,
        )

    @staticmethod
    def visualize_matches(
        img1: np.ndarray,
        feat1: ExtractedFeatures,
        img2: np.ndarray,
        feat2: ExtractedFeatures,
        matches: List[cv2.DMatch],
        max_draw: int = 100,
        title: str = "Feature Matches",
    ) -> np.ndarray:
        """Draws visual matching lines connecting corresponding keypoints in two side-by-side images."""
        draw_matches = matches
        if len(matches) > max_draw:
            # Subsample evenly across match list
            step = len(matches) / float(max_draw)
            draw_matches = [matches[int(i * step)] for i in range(max_draw)]

        vis = cv2.drawMatches(
            img1,
            feat1.keypoints,
            img2,
            feat2.keypoints,
            draw_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(128, 128, 128),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        header = f"{title}: {len(matches)} matches shown (top {len(draw_matches)})"
        cv2.putText(vis, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

