"""
Matching Quality Analyzer Module
Evaluates SIFT descriptor matching quality, Lowe's ratio test pass rates,
and epipolar RANSAC consistency for keypoint pairs inside vs. outside object masks.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class PairwiseMatchingRecord:
    """Pairwise matching and geometric verification statistics between two camera views."""
    frame_idx1: int
    frame_idx2: int
    image_name1: str
    image_name2: str
    total_raw_matches: int
    # Inside mask metrics
    inside_raw_matches: int
    inside_ratio_matches: int
    inside_ratio_pass_pct: float
    inside_epipolar_inliers: int
    inside_epipolar_inlier_pct: float
    inside_median_epipolar_err_px: float
    # Outside mask metrics
    outside_raw_matches: int
    outside_ratio_matches: int
    outside_ratio_pass_pct: float
    outside_epipolar_inliers: int
    outside_epipolar_inlier_pct: float
    outside_median_epipolar_err_px: float

    def to_dict(self) -> dict:
        return asdict(self)


class MatchingQualityAnalyzer:
    """
    Performs pairwise descriptor matching and epipolar verification between frames,
    comparing features lying inside the object mask against features in the background scene.
    """

    def __init__(
        self,
        lowe_ratio: float = 0.75,
        ransac_threshold_px: float = 3.0,
        matcher_type: str = "FLANN",
    ):
        self.lowe_ratio = lowe_ratio
        self.ransac_threshold_px = ransac_threshold_px
        self.matcher_type = matcher_type

    def _create_matcher(self) -> cv2.DescriptorMatcher:
        if self.matcher_type == "FLANN":
            index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def evaluate_pair(
        self,
        frame_idx1: int,
        frame_idx2: int,
        img_name1: str,
        img_name2: str,
        kps1: List[cv2.KeyPoint],
        descs1: np.ndarray,
        mask1: Optional[np.ndarray],
        kps2: List[cv2.KeyPoint],
        descs2: np.ndarray,
        mask2: Optional[np.ndarray],
    ) -> PairwiseMatchingRecord:
        """
        Matches descriptors between view 1 and view 2, segregates correspondences
        by mask occupancy, and performs epipolar RANSAC on both sets.
        """
        if descs1 is None or len(descs1) < 2 or descs2 is None or len(descs2) < 2:
            return PairwiseMatchingRecord(
                frame_idx1=frame_idx1,
                frame_idx2=frame_idx2,
                image_name1=img_name1,
                image_name2=img_name2,
                total_raw_matches=0,
                inside_raw_matches=0,
                inside_ratio_matches=0,
                inside_ratio_pass_pct=0.0,
                inside_epipolar_inliers=0,
                inside_epipolar_inlier_pct=0.0,
                inside_median_epipolar_err_px=0.0,
                outside_raw_matches=0,
                outside_ratio_matches=0,
                outside_ratio_pass_pct=0.0,
                outside_epipolar_inliers=0,
                outside_epipolar_inlier_pct=0.0,
                outside_median_epipolar_err_px=0.0,
            )

        matcher = self._create_matcher()
        knn_matches = matcher.knnMatch(descs1, descs2, k=2)

        # Separate into inside/inside vs outside/outside
        in_raw, in_ratio = 0, 0
        out_raw, out_ratio = 0, 0
        pts1_in, pts2_in = [], []
        pts1_out, pts2_out = [], []

        h1, w1 = mask1.shape[:2] if mask1 is not None else (1920, 1080)
        h2, w2 = mask2.shape[:2] if mask2 is not None else (1920, 1080)

        for m_pair in knn_matches:
            if len(m_pair) < 2:
                continue
            m, n = m_pair[0], m_pair[1]
            p1 = kps1[m.queryIdx].pt
            p2 = kps2[m.trainIdx].pt

            x1_c = min(max(0, int(round(p1[0]))), w1 - 1)
            y1_c = min(max(0, int(round(p1[1]))), h1 - 1)
            x2_c = min(max(0, int(round(p2[0]))), w2 - 1)
            y2_c = min(max(0, int(round(p2[1]))), h2 - 1)

            in1 = (mask1[y1_c, x1_c] > 0) if mask1 is not None else False
            in2 = (mask2[y2_c, x2_c] > 0) if mask2 is not None else False

            is_ratio_pass = (m.distance < self.lowe_ratio * n.distance)

            if in1 and in2:
                in_raw += 1
                if is_ratio_pass:
                    in_ratio += 1
                    pts1_in.append(p1)
                    pts2_in.append(p2)
            elif (not in1) and (not in2):
                out_raw += 1
                if is_ratio_pass:
                    out_ratio += 1
                    pts1_out.append(p1)
                    pts2_out.append(p2)

        # Epipolar RANSAC for inside
        in_inliers = 0
        in_inlier_pct = 0.0
        in_median_err = 0.0
        if len(pts1_in) >= 8:
            pts1_arr = np.array(pts1_in, dtype=np.float32)
            pts2_arr = np.array(pts2_in, dtype=np.float32)
            F_in, mask_in = cv2.findFundamentalMat(
                pts1_arr, pts2_arr, cv2.FM_RANSAC, self.ransac_threshold_px
            )
            if mask_in is not None and F_in is not None:
                in_inliers = int(mask_in.sum())
                in_inlier_pct = (in_inliers / len(pts1_in)) * 100.0
                # Compute epipolar line error for inliers
                in_median_err = self._compute_median_epipolar_distance(
                    pts1_arr[mask_in.ravel() == 1],
                    pts2_arr[mask_in.ravel() == 1],
                    F_in
                )

        # Epipolar RANSAC for outside
        out_inliers = 0
        out_inlier_pct = 0.0
        out_median_err = 0.0
        if len(pts1_out) >= 8:
            pts1_arr = np.array(pts1_out, dtype=np.float32)
            pts2_arr = np.array(pts2_out, dtype=np.float32)
            F_out, mask_out = cv2.findFundamentalMat(
                pts1_arr, pts2_arr, cv2.FM_RANSAC, self.ransac_threshold_px
            )
            if mask_out is not None and F_out is not None:
                out_inliers = int(mask_out.sum())
                out_inlier_pct = (out_inliers / len(pts1_out)) * 100.0
                out_median_err = self._compute_median_epipolar_distance(
                    pts1_arr[mask_out.ravel() == 1],
                    pts2_arr[mask_out.ravel() == 1],
                    F_out
                )

        in_ratio_pct = (in_ratio / in_raw * 100.0) if in_raw > 0 else 0.0
        out_ratio_pct = (out_ratio / out_raw * 100.0) if out_raw > 0 else 0.0

        return PairwiseMatchingRecord(
            frame_idx1=frame_idx1,
            frame_idx2=frame_idx2,
            image_name1=img_name1,
            image_name2=img_name2,
            total_raw_matches=len(knn_matches),
            inside_raw_matches=in_raw,
            inside_ratio_matches=in_ratio,
            inside_ratio_pass_pct=round(in_ratio_pct, 2),
            inside_epipolar_inliers=in_inliers,
            inside_epipolar_inlier_pct=round(in_inlier_pct, 2),
            inside_median_epipolar_err_px=round(in_median_err, 3),
            outside_raw_matches=out_raw,
            outside_ratio_matches=out_ratio,
            outside_ratio_pass_pct=round(out_ratio_pct, 2),
            outside_epipolar_inliers=out_inliers,
            outside_epipolar_inlier_pct=round(out_inlier_pct, 2),
            outside_median_epipolar_err_px=round(out_median_err, 3),
        )

    @staticmethod
    def _compute_median_epipolar_distance(
        pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray
    ) -> float:
        """Computes median symmetric epipolar distance d(x2, F*x1) + d(x1, F.T*x2)."""
        if len(pts1) == 0:
            return 0.0
        # Homogeneous coordinates
        p1_h = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float32)])
        p2_h = np.hstack([pts2, np.ones((len(pts2), 1), dtype=np.float32)])

        # Epipolar lines in image 2: l2 = F * p1
        lines2 = (F @ p1_h.T).T  # (N, 3)
        # Distance from p2 to l2: |a*x + b*y + c| / sqrt(a^2 + b^2)
        numer2 = np.abs(np.sum(lines2 * p2_h, axis=1))
        denom2 = np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2) + 1e-8
        d2 = numer2 / denom2

        return float(np.median(d2))

