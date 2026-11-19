"""
Video Capture Quality Assistant Module
Provides automated pre-reconstruction validation for handheld video captures:
1. Frame Sharpness Detection (Laplacian variance blur estimation)
2. Frame Redundancy Detection (dHash perceptual deduplication & overlap preservation)
3. Feature Availability (SIFT/ORB keypoint distribution and texture density)
4. Camera Movement Assessment (Disparity and epipolar baseline tracking)
5. Angular Coverage Assessment (Clear distinction between estimated epipolar rotation vs. verified 3D orbit)
6. Actionable recommendations and overall reconstruction readiness scoring.

Designed for CPU-only execution with minimal RAM footprint (< 500 MB) and configurable sampling.
"""

from dataclasses import dataclass, asdict, field
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger("CaptureQualityAssistant")


@dataclass
class FrameQualityRecord:
    """Detailed quality measurements for an evaluated candidate frame."""
    frame_index: int
    timestamp_seconds: float
    blur_score: float
    is_blurry: bool
    is_redundant: bool
    hamming_distance_to_prev: Optional[int]
    keypoint_count: int
    feature_status: str              # "SUFFICIENT", "MARGINAL", "INSUFFICIENT"
    movement_status: str             # "OPTIMAL", "STATIONARY", "EXCESSIVE", "FIRST_FRAME", "NO_MATCH"
    mean_displacement_px: float      # Inter-frame disparity displacement
    inliers_count: int
    is_selected: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityAssessmentSummary:
    """Complete diagnostic report and reconstruction suitability score."""
    total_video_frames: int
    candidate_frames_evaluated: int
    selected_frames: int
    rejected_frames: int
    selection_rate_pct: float
    # Sharpness
    average_sharpness: float
    min_sharpness: float
    max_sharpness: float
    blurry_frames_count: int
    blur_percentage: float
    # Features
    average_keypoint_count: float
    min_keypoint_count: int
    feature_deficient_frames_count: int
    # Redundancy
    redundant_frames_count: int
    redundancy_percentage: float
    # Camera Movement & Overlap
    movement_breakdown: Dict[str, int]
    optimal_overlap_pairs_count: int
    stationary_pairs_count: int
    excessive_motion_pairs_count: int
    # Angular Coverage
    estimated_camera_coverage_deg: float
    estimated_coverage_description: str
    verified_coverage_available: bool
    verified_camera_coverage_deg: Optional[float]
    coverage_coverage_tier: str       # "NARROW_ARC", "MEDIUM_ARC", "WIDE_ARC", "FULL_ORBIT_CANDIDATE"
    # Status & Recommendations
    overall_status: str               # "GOOD", "WARNING", "POOR"
    status_headline: str
    status_message: str
    recommendations: List[str]
    per_frame_records: List[FrameQualityRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_frame_records"] = [r.to_dict() for r in self.per_frame_records]
        return d


class VideoCaptureQualityAssistant:
    """
    Evaluates video input quality against Structure from Motion and 3D reconstruction prerequisites.
    
    Operates strictly on CPU using optimized subsampling and downscaled metric extraction
    to ensure smooth performance on typical laptop processors.
    """

    def __init__(
        self,
        blur_threshold: float = 12.0,
        min_keypoints: int = 150,
        good_keypoints: int = 300,
        max_redundancy_hamming: int = 3,
        min_motion_displacement_px: float = 10.0,
        max_motion_displacement_px: float = 160.0,
        min_inliers_count: int = 15,
        target_eval_dimension: int = 640,
        sample_stride: int = 4,
    ):
        self.blur_threshold = blur_threshold
        self.min_keypoints = min_keypoints
        self.good_keypoints = good_keypoints
        self.max_redundancy_hamming = max_redundancy_hamming
        self.min_motion_displacement = min_motion_displacement_px
        self.max_motion_displacement = max_motion_displacement_px
        self.min_inliers_count = min_inliers_count
        self.target_eval_dimension = target_eval_dimension
        self.sample_stride = sample_stride

        # CPU-optimized SIFT detector with capped feature limit
        self.sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.04, edgeThreshold=10.0)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))

    @staticmethod
    def compute_dhash(gray: np.ndarray, hash_size: int = 8) -> int:
        """Computes 64-bit difference hash (dHash) to detect identical or near-duplicate frames."""
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        hash_val = 0
        for bit in diff.flatten():
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Computes bitwise Hamming distance between two 64-bit integer hashes."""
        return bin(h1 ^ h2).count("1")

    def _estimate_relative_rotation_yaw(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        img_width: int,
        img_height: int,
    ) -> float:
        """
        Estimates relative yaw rotation angle in degrees from two-view epipolar geometry.
        Uses an assumed focal length if uncalibrated.
        """
        if len(pts1) < 8:
            return 0.0

        # Construct approximate intrinsics matrix
        focal_est = 1.2 * max(img_width, img_height)
        K = np.array([
            [focal_est, 0.0, img_width / 2.0],
            [0.0, focal_est, img_height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        try:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=2.0
            )
            if E is None or mask is None or mask.sum() < 8:
                return 0.0

            # Decompose Essential Matrix into R and t
            _, R, _, inlier_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
            if R is None:
                return 0.0

            # Extract yaw angle around vertical Y axis
            yaw_rad = np.arctan2(R[0, 2], R[2, 2])
            return abs(float(np.degrees(yaw_rad)))
        except Exception:
            return 0.0

    def assess_video(
        self,
        video_path: Union[str, Path],
        verified_sfm_dir: Optional[Union[str, Path]] = None,
        max_eval_frames: Optional[int] = 120,
    ) -> QualityAssessmentSummary:
        """
        Executes end-to-end quality assessment on a video file.
        
        Args:
            video_path: Path to the MP4/MOV/AVI video file.
            verified_sfm_dir: Optional directory with existing calibrated SfM outputs to cross-verify orbit.
            max_eval_frames: Maximum number of candidate frames to evaluate (guards CPU time).
        """
        v_path = Path(video_path).resolve()
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file with OpenCV: {v_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Adjust stride if video is very long to cap runtime
        stride = self.sample_stride
        if max_eval_frames and total_frames > 0 and (total_frames // stride) > max_eval_frames:
            stride = max(1, total_frames // max_eval_frames)

        records: List[FrameQualityRecord] = []
        blur_scores: List[float] = []
        keypoint_counts: List[int] = []

        prev_gray: Optional[np.ndarray] = None
        prev_hash: Optional[int] = None
        prev_kp = None
        prev_des = None

        accumulated_yaw_degrees = 0.0
        consecutive_pairs_evaluated = 0

        frame_idx = 0
        candidate_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx % stride) != 0:
                frame_idx += 1
                continue

            candidate_count += 1
            t_sec = float(frame_idx / fps)

            # Resize frame for CPU efficiency
            h, w = frame.shape[:2]
            scale = min(1.0, float(self.target_eval_dimension) / max(h, w))
            if scale < 1.0:
                eval_w, eval_h = int(w * scale), int(h * scale)
                small_frame = cv2.resize(frame, (eval_w, eval_h), interpolation=cv2.INTER_AREA)
            else:
                small_frame = frame
                eval_w, eval_h = w, h

            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # -------------------------------------------------------------
            # A. Frame Sharpness Detection (Laplacian Variance)
            # -------------------------------------------------------------
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_scores.append(lap_var)
            is_blurry = lap_var < self.blur_threshold

            # -------------------------------------------------------------
            # B. Frame Redundancy Detection (dHash)
            # -------------------------------------------------------------
            curr_hash = self.compute_dhash(gray)
            is_redundant = False
            hamm_dist: Optional[int] = None

            if prev_hash is not None:
                hamm_dist = self.hamming_distance(prev_hash, curr_hash)
                if hamm_dist <= self.max_redundancy_hamming:
                    is_redundant = True

            # -------------------------------------------------------------
            # C. Feature Availability (SIFT Keypoints)
            # -------------------------------------------------------------
            kp, des = self.sift.detectAndCompute(gray, None)
            kp_count = len(kp) if kp is not None else 0
            keypoint_counts.append(kp_count)

            if kp_count >= self.good_keypoints:
                feat_status = "SUFFICIENT"
            elif kp_count >= self.min_keypoints:
                feat_status = "MARGINAL"
            else:
                feat_status = "INSUFFICIENT"

            # -------------------------------------------------------------
            # D. Camera Movement & Baseline Overlap Assessment
            # -------------------------------------------------------------
            movement_status = "FIRST_FRAME"
            mean_disp = 0.0
            inliers_count = 0

            if prev_des is not None and des is not None and len(prev_des) >= 2 and len(des) >= 2:
                consecutive_pairs_evaluated += 1
                try:
                    matches = self.flann.knnMatch(prev_des, des, k=2)
                    good_matches = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]

                    if len(good_matches) >= 8:
                        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
                        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

                        # Epipolar RANSAC verification
                        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.5)
                        if mask is not None:
                            inliers_mask = mask.ravel() == 1
                            inliers_count = int(inliers_mask.sum())
                            if inliers_count >= 8:
                                inlier_pts1 = pts1[inliers_mask]
                                inlier_pts2 = pts2[inliers_mask]
                                displacements = np.linalg.norm(inlier_pts2 - inlier_pts1, axis=1)
                                mean_disp = float(np.mean(displacements))

                                # -------------------------------------------------
                                # E. Angular Coverage Estimation (Relative Yaw)
                                # -------------------------------------------------
                                pair_yaw = self._estimate_relative_rotation_yaw(
                                    inlier_pts1, inlier_pts2, eval_w, eval_h
                                )
                                # Clamp realistic single-step camera rotation (< 30 deg/step)
                                if pair_yaw < 30.0:
                                    accumulated_yaw_degrees += pair_yaw

                        # Movement categorization
                        if inliers_count < self.min_inliers_count:
                            movement_status = "EXCESSIVE"  # Low tracking overlap
                        elif mean_disp < self.min_motion_displacement:
                            movement_status = "STATIONARY" # Too little baseline
                        elif mean_disp > self.max_motion_displacement:
                            movement_status = "EXCESSIVE"  # Jumped too fast
                        else:
                            movement_status = "OPTIMAL"    # Healthy parallax & overlap
                    else:
                        movement_status = "NO_MATCH"
                except Exception:
                    movement_status = "NO_MATCH"

            # -------------------------------------------------------------
            # Selection Decision
            # -------------------------------------------------------------
            rejection_reasons = []
            if is_blurry:
                rejection_reasons.append(f"Excessive blur ({lap_var:.1f} < {self.blur_threshold:.1f})")
            if is_redundant:
                rejection_reasons.append(f"Near-duplicate stationary frame (dHash dist={hamm_dist})")
            if feat_status == "INSUFFICIENT":
                rejection_reasons.append(f"Texture deficient ({kp_count} features < {self.min_keypoints})")

            is_selected = (len(rejection_reasons) == 0)
            rej_reason_str = "; ".join(rejection_reasons) if rejection_reasons else None

            record = FrameQualityRecord(
                frame_index=frame_idx,
                timestamp_seconds=round(t_sec, 3),
                blur_score=round(lap_var, 2),
                is_blurry=is_blurry,
                is_redundant=is_redundant,
                hamming_distance_to_prev=hamm_dist,
                keypoint_count=kp_count,
                feature_status=feat_status,
                movement_status=movement_status,
                mean_displacement_px=round(mean_disp, 2),
                inliers_count=inliers_count,
                is_selected=is_selected,
                rejection_reason=rej_reason_str,
            )
            records.append(record)

            # Update history only with non-blurry, informative frames to maintain reference baseline
            if not is_blurry:
                prev_gray = gray
                prev_hash = curr_hash
                prev_kp = kp
                prev_des = des

            frame_idx += 1

        cap.release()

        # -------------------------------------------------------------
        # Aggregate Quality Analytics
        # -------------------------------------------------------------
        total_eval = len(records)
        selected_count = sum(1 for r in records if r.is_selected)
        rejected_count = total_eval - selected_count
        selection_rate = (selected_count / total_eval * 100.0) if total_eval > 0 else 0.0

        avg_sharpness = float(np.mean(blur_scores)) if blur_scores else 0.0
        min_sharpness = float(np.min(blur_scores)) if blur_scores else 0.0
        max_sharpness = float(np.max(blur_scores)) if blur_scores else 0.0
        blurry_count = sum(1 for r in records if r.is_blurry)
        blur_pct = (blurry_count / total_eval * 100.0) if total_eval > 0 else 0.0

        avg_kp = float(np.mean(keypoint_counts)) if keypoint_counts else 0.0
        min_kp = int(np.min(keypoint_counts)) if keypoint_counts else 0
        feat_def_count = sum(1 for r in records if r.feature_status == "INSUFFICIENT")

        redundant_count = sum(1 for r in records if r.is_redundant)
        redundancy_pct = (redundant_count / total_eval * 100.0) if total_eval > 0 else 0.0

        # Movement breakdown
        move_breakdown = {
            "OPTIMAL": sum(1 for r in records if r.movement_status == "OPTIMAL"),
            "STATIONARY": sum(1 for r in records if r.movement_status == "STATIONARY"),
            "EXCESSIVE": sum(1 for r in records if r.movement_status == "EXCESSIVE"),
            "NO_MATCH": sum(1 for r in records if r.movement_status == "NO_MATCH"),
        }

        # -------------------------------------------------------------
        # E. Angular Coverage Categorization & Cross-Verification
        # -------------------------------------------------------------
        verified_coverage_available = False
        verified_coverage_deg: Optional[float] = None

        if verified_sfm_dir:
            v_dir = Path(verified_sfm_dir)
            cams_file = v_dir / "cameras.json"
            if cams_file.exists():
                try:
                    with open(cams_file, "r") as f:
                        cams_data = json.load(f)
                    reg_cams = [c for c in cams_data.values() if c.get("is_registered", False)]
                    if len(reg_cams) >= 3:
                        # Extract verified azimuth orbit
                        azimuths = []
                        for c in reg_cams:
                            cc = c["camera_center"]
                            azimuths.append(np.arctan2(cc[0], cc[2]))
                        unwrapped = np.unwrap(azimuths)
                        verified_coverage_deg = float(np.degrees(np.max(unwrapped) - np.min(unwrapped)))
                        verified_coverage_available = True
                except Exception as e:
                    logger.debug(f"Could not parse verified camera orbit: {e}")

        # Formulate estimated coverage text (distinguishing estimated from verified)
        est_deg = round(float(accumulated_yaw_degrees), 1)
        if est_deg < 50.0:
            cov_tier = "NARROW_ARC"
            cov_desc = f"Narrow Viewing Arc (~{est_deg}° estimated). Limited parallax; ideal for 2.5D relief only."
        elif est_deg < 140.0:
            cov_tier = "MEDIUM_ARC"
            cov_desc = f"Medium Viewing Arc (~{est_deg}° estimated). Good multi-view baseline for front and sides."
        elif est_deg < 260.0:
            cov_tier = "WIDE_ARC"
            cov_desc = f"Wide Viewing Arc (~{est_deg}° estimated). Strong multi-angle coverage across majority of object."
        else:
            cov_tier = "FULL_ORBIT_CANDIDATE"
            cov_desc = f"Full Orbit Candidate (~{est_deg}° estimated). Approaching complete 360° encirclement."

        # -------------------------------------------------------------
        # Overall Suitability Decision Engine
        # -------------------------------------------------------------
        penalty_score = 0
        recommendations = []

        # Sharpness rules
        if blur_pct > 40.0:
            penalty_score += 3
            recommendations.append("Move the camera more slowly to reduce motion blur; keep the device stable.")
        elif blur_pct > 15.0:
            penalty_score += 1
            recommendations.append("Moderate blur detected in some frames. Consider holding the smartphone with two hands.")

        # Redundancy rules
        if redundancy_pct > 45.0:
            penalty_score += 2
            recommendations.append("Avoid pausing or holding the camera stationary. Maintain continuous, steady orbit.")
        elif redundancy_pct > 25.0:
            penalty_score += 1

        # Feature density rules
        if avg_kp < self.min_keypoints or feat_def_count > (0.35 * total_eval):
            penalty_score += 3
            recommendations.append("Ensure adequate, uniform lighting and include surrounding textured surfaces or background cues.")
        elif avg_kp < self.good_keypoints:
            penalty_score += 1

        # Movement & baseline rules
        excessive_ratio = (move_breakdown["EXCESSIVE"] + move_breakdown["NO_MATCH"]) / max(1, consecutive_pairs_evaluated)
        if excessive_ratio > 0.40:
            penalty_score += 3
            recommendations.append("Camera moved too rapidly between frames, breaking feature tracks. Maintain 60-80% visual overlap.")
        elif excessive_ratio > 0.20:
            penalty_score += 1

        # Angular coverage rules
        if est_deg < 45.0:
            penalty_score += 1
            recommendations.append(f"Current viewing arc is only ~{est_deg}°. Capture multiple viewing angles around the object.")

        # Always include universal best practice
        recommendations.append("Avoid optical or digital zooming during recording; maintain a constant focal length.")

        # Final grading
        if penalty_score <= 1 and selected_count >= 10:
            overall_status = "GOOD"
            headline = "Suitable for 3D Reconstruction"
            msg = "The video exhibits sharp focus, rich feature density, and healthy inter-frame baseline disparity. Ready for SfM!"
        elif penalty_score <= 4 and selected_count >= 6:
            overall_status = "WARNING"
            headline = "Reconstruction Possible, but Input Quality Should Improve"
            msg = "Reconstruction can proceed, but results may exhibit localized gaps or noise due to partial blur or narrow viewing angles."
        else:
            overall_status = "POOR"
            headline = "Not Recommended for Reconstruction — Capture Another Video"
            msg = "High rejection rate due to motion blur, texture deficiency, or erratic tracking. Re-recording with the recommended settings is advised."

        return QualityAssessmentSummary(
            total_video_frames=total_frames,
            candidate_frames_evaluated=total_eval,
            selected_frames=selected_count,
            rejected_frames=rejected_count,
            selection_rate_pct=round(selection_rate, 1),
            average_sharpness=round(avg_sharpness, 2),
            min_sharpness=round(min_sharpness, 2),
            max_sharpness=round(max_sharpness, 2),
            blurry_frames_count=blurry_count,
            blur_percentage=round(blur_pct, 1),
            average_keypoint_count=round(avg_kp, 1),
            min_keypoint_count=min_kp,
            feature_deficient_frames_count=feat_def_count,
            redundant_frames_count=redundant_count,
            redundancy_percentage=round(redundancy_pct, 1),
            movement_breakdown=move_breakdown,
            optimal_overlap_pairs_count=move_breakdown["OPTIMAL"],
            stationary_pairs_count=move_breakdown["STATIONARY"],
            excessive_motion_pairs_count=move_breakdown["EXCESSIVE"],
            estimated_camera_coverage_deg=est_deg,
            estimated_coverage_description=cov_desc,
            verified_coverage_available=verified_coverage_available,
            verified_camera_coverage_deg=round(verified_coverage_deg, 1) if verified_coverage_deg is not None else None,
            coverage_coverage_tier=cov_tier,
            overall_status=overall_status,
            status_headline=headline,
            status_message=msg,
            recommendations=recommendations,
            per_frame_records=records,
        )

