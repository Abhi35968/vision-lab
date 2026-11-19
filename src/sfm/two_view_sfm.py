"""
Two-View Structure from Motion Pipeline
Integrates feature extraction, matching, geometric verification, pose recovery,
DLT triangulation, and pair evaluation into an end-to-end reconstruction workflow.
"""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.sfm.feature_extractor import FeatureExtractor, ExtractedFeatures
from src.sfm.feature_matcher import FeatureMatcher, MatchResult
from src.sfm.geometry import GeometricVerifier, GeometricVerificationResult
from src.sfm.pose_estimator import PoseEstimator, PoseEstimationResult
from src.sfm.triangulator import Triangulator, TriangulationResult


class TwoViewSfM:
    """
    Executes calibrated two-view epipolar reconstruction and keyframe pair evaluation.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "config/pipeline_config.yaml",
        detector_name: str = "SIFT",
        n_features: int = 4000,
        lowe_ratio: float = 0.75,
        ransac_threshold: float = 2.0,
        max_reprojection_error: float = 4.0,
        calibration_source: str = "auto",  # "auto", "calibrated", "estimated", "uncalibrated"
    ):
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config()

        # Extract features config overrides or use arguments
        f_cfg = self.config.get("features", {})
        det = detector_name or f_cfg.get("detector", "SIFT")
        n_feat = n_features or f_cfg.get(det.lower(), {}).get("n_features", 4000)
        l_ratio = lowe_ratio or float(f_cfg.get("lowe_ratio", 0.75))
        r_thresh = ransac_threshold or float(f_cfg.get("ransac_reproj_threshold", 2.0))

        self.extractor = FeatureExtractor(detector=det, n_features=n_feat)
        self.matcher = FeatureMatcher(matcher_type=f_cfg.get("matcher", "FLANN"), lowe_ratio=l_ratio)
        self.verifier = GeometricVerifier(ransac_reproj_threshold=r_thresh)
        self.pose_estimator = PoseEstimator()
        self.triangulator = Triangulator(max_reprojection_error_px=max_reprojection_error)
        self.calibration_source = calibration_source

    def _load_config(self) -> dict:
        """Loads configuration settings from YAML."""
        if self.config_path and self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def resolve_camera_intrinsics(
        self, image_width: int, image_height: int
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Loads calibrated K from config/camera_intrinsics.json or computes an estimated K from FoV.
        
        Returns:
            Tuple of (K matrix or None, status string: 'calibrated' | 'estimated' | 'uncalibrated')
        """
        calib_file = Path("config/camera_intrinsics.json")
        if self.calibration_source in {"auto", "calibrated"} and calib_file.exists():
            with open(calib_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "calibrated" and "camera_matrix_k" in data:
                K = np.array(data["camera_matrix_k"], dtype=np.float64)
                if K.shape == (3, 3) and K[0, 0] > 0:
                    return K, "calibrated"

        if self.calibration_source in {"auto", "estimated"}:
            # Estimate intrinsics based on standard smartphone wide camera field of view (65 degrees)
            fov_deg = 65.0
            fov_rad = np.radians(fov_deg)
            fx = (image_width / 2.0) / np.tan(fov_rad / 2.0)
            fy = fx  # Square pixel assumption
            cx = image_width / 2.0
            cy = image_height / 2.0
            K_est = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            return K_est, "estimated"

        return None, "uncalibrated"

    def reconstruct_pair(
        self,
        img1_path: Union[str, Path],
        img2_path: Union[str, Path],
        output_dir: Union[str, Path] = "outputs/sfm",
        vis_dir: Union[str, Path] = "outputs/visualizations",
    ) -> dict:
        """
        Executes end-to-end two-view SfM on an image pair.
        """
        p1 = Path(img1_path).resolve()
        p2 = Path(img2_path).resolve()
        out_d = Path(output_dir).resolve()
        v_dir = Path(vis_dir).resolve()
        out_d.mkdir(parents=True, exist_ok=True)
        v_dir.mkdir(parents=True, exist_ok=True)

        img1 = cv2.imread(str(p1))
        img2 = cv2.imread(str(p2))
        if img1 is None or img2 is None:
            raise ValueError(f"Failed to load pair images: '{p1}' or '{p2}'")

        h1, w1 = img1.shape[:2]

        # 1. Resolve Intrinsics
        K, calib_status = self.resolve_camera_intrinsics(w1, h1)

        # 2. Extract Features
        feat1 = self.extractor.extract(img1)
        feat2 = self.extractor.extract(img2)

        # 3. Match Features
        match_res = self.matcher.match(feat1, feat2)

        # 4. Geometric Verification (RANSAC)
        geom_res = self.verifier.verify(
            match_res.matched_pts1,
            match_res.matched_pts2,
            camera_matrix_k=K,
            calibration_status=calib_status,
        )

        pose_res: Optional[PoseEstimationResult] = None
        tri_res: Optional[TriangulationResult] = None
        reconstruction_status = "FAILED"
        failure_reason = None

        if not geom_res.is_valid:
            failure_reason = f"Geometric verification failed: {geom_res.rejection_reason}"
        elif K is None:
            failure_reason = "Camera calibration unavailable. Calibrated pose recovery requires K."
        elif geom_res.essential_matrix is None:
            failure_reason = "Essential Matrix estimation failed."
        else:
            # 5. Pose Estimation
            pose_res = self.pose_estimator.estimate_pose(
                geom_res.essential_matrix,
                geom_res.inlier_pts1,
                geom_res.inlier_pts2,
                K,
            )

            if not pose_res.is_valid:
                failure_reason = f"Pose recovery failed: {pose_res.rejection_reason}"
            else:
                # 6. Triangulation
                tri_res = self.triangulator.triangulate(
                    geom_res.inlier_pts1,
                    geom_res.inlier_pts2,
                    K,
                    pose_res.rotation_matrix,
                    pose_res.translation_vector,
                    image1_bgr=img1,
                )

                if tri_res.is_valid:
                    reconstruction_status = "SUCCESS"
                    # Save point cloud assets
                    ply_path = out_d / "sparse_points.ply"
                    npy_path = out_d / "sparse_points.npy"
                    self.triangulator.save_point_cloud(
                        tri_res.points_3d, tri_res.point_colors, ply_path, npy_path
                    )
                else:
                    reconstruction_status = "FAILED"
                    failure_reason = f"Triangulation yielded too few points ({tri_res.points_reconstructed} < {self.triangulator.min_points_threshold})."

        # 7. Generate Visualizations
        vis_paths = self._generate_visualizations(
            img1, feat1, img2, feat2, match_res, geom_res, pose_res, tri_res, v_dir
        )

        # 8. Compile Comprehensive Report
        report = {
            "timestamp": datetime.now().isoformat(),
            "image1": p1.name,
            "image2": p2.name,
            "reconstruction_status": reconstruction_status,
            "failure_reason": failure_reason,
            "calibration": {
                "status": calib_status,
                "camera_matrix_k": K.tolist() if K is not None else None,
                "note": (
                    "Using true calibration from config/camera_intrinsics.json"
                    if calib_status == "calibrated"
                    else ("Using estimated pinhole FOV intrinsics (assumed 65 deg FOV)"
                          if calib_status == "estimated"
                          else "No calibration provided. Euclidean reconstruction unavailable.")
                ),
            },
            "features": {
                "detector": feat1.detector_name,
                "img1_keypoints": feat1.num_keypoints,
                "img2_keypoints": feat2.num_keypoints,
                "extraction_time_ms": round(feat1.extraction_time_ms + feat2.extraction_time_ms, 2),
            },
            "matching": match_res.to_dict(),
            "geometry": geom_res.to_dict(),
            "pose": pose_res.to_dict() if pose_res is not None else None,
            "triangulation": tri_res.to_dict() if tri_res is not None else None,
            "visualizations": vis_paths,
        }

        # Save JSON report
        report_file = out_d / "two_view_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def evaluate_pairs(
        self,
        image_paths: List[Union[str, Path]],
        output_json: Union[str, Path] = "outputs/sfm/pair_evaluation.json",
    ) -> List[dict]:
        """
        Evaluates consecutive and nearby frame pairs to identify optimal baseline pairs.
        """
        eval_records: List[dict] = []
        valid_paths = [Path(p) for p in image_paths if Path(p).exists()]
        if len(valid_paths) < 2:
            return eval_records

        print(f"[INFO] Evaluating {len(valid_paths) - 1} consecutive pairs for reconstruction quality...")

        for i in range(len(valid_paths) - 1):
            p1 = valid_paths[i]
            p2 = valid_paths[i + 1]

            img1 = cv2.imread(str(p1))
            img2 = cv2.imread(str(p2))
            if img1 is None or img2 is None:
                continue

            h1, w1 = img1.shape[:2]
            K, calib_status = self.resolve_camera_intrinsics(w1, h1)

            feat1 = self.extractor.extract(img1)
            feat2 = self.extractor.extract(img2)
            match_res = self.matcher.match(feat1, feat2)
            geom_res = self.verifier.verify(
                match_res.matched_pts1, match_res.matched_pts2, camera_matrix_k=K, calibration_status=calib_status
            )

            tri_pts_count = 0
            mean_reproj_err = 0.0
            pair_status = "WEAK_GEOMETRY"

            if geom_res.is_valid and K is not None and geom_res.essential_matrix is not None:
                pose_res = self.pose_estimator.estimate_pose(
                    geom_res.essential_matrix, geom_res.inlier_pts1, geom_res.inlier_pts2, K
                )
                if pose_res.is_valid:
                    tri_res = self.triangulator.triangulate(
                        geom_res.inlier_pts1, geom_res.inlier_pts2, K, pose_res.rotation_matrix, pose_res.translation_vector
                    )
                    tri_pts_count = tri_res.points_reconstructed
                    mean_reproj_err = tri_res.mean_reprojection_error
                    pair_status = "USABLE" if tri_res.is_valid else "FEW_POINTS"

            eval_records.append({
                "pair_index": i,
                "image1": p1.name,
                "image2": p2.name,
                "kps1": feat1.num_keypoints,
                "kps2": feat2.num_keypoints,
                "matches": match_res.final_matches_count,
                "inliers": geom_res.num_inliers,
                "inlier_ratio": round(geom_res.inlier_ratio, 3),
                "triangulated_points": tri_pts_count,
                "mean_reproj_error_px": round(mean_reproj_err, 3),
                "status": pair_status,
            })

        out_path = Path(output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(eval_records, f, indent=2)

        return eval_records

    def _generate_visualizations(
        self,
        img1: np.ndarray,
        feat1: ExtractedFeatures,
        img2: np.ndarray,
        feat2: ExtractedFeatures,
        match_res: MatchResult,
        geom_res: GeometricVerificationResult,
        pose_res: Optional[PoseEstimationResult],
        tri_res: Optional[TriangulationResult],
        vis_dir: Path,
    ) -> Dict[str, str]:
        """Generates all 2D/3D diagnostic figures for two-view SfM."""
        paths = {}

        # 1. Keypoint Overlays
        kp_vis1 = FeatureExtractor.visualize_keypoints(img1, feat1)
        kp_vis2 = FeatureExtractor.visualize_keypoints(img2, feat2)
        cv2.imwrite(str(vis_dir / "sfm_keypoints_img1.png"), kp_vis1)
        cv2.imwrite(str(vis_dir / "sfm_keypoints_img2.png"), kp_vis2)
        paths["keypoints_img1"] = str(vis_dir / "sfm_keypoints_img1.png")
        paths["keypoints_img2"] = str(vis_dir / "sfm_keypoints_img2.png")

        # 2. Raw Matches Visualization
        raw_vis = FeatureMatcher.visualize_matches(
            img1, feat1, img2, feat2, match_res.final_matches, max_draw=80, title="Lowe's Ratio Matches"
        )
        cv2.imwrite(str(vis_dir / "sfm_raw_matches.png"), raw_vis)
        paths["raw_matches"] = str(vis_dir / "sfm_raw_matches.png")

        # 3. Inlier Matches Visualization
        if geom_res.num_inliers > 0:
            inlier_matches = [
                m for m, is_in in zip(match_res.final_matches, geom_res.inliers_mask) if is_in
            ]
            inlier_vis = FeatureMatcher.visualize_matches(
                img1, feat1, img2, feat2, inlier_matches, max_draw=80, title="RANSAC Epipolar Inliers"
            )
            cv2.imwrite(str(vis_dir / "sfm_inlier_matches.png"), inlier_vis)
            paths["inlier_matches"] = str(vis_dir / "sfm_inlier_matches.png")

        # 4. Epipolar Lines Visualization
        if geom_res.fundamental_matrix is not None and geom_res.num_inliers > 0:
            el1, el2 = GeometricVerifier.visualize_epipolar_lines(
                img1, img2, match_res.matched_pts1, match_res.matched_pts2,
                geom_res.fundamental_matrix, inlier_mask=geom_res.inliers_mask, num_lines=25
            )
            epi_combined = np.hstack([el1, el2])
            cv2.imwrite(str(vis_dir / "sfm_epipolar_lines.png"), epi_combined)
            paths["epipolar_lines"] = str(vis_dir / "sfm_epipolar_lines.png")

        # 5. Camera Poses & Sparse 3D Point Cloud Scatter
        if tri_res is not None and tri_res.points_reconstructed > 0 and pose_res is not None:
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")

            pts = tri_res.points_3d
            ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c="gray", s=8, alpha=0.6, label="Sparse 3D Points")

            # Camera 1 optical center (0, 0, 0)
            c1 = pose_res.camera1_center
            c2 = pose_res.camera2_center
            ax.scatter([c1[0]], [c1[2]], [-c1[1]], c="blue", s=100, marker="^", label="Camera 1 [0,0,0]")
            ax.scatter([c2[0]], [c2[2]], [-c2[1]], c="red", s=100, marker="^", label="Camera 2 Pose")

            # Baseline vector
            ax.plot([c1[0], c2[0]], [c1[2], c2[2]], [-c1[1], -c2[1]], "k--", linewidth=1.5, label="Baseline")

            ax.set_title(f"Two-View Sparse 3D Reconstruction ({len(pts)} points)")
            ax.set_xlabel("X (Width)")
            ax.set_ylabel("Z (Depth)")
            ax.set_zlabel("Y (Height)")
            ax.legend()
            plt.tight_layout()
            cam_vis_p = vis_dir / "sfm_camera_trajectory.png"
            plt.savefig(str(cam_vis_p), dpi=150)
            plt.close()
            paths["camera_trajectory"] = str(cam_vis_p)

        return paths

