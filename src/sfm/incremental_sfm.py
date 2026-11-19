"""
Incremental Multi-View Structure from Motion Orchestrator
Coordinates multi-view matching, initialization, PnP registration, triangulation, BA, and reporting.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.sfm.feature_extractor import FeatureExtractor, ExtractedFeatures
from src.sfm.feature_matcher import FeatureMatcher
from src.sfm.geometry import GeometricVerifier
from src.sfm.feature_tracks import FeatureTrackSystem
from src.sfm.multi_view_matcher import MultiViewMatcher, VerifiedPairMatch
from src.sfm.initialization import InitialPairSelector, InitializationResult
from src.sfm.camera_registration import CameraRegistrar, CameraPose
from src.sfm.multi_view_triangulator import MultiViewTriangulator, MultiViewTriangulationResult
from src.sfm.bundle_adjustment import BundleAdjuster, BundleAdjustmentResult
from src.sfm.reconstruction_metrics import ReconstructionMetrics, ReconstructionDiagnostics
from src.sfm.outlier_diagnostics import OutlierDiagnostics
from src.sfm.visualization import SfMVisualizer

logger = logging.getLogger("IncrementalSfM")


@dataclass
class IncrementalSfMConfig:
    """Configuration parameters for incremental multi-view reconstruction."""
    max_keyframes: int = 10
    detector_type: str = "SIFT"
    max_features: int = 2000
    matching_window: int = 2
    min_inliers_pair: int = 20
    min_track_length: int = 2
    min_pnp_inliers: int = 15
    min_triangulation_angle_deg: float = 1.5
    max_reprojection_error_px: float = 4.0
    enable_bundle_adjustment: bool = True
    ba_max_iterations: int = 50
    forced_seed_pair: Optional[Tuple[int, int]] = None
    output_dir: str = "outputs/sfm/multi_view"


class IncrementalSfM:
    """
    End-to-End Incremental Structure from Motion Pipeline.
    Progressively reconstructs camera poses and 3D point cloud in arbitrary_units from a sequential video stream.
    """

    def __init__(self, config: Optional[IncrementalSfMConfig] = None):
        self.config = config or IncrementalSfMConfig()
        
        # Initialize sub-modules
        self.extractor = FeatureExtractor(
            detector=self.config.detector_type,
            n_features=self.config.max_features,
        )
        matcher_type = "FLANN" if self.config.detector_type.upper() == "SIFT" else "BF"
        self.matcher = FeatureMatcher(
            matcher_type=matcher_type,
            lowe_ratio=0.75,
        )
        self.verifier = GeometricVerifier(
            ransac_reproj_threshold=self.config.max_reprojection_error_px,
        )
        self.multi_matcher = MultiViewMatcher(
            extractor=self.extractor,
            matcher=self.matcher,
            verifier=self.verifier,
            matching_window=self.config.matching_window,
            min_inliers_threshold=self.config.min_inliers_pair,
        )
        self.init_selector = InitialPairSelector(
            min_inliers=self.config.min_inliers_pair,
            min_parallax_deg=self.config.min_triangulation_angle_deg,
            max_reprojection_error_px=self.config.max_reprojection_error_px,
        )
        self.registrar = CameraRegistrar(
            min_pnp_inliers=self.config.min_pnp_inliers,
            max_reprojection_error_px=self.config.max_reprojection_error_px,
        )
        self.triangulator = MultiViewTriangulator(
            max_reprojection_error_px=self.config.max_reprojection_error_px,
            min_parallax_angle_deg=self.config.min_triangulation_angle_deg,
        )
        self.bundle_adjuster = BundleAdjuster(
            max_nfev=self.config.ba_max_iterations,
        )

    @staticmethod
    def load_camera_intrinsics(
        intrinsics_path: Union[str, Path],
        actual_image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int], str]:
        """
        Loads camera calibration matrix K, resolution (W, H), and calibration status.
        If intrinsics file is missing or uncalibrated, estimates K from image dimensions
        assuming standard ~60-degree smartphone field of view (f ≈ 1.2 * max(W, H)).
        """
        path = Path(intrinsics_path)
        if not path.exists():
            if actual_image_size is not None:
                w, h = actual_image_size
            else:
                w, h = (1920, 1080)
            focal = max(w, h) * 1.2
            K = np.array([
                [focal, 0.0, w / 2.0],
                [0.0, focal, h / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            return K, (w, h), "estimated"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        status = data.get("status") or data.get("calibration_status", "estimated")
        w = int(data.get("image_width", actual_image_size[0] if actual_image_size else 1920))
        h = int(data.get("image_height", actual_image_size[1] if actual_image_size else 1080))

        # If actual image dimensions provided differ from metadata, use actual
        if actual_image_size is not None and (actual_image_size[0] != w or actual_image_size[1] != h):
            w, h = actual_image_size

        k_list = data.get("camera_matrix_k") or data.get("camera_matrix") or data.get("K")
        if k_list is not None:
            K_cand = np.asarray(k_list, dtype=np.float64)
            if K_cand.shape == (3, 3) and K_cand[0, 0] > 1e-4 and status.lower() != "uncalibrated":
                return K_cand, (w, h), status

        # Fallback estimation based on image resolution (60 deg FoV approximation)
        focal = max(w, h) * 1.2
        K = np.array([
            [focal, 0.0, w / 2.0],
            [0.0, focal, h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return K, (w, h), "estimated"

    def run(
        self,
        image_paths: List[Union[str, Path]],
        camera_matrix_k: np.ndarray,
        image_size: Tuple[int, int],
        calibration_status: str = "estimated",
        forced_seed_pair: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Dict[int, CameraPose], FeatureTrackSystem, ReconstructionDiagnostics, Optional[BundleAdjustmentResult]]:
        """
        Executes the complete incremental SfM reconstruction pipeline.
        """
        start_time = time.time()
        active_forced_seed = forced_seed_pair if forced_seed_pair is not None else self.config.forced_seed_pair
        # Cap keyframes according to config
        selected_paths = [Path(p) for p in image_paths[: self.config.max_keyframes]]
        N = len(selected_paths)
        print(f"\n==============================================================================")
        print(f" INCREMENTAL MULTI-VIEW STRUCTURE FROM MOTION")
        print(f"==============================================================================")
        print(f"Keyframe Count:         {N} images")
        print(f"Detector / Matcher:     {self.config.detector_type} (Max {self.config.max_features} features)")
        print(f"Sliding Window W:       {self.config.matching_window} (Windowed pair matching)")
        print(f"Calibration Status:     {calibration_status}")
        print(f"Intrinsics Matrix K:\n{camera_matrix_k}")
        print(f"------------------------------------------------------------------------------")

        # Step 1: Feature Extraction
        print(f"[Step 1/6] Extracting features for {N} keyframes...")
        features_list = self.multi_matcher.extract_all_features(selected_paths)
        for i, feat in enumerate(features_list):
            print(f"  Frame {i:02d} ({Path(feat.image_path).name}): {feat.num_keypoints} {feat.detector_name} keypoints ({feat.extraction_time_ms:.1f} ms)")

        # Step 2: Windowed Matching & Feature Track Construction
        print(f"\n[Step 2/6] Matching keyframe pairs within window W = {self.config.matching_window}...")
        verified_pairs, track_system = self.multi_matcher.match_keyframes(
            features_list,
            camera_matrix_k=camera_matrix_k,
            calibration_status=calibration_status,
        )
        valid_pairs = [p for p in verified_pairs if p.is_valid]
        print(f"  Evaluated {len(verified_pairs)} candidate pairs | {len(valid_pairs)} passed geometric verification.")
        for p in valid_pairs:
            print(f"    Pair ({p.img1_idx}, {p.img2_idx}): {p.num_inliers} inliers ({p.inlier_ratio*100:.1f}%), epipolar RMSE: {p.mean_epipolar_error_px:.2f} px")

        track_stats = track_system.get_statistics()
        print(f"  Constructed {track_stats['total_tracks']} feature tracks (Avg length: {track_stats['average_track_length']} views).")
        print(f"  Track length distribution: {track_stats['track_length_distribution']}")
        print(f"  Track conflicts rejected: {track_stats['conflict_rejections_count']}")

        # Step 3: Initial Reconstruction Pair Selection
        print(f"\n[Step 3/6] Selecting optimal initial pair for reconstruction seeding...")
        init_res = self.init_selector.select_and_initialize(
            verified_pairs,
            camera_matrix_k=camera_matrix_k,
            image_size=image_size,
            track_system=track_system,
            forced_seed_pair=active_forced_seed,
        )

        if not init_res.is_valid:
            print(f"  [ERROR] Reconstruction initialization failed: {init_res.rejection_reason}")
            raise RuntimeError(f"Initialization failed: {init_res.rejection_reason}")

        c1_idx, c2_idx = init_res.pair
        print(f"  Selected Seed Pair: Frame {c1_idx} <-> Frame {c2_idx} ({init_res.pair_names[0]} & {init_res.pair_names[1]})")
        print(f"  Triangulated {init_res.num_initial_points} initial 3D seed points (Mean reproj: {init_res.mean_reprojection_error_px:.2f} px).")

        # Establish Registered Cameras dictionary
        registered_cameras: Dict[int, CameraPose] = {}

        # Anchor Camera 1 at Origin
        R1 = np.array(init_res.camera1_pose["rotation_matrix"], dtype=np.float64)
        t1 = np.array(init_res.camera1_pose["translation_vector"], dtype=np.float64).reshape(3, 1)
        rvec1, _ = cv2.Rodrigues(R1)
        registered_cameras[c1_idx] = CameraPose(
            image_id=c1_idx,
            image_name=init_res.pair_names[0],
            rotation_matrix=R1,
            translation_vector=t1,
            camera_center=np.zeros(3, dtype=np.float64),
            rvec=rvec1,
            tvec=t1,
            num_2d_3d_matches=init_res.num_initial_points,
            num_pnp_inliers=init_res.num_initial_points,
            inlier_ratio=1.0,
            mean_reprojection_error_px=init_res.mean_reprojection_error_px,
            is_registered=True,
        )

        # Camera 2 from Essential Matrix relative pose
        R2 = np.array(init_res.camera2_pose["rotation_matrix"], dtype=np.float64)
        t2 = np.array(init_res.camera2_pose["translation_vector"], dtype=np.float64).reshape(3, 1)
        C2 = np.array(init_res.camera2_pose["camera_center"], dtype=np.float64)
        rvec2, _ = cv2.Rodrigues(R2)
        registered_cameras[c2_idx] = CameraPose(
            image_id=c2_idx,
            image_name=init_res.pair_names[1],
            rotation_matrix=R2,
            translation_vector=t2,
            camera_center=C2,
            rvec=rvec2,
            tvec=t2,
            num_2d_3d_matches=init_res.num_initial_points,
            num_pnp_inliers=init_res.num_initial_points,
            inlier_ratio=1.0,
            mean_reprojection_error_px=init_res.mean_reprojection_error_px,
            is_registered=True,
        )

        # Step 4: Incremental Registration & Triangulation Loop
        print(f"\n[Step 4/6] Iterative camera registration and multi-view triangulation...")
        unregistered_ids = [i for i in range(N) if i not in registered_cameras]

        iteration = 0
        while unregistered_ids:
            iteration += 1
            # Next-Best-View (NBV) Selection
            best_id, vis_pts = self.registrar.select_next_best_view(unregistered_ids, track_system)
            if best_id is None or vis_pts < self.config.min_pnp_inliers:
                print(f"  No remaining candidate has sufficient 2D-3D correspondences (Max observed: {vis_pts} < {self.config.min_pnp_inliers}). Stopping registration loop.")
                break

            img_name = selected_paths[best_id].name
            print(f"  Attempting registration: Frame {best_id:02d} ({img_name}) with {vis_pts} visible 3D points...")

            # PnP Pose Estimation
            pose_cand = self.registrar.register_camera(
                image_id=best_id,
                image_name=img_name,
                camera_matrix_k=camera_matrix_k,
                track_system=track_system,
            )

            registered_cameras[best_id] = pose_cand
            unregistered_ids.remove(best_id)

            if not pose_cand.is_registered:
                print(f"    [REJECTED] Frame {best_id:02d}: {pose_cand.rejection_reason}")
                continue

            print(f"    [REGISTERED] Frame {best_id:02d}: {pose_cand.num_pnp_inliers}/{pose_cand.num_2d_3d_matches} inliers ({pose_cand.inlier_ratio*100:.1f}%), reproj RMSE: {pose_cand.mean_reprojection_error_px:.2f} px")
            print(f"    Camera Center C: [{pose_cand.camera_center[0]:.2f}, {pose_cand.camera_center[1]:.2f}, {pose_cand.camera_center[2]:.2f}]")

            # Triangulate newly visible multi-view tracks
            tri_res = self.triangulator.triangulate_unreconstructed_tracks(
                track_system=track_system,
                registered_cameras=registered_cameras,
                camera_matrix_k=camera_matrix_k,
            )
            print(f"    Triangulated {tri_res.newly_triangulated_count} new 3D points (Mean parallax: {tri_res.mean_parallax_deg:.1f}°, reproj: {tri_res.mean_reprojection_error_px:.2f} px).")

        reg_count = sum(1 for c in registered_cameras.values() if c.is_registered)
        print(f"\nRegistration complete: {reg_count}/{N} cameras registered.")

        # Step 5: Bundle Adjustment
        ba_result: Optional[BundleAdjustmentResult] = None
        if self.config.enable_bundle_adjustment and reg_count >= 2:
            print(f"\n[Step 5/6] Executing Global Bundle Adjustment (Huber loss, sparse TRF solver)...")
            ba_result = self.bundle_adjuster.optimize(
                registered_cameras=registered_cameras,
                track_system=track_system,
                camera_matrix_k=camera_matrix_k,
                fix_camera_id=c1_idx,    # Anchor gauge at initial reference camera
                scale_camera_id=c2_idx,  # Scale anchor: preserve seed baseline length
            )
            print(f"  Reprojection RMSE:         {ba_result.initial_rmse_px:.3f} px -> {ba_result.final_rmse_px:.3f} px ({ba_result.error_reduction_percent:.1f}% reduction)")
            print(f"  Mean Euclidean Error:      {ba_result.initial_mean_px:.3f} px -> {ba_result.final_mean_px:.3f} px ({ba_result.mean_reduction_percent:.1f}% reduction)")
            print(f"  Seed Baseline Invariance:  {ba_result.seed_baseline_before:.6f} -> {ba_result.seed_baseline_after:.6f} (Preserved: {ba_result.seed_baseline_preserved})")
            print(f"  Optimized {ba_result.num_cameras} cameras and {ba_result.num_points} 3D points across {ba_result.num_observations_after} observations.")
            print(f"  Optimization Status:       {ba_result.termination_message} ({ba_result.num_iterations} evals in {ba_result.optimization_time_ms:.1f} ms)")
        else:
            print(f"\n[Step 5/6] Bundle Adjustment skipped by configuration or insufficient cameras.")

        # Step 6: Diagnostics, Metrics, Outlier Analysis & Artifact Export
        print(f"\n[Step 6/6] Computing diagnostics, outlier analysis, and exporting reconstruction artifacts...")
        diagnostics = ReconstructionMetrics.compute_diagnostics(
            total_candidates=N,
            registered_cameras=registered_cameras,
            track_system=track_system,
            camera_matrix_k=camera_matrix_k,
            ba_result=ba_result,
        )

        ReconstructionMetrics.export_artifacts(
            output_dir=self.config.output_dir,
            registered_cameras=registered_cameras,
            track_system=track_system,
            diagnostics=diagnostics,
            camera_matrix_k=camera_matrix_k,
            ba_result=ba_result,
        )

        # Fine-grained Outlier Diagnostics Export
        outlier_diag = OutlierDiagnostics(
            threshold_outlier_px=self.config.max_reprojection_error_px,
            threshold_severe_px=10.0,
        )
        obs_recs, pt_summaries, cam_summaries, outlier_master = outlier_diag.analyze(
            registered_cameras=registered_cameras,
            track_system=track_system,
            camera_matrix_k=camera_matrix_k,
        )
        outlier_diag.export_all(
            output_dir=self.config.output_dir,
            observation_records=obs_recs,
            point_summaries=pt_summaries,
            camera_summaries=cam_summaries,
            master_report=outlier_master,
        )

        # Diagnostic Visualizations
        visualizer = SfMVisualizer(
            output_dir=self.config.output_dir,
            secondary_output_dir=None,
        )
        visualizer.generate_all(
            registered_cameras=registered_cameras,
            track_system=track_system,
            camera_matrix_k=camera_matrix_k,
            master_report=outlier_master,
            camera_summaries=cam_summaries,
            keyframe_dir=selected_paths[0].parent if selected_paths else None,
        )

        elapsed_sec = time.time() - start_time
        pct_1 = outlier_master["error_distribution_percentages"]["pct_below_1px"]
        pct_2 = outlier_master["error_distribution_percentages"]["pct_below_2px"]
        pct_4 = outlier_master["error_distribution_percentages"]["pct_below_4px"]
        pct_gt4 = outlier_master["error_distribution_percentages"]["pct_above_4px"]
        count_gt4 = outlier_master["error_distribution_counts"]["above_4px"]
        count_gt10 = outlier_master["error_distribution_counts"]["above_10px"]

        print(f"  Sparse Point Cloud:    {diagnostics.num_sparse_3d_points} points saved to {self.config.output_dir}/sparse_points.ply")
        print(f"  Camera Poses:          {diagnostics.registered_cameras_count} camera poses saved to {self.config.output_dir}/cameras.json")
        print(f"  Feature Tracks:        Saved to {self.config.output_dir}/feature_tracks.json")
        print(f"  Master Report:         Saved to {self.config.output_dir}/reconstruction_report.json")
        print(f"  Outlier Telemetry:     Saved observation_errors.csv, point_errors.csv, camera_errors.csv, outlier_report.json")
        print(f"  Visualizations:        Saved camera_trajectory.png, reprojection_error_distribution.png, point_cloud_outliers.png")
        print(f"  Mean Reprojection Err: {diagnostics.mean_reprojection_error_px:.3f} px (P90: {diagnostics.p90_reprojection_error_px:.3f} px, P95: {diagnostics.p95_reprojection_error_px:.3f} px)")
        print(f"  Observation Accuracy:  {pct_1:.1f}% < 1px | {pct_2:.1f}% < 2px | {pct_4:.1f}% <= 4px | {pct_gt4:.1f}% > 4px ({count_gt4} obs, {count_gt10} severe > 10px)")
        if outlier_master["high_error_summary"]["total_high_error_cameras"] > 0:
            print(f"  [WARNING] High-error cameras: Camera IDs {outlier_master['high_error_summary']['high_error_cameras_list']}")
        else:
            print(f"  Camera Integrity:      All {diagnostics.registered_cameras_count} registered cameras within nominal reprojection bounds.")
        print(f"  Total Pipeline Time:   {elapsed_sec:.2f} s")
        print(f"==============================================================================\n")

        return registered_cameras, track_system, diagnostics, ba_result
