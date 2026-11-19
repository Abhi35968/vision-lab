"""
Incremental Multi-View Structure from Motion Unit and Integration Test Suite
Validates feature tracks, initial pair selection, PnP registration, triangulation, and Bundle Adjustment.
"""

import unittest
from pathlib import Path
import cv2
import numpy as np

from src.sfm.feature_tracks import FeatureTrackSystem, FeatureTrack, TrackObservation
from src.sfm.camera_registration import CameraRegistrar, CameraPose
from src.sfm.multi_view_triangulator import MultiViewTriangulator
from src.sfm.bundle_adjustment import BundleAdjuster
from src.sfm.initialization import InitialPairSelector
from src.sfm.reconstruction_metrics import ReconstructionMetrics
from src.sfm.multi_view_matcher import VerifiedPairMatch
from src.sfm.outlier_diagnostics import OutlierDiagnostics, ObservationErrorRecord
from src.sfm.visualization import SfMVisualizer
import subprocess
import sys
import tempfile


class TestIncrementalSfM(unittest.TestCase):
    """Rigorous unit and geometric integration tests for Phase 3 SfM."""

    @classmethod
    def setUpClass(cls):
        # Standard synthetic camera intrinsics: 640x480 resolution, f=800
        cls.w, cls.h = 640, 480
        cls.K = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Generate 40 synthetic 3D points in front of cameras: X in [-1.2, 1.2], Y in [-0.8, 0.8], Z in [2.5, 4.5]
        np.random.seed(42)
        cls.pts_3d_gt = np.random.uniform(
            low=[-1.2, -0.8, 2.5],
            high=[1.2, 0.8, 4.5],
            size=(40, 3),
        ).astype(np.float64)

        # Generate 4 ground-truth camera poses along horizontal arc
        cls.cams_gt = []
        for i in range(4):
            # Camera translated along X: 0.0, 0.35, 0.70, 1.05
            C = np.array([0.35 * i, 0.0, 0.0], dtype=np.float64)
            yaw_rad = np.radians(3.0 * i)
            R, _ = cv2.Rodrigues(np.array([0.0, -yaw_rad, 0.0]))
            t = (-R @ C).reshape(3, 1)
            rvec, _ = cv2.Rodrigues(R)
            cls.cams_gt.append({"R": R, "t": t, "C": C, "rvec": rvec})

    def project_points(self, cam_idx: int, pts_3d: np.ndarray) -> np.ndarray:
        """Projects 3D points into camera view cam_idx."""
        cam = self.cams_gt[cam_idx]
        P = self.K @ np.hstack([cam["R"], cam["t"]])
        hom = np.hstack([pts_3d, np.ones((len(pts_3d), 1), dtype=np.float64)])
        proj = (P @ hom.T).T
        return proj[:, :2] / proj[:, 2:3]

    # -------------------------------------------------------------------------
    # Test 1 & 2: Feature Track System Creation, Extension & Duplicates
    # -------------------------------------------------------------------------
    def test_feature_track_creation_and_extension(self):
        """Verifies multi-view track creation across consecutive image matches."""
        tracker = FeatureTrackSystem(min_track_length=2)

        # Match (img0, kp10) <-> (img1, kp25)
        added1 = tracker.add_match(
            img1_id=0, img1_name="frame_0.png", kp1_idx=10, pt1=(100.0, 150.0),
            img2_id=1, img2_name="frame_1.png", kp2_idx=25, pt2=(105.0, 152.0),
        )
        self.assertTrue(added1)
        self.assertEqual(len(tracker.tracks), 1)

        # Extend track: (img1, kp25) <-> (img2, kp30)
        added2 = tracker.add_match(
            img1_id=1, img1_name="frame_1.png", kp1_idx=25, pt1=(105.0, 152.0),
            img2_id=2, img2_name="frame_2.png", kp2_idx=30, pt2=(110.0, 153.0),
        )
        self.assertTrue(added2)
        self.assertEqual(len(tracker.tracks), 1)

        track = list(tracker.tracks.values())[0]
        self.assertEqual(track.length, 3)
        self.assertTrue(track.has_image(0))
        self.assertTrue(track.has_image(1))
        self.assertTrue(track.has_image(2))

        # Test duplicate match: re-adding the same correspondence returns True without adding duplicate observation
        added_dup = tracker.add_match(
            img1_id=0, img1_name="frame_0.png", kp1_idx=10, pt1=(100.0, 150.0),
            img2_id=1, img2_name="frame_1.png", kp2_idx=25, pt2=(105.0, 152.0),
        )
        self.assertTrue(added_dup)
        self.assertEqual(track.length, 3)
        self.assertEqual(tracker.duplicate_matches_count, 1)

    # -------------------------------------------------------------------------
    # Test 3 & 4: Duplicate Observation Prevention & Conflict Handling
    # -------------------------------------------------------------------------
    def test_track_conflict_handling(self):
        """Ensures that conflicting observations in the same image are rejected."""
        tracker = FeatureTrackSystem(min_track_length=2)

        # Track A: (img0, kp1) <-> (img1, kp1)
        tracker.add_match(
            img1_id=0, img1_name="frame_0.png", kp1_idx=1, pt1=(50.0, 50.0),
            img2_id=1, img2_name="frame_1.png", kp2_idx=1, pt2=(55.0, 50.0),
        )
        # Track B: (img0, kp2) <-> (img2, kp2)
        tracker.add_match(
            img1_id=0, img1_name="frame_0.png", kp1_idx=2, pt1=(80.0, 80.0),
            img2_id=2, img2_name="frame_2.png", kp2_idx=2, pt2=(85.0, 80.0),
        )
        self.assertEqual(len(tracker.tracks), 2)

        # Attempt to merge Track A and Track B via (img1, kp1) <-> (img2, kp2)
        # This would cause img0 to have TWO different keypoints (kp1 and kp2) in the merged track!
        conflict_res = tracker.add_match(
            img1_id=1, img1_name="frame_1.png", kp1_idx=1, pt1=(55.0, 50.0),
            img2_id=2, img2_name="frame_2.png", kp2_idx=2, pt2=(85.0, 80.0),
        )
        self.assertFalse(conflict_res, "Conflicting track merge must be rejected.")
        self.assertEqual(tracker.conflict_rejections_count, 1)
        self.assertEqual(len(tracker.tracks), 2)

    # -------------------------------------------------------------------------
    # Test 5: Initial Pair Selection
    # -------------------------------------------------------------------------
    def test_initial_pair_selection(self):
        """Tests multi-criteria scoring to choose a wide-baseline pair over a zero-baseline pair."""
        selector = InitialPairSelector(min_inliers=10, min_parallax_deg=1.5)
        tracker = FeatureTrackSystem()

        # Synthetic points projected into view 0, view 1 (good baseline) and view 0_near (near-zero baseline)
        pts0 = self.project_points(0, self.pts_3d_gt)
        pts1 = self.project_points(1, self.pts_3d_gt) # parallax ~ 5.5 deg

        # Zero-baseline camera (translation = 0.001)
        R_zero = np.eye(3)
        t_zero = np.array([[0.0001], [0.0], [0.0]])
        P_zero = self.K @ np.hstack([R_zero, t_zero])
        hom = np.hstack([self.pts_3d_gt, np.ones((len(self.pts_3d_gt), 1))])
        proj_zero = (P_zero @ hom.T).T
        pts_zero = proj_zero[:, :2] / proj_zero[:, 2:3]

        # Candidate 1: Pair (0, 99) with near-zero baseline
        pair_zero = VerifiedPairMatch(
            img1_idx=0, img2_idx=99, img1_name="frame_0.png", img2_name="frame_zero.png",
            raw_matches_count=40, num_inliers=40, inlier_ratio=1.0,
            mean_epipolar_error_px=0.1, median_epipolar_error_px=0.1,
            inlier_pts1=pts0, inlier_pts2=pts_zero,
            inlier_kp_indices1=list(range(40)), inlier_kp_indices2=list(range(40)),
            essential_matrix=np.array([[0, 0, 0], [0, 0, -0.001], [0, 0.001, 0]]),
            fundamental_matrix=None, is_valid=True,
        )

        # Candidate 2: Pair (0, 1) with genuine camera motion
        R_rel = self.cams_gt[1]["R"]
        t_rel = self.cams_gt[1]["t"]
        t_skew = np.array([
            [0, -t_rel[2, 0], t_rel[1, 0]],
            [t_rel[2, 0], 0, -t_rel[0, 0]],
            [-t_rel[1, 0], t_rel[0, 0], 0],
        ])
        E_rel = t_skew @ R_rel

        pair_good = VerifiedPairMatch(
            img1_idx=0, img2_idx=1, img1_name="frame_0.png", img2_name="frame_1.png",
            raw_matches_count=40, num_inliers=40, inlier_ratio=1.0,
            mean_epipolar_error_px=0.2, median_epipolar_error_px=0.2,
            inlier_pts1=pts0, inlier_pts2=pts1,
            inlier_kp_indices1=list(range(40)), inlier_kp_indices2=list(range(40)),
            essential_matrix=E_rel,
            fundamental_matrix=None, is_valid=True,
        )

        # Populate tracker with pair_good
        for k in range(40):
            tracker.add_match(0, "frame_0.png", k, tuple(pts0[k]), 1, "frame_1.png", k, tuple(pts1[k]))

        res = selector.select_and_initialize(
            [pair_zero, pair_good],
            camera_matrix_k=self.K,
            image_size=(self.w, self.h),
            track_system=tracker,
        )

        self.assertTrue(res.is_valid)
        self.assertEqual(res.pair, (0, 1), "Selector must reject near-zero baseline pair and select pair (0, 1).")
        self.assertGreater(res.num_initial_points, 25)

    # -------------------------------------------------------------------------
    # Test 6: PnP Camera Pose Recovery
    # -------------------------------------------------------------------------
    def test_pnp_camera_pose_recovery(self):
        """Verifies that CameraRegistrar accurately recovers camera pose using PnP RANSAC."""
        registrar = CameraRegistrar(min_pnp_inliers=10, max_reprojection_error_px=3.0)
        tracker = FeatureTrackSystem()

        # Project 3D points to camera 2
        pts2 = self.project_points(2, self.pts_3d_gt)

        # Set up 30 triangulated tracks observed by camera 2
        for k in range(30):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(2, "frame_2.png", k, tuple(pts2[k])))
            tracker.tracks[k] = t

        reg_pose = registrar.register_camera(
            image_id=2,
            image_name="frame_2.png",
            camera_matrix_k=self.K,
            track_system=tracker,
        )

        self.assertTrue(reg_pose.is_registered)
        self.assertEqual(reg_pose.num_pnp_inliers, 30)
        self.assertLess(reg_pose.mean_reprojection_error_px, 0.5)

        # Verify recovered optical center C matches ground truth: C_gt = [0.70, 0, 0]
        C_gt = self.cams_gt[2]["C"]
        center_dist = np.linalg.norm(reg_pose.camera_center - C_gt)
        self.assertLess(center_dist, 0.05, f"Camera center error {center_dist:.4f} exceeds 0.05 m tolerance.")

    # -------------------------------------------------------------------------
    # Test 7: Incremental Triangulation
    # -------------------------------------------------------------------------
    def test_incremental_triangulation_synthetic(self):
        """Verifies Linear DLT N-view triangulation across 3 registered cameras."""
        triangulator = MultiViewTriangulator(max_reprojection_error_px=3.0, min_parallax_angle_deg=1.5)
        tracker = FeatureTrackSystem()

        pts0 = self.project_points(0, self.pts_3d_gt)
        pts1 = self.project_points(1, self.pts_3d_gt)
        pts2 = self.project_points(2, self.pts_3d_gt)

        # Build 20 un-triangulated tracks observed across cameras 0, 1, 2
        for k in range(20):
            t = FeatureTrack(track_id=k, is_triangulated=False)
            t.add_observation(TrackObservation(0, "frame_0.png", k, tuple(pts0[k])))
            t.add_observation(TrackObservation(1, "frame_1.png", k, tuple(pts1[k])))
            t.add_observation(TrackObservation(2, "frame_2.png", k, tuple(pts2[k])))
            tracker.tracks[k] = t

        # Setup registered cameras dict
        registered_cams = {}
        for c_id in [0, 1, 2]:
            cam = self.cams_gt[c_id]
            registered_cams[c_id] = CameraPose(
                image_id=c_id,
                image_name=f"frame_{c_id}.png",
                rotation_matrix=cam["R"],
                translation_vector=cam["t"],
                camera_center=cam["C"],
                rvec=cam["rvec"],
                tvec=cam["t"],
                num_2d_3d_matches=20,
                num_pnp_inliers=20,
                inlier_ratio=1.0,
                mean_reprojection_error_px=0.1,
                is_registered=True,
            )

        res = triangulator.triangulate_unreconstructed_tracks(
            track_system=tracker,
            registered_cameras=registered_cams,
            camera_matrix_k=self.K,
        )

        self.assertEqual(res.newly_triangulated_count, 20)
        self.assertLess(res.mean_reprojection_error_px, 0.5)

        # Verify 3D reconstruction error against ground truth
        errors = [
            np.linalg.norm(tracker.tracks[k].point_3d - self.pts_3d_gt[k])
            for k in range(20)
        ]
        mean_3d_error = float(np.mean(errors))
        self.assertLess(mean_3d_error, 0.05, f"Mean 3D point error {mean_3d_error:.4f} exceeds 0.05 m tolerance.")

    # -------------------------------------------------------------------------
    # Test 8: Reprojection Error Calculation & Diagnostics
    # -------------------------------------------------------------------------
    def test_reprojection_error_diagnostics(self):
        """Verifies percentile calculation and depth bounds in ReconstructionMetrics."""
        tracker = FeatureTrackSystem()
        registered_cams = {}

        pts0 = self.project_points(0, self.pts_3d_gt[:10])
        pts1 = self.project_points(1, self.pts_3d_gt[:10])

        for k in range(10):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "frame_0.png", k, tuple(pts0[k])))
            t.add_observation(TrackObservation(1, "frame_1.png", k, tuple(pts1[k])))
            tracker.tracks[k] = t

        for c_id in [0, 1]:
            cam = self.cams_gt[c_id]
            registered_cams[c_id] = CameraPose(
                image_id=c_id, image_name=f"frame_{c_id}.png",
                rotation_matrix=cam["R"], translation_vector=cam["t"],
                camera_center=cam["C"], rvec=cam["rvec"], tvec=cam["t"],
                num_2d_3d_matches=10, num_pnp_inliers=10, inlier_ratio=1.0,
                mean_reprojection_error_px=0.05, is_registered=True,
            )

        diag = ReconstructionMetrics.compute_diagnostics(
            total_candidates=2,
            registered_cameras=registered_cams,
            track_system=tracker,
            camera_matrix_k=self.K,
        )

        self.assertEqual(diag.registered_cameras_count, 2)
        self.assertEqual(diag.num_sparse_3d_points, 10)
        self.assertEqual(diag.num_total_observations, 20)
        self.assertLess(diag.mean_reprojection_error_px, 0.1)
        self.assertGreater(diag.min_point_depth, 2.0)

    # -------------------------------------------------------------------------
    # Test 9: Bundle Adjustment Optimization
    # -------------------------------------------------------------------------
    def test_bundle_adjustment_synthetic(self):
        """Verifies that sparse BA reduces reprojection RMSE while keeping Camera 0 fixed."""
        ba = BundleAdjuster(max_nfev=60)
        tracker = FeatureTrackSystem()

        # Observations for 25 points across 3 cameras
        pts0 = self.project_points(0, self.pts_3d_gt[:25])
        pts1 = self.project_points(1, self.pts_3d_gt[:25])
        pts2 = self.project_points(2, self.pts_3d_gt[:25])

        # Introduce controlled perturbation to 3D points (+0.04 m)
        np.random.seed(123)
        noisy_pts = self.pts_3d_gt[:25] + np.random.normal(0, 0.04, size=(25, 3))

        for k in range(25):
            t = FeatureTrack(track_id=k, point_3d=noisy_pts[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "frame_0.png", k, tuple(pts0[k])))
            t.add_observation(TrackObservation(1, "frame_1.png", k, tuple(pts1[k])))
            t.add_observation(TrackObservation(2, "frame_2.png", k, tuple(pts2[k])))
            tracker.tracks[k] = t

        registered_cams = {}
        for c_id in [0, 1, 2]:
            cam = self.cams_gt[c_id]
            # Perturb camera 1 and 2 poses slightly
            r_pert = cam["rvec"].copy() if c_id == 0 else cam["rvec"] + np.array([[0.01], [0.01], [0.0]])
            t_pert = cam["t"].copy() if c_id == 0 else cam["t"] + np.array([[0.02], [0.0], [0.0]])
            R_p, _ = cv2.Rodrigues(r_pert)
            registered_cams[c_id] = CameraPose(
                image_id=c_id, image_name=f"frame_{c_id}.png",
                rotation_matrix=R_p, translation_vector=t_pert,
                camera_center=(-R_p.T @ t_pert).ravel(),
                rvec=r_pert, tvec=t_pert,
                num_2d_3d_matches=25, num_pnp_inliers=25, inlier_ratio=1.0,
                mean_reprojection_error_px=3.0, is_registered=True,
            )

        # Camera 0 anchor pose before optimization
        c0_center_before = registered_cams[0].camera_center.copy()

        ba_res = ba.optimize(
            registered_cameras=registered_cams,
            track_system=tracker,
            camera_matrix_k=self.K,
            fix_camera_id=0,
        )

        self.assertTrue(ba_res.is_success)
        self.assertLess(ba_res.final_rmse_px, ba_res.initial_rmse_px)
        self.assertGreater(ba_res.error_reduction_percent, 30.0)

        # Verify Camera 0 remained strictly fixed (anchors the world gauge)
        c0_center_after = registered_cams[0].camera_center
        np.testing.assert_allclose(c0_center_before, c0_center_after, atol=1e-6)

    # -------------------------------------------------------------------------
    # Test 10: Camera Registration Failure Handling
    # -------------------------------------------------------------------------
    def test_camera_registration_failure_handling(self):
        """Verifies graceful rejection when correspondences are insufficient or outlier-dominated."""
        registrar = CameraRegistrar(min_pnp_inliers=15)
        tracker = FeatureTrackSystem()

        # Only 5 correspondences (below threshold 15)
        for k in range(5):
            t = FeatureTrack(track_id=k, point_3d=np.array([0.0, 0.0, 3.0]), is_triangulated=True)
            t.add_observation(TrackObservation(3, "frame_3.png", k, (100.0, 100.0)))
            tracker.tracks[k] = t

        pose = registrar.register_camera(
            image_id=3,
            image_name="frame_3.png",
            camera_matrix_k=self.K,
            track_system=tracker,
        )

        self.assertFalse(pose.is_registered)
        self.assertIn("Insufficient 2D-3D correspondences", pose.rejection_reason)

    # -------------------------------------------------------------------------
    # Test 11: End-to-End Synthetic Incremental Reconstruction
    # -------------------------------------------------------------------------
    def test_end_to_end_synthetic_incremental_sfm(self):
        """End-to-end integration test verifying full multi-view reconstruction on synthetic scene."""
        N_cams = 4
        N_pts = 35

        # 1. Project points into all 4 views with controlled pixel noise
        np.random.seed(99)
        pts_3d = self.pts_3d_gt[:N_pts].copy()
        view_pts = []
        for c in range(N_cams):
            proj = self.project_points(c, pts_3d)
            # Add small Gaussian noise (sigma = 0.2 px)
            proj_noisy = proj + np.random.normal(0, 0.2, size=proj.shape)
            view_pts.append(proj_noisy)

        # 2. Build pairwise verified matches for window W = 2
        tracker = FeatureTrackSystem(min_track_length=2)
        verified_pairs = []

        for i in range(N_cams):
            for j in range(i + 1, min(N_cams, i + 3)):
                p1 = view_pts[i]
                p2 = view_pts[j]

                # Compute relative ground-truth E
                R_i = self.cams_gt[i]["R"]
                t_i = self.cams_gt[i]["t"]
                R_j = self.cams_gt[j]["R"]
                t_j = self.cams_gt[j]["t"]

                R_rel = R_j @ R_i.T
                t_rel = t_j - R_rel @ t_i
                t_skew = np.array([
                    [0, -t_rel[2, 0], t_rel[1, 0]],
                    [t_rel[2, 0], 0, -t_rel[0, 0]],
                    [-t_rel[1, 0], t_rel[0, 0], 0],
                ])
                E_rel = t_skew @ R_rel

                pair_match = VerifiedPairMatch(
                    img1_idx=i, img2_idx=j,
                    img1_name=f"frame_{i}.png", img2_name=f"frame_{j}.png",
                    raw_matches_count=N_pts, num_inliers=N_pts, inlier_ratio=1.0,
                    mean_epipolar_error_px=0.2, median_epipolar_error_px=0.2,
                    inlier_pts1=p1, inlier_pts2=p2,
                    inlier_kp_indices1=list(range(N_pts)), inlier_kp_indices2=list(range(N_pts)),
                    essential_matrix=E_rel, fundamental_matrix=None, is_valid=True,
                )
                verified_pairs.append(pair_match)

                for k in range(N_pts):
                    tracker.add_match(
                        i, f"frame_{i}.png", k, tuple(p1[k]),
                        j, f"frame_{j}.png", k, tuple(p2[k]),
                    )

        # 3. Initialization: Seed from Pair (0, 1)
        selector = InitialPairSelector(min_inliers=15, min_parallax_deg=1.0)
        init_res = selector.select_and_initialize(
            verified_pairs, self.K, (self.w, self.h), tracker
        )
        self.assertTrue(init_res.is_valid)

        registered_cams = {}
        c1_idx, c2_idx = init_res.pair

        R1 = np.array(init_res.camera1_pose["rotation_matrix"])
        t1 = np.array(init_res.camera1_pose["translation_vector"]).reshape(3, 1)
        rvec1, _ = cv2.Rodrigues(R1)
        registered_cams[c1_idx] = CameraPose(
            image_id=c1_idx, image_name=f"frame_{c1_idx}.png", rotation_matrix=R1, translation_vector=t1,
            camera_center=np.zeros(3), rvec=rvec1, tvec=t1,
            num_2d_3d_matches=init_res.num_initial_points, num_pnp_inliers=init_res.num_initial_points,
            inlier_ratio=1.0, mean_reprojection_error_px=init_res.mean_reprojection_error_px, is_registered=True,
        )

        R2 = np.array(init_res.camera2_pose["rotation_matrix"])
        t2 = np.array(init_res.camera2_pose["translation_vector"]).reshape(3, 1)
        C2 = np.array(init_res.camera2_pose["camera_center"])
        rvec2, _ = cv2.Rodrigues(R2)
        registered_cams[c2_idx] = CameraPose(
            image_id=c2_idx, image_name=f"frame_{c2_idx}.png", rotation_matrix=R2, translation_vector=t2,
            camera_center=C2, rvec=rvec2, tvec=t2,
            num_2d_3d_matches=init_res.num_initial_points, num_pnp_inliers=init_res.num_initial_points,
            inlier_ratio=1.0, mean_reprojection_error_px=init_res.mean_reprojection_error_px, is_registered=True,
        )

        # 4. Incremental Registration: Register remaining cameras
        registrar = CameraRegistrar(min_pnp_inliers=10, max_reprojection_error_px=4.0)
        triangulator = MultiViewTriangulator(max_reprojection_error_px=4.0, min_parallax_angle_deg=1.0)

        unregistered_ids = [c for c in range(N_cams) if c not in registered_cams]
        for c_id in unregistered_ids:
            # Register camera via PnP
            pose_c = registrar.register_camera(c_id, f"frame_{c_id}.png", self.K, tracker)
            self.assertTrue(pose_c.is_registered, f"Camera {c_id} registration failed: {pose_c.rejection_reason}")
            registered_cams[c_id] = pose_c

            # Triangulate newly observed tracks
            tri_res = triangulator.triangulate_unreconstructed_tracks(tracker, registered_cams, self.K)

        self.assertEqual(len([c for c in registered_cams.values() if c.is_registered]), 4)

        # 5. Global Bundle Adjustment
        ba = BundleAdjuster(max_nfev=50)
        ba_res = ba.optimize(registered_cams, tracker, self.K, fix_camera_id=0)
        self.assertTrue(ba_res.is_success)

        # 6. Verify Diagnostics and Reconstruction Metrics
        diagnostics = ReconstructionMetrics.compute_diagnostics(
            total_candidates=4,
            registered_cameras=registered_cams,
            track_system=tracker,
            camera_matrix_k=self.K,
            ba_result=ba_res,
        )

        self.assertEqual(diagnostics.registered_cameras_count, 4)
        self.assertEqual(diagnostics.registration_success_rate_percent, 100.0)
        self.assertGreater(diagnostics.num_sparse_3d_points, 25)
        self.assertLess(diagnostics.mean_reprojection_error_px, 1.5)
        # Verify gauge freedom: Camera 0 remained at [0, 0, 0]
        np.testing.assert_allclose(registered_cams[0].camera_center, np.zeros(3), atol=1e-6)

    # =========================================================================
    # Phase 3.1 Mathematical Audit & Regression Tests
    # =========================================================================

    def test_baseline_preservation_after_bundle_adjustment(self):
        """Audit 1: Verifies that seed camera pair baseline magnitude is strictly preserved after BA."""
        ba = BundleAdjuster(max_nfev=50)
        tracker = FeatureTrackSystem()

        pts0 = self.project_points(0, self.pts_3d_gt[:20])
        pts1 = self.project_points(1, self.pts_3d_gt[:20])
        pts2 = self.project_points(2, self.pts_3d_gt[:20])

        for k in range(20):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "f0.png", k, tuple(pts0[k])))
            t.add_observation(TrackObservation(1, "f1.png", k, tuple(pts1[k])))
            t.add_observation(TrackObservation(2, "f2.png", k, tuple(pts2[k])))
            tracker.tracks[k] = t

        registered_cams = {}
        for c_id in [0, 1, 2]:
            cam = self.cams_gt[c_id]
            registered_cams[c_id] = CameraPose(
                image_id=c_id, image_name=f"f{c_id}.png",
                rotation_matrix=cam["R"].copy(), translation_vector=cam["t"].copy(),
                camera_center=cam["C"].copy(), rvec=cam["rvec"].copy(), tvec=cam["t"].copy(),
                num_2d_3d_matches=20, num_pnp_inliers=20, inlier_ratio=1.0,
                mean_reprojection_error_px=0.1, is_registered=True,
            )

        # Baseline distance between camera 0 and camera 1 before BA
        d_before = np.linalg.norm(registered_cams[1].camera_center - registered_cams[0].camera_center)
        self.assertAlmostEqual(d_before, 0.35, places=5)

        ba_res = ba.optimize(
            registered_cameras=registered_cams,
            track_system=tracker,
            camera_matrix_k=self.K,
            fix_camera_id=0,
            scale_camera_id=1,
        )

        d_after = np.linalg.norm(registered_cams[1].camera_center - registered_cams[0].camera_center)
        self.assertTrue(ba_res.seed_baseline_preserved)
        self.assertTrue(ba_res.scale_constraint_enforced)
        # Difference must be identically zero to floating point precision
        self.assertAlmostEqual(d_before, d_after, places=6)

    def test_scale_metadata_and_arbitrary_unit_depth_reporting(self):
        """Audit 2: Verifies explicit scale metadata and arbitrary-unit depth labeling."""
        tracker = FeatureTrackSystem()
        registered_cams = {}

        pts0 = self.project_points(0, self.pts_3d_gt[:10])
        pts1 = self.project_points(1, self.pts_3d_gt[:10])

        for k in range(10):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "f0.png", k, tuple(pts0[k])))
            t.add_observation(TrackObservation(1, "f1.png", k, tuple(pts1[k])))
            tracker.tracks[k] = t

        for c_id in [0, 1]:
            cam = self.cams_gt[c_id]
            registered_cams[c_id] = CameraPose(
                image_id=c_id, image_name=f"f{c_id}.png",
                rotation_matrix=cam["R"], translation_vector=cam["t"],
                camera_center=cam["C"], rvec=cam["rvec"], tvec=cam["t"],
                num_2d_3d_matches=10, num_pnp_inliers=10, inlier_ratio=1.0,
                mean_reprojection_error_px=0.05, is_registered=True,
            )

        diag = ReconstructionMetrics.compute_diagnostics(
            total_candidates=2,
            registered_cameras=registered_cams,
            track_system=tracker,
            camera_matrix_k=self.K,
        )
        data = diag.to_dict()

        # Check scale metadata
        self.assertIn("scale_metadata", data)
        scale_meta = data["scale_metadata"]
        self.assertEqual(scale_meta["scale_status"], "unscaled_arbitrary_units")
        self.assertEqual(scale_meta["distance_unit"], "arbitrary_units")
        self.assertFalse(scale_meta["is_metric"])

        # Check depth bounds unit
        self.assertIn("scene_depth_bounds", data)
        self.assertEqual(data["scene_depth_bounds"]["unit"], "arbitrary_units")
        self.assertIn("Do not interpret as physical meters", data["scene_depth_bounds"]["note"])

    def test_consistent_before_and_after_reprojection_metrics(self):
        """Audit 3: Verifies identical observation sets and mathematical consistency (Mean <= RMSE)."""
        ba = BundleAdjuster(max_nfev=40)
        tracker = FeatureTrackSystem()

        pts0 = self.project_points(0, self.pts_3d_gt[:15])
        pts1 = self.project_points(1, self.pts_3d_gt[:15])

        for k in range(15):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "f0.png", k, tuple(pts0[k])))
            t.add_observation(TrackObservation(1, "f1.png", k, tuple(pts1[k])))
            tracker.tracks[k] = t

        registered_cams = {}
        for c_id in [0, 1]:
            cam = self.cams_gt[c_id]
            registered_cams[c_id] = CameraPose(
                image_id=c_id, image_name=f"f{c_id}.png",
                rotation_matrix=cam["R"].copy(), translation_vector=cam["t"].copy(),
                camera_center=cam["C"].copy(), rvec=cam["rvec"].copy(), tvec=cam["t"].copy(),
                num_2d_3d_matches=15, num_pnp_inliers=15, inlier_ratio=1.0,
                mean_reprojection_error_px=0.1, is_registered=True,
            )

        ba_res = ba.optimize(registered_cams, tracker, self.K, fix_camera_id=0, scale_camera_id=1)

        # 1. Observation counts before and after must be identical
        self.assertEqual(ba_res.num_observations_before, 30)
        self.assertEqual(ba_res.num_observations_after, 30)
        self.assertEqual(ba_res.num_observations_before, ba_res.num_observations_after)

        # 2. Mathematical inequality: Mean <= RMSE (Jensen's inequality)
        self.assertLessEqual(ba_res.initial_mean_px, ba_res.initial_rmse_px + 1e-7)
        self.assertLessEqual(ba_res.final_mean_px, ba_res.final_rmse_px + 1e-7)

        # 3. All metrics must be finite and positive
        for val in [ba_res.initial_rmse_px, ba_res.final_rmse_px, ba_res.initial_mean_px,
                    ba_res.final_mean_px, ba_res.initial_median_px, ba_res.final_median_px]:
            self.assertTrue(np.isfinite(val))
            self.assertGreaterEqual(val, 0.0)

    def test_camera_center_and_projection_consistency(self):
        """Audit 4: Verifies projection matrix P = K[R|t] and optical center C = -R^T t."""
        for c_id in range(4):
            cam = self.cams_gt[c_id]
            R = cam["R"]
            t = cam["t"]
            C = cam["C"]

            # Optical center satisfies R * C + t = 0
            res_c = R @ C + t.ravel()
            np.testing.assert_allclose(res_c, np.zeros(3), atol=1e-9)

            # CameraPose class center calculation
            pose = CameraPose(
                image_id=c_id, image_name=f"f{c_id}.png",
                rotation_matrix=R, translation_vector=t,
                camera_center=(-R.T @ t).ravel(),
                rvec=cam["rvec"], tvec=t,
                num_2d_3d_matches=10, num_pnp_inliers=10, inlier_ratio=1.0,
                mean_reprojection_error_px=0.1, is_registered=True,
            )
            np.testing.assert_allclose(pose.camera_center, C, atol=1e-9)

            # Projection matrix P = K [R | t] matches explicit projection
            P = pose.projection_matrix(self.K)
            X = self.pts_3d_gt[0]
            hom_X = np.append(X, 1.0)
            proj_hom = P @ hom_X
            px_P = proj_hom[:2] / proj_hom[2]

            X_cam = R @ X + t.ravel()
            px_explicit = np.array([
                self.K[0, 0] * (X_cam[0] / X_cam[2]) + self.K[0, 2],
                self.K[1, 1] * (X_cam[1] / X_cam[2]) + self.K[1, 2],
            ])
            np.testing.assert_allclose(px_P, px_explicit, atol=1e-7)

    def test_track_to_point_observation_consistency(self):
        """Audit 5: Verifies feature track observation integrity and 1-to-1 mapping."""
        tracker = FeatureTrackSystem()

        # Build 10 valid tracks
        for k in range(10):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "f0.png", k, (100.0 + k, 150.0)))
            t.add_observation(TrackObservation(1, "f1.png", k, (105.0 + k, 152.0)))
            tracker.tracks[k] = t

        integrity = tracker.validate_track_integrity()
        self.assertTrue(integrity["all_tracks_valid"])
        self.assertEqual(integrity["valid_tracks_count"], 10)
        self.assertEqual(integrity["total_observations_count"], 20)
        self.assertEqual(len(integrity["anomalies_detected"]), 0)

        # Corrupt one track with non-finite coordinate
        tracker.tracks[0].observations[0].coords = (np.nan, 150.0)
        integrity_corrupt = tracker.validate_track_integrity()
        self.assertFalse(integrity_corrupt["all_tracks_valid"])
        self.assertGreater(len(integrity_corrupt["anomalies_detected"]), 0)

    def test_bundle_adjustment_failure_handling(self):
        """Audit 6: Verifies graceful rejection when observations are insufficient (< 10)."""
        ba = BundleAdjuster()
        tracker = FeatureTrackSystem()

        # Only 2 points (4 observations total, below threshold 10)
        for k in range(2):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "f0.png", k, (100.0, 150.0)))
            t.add_observation(TrackObservation(1, "f1.png", k, (105.0, 152.0)))
            tracker.tracks[k] = t

        registered_cams = {
            0: CameraPose(0, "f0.png", np.eye(3), np.zeros((3, 1)), np.zeros(3), np.zeros(3), np.zeros((3, 1)), 2, 2, 1.0, 0.1, True),
            1: CameraPose(1, "f1.png", np.eye(3), np.ones((3, 1)), -np.ones(3), np.zeros(3), np.ones((3, 1)), 2, 2, 1.0, 0.1, True),
        }

        res = ba.optimize(registered_cams, tracker, self.K, fix_camera_id=0, scale_camera_id=1)
        self.assertFalse(res.is_success)
        self.assertEqual(res.num_iterations, 0)
        self.assertIn("Insufficient valid observations", res.termination_message)

    # =========================================================================
    # Phase 3.2 Outlier Diagnostics & Validation Tests
    # =========================================================================

    def test_outlier_diagnostics_calculation(self):
        """Phase 3.2 Test 1: Verifies 2D Euclidean reprojection errors and outlier classification flags."""
        tracker = FeatureTrackSystem()
        registered_cams = {}

        # 2 cameras: Camera 0 at origin, Camera 1 at [0.35, 0, 0]
        for c_id in [0, 1]:
            cam = self.cams_gt[c_id]
            registered_cams[c_id] = CameraPose(
                image_id=c_id, image_name=f"f{c_id}.png",
                rotation_matrix=cam["R"].copy(), translation_vector=cam["t"].copy(),
                camera_center=cam["C"].copy(), rvec=cam["rvec"].copy(), tvec=cam["t"].copy(),
                num_2d_3d_matches=5, num_pnp_inliers=5, inlier_ratio=1.0,
                mean_reprojection_error_px=0.1, is_registered=True,
            )

        pts0 = self.project_points(0, self.pts_3d_gt[:5])
        pts1 = self.project_points(1, self.pts_3d_gt[:5])

        for k in range(5):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            u0, v0 = pts0[k]
            u1, v1 = pts1[k]

            if k == 0:
                # Sub-pixel: 0.5 px displacement
                t.add_observation(TrackObservation(0, "f0.png", k, (u0 + 0.3, v0 + 0.4)))
                t.add_observation(TrackObservation(1, "f1.png", k, (u1, v1)))
            elif k == 1:
                # Moderate error: 3-4-5 triangle -> 5.0 px displacement (> 4px outlier gate, < 10px)
                t.add_observation(TrackObservation(0, "f0.png", k, (u0 + 3.0, v0 + 4.0)))
                t.add_observation(TrackObservation(1, "f1.png", k, (u1, v1)))
            elif k == 2:
                # Severe outlier: 9-12-15 triangle -> 15.0 px displacement (> 10px severe gate)
                t.add_observation(TrackObservation(0, "f0.png", k, (u0 + 9.0, v0 + 12.0)))
                t.add_observation(TrackObservation(1, "f1.png", k, (u1, v1)))
            else:
                # Inlier (< 1 px)
                t.add_observation(TrackObservation(0, "f0.png", k, (u0, v0)))
                t.add_observation(TrackObservation(1, "f1.png", k, (u1, v1)))

            tracker.tracks[k] = t

        diag = OutlierDiagnostics(threshold_outlier_px=4.0, threshold_severe_px=10.0)
        recs, pt_sums, cam_sums, master = diag.analyze(registered_cams, tracker, self.K)

        self.assertEqual(len(recs), 10)  # 5 tracks * 2 observations
        self.assertEqual(master["total_observations"], 10)

        # Check track 1 (error ~ 5.0 px)
        rec_track1_cam0 = [r for r in recs if r.track_id == 1 and r.camera_id == 0][0]
        self.assertAlmostEqual(rec_track1_cam0.reprojection_error_px, 5.0, delta=0.1)
        self.assertTrue(rec_track1_cam0.is_outlier_4px)
        self.assertFalse(rec_track1_cam0.is_outlier_10px)

        # Check track 2 (error ~ 15.0 px)
        rec_track2_cam0 = [r for r in recs if r.track_id == 2 and r.camera_id == 0][0]
        self.assertAlmostEqual(rec_track2_cam0.reprojection_error_px, 15.0, delta=0.1)
        self.assertTrue(rec_track2_cam0.is_outlier_4px)
        self.assertTrue(rec_track2_cam0.is_outlier_10px)

        # Check counts
        self.assertEqual(master["error_distribution_counts"]["above_4px"], 2)
        self.assertEqual(master["error_distribution_counts"]["above_10px"], 1)

    def test_camera_and_point_error_aggregation(self):
        """Phase 3.2 Test 2: Verifies aggregation statistics match individual observation errors."""
        tracker = FeatureTrackSystem()
        registered_cams = {}

        for c_id in [0, 1]:
            cam = self.cams_gt[c_id]
            registered_cams[c_id] = CameraPose(
                image_id=c_id, image_name=f"f{c_id}.png",
                rotation_matrix=cam["R"].copy(), translation_vector=cam["t"].copy(),
                camera_center=cam["C"].copy(), rvec=cam["rvec"].copy(), tvec=cam["t"].copy(),
                num_2d_3d_matches=10, num_pnp_inliers=10, inlier_ratio=1.0,
                mean_reprojection_error_px=0.1, is_registered=True,
            )

        pts0 = self.project_points(0, self.pts_3d_gt[:10])
        pts1 = self.project_points(1, self.pts_3d_gt[:10])

        for k in range(10):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            u0, v0 = pts0[k]
            u1, v1 = pts1[k]
            # Vary error across points
            t.add_observation(TrackObservation(0, "f0.png", k, (u0 + k * 0.5, v0)))
            t.add_observation(TrackObservation(1, "f1.png", k, (u1, v1 + k * 0.3)))
            tracker.tracks[k] = t

        diag = OutlierDiagnostics(threshold_outlier_px=4.0, threshold_severe_px=10.0)
        recs, pt_sums, cam_sums, master = diag.analyze(registered_cams, tracker, self.K)

        # Check point summaries
        for p in pt_sums:
            tid = p["track_id"]
            p_recs = [r.reprojection_error_px for r in recs if r.track_id == tid]
            self.assertEqual(p["num_observations"], len(p_recs))
            self.assertAlmostEqual(p["mean_error_px"], np.mean(p_recs), places=2)
            self.assertAlmostEqual(p["max_error_px"], np.max(p_recs), places=2)
            self.assertAlmostEqual(p["min_error_px"], np.min(p_recs), places=2)

        # Check camera summaries
        for c in cam_sums:
            cid = c["camera_id"]
            c_recs = [r.reprojection_error_px for r in recs if r.camera_id == cid]
            self.assertEqual(c["num_observations"], len(c_recs))
            self.assertAlmostEqual(c["mean_error_px"], np.mean(c_recs), places=2)
            self.assertAlmostEqual(c["max_error_px"], np.max(c_recs), places=2)

    def test_error_distribution_monotonicity(self):
        """Phase 3.2 Test 3: Verifies cumulative error distribution monotonicity."""
        tracker = FeatureTrackSystem()
        registered_cams = {}

        for c_id in [0, 1]:
            cam = self.cams_gt[c_id]
            registered_cams[c_id] = CameraPose(
                image_id=c_id, image_name=f"f{c_id}.png",
                rotation_matrix=cam["R"].copy(), translation_vector=cam["t"].copy(),
                camera_center=cam["C"].copy(), rvec=cam["rvec"].copy(), tvec=cam["t"].copy(),
                num_2d_3d_matches=10, num_pnp_inliers=10, inlier_ratio=1.0,
                mean_reprojection_error_px=0.1, is_registered=True,
            )

        pts0 = self.project_points(0, self.pts_3d_gt[:15])
        pts1 = self.project_points(1, self.pts_3d_gt[:15])

        for k in range(15):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            u0, v0 = pts0[k]
            u1, v1 = pts1[k]
            t.add_observation(TrackObservation(0, "f0.png", k, (u0 + k * 0.4, v0)))
            t.add_observation(TrackObservation(1, "f1.png", k, (u1, v1 + k * 0.2)))
            tracker.tracks[k] = t

        diag = OutlierDiagnostics(threshold_outlier_px=4.0, threshold_severe_px=10.0)
        recs, pt_sums, cam_sums, master = diag.analyze(registered_cams, tracker, self.K)

        pcts = master["error_distribution_percentages"]
        # Monotonicity check
        self.assertLessEqual(pcts["pct_below_1px"], pcts["pct_below_2px"])
        self.assertLessEqual(pcts["pct_below_2px"], pcts["pct_below_4px"])
        self.assertLessEqual(pcts["pct_below_4px"], pcts["pct_below_10px"])

        # Complementarity check: pct_below_4px + pct_above_4px == 100%
        self.assertAlmostEqual(pcts["pct_below_4px"] + pcts["pct_above_4px"], 100.0, places=1)

    def test_outlier_telemetry_export(self):
        """Phase 3.2 Test 4: Verifies CSV and JSON export of outlier telemetry."""
        tracker = FeatureTrackSystem()
        registered_cams = {
            0: CameraPose(0, "f0.png", np.eye(3), np.zeros((3, 1)), np.zeros(3), np.zeros(3), np.zeros((3, 1)), 5, 5, 1.0, 0.1, True),
            1: CameraPose(1, "f1.png", np.eye(3), np.array([[0.35], [0], [0]]), np.array([-0.35, 0, 0]), np.zeros(3), np.array([[0.35], [0], [0]]), 5, 5, 1.0, 0.1, True),
        }

        pts0 = self.project_points(0, self.pts_3d_gt[:5])
        pts1 = self.project_points(1, self.pts_3d_gt[:5])

        for k in range(5):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "f0.png", k, tuple(pts0[k])))
            t.add_observation(TrackObservation(1, "f1.png", k, tuple(pts1[k])))
            tracker.tracks[k] = t

        diag = OutlierDiagnostics()
        recs, pt_sums, cam_sums, master = diag.analyze(registered_cams, tracker, self.K)

        with tempfile.TemporaryDirectory() as tmpdir:
            diag.export_all(tmpdir, recs, pt_sums, cam_sums, master)

            obs_csv = Path(tmpdir) / "observation_errors.csv"
            pt_csv = Path(tmpdir) / "point_errors.csv"
            cam_csv = Path(tmpdir) / "camera_errors.csv"
            out_json = Path(tmpdir) / "outlier_report.json"

            self.assertTrue(obs_csv.exists() and obs_csv.stat().st_size > 0)
            self.assertTrue(pt_csv.exists() and pt_csv.stat().st_size > 0)
            self.assertTrue(cam_csv.exists() and cam_csv.stat().st_size > 0)
            self.assertTrue(out_json.exists() and out_json.stat().st_size > 0)

            # Check header format
            with open(obs_csv, "r") as f:
                header = f.readline().strip()
                self.assertIn("reprojection_error_px", header)
                self.assertIn("is_outlier_4px", header)

    def test_visualization_generation(self):
        """Phase 3.2 Test 5: Verifies that SfMVisualizer creates valid diagnostic plot images."""
        tracker = FeatureTrackSystem()
        registered_cams = {
            0: CameraPose(0, "f0.png", np.eye(3), np.zeros((3, 1)), np.zeros(3), np.zeros(3), np.zeros((3, 1)), 5, 5, 1.0, 0.1, True),
            1: CameraPose(1, "f1.png", np.eye(3), np.array([[0.35], [0], [0]]), np.array([-0.35, 0, 0]), np.zeros(3), np.array([[0.35], [0], [0]]), 5, 5, 1.0, 0.1, True),
        }

        pts0 = self.project_points(0, self.pts_3d_gt[:5])
        pts1 = self.project_points(1, self.pts_3d_gt[:5])

        for k in range(5):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            t.add_observation(TrackObservation(0, "f0.png", k, tuple(pts0[k])))
            t.add_observation(TrackObservation(1, "f1.png", k, tuple(pts1[k])))
            tracker.tracks[k] = t

        diag = OutlierDiagnostics()
        recs, pt_sums, cam_sums, master = diag.analyze(registered_cams, tracker, self.K)

        with tempfile.TemporaryDirectory() as tmpdir:
            vis = SfMVisualizer(output_dir=tmpdir, secondary_output_dir=None)
            vis_dict = vis.generate_all(registered_cams, tracker, self.K, master, cam_sums)

            self.assertIn("camera_trajectory", vis_dict)
            self.assertIn("reprojection_error_distribution", vis_dict)
            self.assertIn("point_cloud_outliers", vis_dict)

            for key, paths in vis_dict.items():
                for p in paths:
                    self.assertTrue(p.exists(), f"Visualization {key} was not generated: {p}")
                    self.assertGreater(p.stat().st_size, 1000, f"Visualization {key} is empty: {p}")

    def test_cli_missing_video_validation(self):
        """Phase 3.2 Test 6: Verifies main.py CLI video validation gracefully exits on missing file."""
        cmd = [sys.executable, "main.py", "--video", "non_existent_raw_video.mp4"]
        res = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(res.returncode, 1)
        self.assertIn("[ERROR] Video file not found", res.stdout)

    # =========================================================================
    # Phase 3.3 Consistency Audit & Regression Tests
    # =========================================================================

    def test_phase31_vs_phase32_intrinsics_consistency(self):
        """Phase 3.3 Audit 1: Verifies resolution adaptation in camera intrinsics estimation."""
        from src.sfm.incremental_sfm import IncrementalSfM

        # 1. Unadapted placeholder (1080p fallback) -> Phase 3.1 behavior
        K_1080, size_1080, status_1080 = IncrementalSfM.load_camera_intrinsics(
            "config/camera_intrinsics.json", actual_image_size=None
        )
        self.assertEqual(size_1080, (1920, 1080))
        self.assertAlmostEqual(K_1080[0, 2], 960.0)  # cx
        self.assertAlmostEqual(K_1080[1, 2], 540.0)  # cy
        self.assertAlmostEqual(K_1080[0, 0], 2304.0) # fx

        # 2. Keyframe-adapted resolution (720p keyframes) -> Phase 3.2 behavior
        K_720, size_720, status_720 = IncrementalSfM.load_camera_intrinsics(
            "config/camera_intrinsics.json", actual_image_size=(1280, 720)
        )
        self.assertEqual(size_720, (1280, 720))
        self.assertAlmostEqual(K_720[0, 2], 640.0)   # cx correctly centered
        self.assertAlmostEqual(K_720[1, 2], 360.0)   # cy correctly centered
        self.assertAlmostEqual(K_720[0, 0], 1536.0)  # fx

    def test_outlier_threshold_equality_and_complementarity(self):
        """Phase 3.3 Audit 2: Verifies strict partitioning of observations across error gates."""
        tracker = FeatureTrackSystem()
        registered_cams = {
            0: CameraPose(0, "f0.png", np.eye(3), np.zeros((3, 1)), np.zeros(3), np.zeros(3), np.zeros((3, 1)), 6, 6, 1.0, 0.1, True),
            1: CameraPose(1, "f1.png", np.eye(3), np.array([[0.35], [0], [0]]), np.array([-0.35, 0, 0]), np.zeros(3), np.array([[0.35], [0], [0]]), 6, 6, 1.0, 0.1, True),
        }

        # Build 6 tracks with exact error steps: 0.5, 1.0, 2.0, 4.0, 8.0, 12.0 px
        pts0 = self.project_points(0, self.pts_3d_gt[:6])
        pts1 = self.project_points(1, self.pts_3d_gt[:6])
        offsets = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

        for k in range(6):
            t = FeatureTrack(track_id=k, point_3d=self.pts_3d_gt[k].copy(), is_triangulated=True)
            u0, v0 = pts0[k]
            u1, v1 = pts1[k]
            t.add_observation(TrackObservation(0, "f0.png", k, (u0 + offsets[k], v0)))
            t.add_observation(TrackObservation(1, "f1.png", k, (u1, v1)))
            tracker.tracks[k] = t

        diag = OutlierDiagnostics(threshold_outlier_px=4.0, threshold_severe_px=10.0)
        recs, pt_sums, cam_sums, master = diag.analyze(registered_cams, tracker, self.K)

        counts = master["error_distribution_counts"]
        pcts = master["error_distribution_percentages"]
        total_obs = master["total_observations"]

        self.assertEqual(total_obs, 12)
        # Exact partitions
        self.assertEqual(counts["below_4px"] + counts["above_4px"], total_obs)
        self.assertEqual(counts["below_10px"] + counts["above_10px"], total_obs)
        self.assertAlmostEqual(pcts["pct_below_4px"] + pcts["pct_above_4px"], 100.0, places=2)
        self.assertAlmostEqual(pcts["pct_below_10px"] + pcts["pct_above_10px"], 100.0, places=2)

    def test_cli_unsupported_and_empty_video_validation(self):
        """Phase 3.3 Audit 3: Verifies CLI rejects unsupported formats and empty 0-byte video files."""
        # 1. Unsupported extension (.txt)
        res_ext = subprocess.run(
            [sys.executable, "main.py", "--video", "requirements.txt"],
            capture_output=True, text=True,
        )
        self.assertEqual(res_ext.returncode, 1)
        self.assertIn("[ERROR] Unsupported video format", res_ext.stdout)

        # 2. Empty 0-byte file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as empty_vid:
            empty_path = empty_vid.name

        try:
            res_empty = subprocess.run(
                [sys.executable, "main.py", "--video", empty_path],
                capture_output=True, text=True,
            )
            self.assertEqual(res_empty.returncode, 1)
            self.assertIn("[ERROR] Video file is empty (0 bytes)", res_empty.stdout)
        finally:
            Path(empty_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
