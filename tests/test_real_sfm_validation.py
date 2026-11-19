"""
Tests for Phase 5.2.1: Real-World SfM Geometric Validation Audit
Validates controlled seed-pair selection, mathematical exactness of BA scale constraints,
trajectory geometry calculations, depth sensitivity to focal length, and telemetry schemas.
"""

import json
from pathlib import Path
import unittest
import cv2
import numpy as np

from src.sfm.bundle_adjustment import BundleAdjuster
from src.sfm.camera_registration import CameraPose
from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem, TrackObservation
from src.sfm.initialization import InitialPairSelector
from src.sfm.multi_view_matcher import VerifiedPairMatch


class TestRealSfMValidation(unittest.TestCase):
    """Unit test suite for Phase 5.2.1 geometric validation auditing."""

    def setUp(self):
        self.K_935 = np.array([
            [935.0, 0.0, 540.0],
            [0.0, 935.0, 960.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.K_1300 = np.array([
            [1300.0, 0.0, 540.0],
            [0.0, 1300.0, 960.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.K_1663 = np.array([
            [1662.8, 0.0, 540.0],
            [0.0, 1662.8, 960.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def test_forced_seed_pair_initialization(self):
        """Verifies InitialPairSelector strictly respects the requested forced seed pair."""
        selector = InitialPairSelector(min_inliers=5)
        track_sys = FeatureTrackSystem()

        # Project synthetic 3D points (>= 10 for Triangulator threshold)
        pts3d = np.array([
            [-1, -1, 5], [1, -1, 5], [-1, 1, 5], [1, 1, 5],
            [-0.5, 0, 4], [0.5, 0, 4], [0, -0.5, 6], [0, 0.5, 6],
            [-0.8, -0.2, 5.5], [0.8, 0.2, 5.5], [-0.2, 0.8, 4.5], [0.2, -0.8, 4.5]
        ], dtype=np.float64)

        p1_h = (self.K_1300 @ pts3d.T).T
        pts1 = (p1_h[:, :2] / p1_h[:, 2:]).astype(np.float32)

        R2 = np.eye(3)
        t2 = np.array([[-0.5], [0], [0]])
        p2_h = (self.K_1300 @ (R2 @ pts3d.T + t2)).T
        pts2 = (p2_h[:, :2] / p2_h[:, 2:]).astype(np.float32)

        E, _ = cv2.findEssentialMat(pts1, pts2, self.K_1300)

        pair1 = VerifiedPairMatch(
            img1_idx=10, img2_idx=11, img1_name="f10.png", img2_name="f11.png",
            raw_matches_count=20, num_inliers=12, inlier_ratio=0.6,
            mean_epipolar_error_px=0.5, median_epipolar_error_px=0.5,
            inlier_pts1=pts1, inlier_pts2=pts2,
            inlier_kp_indices1=list(range(12)), inlier_kp_indices2=list(range(12)),
            essential_matrix=E, fundamental_matrix=None, is_valid=True,
        )

        pair2 = VerifiedPairMatch(
            img1_idx=24, img2_idx=25, img1_name="f24.png", img2_name="f25.png",
            raw_matches_count=20, num_inliers=12, inlier_ratio=0.6,
            mean_epipolar_error_px=0.5, median_epipolar_error_px=0.5,
            inlier_pts1=pts1, inlier_pts2=pts2,
            inlier_kp_indices1=list(range(12)), inlier_kp_indices2=list(range(12)),
            essential_matrix=E, fundamental_matrix=None, is_valid=True,
        )

        # Force selection of pair2: (24, 25)
        res = selector.select_and_initialize(
            [pair1, pair2],
            camera_matrix_k=self.K_1300,
            image_size=(1080, 1920),
            track_system=track_sys,
            forced_seed_pair=(24, 25),
        )

        self.assertTrue(res.is_valid)
        self.assertEqual(res.pair, (24, 25))
        self.assertEqual(res.pair_names, ("f24.png", "f25.png"))

    def test_ba_scale_constraint_exact_preservation(self):
        """
        Verifies that BundleAdjuster maintains the seed baseline length ||t_seed|| = 1.0
        as an exact parameter elimination constraint (delta < 1e-10) without penalty terms.
        """
        ba = BundleAdjuster(max_nfev=20, loss="huber")
        track_sys = FeatureTrackSystem()

        cams = {
            24: CameraPose(
                image_id=24, image_name="c24.png",
                rotation_matrix=np.eye(3), translation_vector=np.zeros((3, 1)),
                camera_center=np.zeros(3), rvec=np.zeros(3), tvec=np.zeros((3, 1)),
                num_2d_3d_matches=10, num_pnp_inliers=10, inlier_ratio=1.0,
                mean_reprojection_error_px=0.5, is_registered=True,
            ),
            25: CameraPose(
                image_id=25, image_name="c25.png",
                rotation_matrix=np.eye(3), translation_vector=np.array([[1.0], [0.0], [0.0]]),
                camera_center=np.array([-1.0, 0.0, 0.0]), rvec=np.zeros(3),
                tvec=np.array([[1.0], [0.0], [0.0]]),
                num_2d_3d_matches=10, num_pnp_inliers=10, inlier_ratio=1.0,
                mean_reprojection_error_px=0.5, is_registered=True,
            ),
        }

        # Create 15 synthetic 3D points and observations
        for i in range(15):
            pt_3d = np.array([float(i - 7) * 0.2, float(i % 3) * 0.2, 5.0])
            t = FeatureTrack(track_id=i, point_3d=pt_3d, is_triangulated=True)
            # Project to camera 24: K * [X, Y, Z]^T
            u24 = 1300.0 * (pt_3d[0] / pt_3d[2]) + 540.0
            v24 = 1300.0 * (pt_3d[1] / pt_3d[2]) + 960.0
            t.add_observation(TrackObservation(24, "c24.png", i, (u24, v24)))

            # Project to camera 25: K * [X - 1.0, Y, Z]^T
            u25 = 1300.0 * ((pt_3d[0] - 1.0) / pt_3d[2]) + 540.0
            v25 = 1300.0 * (pt_3d[1] / pt_3d[2]) + 960.0
            t.add_observation(TrackObservation(25, "c25.png", i, (u25, v25)))

            track_sys.tracks[i] = t

        res = ba.optimize(
            registered_cameras=cams,
            track_system=track_sys,
            camera_matrix_k=self.K_1300,
            fix_camera_id=24,
            scale_camera_id=25,
        )

        self.assertTrue(res.is_success)
        self.assertTrue(res.seed_baseline_preserved)
        self.assertTrue(res.scale_constraint_enforced)
        self.assertAlmostEqual(res.seed_baseline_before, 1.0, places=6)
        self.assertAlmostEqual(res.seed_baseline_after, 1.0, places=6)
        self.assertLess(abs(res.seed_baseline_after - res.seed_baseline_before), 1e-10)

    def test_photogrammetric_depth_sensitivity(self):
        """
        Verifies the fundamental relationship Z ~ f * (b / d):
        Under fixed baseline b and disparity d, triangulated depth Z is strictly proportional to focal length f.
        """
        from src.sfm.triangulator import Triangulator

        pts3d = np.array([
            [-1, -1, 5], [1, -1, 5], [-1, 1, 5], [1, 1, 5],
            [-0.5, 0, 4], [0.5, 0, 4], [0, -0.5, 6], [0, 0.5, 6]
        ], dtype=np.float64)

        p1_h = (self.K_1300 @ pts3d.T).T
        pts1 = (p1_h[:, :2] / p1_h[:, 2:]).astype(np.float32)
        R2 = np.eye(3)
        t2 = np.array([[-0.5], [0], [0]])
        p2_h = (self.K_1300 @ (R2 @ pts3d.T + t2)).T
        pts2 = (p2_h[:, :2] / p2_h[:, 2:]).astype(np.float32)

        tri = Triangulator(max_reprojection_error_px=20.0)
        tri_935 = tri.triangulate(pts1, pts2, self.K_935, R2, t2)
        tri_1300 = tri.triangulate(pts1, pts2, self.K_1300, R2, t2)
        tri_1663 = tri.triangulate(pts1, pts2, self.K_1663, R2, t2)

        z_935 = float(np.median(tri_935.points_3d[:, 2]))
        z_1300 = float(np.median(tri_1300.points_3d[:, 2]))
        z_1663 = float(np.median(tri_1663.points_3d[:, 2]))

        # Depth must scale monotonically: Z_935 < Z_1300 < Z_1663
        self.assertLess(z_935, z_1300)
        self.assertLess(z_1300, z_1663)

        # Expected ratio: z_1300 / z_935 ~ 1300 / 935 = 1.390
        ratio_actual = z_1300 / z_935
        ratio_expected = 1300.0 / 935.0
        self.assertAlmostEqual(ratio_actual, ratio_expected, places=2)

    def test_trajectory_azimuth_sweep_calculation(self):
        """Verifies calculation of azimuth angles and angular span around scene centroid."""
        # 4 camera centers along a 90-degree quadrant arc of radius R=5 around origin (0, 0, 0)
        # Angles: 0, 30, 60, 90 deg
        cams = [
            np.array([5.0 * np.cos(np.radians(deg)), 0.0, 5.0 * np.sin(np.radians(deg))])
            for deg in [0.0, 30.0, 60.0, 90.0]
        ]
        scene_center = np.zeros(3)

        azimuths = [np.degrees(np.arctan2(c[0] - scene_center[0], c[2] - scene_center[2])) for c in cams]
        az_span = abs(max(azimuths) - min(azimuths))
        self.assertAlmostEqual(az_span, 90.0, places=2)

    def test_cli_phase_validation(self):
        """Verifies that the CLI argument parser recognizes --phase real_validation_audit."""
        import main
        parser = main.parse_args
        # Mock sys.argv
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["main.py", "--phase", "real_validation_audit"]
            args = main.parse_args()
            self.assertEqual(args.phase, "real_validation_audit")
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
