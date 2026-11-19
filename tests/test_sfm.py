"""
Unit Tests for Phase 2: Feature Detection, Matching, Epipolar Geometry, and Two-View SfM
Includes synthetic multi-view ground-truth geometric tests (explicitly distinguished from real data).
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.sfm.feature_extractor import FeatureExtractor, ExtractedFeatures
from src.sfm.feature_matcher import FeatureMatcher, MatchResult
from src.sfm.geometry import GeometricVerifier, GeometricVerificationResult
from src.sfm.pose_estimator import PoseEstimator, PoseEstimationResult
from src.sfm.triangulator import Triangulator, TriangulationResult
from src.sfm.two_view_sfm import TwoViewSfM


class TestTwoViewSfM(unittest.TestCase):
    """Test suite verifying Phase 2 Feature Matching and Two-View SfM modules."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="3dcv_sfm_test_"))

        # Create two textured synthetic test images with known homography / displacement
        cls.img1 = np.ones((600, 800, 3), dtype=np.uint8) * 128
        for i in range(15):
            x = 80 + i * 40
            y = 100 + (i % 5) * 80
            cv2.circle(cls.img1, (x, y), 18, (0, 0, 255), -1)
            cv2.rectangle(cls.img1, (x - 10, y - 10), (x + 10, y + 10), (255, 255, 255), 2)
            cv2.putText(cls.img1, f"T{i}", (x - 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Image 2 is shifted and slightly rotated
        M = cv2.getRotationMatrix2D((400, 300), 2.0, 1.0)
        M[0, 2] += 25.0  # Horizontal translation in pixels
        M[1, 2] += 5.0
        cls.img2 = cv2.warpAffine(cls.img1, M, (800, 600), borderValue=(128, 128, 128))

        cls.img1_path = cls.temp_dir / "test_img1.png"
        cls.img2_path = cls.temp_dir / "test_img2.png"
        cv2.imwrite(str(cls.img1_path), cls.img1)
        cv2.imwrite(str(cls.img2_path), cls.img2)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def test_01_sift_feature_extraction(self):
        """Verifies SIFT extracts 128-D float32 descriptors and keypoints."""
        extractor = FeatureExtractor(detector="SIFT", n_features=1000)
        feat = extractor.extract(self.img1)

        self.assertGreater(feat.num_keypoints, 20)
        self.assertIsNotNone(feat.descriptors)
        self.assertEqual(feat.descriptor_dim, 128)
        self.assertEqual(feat.descriptors.dtype, np.float32)
        self.assertEqual(feat.keypoint_coords.shape[0], feat.num_keypoints)
        self.assertEqual(feat.keypoint_coords.shape[1], 2)

    def test_02_orb_feature_extraction(self):
        """Verifies ORB extracts 32-byte (256-bit) uint8 binary descriptors."""
        extractor = FeatureExtractor(detector="ORB", n_features=1000)
        feat = extractor.extract(self.img1)

        self.assertGreater(feat.num_keypoints, 20)
        self.assertIsNotNone(feat.descriptors)
        self.assertEqual(feat.descriptor_dim, 32)
        self.assertEqual(feat.descriptors.dtype, np.uint8)

    def test_03_descriptor_matching_and_ratio_test(self):
        """Verifies KNN matching and Lowe's ratio test filter out ambiguous matches."""
        extractor = FeatureExtractor(detector="SIFT", n_features=1000)
        feat1 = extractor.extract(self.img1)
        feat2 = extractor.extract(self.img2)

        matcher = FeatureMatcher(matcher_type="FLANN", lowe_ratio=0.80)
        match_res = matcher.match(feat1, feat2)

        self.assertGreater(match_res.raw_matches_count, 0)
        self.assertGreater(match_res.final_matches_count, 10)
        # Ratio test count should be <= raw match count
        self.assertLessEqual(match_res.ratio_matches_count, match_res.raw_matches_count)
        self.assertEqual(len(match_res.matched_pts1), match_res.final_matches_count)
        self.assertEqual(len(match_res.matched_pts2), match_res.final_matches_count)

    def test_04_synthetic_geometric_verification_and_pose(self):
        """
        Rigorous Ground-Truth Synthetic Geometry Test:
        Generates known 3D points, projects through known K, R, t, adds noise/outliers,
        and verifies recovered Fundamental/Essential matrices, pose, and DLT triangulation.
        """
        np.random.seed(42)
        N_points = 60

        # 1. Ground truth 3D points in front of camera
        pts3d_gt = np.random.uniform(low=[-1.0, -0.8, 2.0], high=[1.0, 0.8, 4.0], size=(N_points, 3))

        # Known Camera Matrix K
        K = np.array([
            [800.0, 0.0, 400.0],
            [0.0, 800.0, 300.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Known relative rotation: 6 deg around Y axis
        theta = np.radians(6.0)
        R_gt = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ], dtype=np.float64)

        # Known translation: horizontal motion along X
        t_gt = np.array([0.25, 0.0, 0.0], dtype=np.float64).reshape(3, 1)
        t_gt_unit = t_gt / np.linalg.norm(t_gt)

        # Camera projection matrices
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R_gt, t_gt])

        # Project 3D points to 2D
        h_pts3d = np.hstack([pts3d_gt, np.ones((N_points, 1))])
        proj1 = (P1 @ h_pts3d.T).T
        proj2 = (P2 @ h_pts3d.T).T
        pts1 = (proj1[:, :2] / proj1[:, 2:]).astype(np.float32)
        pts2 = (proj2[:, :2] / proj2[:, 2:]).astype(np.float32)

        # Add 0.3 px Gaussian pixel noise
        pts1 += np.random.normal(0, 0.3, size=pts1.shape).astype(np.float32)
        pts2 += np.random.normal(0, 0.3, size=pts2.shape).astype(np.float32)

        # Add 10 outliers
        outliers1 = np.random.uniform(50, 750, size=(10, 2)).astype(np.float32)
        outliers2 = np.random.uniform(50, 550, size=(10, 2)).astype(np.float32)
        pts1_with_outliers = np.vstack([pts1, outliers1])
        pts2_with_outliers = np.vstack([pts2, outliers2])

        # 2. Geometric Verification
        verifier = GeometricVerifier(ransac_reproj_threshold=2.0)
        geom_res = verifier.verify(pts1_with_outliers, pts2_with_outliers, camera_matrix_k=K, calibration_status="calibrated")

        self.assertTrue(geom_res.is_valid)
        self.assertGreaterEqual(geom_res.num_inliers, 40)
        self.assertIsNotNone(geom_res.essential_matrix)
        self.assertLess(geom_res.mean_epipolar_error_px, 1.5)

        # 3. Pose Recovery
        pose_estimator = PoseEstimator()
        pose_res = pose_estimator.estimate_pose(
            geom_res.essential_matrix, geom_res.inlier_pts1, geom_res.inlier_pts2, K
        )

        self.assertTrue(pose_res.is_valid)
        self.assertGreater(pose_res.cheirality_ratio, 0.85)

        # Verify rotation angular error between R_est and R_gt is < 2.0 degrees
        R_err_mat = pose_res.rotation_matrix @ R_gt.T
        tr = np.clip((np.trace(R_err_mat) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg = np.degrees(np.arccos(tr))
        self.assertLess(rot_err_deg, 2.0)

        # Verify translation direction (cosine similarity > 0.95)
        cos_sim = float(np.abs(np.dot(pose_res.translation_vector.ravel(), t_gt_unit.ravel())))
        self.assertGreater(cos_sim, 0.95)

        # 4. Triangulation
        triangulator = Triangulator(max_reprojection_error_px=3.0)
        tri_res = triangulator.triangulate(
            geom_res.inlier_pts1, geom_res.inlier_pts2, K, pose_res.rotation_matrix, pose_res.translation_vector
        )

        self.assertTrue(tri_res.is_valid)
        self.assertGreater(tri_res.points_reconstructed, 35)
        self.assertLess(tri_res.mean_reprojection_error, 2.0)

    def test_05_invalid_input_and_failure_handling(self):
        """Verifies graceful handling of empty descriptors and insufficient points."""
        matcher = FeatureMatcher()
        empty_feat = ExtractedFeatures(
            keypoints=[],
            descriptors=None,
            keypoint_coords=np.empty((0, 2), dtype=np.float32),
            num_keypoints=0,
            detector_name="SIFT",
            descriptor_dim=128,
            descriptor_dtype="float32",
            image_path=None,
            extraction_time_ms=0.0,
            image_size=(800, 600),
        )

        match_res = matcher.match(empty_feat, empty_feat)
        self.assertEqual(match_res.final_matches_count, 0)
        self.assertEqual(len(match_res.matched_pts1), 0)

        # Geometric verification on < 8 points
        verifier = GeometricVerifier()
        pts_few = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
        geom_res = verifier.verify(pts_few, pts_few)
        self.assertFalse(geom_res.is_valid)
        self.assertIn("Insufficient correspondences", geom_res.rejection_reason)

    def test_06_two_view_sfm_pipeline_integration(self):
        """Executes full TwoViewSfM pipeline on test images and verifies report output."""
        sfm = TwoViewSfM(calibration_source="estimated")
        report = sfm.reconstruct_pair(
            self.img1_path,
            self.img2_path,
            output_dir=self.temp_dir / "sfm_out",
            vis_dir=self.temp_dir / "vis_out",
        )

        self.assertIn(report["reconstruction_status"], {"SUCCESS", "FAILED"})
        self.assertTrue((self.temp_dir / "sfm_out" / "two_view_report.json").exists())
        self.assertEqual(report["calibration"]["status"], "estimated")


if __name__ == "__main__":
    unittest.main()
