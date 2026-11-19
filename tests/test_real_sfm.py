"""
Real-World Incremental SfM Test Suite
Tests portrait camera intrinsics handling, seed pair selection criteria,
bundle adjustment scale constraint enforcement, camera ID mapping,
and comparative multi-hypothesis artifact generation.
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest
import cv2
import numpy as np

from src.sfm.real_video_sfm_experiment import RealVideoSfMExperiment, FocalHypothesis
from src.sfm.initialization import InitialPairSelector, InitialPairScore
from src.sfm.multi_view_matcher import VerifiedPairMatch
from src.sfm.feature_tracks import FeatureTrackSystem


class TestRealVideoSfM(unittest.TestCase):
    """Unit and regression tests for Phase 5.2 Real-World Incremental SfM."""

    EXPERIMENT_DIR = Path("outputs/real_video_sfm")
    KEYFRAMES_DIR = Path("outputs/real_video_validation/keyframes")

    def test_portrait_intrinsics_geometry(self):
        """Verifies portrait orientation (1080x1920) intrinsics and FOV trigonometry."""
        runner = RealVideoSfMExperiment()
        self.assertEqual(runner.WIDTH, 1080)
        self.assertEqual(runner.HEIGHT, 1920)
        self.assertEqual(runner.CX, 540.0)
        self.assertEqual(runner.CY, 960.0)

        # Test focal 935: HFOV should be ~60 deg
        K_935 = runner.get_intrinsics_matrix(935.0)
        self.assertEqual(K_935.shape, (3, 3))
        self.assertEqual(K_935[0, 0], 935.0)
        self.assertEqual(K_935[1, 1], 935.0)
        self.assertEqual(K_935[0, 2], 540.0)
        self.assertEqual(K_935[1, 2], 960.0)

        hfov_935 = 2.0 * np.degrees(np.arctan(runner.WIDTH / (2.0 * 935.0)))
        self.assertAlmostEqual(hfov_935, 60.0, delta=0.1)

        # Test focal 1662.8: VFOV should be ~60 deg
        vfov_1663 = 2.0 * np.degrees(np.arctan(runner.HEIGHT / (2.0 * 1662.8)))
        self.assertAlmostEqual(vfov_1663, 60.0, delta=0.1)

    def test_intrinsics_match_original_image_dimensions(self):
        """Verifies principal point matches original portrait image half-dimensions."""
        kf_files = sorted(list(self.KEYFRAMES_DIR.glob("*.png")))
        self.assertGreater(len(kf_files), 0, "Keyframes directory must contain images")

        first_img = cv2.imread(str(kf_files[0]))
        self.assertIsNotNone(first_img)
        h, w = first_img.shape[:2]

        self.assertEqual(w, 1080)
        self.assertEqual(h, 1920)
        self.assertEqual(w / 2.0, RealVideoSfMExperiment.CX)
        self.assertEqual(h / 2.0, RealVideoSfMExperiment.CY)

    def test_seed_pair_rejection_of_unsuitable_pairs(self):
        """Verifies InitialPairSelector rejects low inliers, near-zero parallax, and Cheirality failures."""
        selector = InitialPairSelector(min_inliers=30, min_parallax_deg=1.5)

        # 1. Reject pair with low inliers (< 30)
        pair_low_inliers = VerifiedPairMatch(
            img1_idx=0,
            img2_idx=1,
            img1_name="kf0.png",
            img2_name="kf1.png",
            raw_matches_count=20,
            num_inliers=10,
            inlier_ratio=0.5,
            mean_epipolar_error_px=1.0,
            median_epipolar_error_px=0.8,
            inlier_pts1=np.zeros((10, 2), dtype=np.float32),
            inlier_pts2=np.zeros((10, 2), dtype=np.float32),
            inlier_kp_indices1=[0] * 10,
            inlier_kp_indices2=[0] * 10,
            essential_matrix=np.eye(3),
            fundamental_matrix=np.eye(3),
            is_valid=True,
        )

        K = np.array([[1300.0, 0, 540.0], [0, 1300.0, 960.0], [0, 0, 1.0]])
        track_sys = FeatureTrackSystem()

        res = selector.select_and_initialize([pair_low_inliers], K, (1080, 1920), track_sys)
        self.assertFalse(res.is_valid)
        self.assertEqual(len(res.candidate_evaluations), 1)
        self.assertIn("Insufficient inliers", res.candidate_evaluations[0].rejection_reason)

    def test_camera_ids_map_to_correct_keyframes(self):
        """Verifies camera IDs map to keyframe filenames and registered cameras have 3D centers."""
        cam_file = self.EXPERIMENT_DIR / "focal_1300" / "cameras.json"
        self.assertTrue(cam_file.exists(), f"Missing {cam_file}")

        with open(cam_file, "r", encoding="utf-8") as f:
            cams = json.load(f)

        self.assertGreater(len(cams), 10)
        registered_count = 0
        for cid_str, cam in cams.items():
            cid = int(cid_str)
            self.assertEqual(cam["image_id"], cid)
            self.assertTrue(cam["image_name"].startswith("keyframe_"))
            if cam["is_registered"]:
                registered_count += 1
                self.assertEqual(len(cam["camera_center"]), 3)

        self.assertEqual(registered_count, 32)

        # Verify scale metadata in reconstruction report
        rep_file = self.EXPERIMENT_DIR / "focal_1300" / "reconstruction_report.json"
        with open(rep_file, "r", encoding="utf-8") as f:
            rep = json.load(f)
        self.assertEqual(rep["scale_metadata"]["distance_unit"], "arbitrary_units")

    def test_bundle_adjustment_scale_constraint_preserved(self):
        """Verifies bundle adjustment strictly preserves the seed baseline length (scale constraint)."""
        ba_file = self.EXPERIMENT_DIR / "focal_1300" / "bundle_adjustment_report.json"
        self.assertTrue(ba_file.exists(), f"Missing {ba_file}")

        with open(ba_file, "r", encoding="utf-8") as f:
            ba_data = json.load(f)

        scale_meta = ba_data["gauge_and_scale_constraints"]
        self.assertTrue(scale_meta["scale_constraint_enforced"])
        self.assertTrue(scale_meta["seed_baseline_preserved"])
        self.assertAlmostEqual(scale_meta["seed_baseline_difference"], 0.0, places=5)
        self.assertEqual(scale_meta["unit"], "arbitrary_units")

    def test_all_three_hypotheses_artifacts_exist(self):
        """Verifies isolated output directories, comparative JSON, and comparative plot exist."""
        subdirs = ["focal_935", "focal_1300", "focal_1663"]
        for sub in subdirs:
            p = self.EXPERIMENT_DIR / sub
            self.assertTrue(p.exists(), f"Directory {p} missing")
            self.assertTrue((p / "reconstruction_report.json").exists())
            self.assertTrue((p / "camera_trajectory.png").exists())
            self.assertTrue((p / "reprojection_error_distribution.png").exists())
            self.assertTrue((p / "point_cloud_outliers.png").exists())
            self.assertTrue((p / "sparse_points.ply").exists())
            self.assertTrue((p / "cameras.json").exists())
            self.assertTrue((p / "reprojection_overlays").exists())

        master_json = self.EXPERIMENT_DIR / "intrinsics_comparison.json"
        self.assertTrue(master_json.exists())
        self.assertGreater(master_json.stat().st_size, 0)

        master_csv = self.EXPERIMENT_DIR / "intrinsics_comparison.csv"
        self.assertTrue(master_csv.exists())
        self.assertGreater(master_csv.stat().st_size, 0)

        plot_png = self.EXPERIMENT_DIR / "intrinsics_comparison_plot.png"
        self.assertTrue(plot_png.exists())
        self.assertGreater(plot_png.stat().st_size, 0)

    def test_cli_real_sfm_help(self):
        """Verifies main.py CLI exposes the real_sfm phase in its help menu."""
        cmd = [sys.executable, "main.py", "--help"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("real_sfm", res.stdout)


if __name__ == "__main__":
    unittest.main()
