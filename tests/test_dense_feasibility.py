"""
Unit Tests for Phase 5.4 Dense Object Reconstruction Feasibility Investigation
Tests feature density analyzer, silhouette vertical partitioner, track survival calculator,
pairwise matching quality analyzer, dataset limitations analyzer, and MVS comparator.
"""

import unittest
from pathlib import Path
import cv2
import numpy as np

from src.feasibility.feature_density_analyzer import (
    FeatureDensityAnalyzer,
    FrameFeatureDensityRecord,
    SilhouetteDistribution,
    TrackSurvivalSummary,
)
from src.feasibility.matching_quality_analyzer import (
    MatchingQualityAnalyzer,
    PairwiseMatchingRecord,
)
from src.feasibility.dataset_limitations_analyzer import (
    DatasetLimitationsAnalyzer,
    DatasetLimitationsSummary,
    CameraOrbitRecord,
)
from src.feasibility.mvs_comparator import (
    MVSComparator,
    ReconstructionMethodEvaluation,
)


class TestFeatureDensityAnalyzer(unittest.TestCase):
    """Tests for SIFT feature density and silhouette distribution."""

    def setUp(self):
        self.analyzer = FeatureDensityAnalyzer(n_features=500)
        # Create synthetic 400x400 image with a high-contrast pattern
        self.synthetic_img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Add random textured patches
        for y in range(0, 400, 20):
            for x in range(0, 400, 20):
                self.synthetic_img[y:y+10, x:x+10] = (y % 255, x % 255, (x+y) % 255)

        # Create synthetic mask in center: [100, 300] x [150, 250]
        self.synthetic_mask = np.zeros((400, 400), dtype=np.uint8)
        self.synthetic_mask[100:300, 150:250] = 255

    def test_feature_extraction(self):
        kps, descs = self.analyzer.extract_features(self.synthetic_img)
        self.assertGreater(len(kps), 0)
        self.assertEqual(len(kps), len(descs))
        self.assertEqual(descs.shape[1], 128)  # SIFT descriptor dimension

    def test_partition_keypoints_by_mask(self):
        # Create synthetic keypoints
        kp_inside = cv2.KeyPoint(x=200.0, y=200.0, size=10.0)
        kp_outside = cv2.KeyPoint(x=50.0, y=50.0, size=10.0)
        kps = [kp_inside, kp_outside]

        inside, outside = self.analyzer.partition_keypoints_by_mask(kps, self.synthetic_mask)
        self.assertEqual(len(inside), 1)
        self.assertEqual(len(outside), 1)
        self.assertAlmostEqual(inside[0].pt[0], 200.0)
        self.assertAlmostEqual(outside[0].pt[0], 50.0)

    def test_silhouette_vertical_distribution(self):
        # Mask occupies y in [100, 300], height = 200
        # Cap: rel_y in [0.00, 0.20) -> y in [100, 140)
        # Neck: rel_y in [0.20, 0.35) -> y in [140, 170)
        # Body: rel_y in [0.35, 0.80) -> y in [170, 260)
        # Base: rel_y in [0.80, 1.00] -> y in [260, 300]
        kps = [
            cv2.KeyPoint(x=200.0, y=110.0, size=5.0),  # Cap
            cv2.KeyPoint(x=200.0, y=150.0, size=5.0),  # Neck
            cv2.KeyPoint(x=200.0, y=200.0, size=5.0),  # Body
            cv2.KeyPoint(x=200.0, y=280.0, size=5.0),  # Base
        ]
        sil = self.analyzer.compute_silhouette_distribution(kps, self.synthetic_mask)
        self.assertEqual(sil.cap_count, 1)
        self.assertEqual(sil.neck_count, 1)
        self.assertEqual(sil.body_count, 1)
        self.assertEqual(sil.base_count, 1)
        self.assertEqual(sil.total_inside, 4)

    def test_silhouette_empty_mask(self):
        empty_mask = np.zeros((400, 400), dtype=np.uint8)
        kps = [cv2.KeyPoint(x=200.0, y=200.0, size=5.0)]
        sil = self.analyzer.compute_silhouette_distribution(kps, empty_mask)
        self.assertEqual(sil.total_inside, 0)
        self.assertEqual(sil.cap_count, 0)

    def test_analyze_frame(self):
        rec = self.analyzer.analyze_frame(
            frame_idx=0,
            image_name="test.png",
            image=self.synthetic_img,
            mask=self.synthetic_mask,
            is_registered=True,
        )
        self.assertIsInstance(rec, FrameFeatureDensityRecord)
        self.assertTrue(rec.mask_present)
        self.assertGreater(rec.total_keypoints, 0)
        self.assertGreater(rec.mask_area_px, 0)
        self.assertGreaterEqual(rec.density_inside_per_10k_px2, 0.0)
        self.assertGreaterEqual(rec.density_outside_per_10k_px2, 0.0)

    def test_analyze_track_survival(self):
        masks = [self.synthetic_mask, self.synthetic_mask]
        tracks = [
            # Pure object track (both inside mask)
            {"observations": [{"image_id": 0, "coords": [200.0, 200.0]}, {"image_id": 1, "coords": [201.0, 201.0]}]},
            # Pure background track (both outside mask)
            {"observations": [{"image_id": 0, "coords": [10.0, 10.0]}, {"image_id": 1, "coords": [11.0, 11.0]}, {"image_id": 1, "coords": [12.0, 12.0]}]},
            # Mixed track (one in, one out)
            {"observations": [{"image_id": 0, "coords": [200.0, 200.0]}, {"image_id": 1, "coords": [10.0, 10.0]}]},
        ]
        summary = self.analyzer.analyze_track_survival(tracks, masks)
        self.assertEqual(summary.total_tracks, 3)
        self.assertEqual(summary.pure_object_tracks, 1)
        self.assertEqual(summary.pure_background_tracks, 1)
        self.assertEqual(summary.mixed_tracks, 1)


class TestMatchingQualityAnalyzer(unittest.TestCase):
    """Tests for pairwise descriptor matching and epipolar geometry."""

    def setUp(self):
        self.matcher = MatchingQualityAnalyzer(lowe_ratio=0.75, ransac_threshold_px=3.0)

    def test_empty_descriptors(self):
        rec = self.matcher.evaluate_pair(
            frame_idx1=0, frame_idx2=1,
            img_name1="f0.png", img_name2="f1.png",
            kps1=[], descs1=np.zeros((0, 128), dtype=np.float32), mask1=None,
            kps2=[], descs2=np.zeros((0, 128), dtype=np.float32), mask2=None,
        )
        self.assertEqual(rec.total_raw_matches, 0)
        self.assertEqual(rec.inside_raw_matches, 0)
        self.assertEqual(rec.outside_raw_matches, 0)

    def test_matching_synthetic_pairs(self):
        # Create identical descriptors with slight noise
        rng = np.random.RandomState(42)
        descs1 = rng.randn(20, 128).astype(np.float32)
        descs2 = descs1 + rng.randn(20, 128).astype(np.float32) * 0.05
        # Normalize
        descs1 /= np.linalg.norm(descs1, axis=1, keepdims=True)
        descs2 /= np.linalg.norm(descs2, axis=1, keepdims=True)

        kps1 = [cv2.KeyPoint(x=float(i * 10), y=float(i * 10), size=5.0) for i in range(20)]
        kps2 = [cv2.KeyPoint(x=float(i * 10 + 2), y=float(i * 10 + 1), size=5.0) for i in range(20)]

        # Mask covers first 10 points
        mask1 = np.zeros((300, 300), dtype=np.uint8)
        mask1[:100, :100] = 255
        mask2 = mask1.copy()

        rec = self.matcher.evaluate_pair(
            frame_idx1=0, frame_idx2=1,
            img_name1="f0.png", img_name2="f1.png",
            kps1=kps1, descs1=descs1, mask1=mask1,
            kps2=kps2, descs2=descs2, mask2=mask2,
        )
        self.assertGreater(rec.total_raw_matches, 0)
        self.assertGreaterEqual(rec.inside_ratio_matches, 0)
        self.assertGreaterEqual(rec.outside_ratio_matches, 0)


class TestDatasetLimitationsAnalyzer(unittest.TestCase):
    """Tests for camera orbit geometry, motion blur, and dataset limitations."""

    def setUp(self):
        self.centroid = np.array([0.0, 0.0, 5.0])
        self.analyzer = DatasetLimitationsAnalyzer(object_centroid=self.centroid)

    def test_orbit_angular_span(self):
        # Create 4 cameras on a 60-degree arc around (0, 0, 5)
        # radius = 5, angles = 30, 45, 60, 90 deg
        cameras = {}
        angles = [30.0, 45.0, 60.0, 90.0]
        for i, a in enumerate(angles):
            rad = np.radians(a)
            # x = r * sin(rad), z = 5 - r * cos(rad)
            cx = 5.0 * np.sin(rad)
            cz = 5.0 - 5.0 * np.cos(rad)
            cameras[str(i)] = {
                "is_registered": True,
                "camera_center": [cx, 0.0, cz],
                "image_name": f"cam_{i}.png",
            }

        records, min_az, max_az, span = self.analyzer.analyze_camera_orbit(cameras)
        self.assertEqual(len(records), 4)
        self.assertAlmostEqual(span, 60.0, places=1)
        self.assertGreater(records[0].distance_to_object, 0)

    def test_motion_blur_and_dropouts(self):
        # Sharp synthetic image vs blurry synthetic image
        sharp_img = np.zeros((100, 100), dtype=np.uint8)
        sharp_img[::2, ::2] = 255  # checkerboard = high Laplacian variance
        blur_img = cv2.GaussianBlur(sharp_img, (15, 15), 5.0)

        mask = np.ones((100, 100), dtype=np.uint8) * 255
        images = [sharp_img, blur_img]
        masks = [mask, None]  # Frame 1 has dropout

        f_vars, o_vars, dropouts = self.analyzer.evaluate_motion_blur(images, masks)
        self.assertEqual(len(dropouts), 1)
        self.assertEqual(dropouts[0], 1)
        self.assertGreater(f_vars[0], f_vars[1])  # Sharp has higher variance than blur


class TestMVSComparator(unittest.TestCase):
    """Tests for MVS method comparison matrix."""

    def test_method_evaluations_structure(self):
        evals = MVSComparator.get_method_comparisons()
        self.assertGreaterEqual(len(evals), 5)
        method_names = [e.method_name for e in evals]
        self.assertIn("Classical PatchMatch MVS", method_names)
        self.assertIn("COLMAP Dense Reconstruction", method_names)
        self.assertIn("Shape-from-Silhouette / Visual Hull (Voxel Carving)", method_names)

        for e in evals:
            self.assertIn(e.dataset_feasibility_verdict, {"FEASIBLE", "FEASIBLE (with symmetry prior)", "DEGRADED (Orbit Constrained)", "INFEASIBLE"})
            self.assertTrue(len(e.technical_justification) > 20)


if __name__ == "__main__":
    unittest.main()
