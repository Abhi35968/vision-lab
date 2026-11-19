"""
Unit Tests for Phase 5.3.3 Object-Specific Point Cloud Refinement
Validates reprojection statistics, outlier observation detection, positive depth,
triangulation geometry, spatial clustering, and multi-criteria confidence classification.
"""

from pathlib import Path
import unittest
import numpy as np

from src.refinement.refinement_models import (
    ObservationReprojectionInfo,
    PointGeometricProfile,
    RefinementConfig,
)
from src.refinement.reprojection_filter import ReprojectionFilter
from src.refinement.triangulation_analyzer import TriangulationAnalyzer
from src.refinement.spatial_clusterer import SpatialClusterer
from src.refinement.point_refiner import PointRefiner


class TestReprojectionFilter(unittest.TestCase):
    """Tests reprojection error statistics and outlier observation identification."""

    def setUp(self):
        self.filter = ReprojectionFilter(RefinementConfig(outlier_single_obs_thresh_px=10.0))
        self.K = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 4 registered cameras facing along optical axis Z
        self.cams = {}
        for cid in range(4):
            self.cams[cid] = {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [0.0, 0.0, 0.0],
                "camera_center": [0.0, 0.0, 0.0],
                "is_registered": True,
            }

    def test_reprojection_error_calculations(self):
        errors = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = ReprojectionFilter.compute_robust_statistics(errors)
        self.assertAlmostEqual(stats["mean"], 3.0, places=2)
        self.assertAlmostEqual(stats["median"], 3.0, places=2)
        self.assertEqual(stats["count_gt_2px"], 3)
        self.assertEqual(stats["count_gt_4px"], 1)
        self.assertEqual(stats["count_gt_10px"], 0)
        self.assertAlmostEqual(stats["max"], 5.0, places=2)

    def test_outlier_observation_detection(self):
        # 3D point at (0, 0, 5) -> projects to (500, 500)
        # Cam 0, 1, 2 have near-zero error; Cam 3 has huge 50px error
        observations = [
            {"camera_id": 0, "coords": (500.0, 500.0)},
            {"camera_id": 1, "coords": (501.0, 500.0)},
            {"camera_id": 2, "coords": (500.0, 501.0)},
            {"camera_id": 3, "coords": (550.0, 500.0)},  # 50px outlier
        ]

        obs_infos, stats = self.filter.evaluate_point_reprojections(
            point_3d=(0.0, 0.0, 5.0),
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            image_size=(1000, 1000),
        )

        self.assertEqual(stats["num_observations"], 4)
        self.assertEqual(stats["num_outliers"], 1)
        self.assertTrue(obs_infos[3].is_outlier)
        self.assertFalse(obs_infos[0].is_outlier)
        # Filtered median should exclude the 50px outlier
        self.assertLess(stats["filtered"]["median"], 2.0)
        self.assertGreater(stats["raw"]["max"], 40.0)


class TestTriangulationAnalyzer(unittest.TestCase):
    """Tests ray triangulation angles, baseline geometry, and positive depth validation."""

    def setUp(self):
        self.analyzer = TriangulationAnalyzer()
        # Two cameras displaced along X-axis by 2.0 units
        self.cams = {
            0: {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [-1.0, 0.0, 0.0],  # C_0 = (+1, 0, 0)
                "is_registered": True,
            },
            1: {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [1.0, 0.0, 0.0],   # C_1 = (-1, 0, 0)
                "is_registered": True,
            },
        }

    def test_triangulation_angle_calculation(self):
        # Point at (0, 0, 5). Camera 0 is at (1, 0, 0), Camera 1 is at (-1, 0, 0).
        # Triangle from (0,0,5) to (1,0,0) and (-1,0,0): baseline = 2.0, depth = 5.0
        # Angle theta = 2 * arctan(1 / 5) = 2 * 11.31° = 22.62°
        res = self.analyzer.analyze_triangulation(
            point_3d=(0.0, 0.0, 5.0),
            camera_ids=[0, 1],
            registered_cameras=self.cams,
        )

        self.assertEqual(res["num_distinct_cameras"], 2)
        self.assertAlmostEqual(res["max_angle_deg"], 22.62, places=1)
        self.assertAlmostEqual(res["max_camera_baseline"], 2.0, places=2)
        self.assertTrue(res["has_positive_depth_all_views"])
        self.assertFalse(res["is_degenerate"])

    def test_positive_depth_validation(self):
        # Point behind camera plane (Z = -5)
        res = self.analyzer.analyze_triangulation(
            point_3d=(0.0, 0.0, -5.0),
            camera_ids=[0, 1],
            registered_cameras=self.cams,
        )
        self.assertFalse(res["has_positive_depth_all_views"])
        self.assertTrue(res["is_degenerate"])

    def test_degenerate_triangulation_geometry(self):
        # Two cameras with near-zero baseline (displaced by 0.0001)
        cams_degen = {
            0: {"rotation_matrix": np.eye(3).tolist(), "translation_vector": [0.0, 0.0, 0.0], "is_registered": True},
            1: {"rotation_matrix": np.eye(3).tolist(), "translation_vector": [0.0001, 0.0, 0.0], "is_registered": True},
        }
        res = self.analyzer.analyze_triangulation(
            point_3d=(0.0, 0.0, 10.0),
            camera_ids=[0, 1],
            registered_cameras=cams_degen,
        )
        # Parallax is tiny (< 0.1 deg)
        self.assertLess(res["max_angle_deg"], 1.0)
        self.assertTrue(res["is_degenerate"])


class TestSpatialClusterer(unittest.TestCase):
    """Tests k-NN distance distributions, DBSCAN clustering, and spatial outlier removal."""

    def test_nearest_neighbor_distance_calculation(self):
        # 3 collinear points spaced by 1.0 unit: (0,0,0), (1,0,0), (2,0,0)
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        knn_dict, stats = SpatialClusterer.compute_knn_distance_distribution(pts, k_values=(1, 2))
        # 1-NN distances: point 0 -> 1.0, point 1 -> 1.0, point 2 -> 1.0
        np.testing.assert_allclose(knn_dict[1], [1.0, 1.0, 1.0], rtol=1e-5)
        self.assertAlmostEqual(stats["median"], 1.0, places=2)

    def test_clustering_behavior_and_outlier_detection(self):
        # Cluster of 5 points near origin + 1 isolated outlier at (50, 50, 50)
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
            [0.05, 0.05, 0.0],
            [50.0, 50.0, 50.0],  # Severe spatial outlier
        ])

        clusterer = SpatialClusterer(RefinementConfig(dbscan_eps=0.5, dbscan_min_samples=3))
        res = clusterer.cluster_points(pts)

        labels = res["cluster_labels"]
        is_outlier = res["is_spatial_outlier"]

        # Main cluster should be label 0 for the first 5 points
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[4], 0)
        # Outlier should be noise (-1) and flagged as spatial outlier
        self.assertEqual(labels[5], -1)
        self.assertTrue(is_outlier[5])
        self.assertFalse(is_outlier[0])


class TestPointRefinerIntegration(unittest.TestCase):
    """Tests overall refinement logic, edge cases, and classification precedence."""

    def setUp(self):
        self.refiner = PointRefiner()
        self.K = np.array([[1000.0, 0, 500.0], [0, 1000.0, 500.0], [0, 0, 1.0]])
        self.cams = {
            0: {"rotation_matrix": np.eye(3).tolist(), "translation_vector": [-1.0, 0.0, 0.0], "is_registered": True},
            1: {"rotation_matrix": np.eye(3).tolist(), "translation_vector": [1.0, 0.0, 0.0], "is_registered": True},
            2: {"rotation_matrix": np.eye(3).tolist(), "translation_vector": [0.0, -1.0, 0.0], "is_registered": True},
        }

    def test_invalid_3d_coordinates(self):
        tri_stats = self.refiner.tri_analyzer.analyze_triangulation(
            point_3d=(np.nan, 0.0, 5.0),
            camera_ids=[0, 1],
            registered_cameras=self.cams,
        )
        self.assertTrue(tri_stats["is_degenerate"])
        self.assertIn("non-finite", tri_stats["degeneracy_reason"])

    def test_empty_candidate_groups(self):
        profiles = self.refiner.refine_candidate_points(
            candidate_associations=[],
            tracks_by_id={},
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
        )
        self.assertEqual(len(profiles), 0)

    def test_preservation_of_original_classifications(self):
        # 3 points forming a tight cluster near (0, 0, 5)
        candidates = [
            {
                "point_id": 10,
                "track_id": 100,
                "x_world": 0.0,
                "y_world": 0.0,
                "z_world": 5.0,
                "classification": "STRONG_OBJECT_CANDIDATE",
                "mask_support_ratio": 1.0,
                "masked_observations": 3,
                "inside_mask_observations": 3,
                "supporting_cameras": "0;1;2",
            },
            {
                "point_id": 11,
                "track_id": 101,
                "x_world": 0.05,
                "y_world": 0.05,
                "z_world": 5.0,
                "classification": "STRONG_OBJECT_CANDIDATE",
                "mask_support_ratio": 1.0,
                "masked_observations": 3,
                "inside_mask_observations": 3,
                "supporting_cameras": "0;1;2",
            },
            {
                "point_id": 12,
                "track_id": 102,
                "x_world": -0.05,
                "y_world": -0.05,
                "z_world": 5.0,
                "classification": "STRONG_OBJECT_CANDIDATE",
                "mask_support_ratio": 1.0,
                "masked_observations": 3,
                "inside_mask_observations": 3,
                "supporting_cameras": "0;1;2",
            },
        ]

        # For point 10 at (0, 0, 5):
        # Cam 0 (t=[-1, 0, 0]): u = 1000*(-1)/5 + 500 = 300, v = 500
        # Cam 1 (t=[1, 0, 0]):  u = 1000*(1)/5 + 500 = 700, v = 500
        # Cam 2 (t=[0, -1, 0]): u = 500, v = 1000*(-1)/5 + 500 = 300
        tracks = {
            100: {
                "track_id": 100,
                "observations": [
                    {"camera_id": 0, "coords": (300.0, 500.0)},
                    {"camera_id": 1, "coords": (700.0, 500.0)},
                    {"camera_id": 2, "coords": (500.0, 300.0)},
                ],
            },
            101: {
                "track_id": 101,
                "observations": [
                    {"camera_id": 0, "coords": (310.0, 510.0)},
                    {"camera_id": 1, "coords": (710.0, 510.0)},
                    {"camera_id": 2, "coords": (510.0, 310.0)},
                ],
            },
            102: {
                "track_id": 102,
                "observations": [
                    {"camera_id": 0, "coords": (290.0, 490.0)},
                    {"camera_id": 1, "coords": (690.0, 490.0)},
                    {"camera_id": 2, "coords": (490.0, 290.0)},
                ],
            },
        }

        profiles = self.refiner.refine_candidate_points(
            candidate_associations=candidates,
            tracks_by_id=tracks,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
        )

        self.assertEqual(len(profiles), 3)
        p = profiles[0]
        # Verify original classification is preserved unmodified
        self.assertEqual(p.original_classification, "STRONG_OBJECT_CANDIDATE")
        self.assertEqual(p.refined_status, "HIGH_GEOMETRIC_CONFIDENCE")
        self.assertAlmostEqual(p.median_reproj_error_px, 0.0, places=2)

    def test_correct_refined_status_assignment(self):
        # Test outlier assignment for point with 50px reprojection error
        reproj_stats = {
            "raw": {"median": 15.0, "p90": 25.0, "max": 50.0},
            "filtered": {"median": 15.0},
        }
        tri_stats = {
            "max_angle_deg": 10.0,
            "has_positive_depth_all_views": True,
            "is_degenerate": False,
        }

        status, reason = self.refiner.classify_point_confidence(
            orig_class="STRONG_OBJECT_CANDIDATE",
            orig_ratio=1.0,
            num_masked=3,
            num_inside=3,
            num_cams=3,
            reproj_stats=reproj_stats,
            tri_stats=tri_stats,
            is_spatial_outlier=False,
        )

        self.assertEqual(status, "GEOMETRIC_OUTLIER")
        self.assertIn("Reprojection outlier", reason)


if __name__ == "__main__":
    unittest.main()
