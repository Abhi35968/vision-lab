"""
Unit tests for Phase 2: Point Cloud Optimization Module
"""

import numpy as np
import pytest

from src.point_cloud_optimization.point_cloud_optimizer import (
    PointCloudOptimizer,
    OptimizationConfig,
    OptimizationReport,
)


class TestPointCloudOptimizer:
    """Tests for SOR, ROR, Voxel Downsampling, Normal Estimation, and Cluster Cleaning."""

    @pytest.fixture
    def optimizer(self):
        cfg = OptimizationConfig(
            enable_voxel_downsampling=False,
            enable_sor=True,
            sor_nb_neighbors=5,
            sor_std_ratio=1.0,
            enable_ror=False,
            enable_cluster_filtering=False,
            enable_normal_estimation=True,
            normal_k_neighbors=5,
        )
        return PointCloudOptimizer(config=cfg)

    def test_statistical_outlier_removal(self, optimizer):
        """Verifies that an extreme isolated point is removed by SOR."""
        # Dense cluster of 20 points near origin
        np.random.seed(42)
        cluster = np.random.normal(loc=0.0, scale=0.05, size=(30, 3))
        # Add 1 distant outlier at [100.0, 100.0, 100.0]
        outlier = np.array([[100.0, 100.0, 100.0]])
        pts = np.vstack([cluster, outlier])

        clean_pts, _, _, report = optimizer.optimize(pts)
        assert len(clean_pts) <= 30
        # Check that outlier is gone
        distances = np.linalg.norm(clean_pts - outlier, axis=1)
        assert np.all(distances > 10.0)
        assert report.raw_points_count == 31
        assert report.final_points_count < 31

    def test_voxel_downsampling(self):
        """Verifies that tightly spaced duplicate points are reduced to voxel grid."""
        cfg = OptimizationConfig(
            enable_voxel_downsampling=True,
            voxel_size=0.1,
            enable_sor=False,
            enable_ror=False,
            enable_cluster_filtering=False,
            enable_normal_estimation=False,
        )
        opt = PointCloudOptimizer(config=cfg)
        # 10 identical points at origin
        pts = np.zeros((10, 3), dtype=np.float64)
        clean_pts, _, _, report = opt.optimize(pts)
        assert len(clean_pts) == 1

    def test_normal_estimation(self, optimizer):
        """Verifies that normal estimation calculates unit-length normals."""
        # Create a flat planar grid of points on XY plane: z = 0
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        z = np.zeros_like(x)
        pts = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

        clean_pts, _, normals, report = optimizer.optimize(pts)
        assert normals is not None
        assert normals.shape == clean_pts.shape
        # Normals must have unit length norm == 1.0
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    def test_cluster_filtering(self):
        """Verifies that small disconnected clusters are filtered out by DBSCAN."""
        cfg = OptimizationConfig(
            enable_voxel_downsampling=False,
            enable_sor=False,
            enable_ror=False,
            enable_cluster_filtering=True,
            dbscan_eps=0.2,
            dbscan_min_points=5,
            keep_largest_cluster_only=True,
            enable_normal_estimation=False,
        )
        opt = PointCloudOptimizer(config=cfg)
        np.random.seed(42)
        # Main cluster: 40 points
        main_cluster = np.random.normal(loc=0.0, scale=0.05, size=(40, 3))
        # Small noise cluster: 3 points at distance 10
        noise_cluster = np.random.normal(loc=10.0, scale=0.01, size=(3, 3))
        pts = np.vstack([main_cluster, noise_cluster])

        clean_pts, _, _, report = opt.optimize(pts)
        assert len(clean_pts) == 40
        assert np.all(clean_pts[:, 0] < 5.0)

    def test_empty_or_too_small_input_handling(self, optimizer):
        """Verifies graceful handling of empty or singleton point clouds."""
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        clean_pts, _, normals, report = optimizer.optimize(pts)
        assert len(clean_pts) == 1
        assert report.final_points_count == 1

    def test_plane_segmentation_removal(self):
        """Verifies that a dominant flat ground/table plane is identified and excised."""
        cfg = OptimizationConfig(
            enable_voxel_downsampling=False,
            enable_sor=False,
            enable_ror=False,
            enable_cluster_filtering=False,
            enable_plane_removal=True,
            plane_dist_thresh=0.01,
            enable_normal_estimation=False,
        )
        opt = PointCloudOptimizer(config=cfg)
        np.random.seed(42)
        # Create 100 planar points on z=0 (table)
        x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        table_pts = np.column_stack([x.flatten(), y.flatten(), np.zeros(100)])
        # Create 25 object points elevated above table at z=0.5
        obj_pts = np.random.uniform(low=-0.2, high=0.2, size=(25, 3)) + np.array([0, 0, 0.5])
        all_pts = np.vstack([table_pts, obj_pts])

        clean_pts, _, _, report = opt.optimize(all_pts)
        assert report.plane_removed_count >= 95
        assert len(clean_pts) <= 30
        assert np.all(clean_pts[:, 2] > 0.1)

