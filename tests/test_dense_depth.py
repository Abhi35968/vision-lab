"""
Unit tests for Phase 6: Depth Anything V2 Small Monocular Depth Integration
"""

import numpy as np
import pytest
from pathlib import Path

from src.dense_depth.depth_anything_estimator import (
    DepthAnythingEstimator,
    DepthEstimationResult,
)


class TestDepthAnythingEstimator:
    """Tests monocular depth estimation, caching, affine SfM alignment, and unprojection."""

    @pytest.fixture
    def estimator(self, tmp_path):
        return DepthAnythingEstimator(cache_dir=tmp_path)

    @pytest.fixture
    def synthetic_image(self):
        # 240x320 gradient image
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(50, 200, 320, dtype=np.uint8)
        img[:, :, 1] = np.linspace(100, 255, 320, dtype=np.uint8)
        img[:, :, 2] = np.linspace(20, 150, 320, dtype=np.uint8)
        return img

    def test_structural_depth_fallback(self, estimator, synthetic_image):
        """Verifies structural depth fallback generates normalized [0, 1] relative depth."""
        depth = estimator._structural_depth_fallback(synthetic_image)
        assert depth.shape == (240, 320)
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_depth_estimation_and_caching(self, estimator, synthetic_image, tmp_path):
        """Verifies depth estimation execution and disk caching of .npy and colormap .png."""
        depth_map, inf_ms = estimator.estimate_relative_depth(
            synthetic_image, cache_key="frame_test_001"
        )
        assert depth_map.shape == (240, 320)
        assert inf_ms >= 0.0

        cached_npy = tmp_path / "frame_test_001_depth.npy"
        cached_png = tmp_path / "frame_test_001_depth_vis.png"
        assert cached_npy.exists()
        assert cached_png.exists()

        loaded_depth = np.load(str(cached_npy))
        np.testing.assert_allclose(loaded_depth, depth_map, atol=1e-5)

    def test_affine_alignment_with_sfm(self, estimator):
        """Verifies least-squares scale and shift recovery between relative and SfM depths."""
        # Simulated camera at origin looking down +Z
        K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t = np.zeros(3)

        # 3D points with ground-truth depths z = 2.0, 3.0, 4.0, 5.0
        pts_3d = np.array([
            [0.0, 0.0, 2.0],
            [0.2, 0.1, 3.0],
            [-0.1, 0.3, 4.0],
            [0.4, -0.2, 5.0],
        ], dtype=np.float64)

        # Create relative depth map where ground truth relative depth d = (z - 1.0) / 4.0
        # So z = 4.0 * d + 1.0 => scale s = 4.0, shift b = 1.0
        rel_depth = np.zeros((240, 320), dtype=np.float32)
        for pt in pts_3d:
            p_cam = pt  # R=I, t=0
            u = int(round(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]))
            v = int(round(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))
            if 0 <= v < 240 and 0 <= u < 320:
                rel_depth[v, u] = (p_cam[2] - 1.0) / 4.0

        s, b, rmse, aligned = estimator.align_depth_with_sfm(rel_depth, pts_3d, R, t, K, mode="linear")
        assert pytest.approx(s, abs=0.1) == 4.0
        assert pytest.approx(b, abs=0.1) == 1.0
        assert rmse < 0.1
        assert aligned.min() >= 0.0

    def test_align_depth_with_sfm_inverse(self, estimator):
        """Verifies robust inverse-depth alignment: 1/z = s * d + b."""
        K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t = np.zeros(3)

        pts_3d = np.array([
            [0.0, 0.0, 2.0],
            [0.2, 0.1, 2.5],
            [-0.1, 0.3, 3.3333],
            [0.4, -0.2, 5.0],
        ], dtype=np.float64)

        # 1/z values: 0.5, 0.4, 0.3, 0.2
        # Let d = (1/z - 0.1) / 0.5 => 1/z = 0.5 * d + 0.1
        rel_depth = np.zeros((240, 320), dtype=np.float32)
        for pt in pts_3d:
            p_cam = pt
            u = int(round(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]))
            v = int(round(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))
            if 0 <= v < 240 and 0 <= u < 320:
                inv_z = 1.0 / p_cam[2]
                rel_depth[v, u] = (inv_z - 0.1) / 0.5

        s, b, rmse, aligned, meta = estimator.align_depth_with_sfm(
            rel_depth, pts_3d, R, t, K, mode="inverse", return_metadata=True
        )
        assert pytest.approx(s, abs=0.05) == 0.5
        assert pytest.approx(b, abs=0.05) == 0.1
        assert rmse < 0.1
        assert meta["status"] == "success"
        assert aligned.min() > 0.0

    def test_unproject_dense_cloud(self, estimator):
        """Verifies unprojection produces valid (M, 3) 3D coordinate array."""
        K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t = np.zeros(3)
        # Constant depth plane z = 2.5
        depth_plane = np.ones((240, 320), dtype=np.float32) * 2.5

        pts = estimator.unproject_dense_cloud(depth_plane, R, t, K, stride=10, max_points=1000)
        assert len(pts) > 50
        assert pts.shape[1] == 3
        # In camera frame with R=I, t=0, z coordinates should be close to 2.5
        np.testing.assert_allclose(pts[:, 2], 2.5, atol=1e-3)

    def test_multiview_dense_fuser(self, tmp_path):
        """Verifies multi-view dense fusion merges viewpoints, removes table plane, and saves PLY."""
        from src.dense_depth.multi_view_dense_fuser import (
            MultiViewDenseFuser,
            DenseFusionConfig,
        )
        cfg = DenseFusionConfig(
            stride=8,
            max_points_per_view=500,
            voxel_size=0.05,
            enable_sor=True,
            enable_plane_removal=False,
            enable_mesh_generation=False,
            output_dir=str(tmp_path),
        )
        fuser = MultiViewDenseFuser(config=cfg)

        # Create two synthetic views
        img = np.ones((120, 160, 3), dtype=np.uint8) * 128
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[30:90, 40:120] = 255  # Centered box mask
        K = np.array([[150.0, 0, 80], [0, 150.0, 60], [0, 0, 1]], dtype=np.float64)

        views = [
            {"name": "view_0", "image_bgr": img, "mask": mask, "R": np.eye(3), "t": np.zeros(3), "K": K},
            {"name": "view_1", "image_bgr": img, "mask": mask, "R": np.eye(3), "t": np.array([0.1, 0, 0]), "K": K},
        ]
        sparse_pts = np.array([[0, 0, 1.5], [0.1, 0, 1.6], [-0.1, 0, 1.4]], dtype=np.float64)

        res = fuser.fuse_views(views, sparse_pts, tag="test_fusion")
        assert res.total_views_processed == 2
        assert res.clean_points_count > 0
        assert Path(res.raw_ply_path).exists()
        assert Path(res.clean_ply_path).exists()
        assert Path(res.summary_json_path).exists()
        assert res.density_multiplier > 0.0

