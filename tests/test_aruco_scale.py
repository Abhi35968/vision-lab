"""
Unit tests for Phase 1: ArUco-Based Metric Scale Recovery Module
"""

import cv2
import numpy as np
import pytest

from src.metric_scale.aruco_scale_calibrator import (
    ArucoScaleCalibrator,
    MetricDimensions,
    MetricScaleResult,
    MarkerObservation,
)


def create_synthetic_aruco_image(
    marker_id: int = 4,
    marker_size_px: int = 200,
    image_shape: tuple = (480, 640),
) -> np.ndarray:
    """Creates a synthetic white canvas with an embedded ArUco marker."""
    img = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    # Place marker in center
    y_start = (image_shape[0] - marker_size_px) // 2
    x_start = (image_shape[1] - marker_size_px) // 2
    img[y_start : y_start + marker_size_px, x_start : x_start + marker_size_px] = marker_bgr
    return img


class TestArucoScaleCalibrator:
    """Test suite for ArUco marker detection, pose estimation, and metric scaling."""

    @pytest.fixture
    def camera_intrinsics(self):
        K = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        return K, dist

    @pytest.fixture
    def calibrator(self, camera_intrinsics):
        K, dist = camera_intrinsics
        return ArucoScaleCalibrator(dictionary_name="DICT_4X4_50", camera_matrix=K, dist_coeffs=dist)

    def test_marker_detection_on_synthetic_image(self, calibrator, camera_intrinsics):
        """Verifies ArUco marker detection and 6-DoF pose estimation."""
        K, dist = camera_intrinsics
        img = create_synthetic_aruco_image(marker_id=4, marker_size_px=180)

        corners, ids = calibrator.detect_markers(img)
        assert len(ids) == 1
        assert ids[0] == 4

        success, rvec, tvec, err = calibrator.estimate_marker_pose(corners[0], marker_size_cm=5.0, K=K, dist_coeffs=dist)
        assert success is True
        assert tvec.shape == (3, 1)
        assert tvec[2, 0] > 0.0  # In front of camera
        assert err >= 0.0

    def test_detection_on_blank_image(self, calibrator):
        """Verifies graceful handling when no marker exists in image."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        corners, ids = calibrator.detect_markers(blank)
        assert len(corners) == 0
        assert len(ids) == 0

    def test_scale_point_cloud(self, calibrator):
        """Verifies that point cloud vertices are scaled uniformly by scale_factor."""
        pts = np.array([
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 4.0],
        ], dtype=np.float64)

        scaled = calibrator.scale_point_cloud(pts, scale_factor=2.5)
        np.testing.assert_allclose(scaled, pts * 2.5)

    def test_scale_mesh(self, calibrator):
        """Verifies mesh vertices are scaled while topology (faces) is preserved."""
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        sv, sf = calibrator.scale_mesh(v, f, scale_factor=3.0)
        np.testing.assert_allclose(sv, v * 3.0)
        np.testing.assert_array_equal(sf, f)

    def test_measure_object_dimensions(self, calibrator):
        """Verifies accurate bounding box dimension computation."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [5.0, 10.0, 15.0],
        ], dtype=np.float64)

        dims = calibrator.measure_object_dimensions(pts, scale_factor=1.0)
        assert dims is not None
        assert pytest.approx(dims.width_x_cm, rel=1e-3) == 10.0
        assert pytest.approx(dims.height_y_cm, rel=1e-3) == 20.0
        assert pytest.approx(dims.depth_z_cm, rel=1e-3) == 30.0
        assert pytest.approx(dims.estimated_volume_cm3, rel=1e-3) == 4200.0  # 10*20*30*0.7
        assert pytest.approx(dims.estimated_volume_litres, rel=1e-3) == 4.2

    def test_compute_metric_scale_fallback_without_markers(self, calibrator):
        """Verifies calibration fallback behavior and limitation reporting without markers."""
        keyframes = {
            "frame_0.jpg": np.zeros((480, 640, 3), dtype=np.uint8),
            "frame_1.jpg": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        camera_poses = {
            "frame_0.jpg": {"R": np.eye(3), "t": np.zeros(3), "camera_center": np.zeros(3)},
            "frame_1.jpg": {"R": np.eye(3), "t": np.array([1.0, 0.0, 0.0]), "camera_center": np.array([1.0, 0.0, 0.0])},
        }

        result = calibrator.compute_metric_scale(
            keyframes=keyframes,
            camera_poses_sfm=camera_poses,
            marker_size_cm=5.0,
            assumed_focal_px=600.0,
        )
        assert isinstance(result, MetricScaleResult)
        assert result.calibration_status in ["APPROXIMATED", "FAILED"]
        assert len(result.limitations) > 0
        assert result.confidence_score <= 0.35
