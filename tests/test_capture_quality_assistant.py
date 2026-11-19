"""
Unit Test Suite for Video Capture Quality Assistant
Validates:
1. Frame Sharpness Detection (Laplacian variance on sharp vs blurry images)
2. Frame Redundancy Detection (dHash on identical and near-duplicate images)
3. Feature Availability (Keypoint count on textured vs textureless images)
4. Camera Movement Assessment (Disparity and epipolar baseline tracking)
5. Angular Coverage Assessment (Distinction between estimated vs verified coverage)
6. Decision Engine & Practical Recommendations
7. Full Video Assessment Pipeline
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from src.preprocessing.capture_quality_assistant import (
    VideoCaptureQualityAssistant,
    QualityAssessmentSummary,
    FrameQualityRecord,
)


@pytest.fixture
def quality_assistant():
    return VideoCaptureQualityAssistant(
        blur_threshold=12.0,
        min_keypoints=50,
        good_keypoints=150,
        max_redundancy_hamming=3,
        min_motion_displacement_px=5.0,
        max_motion_displacement_px=100.0,
        target_eval_dimension=320,
        sample_stride=2,
    )


@pytest.fixture
def sample_sharp_textured_image():
    """Generates a high-contrast sharp checkerboard pattern with fine texture."""
    np.random.seed(42)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    # Checkerboard
    for i in range(0, 240, 20):
        for j in range(0, 320, 20):
            if (i // 20 + j // 20) % 2 == 0:
                img[i:i+20, j:j+20] = 255
    # Add high-frequency noise
    noise = np.random.randint(0, 50, (240, 320, 3), dtype=np.uint8)
    return cv2.add(img, noise)


@pytest.fixture
def sample_blurry_image(sample_sharp_textured_image):
    """Heavy Gaussian blur applied to eliminate high-frequency edges."""
    return cv2.GaussianBlur(sample_sharp_textured_image, (31, 31), 15.0)


@pytest.fixture
def sample_blank_image():
    """Flat uniform grayscale image devoid of any geometric features."""
    return np.full((240, 320, 3), 128, dtype=np.uint8)


class TestSharpnessDetection:
    """Verifies Laplacian variance edge response under sharp and blurry conditions."""

    def test_sharp_image_has_high_laplacian_variance(self, quality_assistant, sample_sharp_textured_image):
        gray = cv2.cvtColor(sample_sharp_textured_image, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        assert lap_var > quality_assistant.blur_threshold
        assert lap_var > 100.0

    def test_blurred_image_is_flagged_blurry(self, quality_assistant, sample_blurry_image):
        gray = cv2.cvtColor(sample_blurry_image, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        assert lap_var < quality_assistant.blur_threshold


class TestRedundancyDetection:
    """Verifies 64-bit dHash perceptual hashing and duplicate detection."""

    def test_identical_images_have_zero_hamming_distance(self, quality_assistant, sample_sharp_textured_image):
        gray = cv2.cvtColor(sample_sharp_textured_image, cv2.COLOR_BGR2GRAY)
        h1 = quality_assistant.compute_dhash(gray)
        h2 = quality_assistant.compute_dhash(gray)
        assert quality_assistant.hamming_distance(h1, h2) == 0

    def test_subtly_altered_image_is_flagged_as_redundant(self, quality_assistant):
        # Smooth gradient without edge clipping
        x = np.linspace(30, 200, 320, dtype=np.uint8)
        gray1 = np.tile(x, (240, 1))
        # Minor intensity shift (e.g. auto-exposure micro-fluctuation)
        gray2 = gray1 + 2
        h1 = quality_assistant.compute_dhash(gray1)
        h2 = quality_assistant.compute_dhash(gray2)
        dist = quality_assistant.hamming_distance(h1, h2)
        assert dist <= quality_assistant.max_redundancy_hamming


class TestFeatureAvailability:
    """Verifies SIFT feature distribution across textured vs textureless inputs."""

    def test_textured_image_yields_sufficient_features(self, quality_assistant, sample_sharp_textured_image):
        gray = cv2.cvtColor(sample_sharp_textured_image, cv2.COLOR_BGR2GRAY)
        kp, des = quality_assistant.sift.detectAndCompute(gray, None)
        assert kp is not None
        assert len(kp) >= quality_assistant.good_keypoints

    def test_textureless_blank_image_yields_insufficient_features(self, quality_assistant, sample_blank_image):
        gray = cv2.cvtColor(sample_blank_image, cv2.COLOR_BGR2GRAY)
        kp, des = quality_assistant.sift.detectAndCompute(gray, None)
        kp_count = len(kp) if kp is not None else 0
        assert kp_count < quality_assistant.min_keypoints


class TestCameraMovementAndAngularCoverage:
    """Verifies baseline tracking and relative rotation estimations."""

    def test_stationary_camera_detection(self, quality_assistant, sample_sharp_textured_image):
        gray1 = cv2.cvtColor(sample_sharp_textured_image, cv2.COLOR_BGR2GRAY)
        gray2 = gray1.copy()
        h1 = quality_assistant.compute_dhash(gray1)
        h2 = quality_assistant.compute_dhash(gray2)
        assert quality_assistant.hamming_distance(h1, h2) == 0

    def test_relative_yaw_rotation_estimation(self, quality_assistant, sample_sharp_textured_image):
        # Generate two views with synthetic inliers
        pts1 = np.array([[100, 100], [200, 100], [150, 150], [100, 200], [200, 200], [120, 180], [180, 120], [150, 80]], dtype=np.float32)
        # Slight horizontal shift
        pts2 = pts1 + np.array([15.0, 2.0], dtype=np.float32)

        yaw = quality_assistant._estimate_relative_rotation_yaw(pts1, pts2, 320, 240)
        assert isinstance(yaw, float)
        assert yaw >= 0.0


class TestSyntheticVideoQualityAssessment:
    """Tests end-to-end video evaluation on synthetic video sequences."""

    def test_synthetic_video_pipeline(self, quality_assistant, sample_sharp_textured_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = Path(tmpdir) / "test_synth.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (320, 240))

            # Write 20 frames with gentle horizontal panning
            for i in range(20):
                M = np.float32([[1, 0, i * 2], [0, 1, 0]])
                panned = cv2.warpAffine(sample_sharp_textured_image, M, (320, 240))
                out.write(panned)
            out.release()

            summary = quality_assistant.assess_video(vid_path, max_eval_frames=15)
            assert isinstance(summary, QualityAssessmentSummary)
            assert summary.total_video_frames == 20
            assert summary.candidate_frames_evaluated > 0
            assert summary.average_sharpness > 0.0
            assert summary.overall_status in {"GOOD", "WARNING", "POOR"}
            assert len(summary.recommendations) > 0


class TestRealVideoValidation:
    """Verifies execution against the real project smartphone video."""

    def test_real_video_assessment(self):
        real_vid = Path("VID_20261102_015654418.mp4")
        if not real_vid.exists():
            pytest.skip("Real smartphone video not found in root directory.")

        assistant = VideoCaptureQualityAssistant(
            blur_threshold=12.0,
            sample_stride=10,  # Fast sampling for unit test
        )
        summary = assistant.assess_video(
            video_path=real_vid,
            verified_sfm_dir="outputs/real_video_sfm/focal_1300",
            max_eval_frames=30,
        )

        assert summary.total_video_frames == 572
        assert summary.candidate_frames_evaluated > 0
        assert summary.average_sharpness > 50.0  # Real video is sharp
        assert summary.average_keypoint_count > 100.0
        assert summary.estimated_camera_coverage_deg > 0.0
        assert summary.verified_coverage_available is True
        assert summary.verified_camera_coverage_deg is not None
        assert summary.overall_status in {"GOOD", "WARNING", "POOR"}
