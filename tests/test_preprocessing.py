"""
Unit Tests for Phase 1: Video Preprocessing and Camera Calibration Infrastructure
Uses dynamically generated synthetic videos and images (clearly distinguished from real data).
"""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.preprocessing.video_reader import VideoReader, VideoMetadata
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.image_quality import ImageQualityAssessor
from src.preprocessing.frame_filter import FrameFilter
from src.preprocessing.keyframe_selector import KeyframeSelector
from src.preprocessing.camera_calibration import CameraCalibrator


class TestPreprocessingInfrastructure(unittest.TestCase):
    """Test suite verifying Phase 1 video preprocessing components."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary sandbox directory for test artifacts."""
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="3dcv_test_"))
        cls.synthetic_video_path = cls.temp_dir / "synthetic_test_video.mp4"
        cls._create_synthetic_video(cls.synthetic_video_path, num_frames=24, fps=24.0)

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def _create_synthetic_video(path: Path, num_frames: int = 24, fps: float = 24.0):
        """Generates a small synthetic monocular video with moving geometric shapes."""
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 128
            # Add textured moving circle
            center_x = 100 + i * 15
            center_y = 200
            cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), -1)
            # Add grid pattern for sharp feature points
            for y in range(0, height, 40):
                cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)
            for x in range(0, width, 40):
                cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)
            # Frame index label
            cv2.putText(frame, f"Synthetic Frame {i}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)

        writer.release()

    def test_01_video_reader_metadata(self):
        """Verifies VideoReader opens video, reads metadata, and yields frames sequentially."""
        reader = VideoReader(self.synthetic_video_path)
        metadata = reader.metadata

        self.assertTrue(metadata.is_valid)
        self.assertEqual(metadata.width, 640)
        self.assertEqual(metadata.height, 480)
        self.assertEqual(metadata.fps, 24.0)
        self.assertEqual(metadata.total_frames, 24)
        self.assertAlmostEqual(metadata.duration_seconds, 1.0, places=1)

        # Test streaming generator
        frames = list(reader.read_frames())
        self.assertEqual(len(frames), 24)
        idx, ts, frame = frames[0]
        self.assertEqual(idx, 0)
        self.assertEqual(frame.shape, (480, 640, 3))
        reader.release()

    def test_02_frame_extractor(self):
        """Verifies FrameExtractor subsamples frames and saves metadata manifest."""
        out_dir = self.temp_dir / "extracted_test"
        extractor = FrameExtractor(
            output_dir=out_dir,
            method="frame_step",
            frame_step=4,
            image_format="png",
            overwrite=True,
        )

        reader = VideoReader(self.synthetic_video_path)
        records = extractor.extract_from_reader(reader, show_progress=False)
        reader.release()

        # 24 frames total with step 4 -> 6 extracted frames (0, 4, 8, 12, 16, 20)
        self.assertEqual(len(records), 6)
        self.assertTrue((out_dir / "extraction_metadata.json").exists())
        self.assertTrue((out_dir / records[0].filename).exists())

    def test_03_blur_detection(self):
        """Verifies Laplacian variance detects difference between sharp and blurred images."""
        assessor = ImageQualityAssessor(blur_threshold=100.0)

        # Create high-contrast sharp checkerboard pattern
        sharp = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(0, 480, 20):
            for x in range(0, 640, 20):
                if ((x // 20) + (y // 20)) % 2 == 0:
                    sharp[y:y+20, x:x+20] = 255

        # Create heavily blurred version using 31x31 Gaussian kernel
        blurred = cv2.GaussianBlur(sharp, (31, 31), 10.0)

        sharp_metrics = assessor.evaluate(sharp)
        blurred_metrics = assessor.evaluate(blurred)

        self.assertFalse(sharp_metrics.is_blurry)
        self.assertTrue(blurred_metrics.is_blurry)
        self.assertGreater(sharp_metrics.blur_score, blurred_metrics.blur_score * 5.0)

    def test_04_brightness_and_contrast_assessment(self):
        """Verifies detection of extreme exposure and low contrast."""
        assessor = ImageQualityAssessor(min_brightness=40.0, max_brightness=220.0, min_contrast=20.0)

        # Severely underexposed image (black)
        dark_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dark_metrics = assessor.evaluate(dark_img)
        self.assertTrue(dark_metrics.is_underexposed)
        self.assertTrue(dark_metrics.is_low_contrast)
        self.assertFalse(dark_metrics.quality_passed)

        # Severely overexposed image (washed out white)
        bright_img = np.ones((480, 640, 3), dtype=np.uint8) * 245
        bright_metrics = assessor.evaluate(bright_img)
        self.assertTrue(bright_metrics.is_overexposed)
        self.assertFalse(bright_metrics.quality_passed)

        # Balanced image with contrast
        normal_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.circle(normal_img, (320, 240), 100, (255, 255, 255), -1)
        normal_metrics = assessor.evaluate(normal_img)
        self.assertFalse(normal_metrics.is_underexposed)
        self.assertFalse(normal_metrics.is_overexposed)

    def test_05_duplicate_and_near_duplicate_filter(self):
        """Verifies exact and perceptual near-duplicate detection."""
        ff = FrameFilter(max_hamming_distance=3)

        img1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        cv2.rectangle(img1, (50, 50), (200, 200), (0, 255, 0), -1)
        img1_copy = img1.copy()

        # Slight noise that shouldn't change the 64-bit dHash
        img1_noisy = img1.copy()
        img1_noisy[0:5, 0:5] = 102

        # Completely different scene
        img2 = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.circle(img2, (400, 300), 120, (0, 0, 255), -1)

        comp_exact = ff.compare_frames(img1, img1_copy)
        self.assertTrue(comp_exact.exact_duplicate)
        self.assertTrue(comp_exact.near_duplicate)
        self.assertEqual(comp_exact.perceptual_distance, 0)

        comp_noisy = ff.compare_frames(img1, img1_noisy)
        self.assertFalse(comp_noisy.exact_duplicate)
        self.assertTrue(comp_noisy.near_duplicate)

        comp_diff = ff.compare_frames(img1, img2)
        self.assertFalse(comp_diff.near_duplicate)
        self.assertGreater(comp_diff.perceptual_distance, 10)

    def test_06_keyframe_selector(self):
        """Verifies that keyframe selector accepts sharp diverse frames and rejects blurred/static ones."""
        selector = KeyframeSelector(
            min_time_delta_seconds=0.1,
            max_time_gap_seconds=2.0,
        )

        # Create balanced, textured sharp frame (gray background with crisp lines)
        sharp_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        for y in range(0, 480, 30):
            cv2.line(sharp_frame, (0, y), (640, y), (255, 255, 255), 3)
        for x in range(0, 640, 30):
            cv2.line(sharp_frame, (x, 0), (x, 480), (50, 50, 50), 3)

        blurred_frame = cv2.GaussianBlur(sharp_frame, (35, 35), 10.0)

        candidates = [
            (0, 0.0, sharp_frame),          # Anchor (Accept)
            (1, 0.05, sharp_frame),         # Too fast temporal delta (Reject)
            (2, 0.3, sharp_frame.copy()),   # Near duplicate of anchor (Reject)
            (3, 0.6, blurred_frame),        # Blur failure (Reject)
        ]

        selected_indices, decisions = selector.process_sequence(candidates)
        self.assertIn(0, selected_indices)
        self.assertNotIn(1, selected_indices)
        self.assertNotIn(2, selected_indices)
        self.assertNotIn(3, selected_indices)

    def test_07_camera_calibration_infrastructure(self):
        """Verifies CameraCalibrator error handling and synthetic pattern calibration."""
        calibrator = CameraCalibrator(checkerboard_size=(7, 5), square_size_mm=30.0)

        # 1. Test error handling when no images provided
        with self.assertRaises(FileNotFoundError):
            calibrator.calibrate_from_images([])

        # 2. Test placeholder generation
        placeholder = CameraCalibrator.get_uncalibrated_placeholder()
        self.assertEqual(placeholder.status, "uncalibrated")
        self.assertEqual(placeholder.fx, 0.0)

        # 3. Test synthetic calibration pattern generation and calibration
        calib_dir = self.temp_dir / "synthetic_calib"
        calib_dir.mkdir(exist_ok=True)
        synthetic_paths = []

        # Generate 8 synthetic images with rotation and perspective tilt (avoids fronto-parallel degeneracy)
        for i in range(8):
            board = np.ones((600, 800), dtype=np.uint8) * 255
            for r in range(6):
                for c in range(8):
                    if (r + c) % 2 == 0:
                        board[100 + r * 50 : 100 + (r + 1) * 50, 100 + c * 50 : 100 + (c + 1) * 50] = 0

            # Apply rotation and slight perspective tilt simulating varied viewpoints
            angle = (i - 3.5) * 5.0
            scale = 0.85 + 0.03 * i
            M_rot = cv2.getRotationMatrix2D((400, 300), angle, scale)
            warped = cv2.warpAffine(board, M_rot, (800, 600), borderValue=255)

            src_pts = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])
            dst_pts = np.float32([[0 + i*4, 0], [800 - i*4, 0 + i*3], [800 - i*2, 600], [0 + i*2, 600 - i*3]])
            M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(warped, M_persp, (800, 600), borderValue=255)

            bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            img_p = calib_dir / f"calib_{i:02d}.png"
            cv2.imwrite(str(img_p), bgr)
            synthetic_paths.append(img_p)

        # Calibrate from synthetic images
        result = calibrator.calibrate_from_images(synthetic_paths)
        self.assertEqual(result.status, "calibrated")
        self.assertGreater(result.fx, 0.0)
        self.assertGreater(result.fy, 0.0)
        self.assertGreaterEqual(result.images_used, 5)


if __name__ == "__main__":
    unittest.main()
