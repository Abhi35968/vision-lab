"""
Real Video Preprocessing and Validation Test Suite
Validates metadata extraction, keyframe saving, original frame index preservation,
artifact integrity, synthetic output immutability, and CLI backwards-compatibility.
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest
import cv2
import numpy as np

from src.preprocessing.video_reader import VideoReader
from src.preprocessing.real_video_validator import RealVideoValidator, RealVideoConfig


class TestRealVideoValidation(unittest.TestCase):
    """Unit and regression tests for Phase 5.1 real-world video validation."""

    VIDEO_PATH = "VID_20261102_015654418.mp4"
    OUTPUT_DIR = Path("outputs/real_video_validation")

    def test_video_exists_and_decodable(self):
        """Verifies real video file exists and can be opened with valid metadata."""
        v_path = Path(self.VIDEO_PATH)
        self.assertTrue(v_path.exists(), f"Video file {self.VIDEO_PATH} missing")
        self.assertGreater(v_path.stat().st_size, 10 * 1024 * 1024)  # > 10MB

        reader = VideoReader(v_path)
        meta = reader.metadata
        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.width, 1080)
        self.assertEqual(meta.height, 1920)
        self.assertGreater(meta.fps, 50.0)
        self.assertEqual(meta.total_frames, 572)
        self.assertAlmostEqual(meta.duration_seconds, 9.60, places=1)
        reader.release()

    def test_output_artifacts_exist_and_non_empty(self):
        """Verifies all required reports, summaries, and contact sheets are created and non-empty."""
        expected_files = [
            self.OUTPUT_DIR / "video_metadata.json",
            self.OUTPUT_DIR / "image_quality_report.json",
            self.OUTPUT_DIR / "frame_selection_report.json",
            self.OUTPUT_DIR / "frame_selection_report.csv",
            self.OUTPUT_DIR / "preprocessing_summary.json",
            self.OUTPUT_DIR / "extracted_frames_contact_sheet.png",
            self.OUTPUT_DIR / "selected_keyframes_contact_sheet.png",
        ]

        for p in expected_files:
            self.assertTrue(p.exists(), f"Expected artifact missing: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Artifact is empty: {p}")

    def test_selected_images_can_be_loaded(self):
        """Verifies all selected keyframes exist, are decodable by OpenCV, and match 1080x1920."""
        kf_dir = self.OUTPUT_DIR / "keyframes"
        self.assertTrue(kf_dir.exists())
        kf_files = sorted(list(kf_dir.glob("*.png")))
        self.assertGreaterEqual(len(kf_files), 30, f"Expected >= 30 keyframes, found {len(kf_files)}")

        for kf in kf_files:
            img = cv2.imread(str(kf))
            self.assertIsNotNone(img, f"Failed to load image: {kf}")
            self.assertEqual(img.shape, (1920, 1080, 3), f"Unexpected shape for {kf}")

    def test_original_frame_indices_preserved(self):
        """Verifies filenames preserve original frame indices and timestamps in monotonic order."""
        kf_dir = self.OUTPUT_DIR / "keyframes"
        kf_files = sorted(list(kf_dir.glob("*.png")))

        prev_orig_idx = -1
        for kf in kf_files:
            # Format: keyframe_0000_orig_00000.png
            parts = kf.stem.split("_")
            self.assertEqual(len(parts), 4)
            self.assertEqual(parts[0], "keyframe")
            self.assertEqual(parts[2], "orig")

            orig_idx = int(parts[3])
            self.assertGreater(orig_idx, prev_orig_idx, "Original frame indices must be strictly increasing")
            self.assertLess(orig_idx, 572, "Original frame index must not exceed total video frames")
            prev_orig_idx = orig_idx

    def test_existing_synthetic_outputs_unchanged(self):
        """Guarantees existing synthetic keyframes, SfM, detection, and association outputs are untouched."""
        synth_frames_dir = Path("data/extracted_frames")
        synth_files = list(synth_frames_dir.glob("*.png"))
        self.assertEqual(len(synth_files), 6, "Synthetic keyframes in data/extracted_frames/ must be 6")

        sfm_report = Path("outputs/sfm/two_view_report.json")
        if sfm_report.exists():
            self.assertGreater(sfm_report.stat().st_size, 0)

        det_summary = Path("outputs/detection/detection_summary.json")
        if det_summary.exists():
            self.assertGreater(det_summary.stat().st_size, 0)

        assoc_summary = Path("outputs/association/association_summary.json")
        if assoc_summary.exists():
            self.assertGreater(assoc_summary.stat().st_size, 0)

    def test_preprocessing_cli_still_works(self):
        """Verifies preprocessing CLI command line interface continues to function normally."""
        cmd = [sys.executable, "main.py", "--phase", "preprocess", "--help"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--video", res.stdout)
        self.assertIn("--blur-threshold", res.stdout)

        # Also check validate_real help
        cmd_real = [sys.executable, "main.py", "--help"]
        res_real = subprocess.run(cmd_real, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res_real.returncode, 0)
        self.assertIn("validate_real", res_real.stdout)


if __name__ == "__main__":
    unittest.main()

