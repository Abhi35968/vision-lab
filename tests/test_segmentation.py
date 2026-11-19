"""
Unit Tests for Phase 5.3.1 Object Segmentation Module
Validates dataclasses, mask processor, instance tracker, YOLO segmenter, and visualization tools.
"""

from pathlib import Path
import unittest
import numpy as np

from src.segmentation.segmentation_models import (
    InstanceMask,
    FrameSegmentation,
    SegmentationSummary,
)
from src.segmentation.mask_processor import MaskProcessor
from src.segmentation.instance_tracker import InstanceTracker, InstanceTrackerConfig
from src.segmentation.segmenter import YOLOSegmenter, SegmenterConfig


class TestSegmentationModels(unittest.TestCase):
    """Tests segmentation dataclasses, geometry properties, and serialization."""

    def setUp(self):
        self.instance = InstanceMask(
            instance_id=0,
            camera_id=1,
            image_filename="keyframe_0001.png",
            class_id=39,
            class_name="bottle",
            confidence=0.8543,
            bbox=(100.0, 200.0, 400.0, 1200.0),  # w=300, h=1000
            mask_area_px=250000,
            mask_area_pct=12.057,
            centroid=(250.0, 700.0),
            touches_border=False,
            num_components=1,
            largest_component_ratio=1.0,
            is_target=True,
            is_ambiguous=False,
            tracking_score=0.912,
        )

    def test_instance_mask_properties(self):
        self.assertEqual(self.instance.width, 300.0)
        self.assertEqual(self.instance.height, 1000.0)
        self.assertAlmostEqual(self.instance.aspect_ratio, 1000.0 / 300.0, places=2)

    def test_instance_mask_serialization(self):
        d = self.instance.to_dict()
        self.assertEqual(d["instance_id"], 0)
        self.assertEqual(d["class_name"], "bottle")
        self.assertEqual(d["confidence"], 0.8543)
        self.assertEqual(d["bbox_xyxy"], [100.0, 200.0, 400.0, 1200.0])
        self.assertTrue(d["is_target"])
        self.assertFalse(d["is_ambiguous"])

    def test_frame_segmentation_serialization(self):
        frame_seg = FrameSegmentation(
            frame_idx=1,
            image_name="keyframe_0001.png",
            image_path="outputs/keyframes/keyframe_0001.png",
            image_width=1080,
            image_height=1920,
            is_registered_sfm=True,
            target_detected=True,
            target_status="VALID",
            target_instance=self.instance,
            all_instances=[self.instance],
            inference_time_ms=95.4,
            mask_path="outputs/masks/keyframe_0001_mask.png",
        )
        d = frame_seg.to_dict()
        self.assertEqual(d["frame_idx"], 1)
        self.assertEqual(d["target_status"], "VALID")
        self.assertEqual(d["resolution"], [1080, 1920])
        self.assertIsNotNone(d["target_instance"])

        row = frame_seg.to_csv_row()
        self.assertEqual(row["frame_idx"], 1)
        self.assertEqual(row["target_status"], "VALID")
        self.assertEqual(row["target_class"], "bottle")
        self.assertEqual(row["is_registered_sfm"], 1)

    def test_segmentation_summary_serialization(self):
        summary = SegmentationSummary(
            total_frames=47,
            sfm_registered_frames=32,
            valid_target_frames=42,
            sfm_valid_target_frames=30,
            ambiguous_frames=2,
            no_target_frames=3,
            excluded_frames=0,
            mean_inference_time_ms=105.2,
            mean_target_mask_area_pct=15.3,
            mean_target_confidence=0.68,
            border_touching_count=4,
            target_classes_detected={"bottle": 25, "vase": 17},
        )
        d = summary.to_dict()
        self.assertEqual(d["total_frames"], 47)
        self.assertAlmostEqual(d["target_coverage_overall_pct"], 89.36, places=1)
        self.assertAlmostEqual(d["target_coverage_sfm_pct"], 93.75, places=1)


class TestMaskProcessor(unittest.TestCase):
    """Tests image mask processing, resizing, morphology, and geometric diagnostics."""

    def test_resize_to_target_resolution(self):
        raw_mask = np.zeros((64, 36), dtype=np.float32)
        raw_mask[10:50, 10:30] = 0.9  # Positive block

        target_shape = (1920, 1080)
        resized = MaskProcessor.resize_to_target_resolution(raw_mask, target_shape, threshold=0.5)

        self.assertEqual(resized.shape, target_shape)
        self.assertEqual(resized.dtype, np.uint8)
        unique_vals = set(np.unique(resized))
        self.assertTrue(unique_vals.issubset({0, 255}))
        self.assertIn(255, unique_vals)

    def test_check_touches_border(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        self.assertFalse(MaskProcessor.check_touches_border(mask, margin=2))

        # Top border
        mask_top = mask.copy()
        mask_top[0, 50] = 255
        self.assertTrue(MaskProcessor.check_touches_border(mask_top, margin=2))

        # Left border
        mask_left = mask.copy()
        mask_left[50, 0] = 255
        self.assertTrue(MaskProcessor.check_touches_border(mask_left, margin=2))

    def test_analyze_connected_components(self):
        # Empty mask
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        num_c, ratio = MaskProcessor.analyze_connected_components(empty_mask)
        self.assertEqual(num_c, 0)
        self.assertEqual(ratio, 0.0)

        # Single component
        single_mask = np.zeros((100, 100), dtype=np.uint8)
        single_mask[10:30, 10:30] = 255
        num_c, ratio = MaskProcessor.analyze_connected_components(single_mask)
        self.assertEqual(num_c, 1)
        self.assertEqual(ratio, 1.0)

        # Two disjoint components: one 20x20=400px, one 10x10=100px. Total 500px.
        dual_mask = single_mask.copy()
        dual_mask[70:80, 70:80] = 255
        num_c, ratio = MaskProcessor.analyze_connected_components(dual_mask)
        self.assertEqual(num_c, 2)
        self.assertAlmostEqual(ratio, 400.0 / 500.0, places=2)

    def test_compute_mask_centroid_and_bbox(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 60:160] = 255  # Box from x=[60, 159], y=[50, 149]

        cx, cy = MaskProcessor.compute_mask_centroid(mask)
        self.assertAlmostEqual(cx, 109.5, delta=1.5)
        self.assertAlmostEqual(cy, 99.5, delta=1.5)

        bbox = MaskProcessor.compute_mask_bbox(mask)
        self.assertIsNotNone(bbox)
        x1, y1, x2, y2 = bbox
        self.assertEqual(x1, 60.0)
        self.assertEqual(y1, 50.0)
        self.assertEqual(x2, 159.0)
        self.assertEqual(y2, 149.0)

    def test_compute_iou(self):
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_b = np.zeros((100, 100), dtype=np.uint8)

        # Identical
        mask_a[20:60, 20:60] = 255
        mask_b[20:60, 20:60] = 255
        self.assertEqual(MaskProcessor.compute_mask_iou(mask_a, mask_b), 1.0)

        # Disjoint
        mask_c = np.zeros((100, 100), dtype=np.uint8)
        mask_c[70:90, 70:90] = 255
        self.assertEqual(MaskProcessor.compute_mask_iou(mask_a, mask_c), 0.0)

        # BBox IoU
        b1 = (0.0, 0.0, 10.0, 10.0)  # Area = 100
        b2 = (5.0, 0.0, 15.0, 10.0)  # Area = 100, Inter = 50, Union = 150 -> 0.333
        self.assertAlmostEqual(MaskProcessor.compute_bbox_iou(b1, b2), 50.0 / 150.0, places=3)


class TestInstanceTracker(unittest.TestCase):
    """Tests target instance tracking, class aliasing resolution, and ambiguity detection."""

    def setUp(self):
        self.tracker = InstanceTracker(config=InstanceTrackerConfig())
        self.frame_shape = (1920, 1080)

    def test_filter_and_deduplicate_aliased_classes(self):
        # Create dual detection on same physical object: vase (0.65) and bottle (0.50)
        inst_vase = InstanceMask(
            instance_id=0, camera_id=0, image_filename="f0.png",
            class_id=75, class_name="vase", confidence=0.65,
            bbox=(200.0, 300.0, 600.0, 1600.0),  # w=400, h=1300, ar=3.25
            mask_area_px=400000, mask_area_pct=19.3,
            centroid=(400.0, 950.0), touches_border=False,
            num_components=1, largest_component_ratio=1.0,
        )
        inst_bottle = InstanceMask(
            instance_id=1, camera_id=0, image_filename="f0.png",
            class_id=39, class_name="bottle", confidence=0.50,
            bbox=(205.0, 305.0, 595.0, 1595.0),  # Almost identical bbox
            mask_area_px=390000, mask_area_pct=18.8,
            centroid=(400.0, 950.0), touches_border=False,
            num_components=1, largest_component_ratio=1.0,
        )
        inst_book = InstanceMask(
            instance_id=2, camera_id=0, image_filename="f0.png",
            class_id=73, class_name="book", confidence=0.75,
            bbox=(700.0, 800.0, 950.0, 1100.0),
            mask_area_px=50000, mask_area_pct=2.4,
            centroid=(825.0, 950.0), touches_border=False,
            num_components=1, largest_component_ratio=1.0,
        )

        filtered = self.tracker.filter_and_deduplicate_candidates(
            [inst_vase, inst_bottle, inst_book], self.frame_shape
        )

        # Book should be filtered out, and vase+bottle merged into single candidate
        self.assertEqual(len(filtered), 1)
        # Preferred class 'bottle' is chosen because confidence is within margin
        self.assertEqual(filtered[0].class_name, "bottle")

    def test_tracking_continuity_and_status(self):
        # Frame 0
        cand0 = InstanceMask(
            instance_id=0, camera_id=0, image_filename="f0.png",
            class_id=39, class_name="bottle", confidence=0.70,
            bbox=(200.0, 300.0, 600.0, 1600.0),
            mask_area_px=400000, mask_area_pct=19.3,
            centroid=(400.0, 950.0), touches_border=False,
            num_components=1, largest_component_ratio=1.0,
        )
        target0, status0, _ = self.tracker.track_frame(
            frame_idx=0, all_instances=[cand0], frame_shape=self.frame_shape
        )
        self.assertIsNotNone(target0)
        self.assertEqual(status0, "VALID")
        self.assertTrue(target0.is_target)

        # Frame 1: slight camera movement (continuous)
        cand1 = InstanceMask(
            instance_id=0, camera_id=1, image_filename="f1.png",
            class_id=75, class_name="vase", confidence=0.68,
            bbox=(220.0, 310.0, 620.0, 1610.0),
            mask_area_px=400000, mask_area_pct=19.3,
            centroid=(420.0, 960.0), touches_border=False,
            num_components=1, largest_component_ratio=1.0,
        )
        target1, status1, _ = self.tracker.track_frame(
            frame_idx=1, all_instances=[cand1], frame_shape=self.frame_shape
        )
        self.assertIsNotNone(target1)
        self.assertEqual(status1, "VALID")
        self.assertGreater(target1.tracking_score, 0.6)

    def test_ambiguity_detection(self):
        # Two spatially disjoint candidate containers with identical high scores
        cand_a = InstanceMask(
            instance_id=0, camera_id=0, image_filename="f0.png",
            class_id=39, class_name="bottle", confidence=0.60,
            bbox=(50.0, 300.0, 450.0, 1600.0),  # Left
            mask_area_px=400000, mask_area_pct=19.3,
            centroid=(250.0, 950.0), touches_border=False,
            num_components=1, largest_component_ratio=1.0,
        )
        cand_b = InstanceMask(
            instance_id=1, camera_id=0, image_filename="f0.png",
            class_id=39, class_name="bottle", confidence=0.60,
            bbox=(600.0, 300.0, 1000.0, 1600.0),  # Right
            mask_area_px=400000, mask_area_pct=19.3,
            centroid=(800.0, 950.0), touches_border=False,
            num_components=1, largest_component_ratio=1.0,
        )
        target, status, _ = self.tracker.track_frame(
            frame_idx=0, all_instances=[cand_a, cand_b], frame_shape=self.frame_shape
        )
        self.assertEqual(status, "AMBIGUOUS")
        self.assertTrue(cand_a.is_ambiguous or cand_b.is_ambiguous)


class TestYOLOSegmenter(unittest.TestCase):
    """Tests YOLOSegmenter model loading and CPU inference execution."""

    @classmethod
    def setUpClass(cls):
        cls.weights_path = Path("yolo11n-seg.pt")
        if not cls.weights_path.exists():
            raise unittest.SkipTest("yolo11n-seg.pt not found.")
        cls.config = SegmenterConfig(
            model_weights=str(cls.weights_path),
            conf_threshold=0.15,
            device="cpu",
            imgsz=640,
        )
        cls.segmenter = YOLOSegmenter(cls.config)

    def test_segment_frame_synthetic(self):
        # Synthetic RGB canvas 1080x1920
        synthetic_img = np.zeros((1920, 1080, 3), dtype=np.uint8)
        # Draw a synthetic bright cylinder/rectangle
        synthetic_img[300:1500, 300:700] = (200, 200, 200)

        frame_seg, target_mask = self.segmenter.segment_frame(
            image_input=synthetic_img,
            camera_id=0,
            frame_idx=0,
            image_name="test_synth.png",
            is_registered_sfm=True,
        )

        self.assertEqual(frame_seg.image_width, 1080)
        self.assertEqual(frame_seg.image_height, 1920)
        self.assertGreater(frame_seg.inference_time_ms, 0.0)
        self.assertIn(frame_seg.target_status, ["VALID", "AMBIGUOUS", "NO_TARGET"])


if __name__ == "__main__":
    unittest.main()

