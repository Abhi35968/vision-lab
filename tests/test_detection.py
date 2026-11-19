"""
2D Object Detection Unit and Integration Test Suite
Validates detection data models, coordinate computations, serialization,
summary aggregations, visual annotations, error handling, and mocked CPU inference.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.detection.detection_models import (
    Detection,
    ImageDetections,
    DetectionSummary,
)
from src.detection.detection_visualizer import DetectionVisualizer
from src.detection.yolo_detector import (
    YOLODetector,
    YOLOConfig,
)


class TestDetectionModule(unittest.TestCase):
    """Rigorous unit tests for Phase 4.1 YOLO 2D Object Detection."""

    def test_detection_from_xyxy_and_properties(self):
        """Verifies bounding box center, width, height computation and unordered bounds handling."""
        # Standard ordered coords
        det = Detection.from_xyxy(
            detection_id=1,
            camera_id=0,
            image_filename="frame_0.png",
            class_id=56,
            class_name="chair",
            confidence=0.87654,
            x1=100.0,
            y1=50.0,
            x2=200.0,
            y2=250.0,
        )

        self.assertEqual(det.detection_id, 1)
        self.assertEqual(det.camera_id, 0)
        self.assertEqual(det.class_name, "chair")
        self.assertEqual(det.confidence, 0.8765)
        self.assertEqual(det.width, 100.0)
        self.assertEqual(det.height, 200.0)
        self.assertEqual(det.center_x, 150.0)
        self.assertEqual(det.center_y, 150.0)
        self.assertEqual(det.bbox_xyxy, (100.0, 50.0, 200.0, 250.0))
        self.assertEqual(det.bbox_center, (150.0, 150.0))
        self.assertEqual(det.bbox_dimensions, (100.0, 200.0))

        # Inverted coords (x1 > x2, y1 > y2)
        det_inv = Detection.from_xyxy(
            detection_id=2,
            camera_id=1,
            image_filename="frame_1.png",
            class_id=56,
            class_name="chair",
            confidence=0.9,
            x1=200.0,
            y1=250.0,
            x2=100.0,
            y2=50.0,
        )
        self.assertEqual(det_inv.x1, 100.0)
        self.assertEqual(det_inv.x2, 200.0)
        self.assertEqual(det_inv.y1, 50.0)
        self.assertEqual(det_inv.y2, 250.0)

    def test_detection_serialization_json_and_csv(self):
        """Verifies JSON dictionary schema and CSV line formatting."""
        det = Detection.from_xyxy(
            detection_id=0,
            camera_id=2,
            image_filename="path/to/keyframe_0002.png",
            class_id=60,
            class_name="dining table",
            confidence=0.9234,
            x1=50.5,
            y1=60.5,
            x2=250.5,
            y2=360.5,
        )

        d_dict = det.to_dict()
        self.assertEqual(d_dict["detection_id"], 0)
        self.assertEqual(d_dict["camera_id"], 2)
        self.assertEqual(d_dict["image_filename"], "keyframe_0002.png")
        self.assertEqual(d_dict["class_name"], "dining table")
        self.assertEqual(d_dict["bbox_xyxy"], [50.5, 60.5, 250.5, 360.5])
        self.assertEqual(d_dict["bbox_center"], [150.5, 210.5])
        self.assertEqual(d_dict["bbox_dimensions"], [200.0, 300.0])

        csv_row = det.to_csv_row()
        parts = csv_row.strip().split(",")
        self.assertEqual(parts[0], "0")                 # detection_id
        self.assertEqual(parts[1], "2")                 # camera_id
        self.assertEqual(parts[2], "keyframe_0002.png") # image_filename
        self.assertEqual(parts[3], "60")                # class_id
        self.assertEqual(parts[4], "dining table")      # class_name
        self.assertAlmostEqual(float(parts[5]), 0.9234) # confidence
        self.assertAlmostEqual(float(parts[6]), 50.5)   # x1
        self.assertAlmostEqual(float(parts[7]), 60.5)   # y1
        self.assertAlmostEqual(float(parts[8]), 250.5)  # x2
        self.assertAlmostEqual(float(parts[9]), 360.5)  # y2
        self.assertAlmostEqual(float(parts[10]), 150.5) # center_x
        self.assertAlmostEqual(float(parts[11]), 210.5) # center_y
        self.assertAlmostEqual(float(parts[12]), 200.0) # width
        self.assertAlmostEqual(float(parts[13]), 300.0) # height

    def test_image_detections_container(self):
        """Verifies ImageDetections container for non-empty and empty cases."""
        # Non-empty
        det = Detection.from_xyxy(0, 0, "f0.png", 0, "person", 0.95, 10, 20, 50, 100)
        img_dets = ImageDetections("f0.png", 0, 1280, 720, 42.5, [det])
        self.assertEqual(img_dets.detection_count, 1)
        self.assertEqual(img_dets.to_dict()["image_dimensions"], [1280, 720])

        # Empty
        empty_dets = ImageDetections("f1.png", 1, 1280, 720, 35.0, [])
        self.assertEqual(empty_dets.detection_count, 0)
        self.assertEqual(empty_dets.to_dict()["detections"], [])

    def test_empty_detection_results_summary(self):
        """Verifies summary statistics when all images yield zero detections."""
        img_dets = [
            ImageDetections("f0.png", 0, 640, 480, 30.0, []),
            ImageDetections("f1.png", 1, 640, 480, 28.0, []),
        ]
        times = {"f0.png": 30.0, "f1.png": 28.0}
        no_dets = ["f0.png", "f1.png"]

        detector = YOLODetector.__new__(YOLODetector)
        detector.config = YOLOConfig(model_weights="mock.pt", conf_threshold=0.25, iou_threshold=0.45, imgsz=640, device="cpu")

        summary = detector.compute_summary(img_dets, times, no_dets, total_inference_sec=0.058)
        self.assertEqual(summary.total_images_processed, 2)
        self.assertEqual(summary.total_detections, 0)
        self.assertEqual(summary.unique_object_classes, [])
        self.assertEqual(summary.detection_count_per_class, {})
        self.assertEqual(summary.average_confidence_per_class, {})
        self.assertEqual(summary.images_with_no_detections, ["f0.png", "f1.png"])

    def test_class_aggregation_in_summary(self):
        """Verifies multi-class frequency and average confidence aggregations in DetectionSummary."""
        d1 = Detection.from_xyxy(0, 0, "f0.png", 0, "chair", 0.80, 10, 10, 50, 50)
        d2 = Detection.from_xyxy(1, 0, "f0.png", 0, "chair", 0.90, 60, 10, 100, 50)
        d3 = Detection.from_xyxy(2, 1, "f1.png", 0, "chair", 0.70, 20, 20, 70, 70)
        d4 = Detection.from_xyxy(3, 1, "f1.png", 1, "table", 0.60, 100, 100, 200, 200)
        d5 = Detection.from_xyxy(4, 2, "f2.png", 1, "table", 0.80, 110, 110, 210, 210)

        img_dets = [
            ImageDetections("f0.png", 0, 640, 480, 30.0, [d1, d2]),
            ImageDetections("f1.png", 1, 640, 480, 25.0, [d3, d4]),
            ImageDetections("f2.png", 2, 640, 480, 22.0, [d5]),
            ImageDetections("f3.png", 3, 640, 480, 20.0, []),
        ]
        times = {"f0.png": 30.0, "f1.png": 25.0, "f2.png": 22.0, "f3.png": 20.0}
        no_dets = ["f3.png"]

        detector = YOLODetector.__new__(YOLODetector)
        detector.config = YOLOConfig(model_weights="mock.pt", conf_threshold=0.25, iou_threshold=0.45, imgsz=640, device="cpu")

        summary = detector.compute_summary(img_dets, times, no_dets, total_inference_sec=0.097)
        self.assertEqual(summary.total_images_processed, 4)
        self.assertEqual(summary.total_detections, 5)
        self.assertEqual(summary.unique_object_classes, ["chair", "table"])
        self.assertEqual(summary.detection_count_per_class, {"chair": 3, "table": 2})
        self.assertAlmostEqual(summary.average_confidence_per_class["chair"], 0.80, places=4)
        self.assertAlmostEqual(summary.average_confidence_per_class["table"], 0.70, places=4)
        self.assertEqual(summary.images_with_no_detections, ["f3.png"])

    def test_visualizer_annotation(self):
        """Verifies bounding box drawing on OpenCV canvas."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        det = Detection.from_xyxy(0, 0, "test.png", 1, "chair", 0.85, 20, 20, 80, 80)

        vis = DetectionVisualizer()
        annotated = vis.annotate_frame(canvas, [det])

        self.assertEqual(annotated.shape, (200, 200, 3))
        # Ensure canvas is no longer completely zero
        self.assertGreater(int(np.sum(annotated)), 0)

    def test_missing_image_directory_and_invalid_inputs(self):
        """Verifies error handling on non-existent or empty directories."""
        detector = YOLODetector.__new__(YOLODetector)
        detector.config = YOLOConfig()

        # 1. Non-existent directory
        with self.assertRaises(FileNotFoundError):
            detector.detect_directory("data/totally_missing_dir_12345")

        # 2. Path is a file, not a directory
        with self.assertRaises(ValueError):
            detector.detect_directory("main.py")

        # 3. Empty directory
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(ValueError):
                detector.detect_directory(empty_dir)

    def test_invalid_model_weights(self):
        """Verifies exception handling when invalid model weights are provided."""
        with self.assertRaises(ValueError):
            YOLODetector(config=YOLOConfig(model_weights=""))

        with self.assertRaises(RuntimeError):
            YOLODetector(config=YOLOConfig(model_weights="non_existent_model_weights_xyz123.pt"))

    def test_camera_id_parsing(self):
        """Verifies regex extraction of camera/keyframe index from standard naming patterns."""
        self.assertEqual(YOLODetector.parse_camera_id("keyframe_0003_orig_00046.png"), 3)
        self.assertEqual(YOLODetector.parse_camera_id("keyframe_0000_orig_00000.png"), 0)
        self.assertEqual(YOLODetector.parse_camera_id("frame_04.jpg"), 4)
        self.assertEqual(YOLODetector.parse_camera_id("cam_12.png"), 12)
        self.assertEqual(YOLODetector.parse_camera_id("arbitrary_scene_name.png", fallback_index=7), 7)

    @patch("src.detection.yolo_detector.YOLO")
    def test_cpu_configuration_and_mocked_inference(self, mock_yolo_cls):
        """Verifies CPU inference configuration and bounding box parsing using a mocked YOLO model."""
        mock_instance = MagicMock()
        mock_yolo_cls.return_value = mock_instance
        mock_instance.names = {0: "person", 56: "chair"}

        # Create mock prediction result
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 56)]
        mock_box.conf = [MagicMock(item=lambda: 0.88)]
        mock_box.xyxy = [np.array([50.0, 60.0, 150.0, 200.0])]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_instance.predict.return_value = [mock_result]

        config = YOLOConfig(model_weights="mock_yolo.pt", device="cpu", conf_threshold=0.3, iou_threshold=0.5, imgsz=640)
        detector = YOLODetector(config=config)

        # Create synthetic test image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(tmp_path, img)

        try:
            img_dets = detector.detect_image(tmp_path, camera_id=1)
            self.assertEqual(img_dets.camera_id, 1)
            self.assertEqual(img_dets.detection_count, 1)
            det = img_dets.detections[0]
            self.assertEqual(det.class_name, "chair")
            self.assertEqual(det.confidence, 0.88)
            self.assertEqual(det.bbox_xyxy, (50.0, 60.0, 150.0, 200.0))

            # Verify that predict was called with device="cpu"
            mock_instance.predict.assert_called_once()
            call_kwargs = mock_instance.predict.call_args[1]
            self.assertEqual(call_kwargs["device"], "cpu")
            self.assertEqual(call_kwargs["conf"], 0.3)
            self.assertEqual(call_kwargs["iou"], 0.5)
            self.assertEqual(call_kwargs["imgsz"], 640)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

