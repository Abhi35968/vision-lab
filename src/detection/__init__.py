"""
2D Object Detection Package
Implements YOLO-based multi-view object detection, bounding box telemetry, and visual annotation.
"""

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

__all__ = [
    "Detection",
    "ImageDetections",
    "DetectionSummary",
    "DetectionVisualizer",
    "YOLODetector",
    "YOLOConfig",
]

