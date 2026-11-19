"""
Detection Data Models Module
Defines structured dataclasses for 2D object detections, per-image detection sets,
and aggregated detection summaries with rigorous coordinate tracking and serialization.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class Detection:
    """Represents a single 2D bounding box detection in original image pixel coordinates."""
    detection_id: int
    camera_id: int
    image_filename: str
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float

    @classmethod
    def from_xyxy(
        cls,
        detection_id: int,
        camera_id: int,
        image_filename: str,
        class_id: int,
        class_name: str,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> "Detection":
        """Constructs a Detection object and automatically computes center, width, and height."""
        # Ensure ordered bounds
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        w = float(x_max - x_min)
        h = float(y_max - y_min)
        cx = float(x_min + w / 2.0)
        cy = float(y_min + h / 2.0)

        return cls(
            detection_id=detection_id,
            camera_id=camera_id,
            image_filename=str(Path(image_filename).name),
            class_id=int(class_id),
            class_name=str(class_name),
            confidence=round(float(confidence), 4),
            x1=round(float(x_min), 2),
            y1=round(float(y_min), 2),
            x2=round(float(x_max), 2),
            y2=round(float(y_max), 2),
            center_x=round(cx, 2),
            center_y=round(cy, 2),
            width=round(w, 2),
            height=round(h, 2),
        )

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """Returns (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def bbox_center(self) -> Tuple[float, float]:
        """Returns (center_x, center_y)."""
        return (self.center_x, self.center_y)

    @property
    def bbox_dimensions(self) -> Tuple[float, float]:
        """Returns (width, height)."""
        return (self.width, self.height)

    def to_dict(self) -> dict:
        """Serializes detection to JSON-compatible dictionary."""
        return {
            "detection_id": self.detection_id,
            "camera_id": self.camera_id,
            "image_filename": self.image_filename,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": [self.x1, self.y1, self.x2, self.y2],
            "bbox_center": [self.center_x, self.center_y],
            "bbox_dimensions": [self.width, self.height],
        }

    def to_csv_row(self) -> str:
        """Formats detection as a single CSV record row."""
        return (
            f"{self.detection_id},{self.camera_id},{self.image_filename},"
            f"{self.class_id},{self.class_name},{self.confidence:.4f},"
            f"{self.x1:.2f},{self.y1:.2f},{self.x2:.2f},{self.y2:.2f},"
            f"{self.center_x:.2f},{self.center_y:.2f},{self.width:.2f},{self.height:.2f}\n"
        )


@dataclass
class ImageDetections:
    """Container for all 2D detections extracted from a single keyframe."""
    image_filename: str
    camera_id: int
    image_width: int
    image_height: int
    inference_time_ms: float
    detections: List[Detection] = field(default_factory=list)

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict:
        return {
            "image_filename": self.image_filename,
            "camera_id": self.camera_id,
            "image_dimensions": [self.image_width, self.image_height],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "detection_count": self.detection_count,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class DetectionSummary:
    """Aggregated statistical summary across an entire image sequence."""
    total_images_processed: int
    total_detections: int
    unique_object_classes: List[str]
    detection_count_per_class: Dict[str, int]
    average_confidence_per_class: Dict[str, float]
    images_with_no_detections: List[str]
    inference_time_per_image: Dict[str, float]
    total_inference_time: float
    model_name: str
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    device: str

    def to_dict(self) -> dict:
        return {
            "model_metadata": {
                "model_name": self.model_name,
                "device": self.device,
                "confidence_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "input_resolution": self.imgsz,
            },
            "summary_metrics": {
                "total_images_processed": self.total_images_processed,
                "total_detections": self.total_detections,
                "unique_object_classes": self.unique_object_classes,
                "unique_class_count": len(self.unique_object_classes),
                "detection_count_per_class": self.detection_count_per_class,
                "average_confidence_per_class": self.average_confidence_per_class,
                "images_with_no_detections": self.images_with_no_detections,
                "no_detection_count": len(self.images_with_no_detections),
            },
            "timing_benchmarks": {
                "total_inference_time_seconds": round(self.total_inference_time, 3),
                "average_inference_time_ms": round(
                    (self.total_inference_time * 1000.0) / max(1, self.total_images_processed), 2
                ),
                "inference_time_per_image_ms": self.inference_time_per_image,
            },
        }

