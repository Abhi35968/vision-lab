"""
YOLO Object Detection Module
Coordinates 2D object detection across single images and keyframe sequences using Ultralytics YOLO.
Exports structured JSON/CSV telemetry, per-class summaries, and annotated visualization frames.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.detection.detection_models import Detection, ImageDetections, DetectionSummary
from src.detection.detection_visualizer import DetectionVisualizer

logger = logging.getLogger("YOLODetector")


@dataclass
class YOLOConfig:
    """Configuration options for YOLO object detection."""
    model_weights: str = "yolo26n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = "cpu"
    classes: Optional[List[int]] = None
    output_dir: str = "outputs/detection"


class YOLODetector:
    """
    Modular 2D object detector powered by Ultralytics YOLO.
    Optimized for CPU-based multi-view keyframe evaluation.
    """

    def __init__(self, config: Optional[YOLOConfig] = None):
        if YOLO is None:
            raise ImportError(
                "Ultralytics YOLO is not installed. Please install it via: pip install ultralytics"
            )

        self.config = config or YOLOConfig()
        self._init_model()
        self.visualizer = DetectionVisualizer()

    def _init_model(self) -> None:
        """Initializes the YOLO model from specified weights with error handling."""
        weights = str(self.config.model_weights).strip()
        if not weights:
            raise ValueError("Model weights name or path cannot be empty.")

        try:
            logger.info("Loading YOLO model: %s on device: %s", weights, self.config.device)
            self.model = YOLO(weights)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model weights '{weights}'. "
                f"Ensure the model name is valid or path exists. Underlying error: {e}"
            ) from e

    @staticmethod
    def parse_camera_id(filename: Union[str, Path], fallback_index: int = 0) -> int:
        """
        Parses camera/keyframe integer ID from standard file naming schemes:
        - 'keyframe_0003_orig_00046.png' -> 3
        - 'frame_02.png' -> 2
        - 'cam_5.jpg' -> 5
        Falls back to sequential index if no pattern is found.
        """
        name = Path(filename).stem.lower()
        patterns = [
            r"keyframe_(\d+)",
            r"frame_(\d+)",
            r"cam_(\d+)",
            r"camera_(\d+)",
            r"view_(\d+)",
            r"img_(\d+)",
        ]
        for pat in patterns:
            match = re.search(pat, name)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        return fallback_index

    def detect_image(
        self,
        image_path: Union[str, Path],
        camera_id: Optional[int] = None,
        detection_id_offset: int = 0,
    ) -> ImageDetections:
        """
        Runs object detection on a single image and maps coordinates to original resolution.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image via OpenCV: {path}")

        h_orig, w_orig = img.shape[:2]
        cam_id = camera_id if camera_id is not None else self.parse_camera_id(path)

        # Run inference on CPU
        t0 = time.time()
        results = self.model.predict(
            source=str(path),
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
            classes=self.config.classes,
            verbose=False,
        )
        t_elapsed_ms = (time.time() - t0) * 1000.0

        detections: List[Detection] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            res = results[0]
            boxes = res.boxes

            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                # Clamp coordinates to original image boundaries
                x1 = max(0.0, min(float(w_orig), float(xyxy[0])))
                y1 = max(0.0, min(float(h_orig), float(xyxy[1])))
                x2 = max(0.0, min(float(w_orig), float(xyxy[2])))
                y2 = max(0.0, min(float(h_orig), float(xyxy[3])))

                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                det = Detection.from_xyxy(
                    detection_id=detection_id_offset + idx,
                    camera_id=cam_id,
                    image_filename=path.name,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
                detections.append(det)

        return ImageDetections(
            image_filename=path.name,
            camera_id=cam_id,
            image_width=w_orig,
            image_height=h_orig,
            inference_time_ms=t_elapsed_ms,
            detections=detections,
        )

    def detect_directory(
        self,
        image_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_annotated: bool = True,
    ) -> Tuple[List[ImageDetections], DetectionSummary]:
        """
        Runs batch object detection across all keyframes in a directory.
        Saves detections.json, detections.csv, detection_summary.json, and annotated frames.
        """
        dir_path = Path(image_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Input image directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Provided path is not a directory: {dir_path}")

        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        image_files = []
        for ext in extensions:
            image_files.extend(list(dir_path.glob(ext)))
        image_files = sorted(list(set(image_files)))

        if not image_files:
            raise ValueError(f"No image files found in directory: {dir_path}")

        target_out = Path(output_dir) if output_dir else Path(self.config.output_dir)
        target_out.mkdir(parents=True, exist_ok=True)
        annotated_dir = target_out / "annotated_frames"
        if save_annotated:
            annotated_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==============================================================================")
        print(f" 2D OBJECT DETECTION (Ultralytics YOLO)")
        print(f"==============================================================================")
        print(f"Model Weights:         {self.config.model_weights}")
        print(f"Inference Device:      {self.config.device.upper()}")
        print(f"Confidence Threshold:  {self.config.conf_threshold}")
        print(f"IoU NMS Threshold:     {self.config.iou_threshold}")
        print(f"Input Resolution:      {self.config.imgsz}")
        print(f"Input Images:          {len(image_files)} keyframes from {dir_path}")
        print(f"------------------------------------------------------------------------------")

        all_image_detections: List[ImageDetections] = []
        all_detections: List[Detection] = []
        time_per_image: Dict[str, float] = {}
        images_no_det: List[str] = []
        detection_counter = 0

        total_t0 = time.time()

        for idx, img_path in enumerate(image_files):
            cam_id = self.parse_camera_id(img_path, fallback_index=idx)
            img_dets = self.detect_image(
                image_path=img_path,
                camera_id=cam_id,
                detection_id_offset=detection_counter,
            )
            all_image_detections.append(img_dets)
            all_detections.extend(img_dets.detections)
            detection_counter += img_dets.detection_count

            time_per_image[img_path.name] = round(img_dets.inference_time_ms, 2)
            if img_dets.detection_count == 0:
                images_no_det.append(img_path.name)

            # Save annotated frame
            if save_annotated:
                self.visualizer.save_annotated_frame(
                    image_or_path=img_path,
                    detections=img_dets.detections,
                    output_filename=img_path.name,
                    custom_output_dir=annotated_dir,
                )

            det_summary_str = (
                f"{img_dets.detection_count} detections"
                if img_dets.detection_count > 0
                else "No detections"
            )
            classes_str = ", ".join(sorted(list(set(d.class_name for d in img_dets.detections))))
            classes_info = f" ({classes_str})" if classes_str else ""
            print(
                f"  [{idx + 1:02d}/{len(image_files):02d}] {img_path.name}: "
                f"{det_summary_str}{classes_info} in {img_dets.inference_time_ms:.1f} ms"
            )

        total_inference_sec = time.time() - total_t0

        # Compute summary aggregations
        summary = self.compute_summary(
            image_detections=all_image_detections,
            time_per_image=time_per_image,
            images_no_det=images_no_det,
            total_inference_sec=total_inference_sec,
        )

        # Export all artifacts
        self.export_artifacts(
            output_dir=target_out,
            image_detections=all_image_detections,
            all_detections=all_detections,
            summary=summary,
        )

        print(f"------------------------------------------------------------------------------")
        print(f"Total Detections:      {summary.total_detections} objects across {len(summary.unique_object_classes)} classes")
        print(f"Classes Detected:      {', '.join(summary.unique_object_classes) if summary.unique_object_classes else 'None'}")
        print(f"Annotated Frames:      Saved to {annotated_dir}/")
        print(f"JSON / CSV Outputs:    Saved to {target_out}/")
        print(f"Total Inference Time:  {total_inference_sec:.2f} s (Avg: {summary.to_dict()['timing_benchmarks']['average_inference_time_ms']:.1f} ms/frame)")
        print(f"==============================================================================\n")

        return all_image_detections, summary

    def compute_summary(
        self,
        image_detections: List[ImageDetections],
        time_per_image: Dict[str, float],
        images_no_det: List[str],
        total_inference_sec: float,
    ) -> DetectionSummary:
        """Computes statistical breakdown across all detections."""
        all_dets: List[Detection] = []
        for img_det in image_detections:
            all_dets.extend(img_det.detections)

        class_counts: Dict[str, int] = {}
        class_conf_sums: Dict[str, float] = {}

        for d in all_dets:
            c = d.class_name
            class_counts[c] = class_counts.get(c, 0) + 1
            class_conf_sums[c] = class_conf_sums.get(c, 0.0) + d.confidence

        avg_conf_per_class: Dict[str, float] = {}
        for c, count in class_counts.items():
            avg_conf_per_class[c] = round(class_conf_sums[c] / max(1, count), 4)

        unique_classes = sorted(list(class_counts.keys()))

        return DetectionSummary(
            total_images_processed=len(image_detections),
            total_detections=len(all_dets),
            unique_object_classes=unique_classes,
            detection_count_per_class=class_counts,
            average_confidence_per_class=avg_conf_per_class,
            images_with_no_detections=images_no_det,
            inference_time_per_image=time_per_image,
            total_inference_time=total_inference_sec,
            model_name=str(self.config.model_weights),
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
        )

    def export_artifacts(
        self,
        output_dir: Union[str, Path],
        image_detections: List[ImageDetections],
        all_detections: List[Detection],
        summary: DetectionSummary,
    ) -> None:
        """Exports detections.json, detections.csv, and detection_summary.json."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. detections.json
        det_json_path = out_path / "detections.json"
        with open(det_json_path, "w", encoding="utf-8") as f:
            json.dump([img_d.to_dict() for img_d in image_detections], f, indent=2)

        # 2. detections.csv
        det_csv_path = out_path / "detections.csv"
        with open(det_csv_path, "w", encoding="utf-8") as f:
            f.write(
                "detection_id,camera_id,image_filename,class_id,class_name,confidence,"
                "x1,y1,x2,y2,center_x,center_y,width,height\n"
            )
            for d in all_detections:
                f.write(d.to_csv_row())

        # 3. detection_summary.json
        sum_json_path = out_path / "detection_summary.json"
        with open(sum_json_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

