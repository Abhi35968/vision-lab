"""
Detection Visualizer Module
Renders high-contrast bounding boxes, class banners, and centroid markers
onto keyframe images and exports annotated frame artifacts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.detection.detection_models import Detection, ImageDetections


class DetectionVisualizer:
    """
    Renders 2D bounding boxes and class telemetry on OpenCV images.
    Uses deterministic HSV color hashing to ensure identical colors per class across all views.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_class_color(class_id: int) -> Tuple[int, int, int]:
        """Generates a distinctive, deterministic BGR color for any class ID."""
        # Use golden ratio hue spacing for high visual contrast
        golden_ratio = 0.618033988749895
        hue = int(((class_id * golden_ratio) % 1.0) * 180)
        hsv_pixel = np.uint8([[[hue, 220, 240]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

    def annotate_frame(
        self,
        image_or_path: Union[np.ndarray, str, Path],
        detections: List[Detection],
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draws bounding boxes, labels, and center markers on an image.
        
        Returns:
            Annotated BGR image as a NumPy array.
        """
        if isinstance(image_or_path, (str, Path)):
            img = cv2.imread(str(image_or_path))
            if img is None:
                raise FileNotFoundError(f"Failed to read image at: {image_or_path}")
        else:
            img = image_or_path.copy()

        h, w = img.shape[:2]

        for det in detections:
            color = self.get_class_color(det.class_id)

            # Clamp coordinates to valid image bounds
            x1 = max(0, min(w - 1, int(round(det.x1))))
            y1 = max(0, min(h - 1, int(round(det.y1))))
            x2 = max(0, min(w - 1, int(round(det.x2))))
            y2 = max(0, min(h - 1, int(round(det.y2))))

            # 1. Bounding box rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

            # 2. Centroid marker
            cx = max(0, min(w - 1, int(round(det.center_x))))
            cy = max(0, min(h - 1, int(round(det.center_y))))
            cv2.circle(img, (cx, cy), 3, color, -1)

            # 3. Text label tag with background box
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Determine label banner position (place above box if space permits, else inside)
            tag_y1 = max(0, y1 - text_h - baseline - 4)
            tag_y2 = y1 if y1 - text_h - baseline - 4 >= 0 else y1 + text_h + baseline + 4
            tag_x2 = min(w - 1, x1 + text_w + 6)

            cv2.rectangle(img, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            text_pos_y = tag_y2 - baseline - 2 if tag_y2 == y1 else tag_y2 - 2
            cv2.putText(
                img,
                label,
                (x1 + 3, text_pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return img

    def save_annotated_frame(
        self,
        image_or_path: Union[np.ndarray, str, Path],
        detections: List[Detection],
        output_filename: str,
        custom_output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Draws annotations and saves image to disk."""
        annotated = self.annotate_frame(image_or_path, detections)
        save_dir = Path(custom_output_dir) if custom_output_dir else self.output_dir
        if save_dir is None:
            raise ValueError("No output directory specified for saving annotated frame.")
        save_dir.mkdir(parents=True, exist_ok=True)

        out_path = save_dir / output_filename
        cv2.imwrite(str(out_path), annotated)
        return out_path

