"""
Mask Processor Module
Handles mask resizing, thresholding, connected components analysis,
border contact detection, centroid calculation, morphological refinement,
and pixel-level IoU computation for instance masks.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np


class MaskProcessor:
    """Utility class for processing and analyzing 2D binary segmentation masks."""

    @staticmethod
    def resize_to_target_resolution(
        raw_mask: np.ndarray,
        target_shape: Tuple[int, int],  # (height, width)
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Resizes a model-output mask to the original target resolution.

        Args:
            raw_mask: Raw mask array of shape (H_raw, W_raw) with float [0, 1] or binary values.
            target_shape: Desired output dimensions as (height, width), e.g. (1920, 1080).
            threshold: Probability threshold above which pixels are marked positive.

        Returns:
            Binary uint8 array of shape (height, width) with values strictly in {0, 255}.
        """
        target_h, target_w = target_shape

        if raw_mask.shape[:2] == (target_h, target_w):
            # Already target size
            if raw_mask.dtype == bool:
                return (raw_mask.astype(np.uint8)) * 255
            elif np.issubdtype(raw_mask.dtype, np.floating):
                return ((raw_mask > threshold).astype(np.uint8)) * 255
            elif np.array_equal(np.unique(raw_mask), [0, 1]):
                return (raw_mask * 255).astype(np.uint8)
            return raw_mask.astype(np.uint8)

        # Bilinear interpolation for smooth boundary representation
        mask_float = raw_mask.astype(np.float32)
        if mask_float.max() > 1.0:
            mask_float /= 255.0

        # OpenCV takes dsize as (width, height)
        resized = cv2.resize(mask_float, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = ((resized > threshold).astype(np.uint8)) * 255

        return binary_mask

    @staticmethod
    def check_touches_border(
        mask: np.ndarray,
        margin: int = 2,
    ) -> bool:
        """
        Determines whether the positive mask touches or falls within margin of image boundaries.

        Args:
            mask: Binary mask of shape (H, W).
            margin: Pixel margin from borders to check.

        Returns:
            True if any mask pixels lie within margin of top, bottom, left, or right edges.
        """
        h, w = mask.shape[:2]
        if margin <= 0:
            margin = 1

        top_slice = mask[:margin, :]
        bottom_slice = mask[max(0, h - margin):, :]
        left_slice = mask[:, :margin]
        right_slice = mask[:, max(0, w - margin):]

        return bool(
            np.any(top_slice > 0)
            or np.any(bottom_slice > 0)
            or np.any(left_slice > 0)
            or np.any(right_slice > 0)
        )

    @staticmethod
    def analyze_connected_components(
        mask: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Analyzes connected components in the mask.

        Args:
            mask: Binary mask with values in {0, 255}.

        Returns:
            Tuple of:
              - num_components: int (excluding background)
              - largest_component_ratio: float (ratio of largest component area to total area)
        """
        bin_mask = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8
        )

        if num_labels <= 1:
            return 0, 0.0

        # Component 0 is background; components 1..num_labels-1 are foreground
        areas = stats[1:, cv2.CC_STAT_AREA]
        total_area = int(np.sum(areas))

        if total_area == 0:
            return 0, 0.0

        largest_area = int(np.max(areas))
        largest_ratio = float(largest_area / total_area)
        num_components = len(areas)

        return num_components, round(largest_ratio, 4)

    @staticmethod
    def compute_mask_centroid(
        mask: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Computes the spatial centroid (center of mass) of the mask.

        Args:
            mask: Binary mask.

        Returns:
            (center_x, center_y) in pixels. Defaults to (0.0, 0.0) if mask is empty.
        """
        moments = cv2.moments((mask > 0).astype(np.uint8))
        if moments["m00"] > 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            return round(cx, 2), round(cy, 2)
        return 0.0, 0.0

    @staticmethod
    def compute_mask_bbox(
        mask: np.ndarray,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Computes the tight bounding box containing all positive mask pixels.

        Args:
            mask: Binary mask.

        Returns:
            (x1, y1, x2, y2) or None if mask is empty.
        """
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        x1 = float(np.min(x_indices))
        y1 = float(np.min(y_indices))
        x2 = float(np.max(x_indices))
        y2 = float(np.max(y_indices))

        return round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)

    @staticmethod
    def clean_mask_morphology(
        mask: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        Applies morphological closing to bridge specular gaps and remove isolated noise specks.

        Args:
            mask: Binary mask with values in {0, 255}.
            kernel_size: Kernel dimension for structuring element.

        Returns:
            Refined binary mask.
        """
        if kernel_size <= 1:
            return mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return closed

    @staticmethod
    def erode_mask(
        mask: np.ndarray,
        kernel_size: int = 3,
        shape: str = "ellipse",
        iterations: int = 1,
        frame_id: Optional[Union[int, str]] = None,
        object_id: Optional[Union[int, str]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Applies morphological boundary erosion using OpenCV to eliminate background bleeding.

        Supports 3x3 and 5x5 structuring elements (cv2.MORPH_ELLIPSE or cv2.MORPH_RECT).
        Returns a tuple of (eroded_mask, metadata_dict).
        Preserves original mask untouched while generating eroded mask.
        Guards against over-erosion destroying thin object structures.

        Args:
            mask: Input binary mask (H, W).
            kernel_size: Structuring element diameter (3 or 5 recommended).
            shape: Kernel geometry: "ellipse" (cv2.MORPH_ELLIPSE) or "rect".
            iterations: Number of erosion passes.
            frame_id: Optional frame identifier for metadata export.
            object_id: Optional object / track identifier for metadata export.

        Returns:
            Tuple of:
              - eroded_mask: np.ndarray of shape (H, W) uint8 in {0, 255}.
              - metadata: dict containing frame ID, object ID, mask area, erosion kernel, resolution, retention.
        """
        if mask is None:
            return np.zeros((1, 1), dtype=np.uint8), {
                "frame_id": frame_id,
                "object_id": object_id,
                "original_area_px": 0,
                "eroded_area_px": 0,
                "erosion_kernel": f"{kernel_size}x{kernel_size}_{shape}",
                "resolution": (0, 0),
                "area_retention_ratio": 0.0,
            }

        h, w = mask.shape[:2]
        bin_mask = (mask > 0).astype(np.uint8) * 255
        orig_area = int(np.sum(bin_mask > 0))

        if kernel_size <= 1 or orig_area == 0:
            meta = {
                "frame_id": frame_id,
                "object_id": object_id,
                "original_area_px": orig_area,
                "eroded_area_px": orig_area,
                "erosion_kernel": f"none",
                "resolution": (w, h),
                "area_retention_ratio": 1.0,
            }
            return bin_mask.copy(), meta

        morph_shape = cv2.MORPH_ELLIPSE if shape.lower() == "ellipse" else cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(morph_shape, (kernel_size, kernel_size))
        eroded = cv2.erode(bin_mask, kernel, iterations=iterations)
        eroded_area = int(np.sum(eroded > 0))

        # Thin-structure protection guard: if erosion completely wipes out the mask, fall back to milder erosion or original
        if eroded_area == 0 and orig_area > 0:
            if kernel_size > 3:
                k3 = cv2.getStructuringElement(morph_shape, (3, 3))
                eroded = cv2.erode(bin_mask, k3, iterations=1)
                eroded_area = int(np.sum(eroded > 0))
            if eroded_area == 0:
                eroded = bin_mask.copy()
                eroded_area = orig_area

        metadata = {
            "frame_id": frame_id,
            "object_id": object_id,
            "original_area_px": orig_area,
            "eroded_area_px": eroded_area,
            "erosion_kernel": f"{kernel_size}x{kernel_size}_{shape}",
            "resolution": (w, h),
            "area_retention_ratio": round(float(eroded_area / max(1, orig_area)), 4),
        }
        return eroded, metadata

    @staticmethod
    def compute_mask_iou(
        mask_a: np.ndarray,
        mask_b: np.ndarray,
    ) -> float:
        """
        Computes pixel-wise Intersection-over-Union between two binary masks.

        Args:
            mask_a: First binary mask.
            mask_b: Second binary mask of the same dimensions.

        Returns:
            IoU score in [0.0, 1.0].
        """
        if mask_a.shape != mask_b.shape:
            raise ValueError(
                f"Mask shape mismatch: {mask_a.shape} vs {mask_b.shape}"
            )

        bin_a = mask_a > 0
        bin_b = mask_b > 0

        intersection = np.logical_and(bin_a, bin_b).sum()
        union = np.logical_or(bin_a, bin_b).sum()

        if union == 0:
            return 0.0

        return round(float(intersection / union), 4)

    @staticmethod
    def compute_bbox_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        """
        Computes Intersection-over-Union between two bounding boxes (x1, y1, x2, y2).
        """
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0

        return round(float(inter_area / union), 4)

