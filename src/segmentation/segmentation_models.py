"""
Segmentation Data Models Module
Defines structured dataclasses for instance masks, per-frame segmentation sets,
and aggregated segmentation summaries with pixel-accurate coordinates, mask diagnostics,
and CSV/JSON serialization.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class InstanceMask:
    """Represents a single segmented instance mask in original image pixel coordinates."""
    instance_id: int
    camera_id: int
    image_filename: str
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    mask_area_px: int
    mask_area_pct: float  # Percentage of total frame area
    centroid: Tuple[float, float]  # (center_x, center_y) in pixels
    touches_border: bool
    num_components: int
    largest_component_ratio: float
    is_target: bool = False
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    tracking_score: float = 0.0
    erosion_kernel: Optional[str] = None
    eroded_mask_area_px: Optional[int] = None
    mask_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        """Bounding box width in pixels."""
        return max(0.0, float(self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        """Bounding box height in pixels."""
        return max(0.0, float(self.bbox[3] - self.bbox[1]))

    @property
    def aspect_ratio(self) -> float:
        """Height-to-width aspect ratio."""
        w = self.width
        return round(float(self.height / w), 3) if w > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes instance mask metadata to a JSON-compatible dictionary."""
        return {
            "instance_id": self.instance_id,
            "camera_id": self.camera_id,
            "image_filename": self.image_filename,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox],
            "mask_area_px": int(self.mask_area_px),
            "mask_area_pct": round(float(self.mask_area_pct), 3),
            "centroid": [round(float(c), 2) for c in self.centroid],
            "touches_border": bool(self.touches_border),
            "num_components": int(self.num_components),
            "largest_component_ratio": round(float(self.largest_component_ratio), 4),
            "is_target": bool(self.is_target),
            "is_ambiguous": bool(self.is_ambiguous),
            "ambiguity_reason": self.ambiguity_reason,
            "tracking_score": round(float(self.tracking_score), 4),
            "aspect_ratio": self.aspect_ratio,
            "erosion_kernel": self.erosion_kernel,
            "eroded_mask_area_px": self.eroded_mask_area_px,
            "mask_metadata": self.mask_metadata,
        }


@dataclass
class FrameSegmentation:
    """Represents segmentation results and target tracking status for a single keyframe."""
    frame_idx: int
    image_name: str
    image_path: str
    image_width: int
    image_height: int
    is_registered_sfm: bool
    target_detected: bool
    target_status: str  # "VALID", "AMBIGUOUS", "NO_TARGET", "EXCLUDED"
    target_instance: Optional[InstanceMask]
    all_instances: List[InstanceMask] = field(default_factory=list)
    inference_time_ms: float = 0.0
    mask_path: Optional[str] = None
    overlay_path: Optional[str] = None
    side_by_side_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes frame segmentation record to JSON-compatible dictionary."""
        return {
            "frame_idx": self.frame_idx,
            "image_name": self.image_name,
            "image_path": str(self.image_path),
            "resolution": [self.image_width, self.image_height],
            "is_registered_sfm": bool(self.is_registered_sfm),
            "target_detected": bool(self.target_detected),
            "target_status": self.target_status,
            "inference_time_ms": round(float(self.inference_time_ms), 2),
            "mask_path": self.mask_path,
            "overlay_path": self.overlay_path,
            "side_by_side_path": self.side_by_side_path,
            "target_instance": self.target_instance.to_dict() if self.target_instance else None,
            "num_total_instances": len(self.all_instances),
            "all_instances": [inst.to_dict() for inst in self.all_instances],
        }

    def to_csv_row(self) -> Dict[str, Any]:
        """Produces flattened dictionary for tabular summary CSV export."""
        inst = self.target_instance
        return {
            "frame_idx": self.frame_idx,
            "image_name": self.image_name,
            "is_registered_sfm": int(self.is_registered_sfm),
            "target_detected": int(self.target_detected),
            "target_status": self.target_status,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "target_class": inst.class_name if inst else "none",
            "confidence": round(inst.confidence, 4) if inst else 0.0,
            "bbox_x1": round(inst.bbox[0], 1) if inst else "",
            "bbox_y1": round(inst.bbox[1], 1) if inst else "",
            "bbox_x2": round(inst.bbox[2], 1) if inst else "",
            "bbox_y2": round(inst.bbox[3], 1) if inst else "",
            "mask_area_px": inst.mask_area_px if inst else 0,
            "mask_area_pct": round(inst.mask_area_pct, 2) if inst else 0.0,
            "touches_border": int(inst.touches_border) if inst else 0,
            "num_components": inst.num_components if inst else 0,
            "largest_comp_ratio": round(inst.largest_component_ratio, 3) if inst else 0.0,
            "aspect_ratio": inst.aspect_ratio if inst else 0.0,
            "tracking_score": round(inst.tracking_score, 3) if inst else 0.0,
            "erosion_kernel": inst.erosion_kernel if inst and inst.erosion_kernel else "none",
            "eroded_mask_area_px": inst.eroded_mask_area_px if inst and inst.eroded_mask_area_px is not None else (inst.mask_area_px if inst else 0),
            "mask_path": self.mask_path or "",
        }


@dataclass
class SegmentationSummary:
    """Aggregates quantitative segmentation statistics across all evaluated frames."""
    total_frames: int
    sfm_registered_frames: int
    valid_target_frames: int
    sfm_valid_target_frames: int
    ambiguous_frames: int
    no_target_frames: int
    excluded_frames: int
    mean_inference_time_ms: float
    mean_target_mask_area_pct: float
    mean_target_confidence: float
    border_touching_count: int
    target_classes_detected: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes summary to JSON-compatible dictionary."""
        return {
            "total_frames": self.total_frames,
            "sfm_registered_frames": self.sfm_registered_frames,
            "valid_target_frames": self.valid_target_frames,
            "sfm_valid_target_frames": self.sfm_valid_target_frames,
            "ambiguous_frames": self.ambiguous_frames,
            "no_target_frames": self.no_target_frames,
            "excluded_frames": self.excluded_frames,
            "target_coverage_overall_pct": round(
                (self.valid_target_frames / max(1, self.total_frames)) * 100.0, 2
            ),
            "target_coverage_sfm_pct": round(
                (self.sfm_valid_target_frames / max(1, self.sfm_registered_frames)) * 100.0, 2
            ),
            "mean_inference_time_ms": round(float(self.mean_inference_time_ms), 2),
            "mean_target_mask_area_pct": round(float(self.mean_target_mask_area_pct), 2),
            "mean_target_confidence": round(float(self.mean_target_confidence), 4),
            "border_touching_count": self.border_touching_count,
            "target_classes_detected": self.target_classes_detected,
        }

