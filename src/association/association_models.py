"""
Association Data Models Module
Defines structured dataclasses and serialization schemas for observation-level matches,
multi-view track associations, candidate 3D object point clusters, and diagnostic summaries.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np


class AssociationStatus(str, Enum):
    """Classification states for multi-view feature track association."""
    STRONG = "STRONG"              # Supported by >= 2 distinct camera views with high agreement
    WEAK = "WEAK"                  # Inside detection, but supported by only 1 view or low matching ratio
    AMBIGUOUS = "AMBIGUOUS"        # Multi-class conflict or overlapping detection ambiguity
    UNASSOCIATED = "UNASSOCIATED"  # No observations fell inside any detection bounding box


@dataclass
class ObservationAssociation:
    """Telemetry for a single SfM 2D feature observation mapped to 2D object detections."""
    track_id: int
    camera_id: int
    image_filename: str
    coords: Tuple[float, float]
    point_id: Optional[int] = None
    is_matched: bool = False
    detection_id: Optional[int] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    is_ambiguous: bool = False
    containing_detection_ids: List[int] = field(default_factory=list)
    containing_classes: List[str] = field(default_factory=list)
    ambiguity_reason: Optional[str] = None
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    bbox_area: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "point_id": self.point_id,
            "camera_id": self.camera_id,
            "image_filename": self.image_filename,
            "coords": [round(self.coords[0], 2), round(self.coords[1], 2)],
            "is_matched": self.is_matched,
            "detection_id": self.detection_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "is_ambiguous": self.is_ambiguous,
            "containing_detection_ids": self.containing_detection_ids,
            "containing_classes": self.containing_classes,
            "ambiguity_reason": self.ambiguity_reason,
            "bbox_xyxy": [round(c, 2) for c in self.bbox_xyxy] if self.bbox_xyxy else None,
        }


@dataclass
class TrackAssociation:
    """Aggregated multi-view association metrics and status for a single feature track."""
    track_id: int
    point_id: Optional[int] = None
    is_triangulated: bool = False
    point_3d: Optional[List[float]] = None
    total_observations: int = 0
    matched_observations: int = 0
    matching_ratio: float = 0.0
    association_status: str = AssociationStatus.UNASSOCIATED.value
    assigned_class_id: Optional[int] = None
    assigned_class_name: Optional[str] = None
    dominant_class_count: int = 0
    dominant_class_views_count: int = 0
    dominant_class_views: List[int] = field(default_factory=list)
    mean_confidence: float = 0.0
    has_conflicting_classes: bool = False
    class_observation_counts: Dict[str, int] = field(default_factory=dict)
    class_supporting_views: Dict[str, List[int]] = field(default_factory=dict)
    ambiguity_notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "point_id": self.point_id,
            "is_triangulated": self.is_triangulated,
            "point_3d": [round(x, 4) for x in self.point_3d] if self.point_3d else None,
            "unit": "arbitrary_units",
            "total_observations": self.total_observations,
            "matched_observations": self.matched_observations,
            "matching_ratio": round(self.matching_ratio, 3),
            "association_status": self.association_status,
            "assigned_class_id": self.assigned_class_id,
            "assigned_class_name": self.assigned_class_name,
            "dominant_class_count": self.dominant_class_count,
            "dominant_class_views_count": self.dominant_class_views_count,
            "dominant_class_views": self.dominant_class_views,
            "mean_confidence": round(self.mean_confidence, 4),
            "has_conflicting_classes": self.has_conflicting_classes,
            "class_observation_counts": self.class_observation_counts,
            "class_supporting_views": self.class_supporting_views,
            "ambiguity_notes": self.ambiguity_notes,
        }


@dataclass
class ObjectPointGroup:
    """Group of reconstructed 3D points associated with a candidate object class."""
    class_id: int
    class_name: str
    total_points: int = 0
    point_ids: List[int] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    strong_point_count: int = 0
    weak_point_count: int = 0
    ambiguous_point_count: int = 0
    contributing_camera_ids: List[int] = field(default_factory=list)
    contributing_detection_ids: Dict[str, List[int]] = field(default_factory=dict)
    mean_association_confidence: float = 0.0
    spatial_bounds: Dict[str, Union[str, List[float]]] = field(default_factory=dict)
    academic_note: str = (
        "Candidate 3D point cluster sharing class detection evidence. Does not imply "
        "single instance segmentation or persistent object identity across disjoint physical objects."
    )

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "total_points": self.total_points,
            "breakdown": {
                "strong_points": self.strong_point_count,
                "weak_points": self.weak_point_count,
                "ambiguous_points": self.ambiguous_point_count,
            },
            "point_ids": self.point_ids,
            "track_ids": self.track_ids,
            "contributing_camera_ids": self.contributing_camera_ids,
            "contributing_detection_ids": self.contributing_detection_ids,
            "mean_association_confidence": round(self.mean_association_confidence, 4),
            "spatial_bounds": self.spatial_bounds,
            "academic_note": self.academic_note,
        }


@dataclass
class AssociationSummary:
    """Consolidated statistics and quality telemetry for Phase 4.2 association."""
    total_feature_tracks: int
    total_3d_points: int
    total_observations: int
    matched_observations: int
    unmatched_observations: int
    ambiguous_observations: int
    strongly_associated_tracks: int
    weakly_associated_tracks: int
    ambiguous_tracks: int
    unassociated_tracks: int
    track_counts_per_class: Dict[str, int]
    point_counts_per_class: Dict[str, int]
    confidence_statistics: Dict[str, float]
    thresholds_used: Dict[str, Union[int, float, str]]

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds_used,
            "observation_level_metrics": {
                "total_observations": self.total_observations,
                "matched_observations": self.matched_observations,
                "unmatched_observations": self.unmatched_observations,
                "ambiguous_observations": self.ambiguous_observations,
                "match_rate_percent": round(
                    (self.matched_observations / max(1, self.total_observations)) * 100.0, 2
                ),
            },
            "track_level_metrics": {
                "total_feature_tracks": self.total_feature_tracks,
                "strongly_associated_tracks": self.strongly_associated_tracks,
                "weakly_associated_tracks": self.weakly_associated_tracks,
                "ambiguous_tracks": self.ambiguous_tracks,
                "unassociated_tracks": self.unassociated_tracks,
                "association_rate_percent": round(
                    ((self.strongly_associated_tracks + self.weakly_associated_tracks) /
                     max(1, self.total_feature_tracks)) * 100.0, 2
                ),
                "track_counts_per_class": self.track_counts_per_class,
            },
            "point_3d_metrics": {
                "total_3d_points": self.total_3d_points,
                "point_counts_per_class": self.point_counts_per_class,
                "associated_points_count": sum(self.point_counts_per_class.values()),
            },
            "association_confidence_statistics": self.confidence_statistics,
            "limitations_note": (
                "Bounding-box membership indicates spatial inclusion, not proven physical object ownership. "
                "Bounding boxes encompass foreground objects and incidental background features."
            ),
        }

