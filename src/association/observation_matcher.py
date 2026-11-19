"""
Observation Matcher Module
Performs point-in-bounding-box testing between 2D SfM feature observations
and 2D YOLO object detections, recording spatial inclusion and multi-box ambiguity.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.detection.detection_models import Detection, ImageDetections
from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem, TrackObservation
from src.association.association_models import ObservationAssociation


class ObservationMatcher:
    """
    Evaluates geometric inclusion of 2D SfM observations within 2D YOLO detection bounding boxes.
    Handles overlapping bounding boxes via configurable disambiguation policies.
    """

    VALID_POLICIES = {"smallest_box", "highest_confidence", "none"}

    def __init__(self, ambiguity_policy: str = "smallest_box"):
        if ambiguity_policy not in self.VALID_POLICIES:
            raise ValueError(
                f"Unknown ambiguity policy '{ambiguity_policy}'. "
                f"Supported: {sorted(list(self.VALID_POLICIES))}"
            )
        self.ambiguity_policy = ambiguity_policy

    @staticmethod
    def is_point_inside_box(
        x: float,
        y: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> bool:
        """
        Inclusive bounding-box containment test: x1 <= x <= x2 and y1 <= y <= y2.
        Handles coordinate order automatically.
        """
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        return bool(min_x <= x <= max_x and min_y <= y <= max_y)

    def match_observation(
        self,
        obs_x: float,
        obs_y: float,
        detections: List[Detection],
        track_id: int,
        camera_id: int,
        image_filename: str,
        point_id: Optional[int] = None,
    ) -> ObservationAssociation:
        """
        Tests a single (x, y) observation against all detections present in that camera view.
        """
        containing: List[Detection] = []
        for det in detections:
            if self.is_point_inside_box(obs_x, obs_y, det.x1, det.y1, det.x2, det.y2):
                containing.append(det)

        coords = (float(obs_x), float(obs_y))

        # Case 1: Point is outside all detection boxes
        if not containing:
            return ObservationAssociation(
                track_id=track_id,
                camera_id=camera_id,
                image_filename=image_filename,
                coords=coords,
                point_id=point_id,
                is_matched=False,
            )

        containing_ids = [d.detection_id for d in containing]
        containing_classes = [d.class_name for d in containing]

        # Case 2: Exactly one bounding box contains the observation
        if len(containing) == 1:
            det = containing[0]
            return ObservationAssociation(
                track_id=track_id,
                camera_id=camera_id,
                image_filename=image_filename,
                coords=coords,
                point_id=point_id,
                is_matched=True,
                detection_id=det.detection_id,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                is_ambiguous=False,
                containing_detection_ids=containing_ids,
                containing_classes=containing_classes,
                ambiguity_reason=None,
                bbox_xyxy=det.bbox_xyxy,
                bbox_area=det.width * det.height,
            )

        # Case 3: Multiple overlapping bounding boxes contain the observation (Ambiguous)
        ambiguity_msg = (
            f"Observation ({obs_x:.1f}, {obs_y:.1f}) is inside {len(containing)} overlapping "
            f"detections: {containing_classes} (IDs: {containing_ids})."
        )

        selected_det: Optional[Detection] = None
        if self.ambiguity_policy == "smallest_box":
            # Select detection with smallest spatial bounding box area (tighter object boundary)
            # Tie-break by highest confidence
            selected_det = min(
                containing, key=lambda d: (d.width * d.height, -d.confidence)
            )
        elif self.ambiguity_policy == "highest_confidence":
            # Select detection with highest detector confidence score
            # Tie-break by smallest box area
            selected_det = min(
                containing, key=lambda d: (-d.confidence, d.width * d.height)
            )
        elif self.ambiguity_policy == "none":
            selected_det = None

        return ObservationAssociation(
            track_id=track_id,
            camera_id=camera_id,
            image_filename=image_filename,
            coords=coords,
            point_id=point_id,
            is_matched=True,
            detection_id=selected_det.detection_id if selected_det else None,
            class_id=selected_det.class_id if selected_det else None,
            class_name=selected_det.class_name if selected_det else None,
            confidence=selected_det.confidence if selected_det else None,
            is_ambiguous=True,
            containing_detection_ids=containing_ids,
            containing_classes=containing_classes,
            ambiguity_reason=ambiguity_msg,
            bbox_xyxy=selected_det.bbox_xyxy if selected_det else None,
            bbox_area=(selected_det.width * selected_det.height) if selected_det else None,
        )

    def match_all_observations(
        self,
        tracks: Union[FeatureTrackSystem, Dict[int, FeatureTrack], List[FeatureTrack]],
        detections_by_image: Dict[str, List[Detection]],
        track_to_point_id: Optional[Dict[int, int]] = None,
    ) -> List[ObservationAssociation]:
        """
        Evaluates all observations across all feature tracks against keyframe detections.
        """
        track_dict = tracks.tracks if isinstance(tracks, FeatureTrackSystem) else tracks
        if isinstance(track_dict, list):
            track_dict = {t.track_id: t for t in track_dict}

        pt_map = track_to_point_id or {}
        associations: List[ObservationAssociation] = []

        for track_id, track in sorted(track_dict.items()):
            point_id = pt_map.get(track_id)
            if point_id is None and track.is_triangulated and track.point_3d is not None:
                # Fallback to track_id if no separate point index is provided
                point_id = track_id

            for cam_id, obs in sorted(track.observations.items()):
                # Lookup detections by image filename (try basename or matching key)
                img_name = obs.image_name
                dets = detections_by_image.get(img_name)
                if dets is None:
                    # Try matching by filename stem or fallback
                    for k, v in detections_by_image.items():
                        if k == img_name or Path(k).name == Path(img_name).name:
                            dets = v
                            break

                img_dets = dets or []
                match_rec = self.match_observation(
                    obs_x=float(obs.coords[0]),
                    obs_y=float(obs.coords[1]),
                    detections=img_dets,
                    track_id=track_id,
                    camera_id=cam_id,
                    image_filename=img_name,
                    point_id=point_id,
                )
                associations.append(match_rec)

        return associations

