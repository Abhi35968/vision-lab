"""
Track Associator Module
Aggregates observation-level detection evidence across multi-view feature tracks,
classifies association strength (STRONG, WEAK, AMBIGUOUS, UNASSOCIATED),
and groups candidate 3D points by object class in arbitrary_units.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem
from src.association.association_models import (
    AssociationStatus,
    ObservationAssociation,
    TrackAssociation,
    ObjectPointGroup,
    AssociationSummary,
)

logger = logging.getLogger("TrackAssociator")


@dataclass
class AssociationConfig:
    """Configurable thresholds for multi-view feature track association."""
    min_supporting_views: int = 2        # Minimum distinct camera views required for STRONG status
    min_matching_ratio: float = 0.50     # Minimum fraction of track observations inside detections
    min_confidence: float = 0.05         # Minimum average detector confidence threshold
    min_class_purity: float = 0.70       # Minimum vote fraction required for non-ambiguous class assignment
    ambiguity_policy: str = "smallest_box"
    output_dir: str = "outputs/association"


class TrackAssociator:
    """
    Evaluates multi-view consistency of 2D object detections across SfM feature tracks.
    Enforces conservative multi-view gating so no single-view observation can falsely promote a track.
    """

    def __init__(self, config: Optional[AssociationConfig] = None):
        self.config = config or AssociationConfig()

    def associate_tracks(
        self,
        tracks: Union[FeatureTrackSystem, Dict[int, FeatureTrack], List[FeatureTrack]],
        observation_associations: List[ObservationAssociation],
        track_to_point_id: Optional[Dict[int, int]] = None,
    ) -> List[TrackAssociation]:
        """
        Aggregates observation matches per track and determines multi-view association status.
        """
        track_dict = tracks.tracks if isinstance(tracks, FeatureTrackSystem) else tracks
        if isinstance(track_dict, list):
            track_dict = {t.track_id: t for t in track_dict}

        pt_map = track_to_point_id or {}

        # Index observation associations by track_id
        obs_by_track: Dict[int, List[ObservationAssociation]] = {
            t_id: [] for t_id in track_dict.keys()
        }
        for obs_assoc in observation_associations:
            if obs_assoc.track_id in obs_by_track:
                obs_by_track[obs_assoc.track_id].append(obs_assoc)

        track_associations: List[TrackAssociation] = []

        for track_id, track in sorted(track_dict.items()):
            obs_list = obs_by_track.get(track_id, [])
            total_obs = len(obs_list) if obs_list else len(track.observations)
            matched_obs = [o for o in obs_list if o.is_matched and o.class_name is not None]
            n_matched = len(matched_obs)
            ratio = (n_matched / max(1, total_obs))

            point_id = pt_map.get(track_id)
            if point_id is None and track.is_triangulated and track.point_3d is not None:
                point_id = track_id

            p3d = track.point_3d.tolist() if track.is_triangulated and track.point_3d is not None else None

            # Case 1: Unassociated track (0 matched observations)
            if n_matched == 0:
                track_assoc = TrackAssociation(
                    track_id=track_id,
                    point_id=point_id,
                    is_triangulated=track.is_triangulated,
                    point_3d=p3d,
                    total_observations=total_obs,
                    matched_observations=0,
                    matching_ratio=0.0,
                    association_status=AssociationStatus.UNASSOCIATED.value,
                    ambiguity_notes="No observations fell inside any detection bounding box.",
                )
                track_associations.append(track_assoc)
                continue

            # Tally votes per class
            class_counts: Dict[str, int] = {}
            class_views: Dict[str, Set[int]] = {}
            class_confs: Dict[str, List[float]] = {}
            class_ids: Dict[str, int] = {}

            for o in matched_obs:
                c = o.class_name
                class_counts[c] = class_counts.get(c, 0) + 1
                if c not in class_views:
                    class_views[c] = set()
                class_views[c].add(o.camera_id)
                if c not in class_confs:
                    class_confs[c] = []
                if o.confidence is not None:
                    class_confs[c].append(o.confidence)
                if o.class_id is not None:
                    class_ids[c] = o.class_id

            # Find dominant class
            sorted_classes = sorted(
                class_counts.items(),
                key=lambda kv: (kv[1], len(class_views.get(kv[0], set()))),
                reverse=True,
            )
            top_class, top_count = sorted_classes[0]
            top_views = sorted(list(class_views.get(top_class, set())))
            top_views_count = len(top_views)
            top_confs = class_confs.get(top_class, [0.0])
            top_mean_conf = float(np.mean(top_confs)) if top_confs else 0.0
            top_class_id = class_ids.get(top_class, 0)

            # Check class conflict
            has_conflict = False
            purity = top_count / max(1, n_matched)
            if len(sorted_classes) > 1:
                second_class, second_count = sorted_classes[1]
                second_views_count = len(class_views.get(second_class, set()))
                if second_views_count >= self.config.min_supporting_views and second_count >= top_count:
                    has_conflict = True
                elif purity < self.config.min_class_purity:
                    has_conflict = True

            # Convert sets to serializable lists
            views_dict = {c: sorted(list(v_set)) for c, v_set in class_views.items()}

            # Status classification
            if has_conflict:
                status = AssociationStatus.AMBIGUOUS.value
                notes = (
                    f"Conflicting multi-view class assignments: {class_counts} "
                    f"(purity={purity:.2f} < {self.config.min_class_purity:.2f})."
                )
            elif (
                top_views_count >= self.config.min_supporting_views
                and ratio >= self.config.min_matching_ratio
                and top_mean_conf >= self.config.min_confidence
            ):
                status = AssociationStatus.STRONG.value
                notes = (
                    f"Strong multi-view support: {top_views_count} views {top_views}, "
                    f"matching ratio: {ratio:.2f}, confidence: {top_mean_conf:.3f}."
                )
            else:
                status = AssociationStatus.WEAK.value
                reasons = []
                if top_views_count < self.config.min_supporting_views:
                    reasons.append(f"insufficient views ({top_views_count} < {self.config.min_supporting_views})")
                if ratio < self.config.min_matching_ratio:
                    reasons.append(f"low matching ratio ({ratio:.2f} < {self.config.min_matching_ratio})")
                if top_mean_conf < self.config.min_confidence:
                    reasons.append(f"low confidence ({top_mean_conf:.3f} < {self.config.min_confidence})")
                notes = f"Weak association due to: {', '.join(reasons)}."

            track_assoc = TrackAssociation(
                track_id=track_id,
                point_id=point_id,
                is_triangulated=track.is_triangulated,
                point_3d=p3d,
                total_observations=total_obs,
                matched_observations=n_matched,
                matching_ratio=ratio,
                association_status=status,
                assigned_class_id=top_class_id,
                assigned_class_name=top_class,
                dominant_class_count=top_count,
                dominant_class_views_count=top_views_count,
                dominant_class_views=top_views,
                mean_confidence=top_mean_conf,
                has_conflicting_classes=has_conflict,
                class_observation_counts=class_counts,
                class_supporting_views=views_dict,
                ambiguity_notes=notes,
            )
            track_associations.append(track_assoc)

        return track_associations

    def group_object_points(
        self,
        track_associations: List[TrackAssociation],
        observation_associations: List[ObservationAssociation],
    ) -> Dict[str, ObjectPointGroup]:
        """
        Groups reconstructed 3D points by their supported candidate object class.
        Computes 3D spatial extents and bounding bounds strictly in arbitrary_units.
        """
        # Map (track_id, class_name) to detection IDs
        det_ids_by_track_class: Dict[Tuple[int, str], List[Tuple[str, int]]] = {}
        for o in observation_associations:
            if o.is_matched and o.class_name and o.detection_id is not None:
                key = (o.track_id, o.class_name)
                if key not in det_ids_by_track_class:
                    det_ids_by_track_class[key] = []
                det_ids_by_track_class[key].append((o.image_filename, o.detection_id))

        # Filter to triangulated tracks with candidate class assignment
        groups: Dict[str, ObjectPointGroup] = {}

        for ta in track_associations:
            # Only group points that have 3D geometry and a valid class candidate
            if not ta.is_triangulated or ta.point_3d is None or not ta.assigned_class_name:
                continue
            if ta.association_status == AssociationStatus.UNASSOCIATED.value:
                continue

            c_name = ta.assigned_class_name
            c_id = ta.assigned_class_id or 0

            if c_name not in groups:
                groups[c_name] = ObjectPointGroup(class_id=c_id, class_name=c_name)

            grp = groups[c_name]
            grp.total_points += 1
            if ta.point_id is not None:
                grp.point_ids.append(ta.point_id)
            grp.track_ids.append(ta.track_id)

            if ta.association_status == AssociationStatus.STRONG.value:
                grp.strong_point_count += 1
            elif ta.association_status == AssociationStatus.WEAK.value:
                grp.weak_point_count += 1
            elif ta.association_status == AssociationStatus.AMBIGUOUS.value:
                grp.ambiguous_point_count += 1

            for cam in ta.dominant_class_views:
                if cam not in grp.contributing_camera_ids:
                    grp.contributing_camera_ids.append(cam)

            # Record contributing detection IDs per image
            key = (ta.track_id, c_name)
            for img_name, det_id in det_ids_by_track_class.get(key, []):
                if img_name not in grp.contributing_detection_ids:
                    grp.contributing_detection_ids[img_name] = []
                if det_id not in grp.contributing_detection_ids[img_name]:
                    grp.contributing_detection_ids[img_name].append(det_id)

        # Compute 3D spatial extents and confidence averages per group
        for c_name, grp in groups.items():
            grp.contributing_camera_ids.sort()
            for img in grp.contributing_detection_ids:
                grp.contributing_detection_ids[img].sort()

            # Gather 3D coordinates of points in this group
            pts = [
                ta.point_3d for ta in track_associations
                if ta.track_id in grp.track_ids and ta.point_3d is not None
            ]
            confs = [
                ta.mean_confidence for ta in track_associations
                if ta.track_id in grp.track_ids and ta.mean_confidence > 0
            ]
            grp.mean_association_confidence = float(np.mean(confs)) if confs else 0.0

            if pts:
                pts_arr = np.array(pts, dtype=np.float64)
                c_min = np.min(pts_arr, axis=0)
                c_max = np.max(pts_arr, axis=0)
                c_mid = np.mean(pts_arr, axis=0)
                ext = c_max - c_min

                grp.spatial_bounds = {
                    "unit": "arbitrary_units",
                    "centroid": [round(float(x), 4) for x in c_mid],
                    "min_bounds": [round(float(x), 4) for x in c_min],
                    "max_bounds": [round(float(x), 4) for x in c_max],
                    "extents": [round(float(x), 4) for x in ext],
                    "point_count": len(pts),
                }

        return groups

    def compute_summary(
        self,
        tracks: Union[FeatureTrackSystem, Dict[int, FeatureTrack], List[FeatureTrack]],
        track_associations: List[TrackAssociation],
        observation_associations: List[ObservationAssociation],
        object_point_groups: Dict[str, ObjectPointGroup],
    ) -> AssociationSummary:
        """Computes consolidated telemetry and audit statistics."""
        track_dict = tracks.tracks if isinstance(tracks, FeatureTrackSystem) else tracks
        if isinstance(track_dict, list):
            track_dict = {t.track_id: t for t in track_dict}

        total_tracks = len(track_dict)
        tri_tracks = sum(1 for t in track_dict.values() if t.is_triangulated and t.point_3d is not None)

        total_obs = len(observation_associations)
        matched_obs = sum(1 for o in observation_associations if o.is_matched)
        unmatched_obs = total_obs - matched_obs
        ambiguous_obs = sum(1 for o in observation_associations if o.is_ambiguous)

        n_strong = sum(1 for ta in track_associations if ta.association_status == AssociationStatus.STRONG.value)
        n_weak = sum(1 for ta in track_associations if ta.association_status == AssociationStatus.WEAK.value)
        n_ambig = sum(1 for ta in track_associations if ta.association_status == AssociationStatus.AMBIGUOUS.value)
        n_unassoc = sum(1 for ta in track_associations if ta.association_status == AssociationStatus.UNASSOCIATED.value)

        track_counts_per_class: Dict[str, int] = {}
        point_counts_per_class: Dict[str, int] = {}

        for ta in track_associations:
            if ta.association_status in {AssociationStatus.STRONG.value, AssociationStatus.WEAK.value}:
                c = ta.assigned_class_name
                if c:
                    track_counts_per_class[c] = track_counts_per_class.get(c, 0) + 1
                    if ta.is_triangulated and ta.point_3d is not None:
                        point_counts_per_class[c] = point_counts_per_class.get(c, 0) + 1

        confs = [
            ta.mean_confidence for ta in track_associations
            if ta.association_status in {AssociationStatus.STRONG.value, AssociationStatus.WEAK.value}
            and ta.mean_confidence > 0
        ]
        conf_stats = {
            "mean": round(float(np.mean(confs)), 4) if confs else 0.0,
            "median": round(float(np.median(confs)), 4) if confs else 0.0,
            "min": round(float(np.min(confs)), 4) if confs else 0.0,
            "max": round(float(np.max(confs)), 4) if confs else 0.0,
        }

        thresholds = {
            "min_supporting_views": self.config.min_supporting_views,
            "min_matching_ratio": self.config.min_matching_ratio,
            "min_confidence": self.config.min_confidence,
            "min_class_purity": self.config.min_class_purity,
            "ambiguity_policy": self.config.ambiguity_policy,
        }

        return AssociationSummary(
            total_feature_tracks=total_tracks,
            total_3d_points=tri_tracks,
            total_observations=total_obs,
            matched_observations=matched_obs,
            unmatched_observations=unmatched_obs,
            ambiguous_observations=ambiguous_obs,
            strongly_associated_tracks=n_strong,
            weakly_associated_tracks=n_weak,
            ambiguous_tracks=n_ambig,
            unassociated_tracks=n_unassoc,
            track_counts_per_class=track_counts_per_class,
            point_counts_per_class=point_counts_per_class,
            confidence_statistics=conf_stats,
            thresholds_used=thresholds,
        )

    def export_all(
        self,
        output_dir: Union[str, Path],
        observation_associations: List[ObservationAssociation],
        track_associations: List[TrackAssociation],
        object_point_groups: Dict[str, ObjectPointGroup],
        summary: AssociationSummary,
    ) -> None:
        """Exports observation_associations.json, track_associations.json, object_point_groups.json, and association_summary.json."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. observation_associations.json
        with open(out_path / "observation_associations.json", "w", encoding="utf-8") as f:
            json.dump([o.to_dict() for o in observation_associations], f, indent=2)

        # 2. track_associations.json
        with open(out_path / "track_associations.json", "w", encoding="utf-8") as f:
            json.dump([ta.to_dict() for ta in track_associations], f, indent=2)

        # 3. object_point_groups.json
        with open(out_path / "object_point_groups.json", "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in object_point_groups.items()}, f, indent=2)

        # 4. association_summary.json
        with open(out_path / "association_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

