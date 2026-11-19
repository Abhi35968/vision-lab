"""
Feature Track System Module
Maintains multi-view feature correspondences across keyframes with conflict detection.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np


@dataclass
class TrackObservation:
    """A single 2D keypoint observation belonging to a multi-view track."""
    image_id: int
    image_name: str
    keypoint_idx: int
    coords: Tuple[float, float]               # (x, y) subpixel coordinates
    descriptor: Optional[np.ndarray] = None   # Optional descriptor vector

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "keypoint_idx": self.keypoint_idx,
            "coords": [round(float(self.coords[0]), 3), round(float(self.coords[1]), 3)],
        }


@dataclass
class FeatureTrack:
    """
    Represents a single physical 3D scene point observed across multiple camera views.
    Enforces that each camera view (image_id) contains at most ONE observation.
    """
    track_id: int
    observations: Dict[int, TrackObservation] = field(default_factory=dict) # image_id -> TrackObservation
    point_3d: Optional[np.ndarray] = None                                    # (3,) float64 in world frame
    reprojection_error: Optional[float] = None
    is_triangulated: bool = False

    @property
    def length(self) -> int:
        """Number of distinct camera views observing this 3D point."""
        return len(self.observations)

    def has_image(self, image_id: int) -> bool:
        return image_id in self.observations

    def add_observation(self, obs: TrackObservation) -> bool:
        """Adds an observation. Returns False if this camera already observes this track."""
        if obs.image_id in self.observations:
            return False
        self.observations[obs.image_id] = obs
        return True

    def get_image_ids(self) -> Set[int]:
        return set(self.observations.keys())

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "track_length": self.length,
            "is_triangulated": self.is_triangulated,
            "point_3d": self.point_3d.tolist() if self.point_3d is not None else None,
            "reprojection_error_px": round(float(self.reprojection_error), 3) if self.reprojection_error is not None else None,
            "observations": [obs.to_dict() for obs in self.observations.values()],
        }


class FeatureTrackSystem:
    """
    Disjoint-Set and graph-based tracking system for multi-view feature correspondences.
    
    Responsibilities:
    -----------------
    1. Aggregates pairwise matches across consecutive and nearby keyframes.
    2. Builds consistent multi-view feature tracks across N images.
    3. Strictly enforces 1-to-1 consistency: no single camera can observe the same track
       at multiple distinct 2D keypoints (duplicate/conflict prevention).
    4. Handles track merging when two previously separate tracks are connected by a
       verified pairwise correspondence without internal conflicts.
    5. Rejects conflicting matches and logs conflict statistics.
    6. Stores and indexes observations for fast 2D-3D correspondence lookup during PnP.
    """

    def __init__(self, min_track_length: int = 2):
        self.min_track_length = min_track_length
        self.tracks: Dict[int, FeatureTrack] = {}
        self._next_track_id: int = 0
        
        # Mapping from (image_id, keypoint_idx) -> track_id
        self._keypoint_to_track: Dict[Tuple[int, int], int] = {}
        
        # Telemetry counters
        self.total_matches_processed: int = 0
        self.duplicate_matches_count: int = 0
        self.conflict_rejections_count: int = 0
        self.merge_success_count: int = 0
        self.new_tracks_created: int = 0

    def add_match(
        self,
        img1_id: int,
        img1_name: str,
        kp1_idx: int,
        pt1: Tuple[float, float],
        img2_id: int,
        img2_name: str,
        kp2_idx: int,
        pt2: Tuple[float, float],
        desc1: Optional[np.ndarray] = None,
        desc2: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Integrates a verified pairwise correspondence between Image 1 and Image 2.
        
        Returns:
            True if match was successfully added or merged into a track.
            False if rejected due to conflict or duplicate.
        """
        self.total_matches_processed += 1
        node1 = (img1_id, kp1_idx)
        node2 = (img2_id, kp2_idx)

        t1_id = self._keypoint_to_track.get(node1)
        t2_id = self._keypoint_to_track.get(node2)

        # Case 1: Neither keypoint is currently part of any track
        if t1_id is None and t2_id is None:
            new_id = self._next_track_id
            self._next_track_id += 1
            track = FeatureTrack(track_id=new_id)
            track.add_observation(TrackObservation(img1_id, img1_name, kp1_idx, pt1, desc1))
            track.add_observation(TrackObservation(img2_id, img2_name, kp2_idx, pt2, desc2))
            self.tracks[new_id] = track
            self._keypoint_to_track[node1] = new_id
            self._keypoint_to_track[node2] = new_id
            self.new_tracks_created += 1
            return True

        # Case 2: Node 1 is in track T1, Node 2 is unassigned
        if t1_id is not None and t2_id is None:
            track1 = self.tracks[t1_id]
            # Conflict check: does track1 already have an observation for img2?
            if track1.has_image(img2_id):
                # Conflict: track already has a DIFFERENT keypoint in img2
                self.conflict_rejections_count += 1
                return False
            # Extend track
            track1.add_observation(TrackObservation(img2_id, img2_name, kp2_idx, pt2, desc2))
            self._keypoint_to_track[node2] = t1_id
            return True

        # Case 3: Node 2 is in track T2, Node 1 is unassigned
        if t1_id is None and t2_id is not None:
            track2 = self.tracks[t2_id]
            # Conflict check: does track2 already have an observation for img1?
            if track2.has_image(img1_id):
                self.conflict_rejections_count += 1
                return False
            # Extend track
            track2.add_observation(TrackObservation(img1_id, img1_name, kp1_idx, pt1, desc1))
            self._keypoint_to_track[node1] = t2_id
            return True

        # Case 4: Both are already in tracks
        if t1_id == t2_id:
            # Redundant/duplicate match inside the same track
            self.duplicate_matches_count += 1
            return True

        # Case 5: Different tracks T1 and T2 - Candidate track merge
        track1 = self.tracks[t1_id]
        track2 = self.tracks[t2_id]

        # Conflict check: do track1 and track2 share any images?
        common_images = track1.get_image_ids().intersection(track2.get_image_ids())
        if len(common_images) > 0:
            # Overlapping images would cause an image to have 2 distinct keypoints for the same track
            self.conflict_rejections_count += 1
            return False

        # Safe to merge: Merge smaller track into larger track
        if track1.length < track2.length:
            source_track, target_track = track1, track2
            source_id, target_id = t1_id, t2_id
        else:
            source_track, target_track = track2, track1
            source_id, target_id = t2_id, t1_id

        for obs in source_track.observations.values():
            target_track.add_observation(obs)
            self._keypoint_to_track[(obs.image_id, obs.keypoint_idx)] = target_id

        # Preserve 3D point if target lacked one but source had one
        if target_track.point_3d is None and source_track.point_3d is not None:
            target_track.point_3d = source_track.point_3d
            target_track.is_triangulated = source_track.is_triangulated
            target_track.reprojection_error = source_track.reprojection_error

        del self.tracks[source_id]
        self.merge_success_count += 1
        return True

    def get_track(self, track_id: int) -> Optional[FeatureTrack]:
        return self.tracks.get(track_id)

    def get_track_for_keypoint(self, image_id: int, keypoint_idx: int) -> Optional[FeatureTrack]:
        track_id = self._keypoint_to_track.get((image_id, keypoint_idx))
        if track_id is not None:
            return self.tracks.get(track_id)
        return None

    def get_tracks_for_image(self, image_id: int) -> List[FeatureTrack]:
        """Returns all tracks observed in a given camera view."""
        return [t for t in self.tracks.values() if t.has_image(image_id)]

    def get_common_tracks(self, img1_id: int, img2_id: int) -> List[FeatureTrack]:
        """Returns all tracks co-observed by both cameras."""
        return [t for t in self.tracks.values() if t.has_image(img1_id) and t.has_image(img2_id)]

    def get_triangulated_tracks_for_image(self, image_id: int) -> List[FeatureTrack]:
        """Returns all already-triangulated 3D tracks visible in this image."""
        return [t for t in self.tracks.values() if t.has_image(image_id) and t.is_triangulated]

    def get_valid_tracks(self, min_length: Optional[int] = None) -> List[FeatureTrack]:
        """Returns tracks meeting the minimum view support threshold."""
        threshold = min_length if min_length is not None else self.min_track_length
        return [t for t in self.tracks.values() if t.length >= threshold]

    def get_statistics(self) -> dict:
        """Computes comprehensive track statistics and length distributions."""
        tracks = list(self.tracks.values())
        total_tracks = len(tracks)

        # Track length distribution
        length_dist: Dict[str, int] = {}
        obs_per_image: Dict[int, int] = {}
        triangulated_count = 0

        for t in tracks:
            l = t.length
            bucket = str(l) if l < 6 else "6+"
            length_dist[bucket] = length_dist.get(bucket, 0) + 1
            if t.is_triangulated:
                triangulated_count += 1
            for img_id in t.observations.keys():
                obs_per_image[img_id] = obs_per_image.get(img_id, 0) + 1

        avg_length = float(np.mean([t.length for t in tracks])) if tracks else 0.0

        return {
            "total_tracks": total_tracks,
            "min_track_length_filter": self.min_track_length,
            "valid_tracks_count": len([t for t in tracks if t.length >= self.min_track_length]),
            "triangulated_tracks_count": triangulated_count,
            "average_track_length": round(avg_length, 2),
            "track_length_distribution": dict(sorted(length_dist.items(), key=lambda x: str(x[0]))),
            "observations_per_image": {str(k): v for k, v in sorted(obs_per_image.items())},
            "total_matches_processed": self.total_matches_processed,
            "duplicate_matches_count": self.duplicate_matches_count,
            "conflict_rejections_count": self.conflict_rejections_count,
            "merge_success_count": self.merge_success_count,
            "new_tracks_created": self.new_tracks_created,
        }

    def validate_track_integrity(self) -> dict:
        """
        Validates mathematical integrity of all feature tracks.
        
        Verifies:
        1. Every observation has non-negative keypoint index and finite (u, v) coordinates.
        2. No track contains duplicate observations from the same camera view.
        3. For triangulated tracks, point_3d is finite with shape (3,).
        4. Stored observation count matches reported track length.
        """
        tri_tracks = [t for t in self.tracks.values() if t.is_triangulated]
        total_tri_points = len(tri_tracks)

        anomalies = []
        valid_tri_count = 0
        total_stored_obs = 0

        for t in tri_tracks:
            # Check 3D point finiteness
            if t.point_3d is None or not np.all(np.isfinite(t.point_3d)):
                anomalies.append(f"Track {t.track_id}: non-finite or missing 3D point")
                continue
            if t.point_3d.shape != (3,):
                anomalies.append(f"Track {t.track_id}: point_3d shape != (3,)")
                continue

            # Check observations
            obs_images = set()
            track_valid = True
            for img_id, obs in t.observations.items():
                total_stored_obs += 1
                if img_id in obs_images:
                    anomalies.append(f"Track {t.track_id}: duplicate observation for image {img_id}")
                    track_valid = False
                    break
                obs_images.add(img_id)

                if obs.keypoint_idx < 0:
                    anomalies.append(f"Track {t.track_id}: invalid keypoint index {obs.keypoint_idx}")
                    track_valid = False
                    break

                u, v = obs.coords
                if not (np.isfinite(u) and np.isfinite(v)):
                    anomalies.append(f"Track {t.track_id}: non-finite pixel coordinate ({u}, {v})")
                    track_valid = False
                    break

            if track_valid and len(t.observations) == t.length:
                valid_tri_count += 1

        track_lengths = [t.length for t in tri_tracks]
        return {
            "total_triangulated_tracks": total_tri_points,
            "valid_tracks_count": valid_tri_count,
            "total_observations_count": total_stored_obs,
            "all_tracks_valid": len(anomalies) == 0,
            "anomalies_detected": anomalies[:20],
            "track_length_stats": {
                "min_length": int(np.min(track_lengths)) if track_lengths else 0,
                "max_length": int(np.max(track_lengths)) if track_lengths else 0,
                "mean_length": round(float(np.mean(track_lengths)), 2) if track_lengths else 0.0,
                "median_length": round(float(np.median(track_lengths)), 2) if track_lengths else 0.0,
            },
        }

    def export_json(self, output_path: Union[str, Path], include_unregistered: bool = False):
        """Saves tracks and observation statistics to a JSON report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        valid_tracks = [t for t in self.tracks.values() if t.length >= self.min_track_length]
        if not include_unregistered:
            tracks_to_export = [t for t in valid_tracks if t.is_triangulated]
        else:
            tracks_to_export = valid_tracks

        data = {
            "statistics": self.get_statistics(),
            "tracks": [t.to_dict() for t in tracks_to_export[:5000]], # Cap detailed list to 5000 for size
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

