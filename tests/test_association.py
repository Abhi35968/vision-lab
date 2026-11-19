"""
Association Module Unit and Integration Test Suite
Validates point-in-bounding-box testing, boundary conditions, multi-box ambiguity policies,
track-level multi-view gating (STRONG, WEAK, AMBIGUOUS, UNASSOCIATED), 3D point clustering,
serialization schemas, and CLI input validation.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from src.detection.detection_models import Detection
from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem, TrackObservation
from src.association.association_models import (
    AssociationStatus,
    ObservationAssociation,
    TrackAssociation,
    ObjectPointGroup,
    AssociationSummary,
)
from src.association.observation_matcher import ObservationMatcher
from src.association.track_associator import TrackAssociator, AssociationConfig


class TestAssociationModule(unittest.TestCase):
    """Comprehensive unit and integration tests for Phase 4.2 Association."""

    def setUp(self):
        self.matcher = ObservationMatcher(ambiguity_policy="smallest_box")
        self.config = AssociationConfig(
            min_supporting_views=2,
            min_matching_ratio=0.50,
            min_confidence=0.05,
            min_class_purity=0.70,
        )
        self.associator = TrackAssociator(self.config)

    def test_point_inside_bounding_box(self):
        """Verifies point strictly inside bounding box returns True."""
        # Box from (100, 100) to (200, 300)
        self.assertTrue(ObservationMatcher.is_point_inside_box(150.0, 200.0, 100.0, 100.0, 200.0, 300.0))
        # Inverted box coordinates (x2 < x1, y2 < y1)
        self.assertTrue(ObservationMatcher.is_point_inside_box(150.0, 200.0, 200.0, 300.0, 100.0, 100.0))

    def test_point_outside_bounding_box(self):
        """Verifies points outside bounding box return False across all quadrants."""
        # Box from (100, 100) to (200, 200)
        self.assertFalse(ObservationMatcher.is_point_inside_box(50.0, 150.0, 100.0, 100.0, 200.0, 200.0))  # Left
        self.assertFalse(ObservationMatcher.is_point_inside_box(250.0, 150.0, 100.0, 100.0, 200.0, 200.0)) # Right
        self.assertFalse(ObservationMatcher.is_point_inside_box(150.0, 50.0, 100.0, 100.0, 200.0, 200.0))  # Above
        self.assertFalse(ObservationMatcher.is_point_inside_box(150.0, 250.0, 100.0, 100.0, 200.0, 200.0)) # Below

    def test_boundary_coordinates(self):
        """Verifies inclusive boundary checks: edges and exact corners return True."""
        x1, y1, x2, y2 = 100.0, 200.0, 300.0, 400.0
        # Corners
        self.assertTrue(ObservationMatcher.is_point_inside_box(x1, y1, x1, y1, x2, y2))
        self.assertTrue(ObservationMatcher.is_point_inside_box(x2, y1, x1, y1, x2, y2))
        self.assertTrue(ObservationMatcher.is_point_inside_box(x1, y2, x1, y1, x2, y2))
        self.assertTrue(ObservationMatcher.is_point_inside_box(x2, y2, x1, y1, x2, y2))
        # Edges
        self.assertTrue(ObservationMatcher.is_point_inside_box(200.0, y1, x1, y1, x2, y2))
        self.assertTrue(ObservationMatcher.is_point_inside_box(200.0, y2, x1, y1, x2, y2))
        self.assertTrue(ObservationMatcher.is_point_inside_box(x1, 300.0, x1, y1, x2, y2))
        self.assertTrue(ObservationMatcher.is_point_inside_box(x2, 300.0, x1, y1, x2, y2))

    def test_multiple_overlapping_boxes(self):
        """
        Verifies behavior when an observation falls inside multiple overlapping boxes.
        Tests is_ambiguous=True and behavior under smallest_box, highest_confidence, and none policies.
        """
        # Det A: larger box (area = 200 * 200 = 40000), higher confidence (0.95), class "dining table"
        det_a = Detection.from_xyxy(
            detection_id=1, camera_id=0, image_filename="f0.png",
            class_id=60, class_name="dining table", confidence=0.95,
            x1=100.0, y1=100.0, x2=300.0, y2=300.0
        )
        # Det B: smaller box (area = 100 * 100 = 10000), lower confidence (0.75), class "chair"
        det_b = Detection.from_xyxy(
            detection_id=2, camera_id=0, image_filename="f0.png",
            class_id=56, class_name="chair", confidence=0.75,
            x1=120.0, y1=120.0, x2=220.0, y2=220.0
        )
        dets = [det_a, det_b]
        obs_x, obs_y = 150.0, 150.0  # inside both

        # 1. Policy: smallest_box
        matcher_smallest = ObservationMatcher(ambiguity_policy="smallest_box")
        res_smallest = matcher_smallest.match_observation(
            obs_x, obs_y, dets, track_id=10, camera_id=0, image_filename="f0.png"
        )
        self.assertTrue(res_smallest.is_matched)
        self.assertTrue(res_smallest.is_ambiguous)
        self.assertEqual(res_smallest.detection_id, 2)
        self.assertEqual(res_smallest.class_name, "chair")
        self.assertEqual(len(res_smallest.containing_detection_ids), 2)

        # 2. Policy: highest_confidence
        matcher_conf = ObservationMatcher(ambiguity_policy="highest_confidence")
        res_conf = matcher_conf.match_observation(
            obs_x, obs_y, dets, track_id=10, camera_id=0, image_filename="f0.png"
        )
        self.assertTrue(res_conf.is_matched)
        self.assertTrue(res_conf.is_ambiguous)
        self.assertEqual(res_conf.detection_id, 1)
        self.assertEqual(res_conf.class_name, "dining table")

        # 3. Policy: none
        matcher_none = ObservationMatcher(ambiguity_policy="none")
        res_none = matcher_none.match_observation(
            obs_x, obs_y, dets, track_id=10, camera_id=0, image_filename="f0.png"
        )
        self.assertTrue(res_none.is_matched)
        self.assertTrue(res_none.is_ambiguous)
        self.assertIsNone(res_none.detection_id)
        self.assertIsNone(res_none.class_name)

    def test_missing_detection_in_frame(self):
        """Verifies observation in a frame without detections produces an unassociated match record."""
        res = self.matcher.match_observation(
            obs_x=100.0, obs_y=150.0, detections=[],
            track_id=42, camera_id=1, image_filename="empty_frame.png", point_id=5
        )
        self.assertFalse(res.is_matched)
        self.assertFalse(res.is_ambiguous)
        self.assertIsNone(res.detection_id)
        self.assertIsNone(res.class_name)
        self.assertEqual(res.point_id, 5)

    def test_single_observation_track(self):
        """
        Verifies a track with only 1 observation inside a detection is classified as WEAK,
        preventing single-view false promotion.
        """
        # Track 1 has 3 observations in cameras 0, 1, 2. Only camera 0 observation is inside detection.
        track = FeatureTrack(track_id=1)
        track.add_observation(TrackObservation(0, "f0.png", 10, (150.0, 150.0)))
        track.add_observation(TrackObservation(1, "f1.png", 12, (500.0, 500.0)))
        track.add_observation(TrackObservation(2, "f2.png", 14, (600.0, 600.0)))
        track.is_triangulated = True
        track.point_3d = np.array([0.1, 0.2, 1.5])

        obs_assocs = [
            ObservationAssociation(
                track_id=1, camera_id=0, image_filename="f0.png", coords=(150.0, 150.0),
                is_matched=True, detection_id=100, class_id=56, class_name="chair", confidence=0.88
            ),
            ObservationAssociation(
                track_id=1, camera_id=1, image_filename="f1.png", coords=(500.0, 500.0),
                is_matched=False
            ),
            ObservationAssociation(
                track_id=1, camera_id=2, image_filename="f2.png", coords=(600.0, 600.0),
                is_matched=False
            ),
        ]

        tracks = {1: track}
        t_assocs = self.associator.associate_tracks(tracks, obs_assocs)
        self.assertEqual(len(t_assocs), 1)
        ta = t_assocs[0]
        self.assertEqual(ta.association_status, AssociationStatus.WEAK.value)
        self.assertEqual(ta.matched_observations, 1)
        self.assertAlmostEqual(ta.matching_ratio, 1.0 / 3.0, places=2)
        self.assertEqual(ta.assigned_class_name, "chair")
        self.assertEqual(ta.dominant_class_views_count, 1)

    def test_multiview_supported_track(self):
        """
        Verifies a track observed in >= 2 distinct views with >= 50% matching ratio
        and sufficient confidence is classified as STRONG.
        """
        track = FeatureTrack(track_id=2)
        track.add_observation(TrackObservation(0, "f0.png", 10, (150.0, 150.0)))
        track.add_observation(TrackObservation(1, "f1.png", 12, (155.0, 152.0)))
        track.add_observation(TrackObservation(2, "f2.png", 14, (160.0, 158.0)))
        track.is_triangulated = True
        track.point_3d = np.array([0.5, -0.2, 2.0])

        obs_assocs = [
            ObservationAssociation(
                track_id=2, camera_id=0, image_filename="f0.png", coords=(150.0, 150.0),
                is_matched=True, detection_id=101, class_id=56, class_name="chair", confidence=0.85
            ),
            ObservationAssociation(
                track_id=2, camera_id=1, image_filename="f1.png", coords=(155.0, 152.0),
                is_matched=True, detection_id=102, class_id=56, class_name="chair", confidence=0.82
            ),
            ObservationAssociation(
                track_id=2, camera_id=2, image_filename="f2.png", coords=(160.0, 158.0),
                is_matched=False
            ),
        ]

        tracks = {2: track}
        t_assocs = self.associator.associate_tracks(tracks, obs_assocs)
        self.assertEqual(len(t_assocs), 1)
        ta = t_assocs[0]
        self.assertEqual(ta.association_status, AssociationStatus.STRONG.value)
        self.assertEqual(ta.matched_observations, 2)
        self.assertAlmostEqual(ta.matching_ratio, 2.0 / 3.0, places=2)
        self.assertEqual(ta.assigned_class_name, "chair")
        self.assertEqual(ta.dominant_class_views_count, 2)
        self.assertEqual(ta.dominant_class_views, [0, 1])

    def test_conflicting_class_assignments(self):
        """
        Verifies a track with conflicting detections across views (e.g., 2 views chair, 2 views bench)
        or low purity is classified as AMBIGUOUS.
        """
        track = FeatureTrack(track_id=3)
        track.add_observation(TrackObservation(0, "f0.png", 1, (100.0, 100.0)))
        track.add_observation(TrackObservation(1, "f1.png", 2, (105.0, 102.0)))
        track.add_observation(TrackObservation(2, "f2.png", 3, (110.0, 105.0)))
        track.add_observation(TrackObservation(3, "f3.png", 4, (115.0, 108.0)))

        # 2 views chair, 2 views bench -> purity = 2/4 = 0.5 < min_class_purity (0.70)
        obs_assocs = [
            ObservationAssociation(
                track_id=3, camera_id=0, image_filename="f0.png", coords=(100.0, 100.0),
                is_matched=True, detection_id=1, class_id=56, class_name="chair", confidence=0.9
            ),
            ObservationAssociation(
                track_id=3, camera_id=1, image_filename="f1.png", coords=(105.0, 102.0),
                is_matched=True, detection_id=2, class_id=56, class_name="chair", confidence=0.88
            ),
            ObservationAssociation(
                track_id=3, camera_id=2, image_filename="f2.png", coords=(110.0, 105.0),
                is_matched=True, detection_id=3, class_id=15, class_name="bench", confidence=0.85
            ),
            ObservationAssociation(
                track_id=3, camera_id=3, image_filename="f3.png", coords=(115.0, 108.0),
                is_matched=True, detection_id=4, class_id=15, class_name="bench", confidence=0.87
            ),
        ]

        tracks = {3: track}
        t_assocs = self.associator.associate_tracks(tracks, obs_assocs)
        ta = t_assocs[0]
        self.assertEqual(ta.association_status, AssociationStatus.AMBIGUOUS.value)
        self.assertTrue(ta.has_conflicting_classes)

    def test_missing_3d_point(self):
        """
        Verifies un-triangulated tracks (missing 3D coordinates) are handled gracefully
        and are excluded from 3D object point clusters.
        """
        track = FeatureTrack(track_id=4)
        track.add_observation(TrackObservation(0, "f0.png", 1, (100.0, 100.0)))
        track.add_observation(TrackObservation(1, "f1.png", 2, (105.0, 100.0)))
        track.is_triangulated = False
        track.point_3d = None

        obs_assocs = [
            ObservationAssociation(
                track_id=4, camera_id=0, image_filename="f0.png", coords=(100.0, 100.0),
                is_matched=True, detection_id=10, class_id=56, class_name="chair", confidence=0.9
            ),
            ObservationAssociation(
                track_id=4, camera_id=1, image_filename="f1.png", coords=(105.0, 100.0),
                is_matched=True, detection_id=11, class_id=56, class_name="chair", confidence=0.9
            ),
        ]

        tracks = {4: track}
        t_assocs = self.associator.associate_tracks(tracks, obs_assocs)
        self.assertEqual(t_assocs[0].association_status, AssociationStatus.STRONG.value)
        self.assertFalse(t_assocs[0].is_triangulated)
        self.assertIsNone(t_assocs[0].point_3d)

        # When grouping 3D points, non-triangulated tracks should NOT produce 3D clusters
        groups = self.associator.group_object_points(t_assocs, obs_assocs)
        self.assertEqual(len(groups), 0)

    def test_object_point_group_extents(self):
        """Verifies 3D bounding box extents, centroid, and arbitrary_units for grouped 3D points."""
        # Create 2 triangulated tracks for "chair"
        t1 = FeatureTrack(track_id=1)
        t1.add_observation(TrackObservation(0, "f0.png", 1, (100.0, 100.0)))
        t1.add_observation(TrackObservation(1, "f1.png", 2, (105.0, 100.0)))
        t1.is_triangulated = True
        t1.point_3d = np.array([1.0, 2.0, 3.0])

        t2 = FeatureTrack(track_id=2)
        t2.add_observation(TrackObservation(0, "f0.png", 3, (120.0, 120.0)))
        t2.add_observation(TrackObservation(1, "f1.png", 4, (125.0, 120.0)))
        t2.is_triangulated = True
        t2.point_3d = np.array([3.0, 6.0, 5.0])

        obs_assocs = [
            ObservationAssociation(
                track_id=1, camera_id=0, image_filename="f0.png", coords=(100.0, 100.0),
                is_matched=True, detection_id=10, class_id=56, class_name="chair", confidence=0.8
            ),
            ObservationAssociation(
                track_id=1, camera_id=1, image_filename="f1.png", coords=(105.0, 100.0),
                is_matched=True, detection_id=11, class_id=56, class_name="chair", confidence=0.8
            ),
            ObservationAssociation(
                track_id=2, camera_id=0, image_filename="f0.png", coords=(120.0, 120.0),
                is_matched=True, detection_id=10, class_id=56, class_name="chair", confidence=0.85
            ),
            ObservationAssociation(
                track_id=2, camera_id=1, image_filename="f1.png", coords=(125.0, 120.0),
                is_matched=True, detection_id=11, class_id=56, class_name="chair", confidence=0.85
            ),
        ]

        tracks = {1: t1, 2: t2}
        t_assocs = self.associator.associate_tracks(tracks, obs_assocs)
        groups = self.associator.group_object_points(t_assocs, obs_assocs)

        self.assertIn("chair", groups)
        grp = groups["chair"]
        self.assertEqual(grp.total_points, 2)
        self.assertEqual(grp.spatial_bounds["unit"], "arbitrary_units")
        self.assertEqual(grp.spatial_bounds["centroid"], [2.0, 4.0, 4.0])
        self.assertEqual(grp.spatial_bounds["min_bounds"], [1.0, 2.0, 3.0])
        self.assertEqual(grp.spatial_bounds["max_bounds"], [3.0, 6.0, 5.0])
        self.assertEqual(grp.spatial_bounds["extents"], [2.0, 4.0, 2.0])

    def test_empty_detection_results(self):
        """Verifies summary and associator handle zero detections gracefully without division by zero."""
        track = FeatureTrack(track_id=0)
        track.add_observation(TrackObservation(0, "f0.png", 1, (100.0, 100.0)))
        track.is_triangulated = True
        track.point_3d = np.array([0.0, 0.0, 1.0])

        obs_assocs = [
            ObservationAssociation(
                track_id=0, camera_id=0, image_filename="f0.png", coords=(100.0, 100.0),
                is_matched=False
            )
        ]

        tracks = {0: track}
        t_assocs = self.associator.associate_tracks(tracks, obs_assocs)
        groups = self.associator.group_object_points(t_assocs, obs_assocs)
        summary = self.associator.compute_summary(tracks, t_assocs, obs_assocs, groups)

        self.assertEqual(summary.total_feature_tracks, 1)
        self.assertEqual(summary.matched_observations, 0)
        self.assertEqual(summary.unassociated_tracks, 1)
        self.assertEqual(summary.strongly_associated_tracks, 0)
        self.assertEqual(summary.weakly_associated_tracks, 0)
        self.assertEqual(len(summary.point_counts_per_class), 0)

        s_dict = summary.to_dict()
        self.assertEqual(s_dict["observation_level_metrics"]["match_rate_percent"], 0.0)
        self.assertEqual(s_dict["track_level_metrics"]["association_rate_percent"], 0.0)

    def test_json_serialization(self):
        """Verifies all association dataclasses serialize to JSON-compliant dictionaries."""
        obs = ObservationAssociation(
            track_id=1, camera_id=0, image_filename="f0.png", coords=(10.5, 20.5),
            point_id=100, is_matched=True, detection_id=5, class_id=56, class_name="chair",
            confidence=0.88888, is_ambiguous=False, bbox_xyxy=(5.0, 5.0, 25.0, 35.0)
        )
        d_obs = obs.to_dict()
        self.assertEqual(d_obs["track_id"], 1)
        self.assertEqual(d_obs["point_id"], 100)
        self.assertEqual(d_obs["coords"], [10.5, 20.5])
        self.assertEqual(d_obs["confidence"], 0.8889)
        self.assertEqual(d_obs["bbox_xyxy"], [5.0, 5.0, 25.0, 35.0])
        # Verify valid JSON
        self.assertTrue(len(json.dumps(d_obs)) > 0)

        ta = TrackAssociation(
            track_id=1, point_id=100, is_triangulated=True, point_3d=[1.0, 2.0, 3.0],
            total_observations=3, matched_observations=2, matching_ratio=0.66666,
            association_status="STRONG", assigned_class_id=56, assigned_class_name="chair",
            dominant_class_count=2, dominant_class_views_count=2, dominant_class_views=[0, 1],
            mean_confidence=0.85
        )
        d_ta = ta.to_dict()
        self.assertEqual(d_ta["unit"], "arbitrary_units")
        self.assertEqual(d_ta["matching_ratio"], 0.667)
        self.assertTrue(len(json.dumps(d_ta)) > 0)

    def test_cli_missing_input_validation(self):
        """Verifies CLI exits cleanly with error code when required input files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fake_sfm = tmp_path / "sfm"
            fake_det = tmp_path / "detection"
            fake_sfm.mkdir()
            fake_det.mkdir()

            cmd = [
                sys.executable,
                "main.py",
                "--phase", "association",
                "--sfm-dir", str(fake_sfm),
                "--detection-dir", str(fake_det),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertTrue(
                "[ERROR]" in res.stdout or "[ERROR]" in res.stderr or "Error" in res.stdout or "Error" in res.stderr
            )


if __name__ == "__main__":
    unittest.main()
