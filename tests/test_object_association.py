"""
Unit Tests for Phase 5.3.2 Multi-View 2D-to-3D Object Association
Validates pinhole projection, mask boundary sampling, multi-view consensus classification,
threshold boundary behavior, and rigorous edge cases (negative depth, missing masks, out-of-bounds).
"""

from pathlib import Path
import unittest
import numpy as np

from src.association.mask_projection import CameraProjector, ProjectionResult, MaskSampleResult
from src.association.multiview_mask_associator import (
    MultiViewMaskAssociator,
    MultiViewAssociatorConfig,
    PointMaskAssociation,
)
from src.association.object_point_filter import ObjectPointFilter
from src.association.object_association_metrics import ObjectAssociationMetricsCalculator


class TestCameraProjectionAndMaskSampling(unittest.TestCase):
    """Tests geometric projection and mask occupancy sampling."""

    def setUp(self):
        self.K = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros((3, 1), dtype=np.float64)
        self.image_size = (1000, 1000)  # (width, height)

        # Create 1000x1000 synthetic mask with a positive box in center [300:700, 300:700]
        self.mask = np.zeros((1000, 1000), dtype=np.uint8)
        self.mask[300:700, 300:700] = 255

    def test_point_inside_binary_mask(self):
        # Point right at center (u=500, v=500)
        res = CameraProjector.sample_mask((500.0, 500.0), self.mask)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_inside)
        self.assertEqual(res.mask_value, 255)
        self.assertGreater(res.distance_to_boundary_px, 0.0)

    def test_point_outside_binary_mask(self):
        # Point in background (u=100, v=100)
        res = CameraProjector.sample_mask((100.0, 100.0), self.mask)
        self.assertTrue(res.is_valid)
        self.assertFalse(res.is_inside)
        self.assertEqual(res.mask_value, 0)
        self.assertLess(res.distance_to_boundary_px, 0.0)

    def test_point_on_mask_boundary(self):
        # Pixel at boundary (col=300, row=300)
        res = CameraProjector.sample_mask((300.0, 300.0), self.mask)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_inside)
        self.assertEqual(res.distance_to_boundary_px, 0.5)

    def test_invalid_or_missing_mask(self):
        res_none = CameraProjector.sample_mask((500.0, 500.0), None)
        self.assertFalse(res_none.is_valid)
        self.assertIn("None", res_none.rejection_reason)

        # Non-2D mask
        mask_3d = np.zeros((10, 10, 3), dtype=np.uint8)
        res_3d = CameraProjector.sample_mask((5.0, 5.0), mask_3d)
        self.assertFalse(res_3d.is_valid)

    def test_incorrect_image_dimensions_and_out_of_bounds(self):
        # Sampling out of mask dimensions
        res = CameraProjector.sample_mask((1500.0, 500.0), self.mask)
        self.assertFalse(res.is_valid)
        self.assertIn("out of mask bounds", res.rejection_reason)

        res_neg = CameraProjector.sample_mask((-10.0, 500.0), self.mask)
        self.assertFalse(res_neg.is_valid)

    def test_negative_camera_space_depth(self):
        # 3D point behind camera (Z = -5)
        pt_behind = np.array([0.0, 0.0, -5.0])
        proj = CameraProjector.project_point(
            point_3d_world=pt_behind,
            rotation_matrix=self.R,
            translation_vector=self.t,
            camera_matrix_k=self.K,
            image_size=self.image_size,
        )
        self.assertFalse(proj.is_valid)
        self.assertLess(proj.depth, 0.0)
        self.assertIn("behind or on the camera plane", proj.rejection_reason)

    def test_projection_outside_image_bounds(self):
        # 3D point far to the right (X = 20, Z = 5) -> u = 1000*4 + 500 = 4500 >> 1000
        pt_far = np.array([20.0, 0.0, 5.0])
        proj = CameraProjector.project_point(
            point_3d_world=pt_far,
            rotation_matrix=self.R,
            translation_vector=self.t,
            camera_matrix_k=self.K,
            image_size=self.image_size,
        )
        self.assertFalse(proj.is_valid)
        self.assertFalse(proj.is_inside_image)
        self.assertIn("outside image boundaries", proj.rejection_reason)

    def test_reprojection_error_calculation(self):
        err = CameraProjector.compute_reprojection_error((100.0, 200.0), (103.0, 204.0))
        # sqrt(3^2 + 4^2) = 5.0
        self.assertAlmostEqual(err, 5.0, places=3)


class TestMultiViewMaskAssociator(unittest.TestCase):
    """Tests multi-view evidence aggregation and classification logic."""

    def setUp(self):
        self.associator = MultiViewMaskAssociator(
            config=MultiViewAssociatorConfig(
                min_masked_observations=3,
                min_inside_observations=3,
                min_mask_support_ratio=0.60,
                weak_inside_observations=2,
                weak_min_ratio=0.35,
                image_size=(1000, 1000),
            )
        )
        self.K = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 4 identical registered cameras along z-axis
        self.cams = {}
        for cid in range(4):
            self.cams[cid] = {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [0.0, 0.0, 0.0],
                "is_registered": True,
            }

        # Mask for cameras
        self.mask = np.zeros((1000, 1000), dtype=np.uint8)
        self.mask[400:600, 400:600] = 255  # Box at center
        self.masks = {f"cam_{cid}_mask.png": self.mask for cid in range(4)}
        self.statuses = {f"cam_{cid}.png": "VALID" for cid in range(4)}

    def test_strong_object_candidate_classification(self):
        # 3D point at (0, 0, 5) -> projects to (500, 500), inside mask in all 4 views
        p3d = (0.0, 0.0, 5.0)
        observations = [
            {"image_id": cid, "image_name": f"cam_{cid}.png", "coords": [500.0, 500.0]}
            for cid in range(4)
        ]

        assoc = self.associator.evaluate_point(
            point_id=0,
            track_id=10,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.classification, "STRONG_OBJECT_CANDIDATE")
        self.assertEqual(assoc.inside_mask_observations, 4)
        self.assertEqual(assoc.masked_observations, 4)
        self.assertEqual(assoc.mask_support_ratio, 1.0)
        self.assertEqual(len(assoc.supporting_camera_ids), 4)

    def test_weak_object_candidate_classification(self):
        # Inside in 2 views out of 4 masked views (ratio = 2/4 = 0.50)
        p3d = (0.0, 0.0, 5.0)
        observations = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 1, "image_name": "cam_1.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 2, "image_name": "cam_2.png", "coords": [100.0, 100.0]},  # Outside
            {"image_id": 3, "image_name": "cam_3.png", "coords": [100.0, 100.0]},  # Outside
        ]

        assoc = self.associator.evaluate_point(
            point_id=1,
            track_id=11,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.classification, "WEAK_OBJECT_CANDIDATE")
        self.assertEqual(assoc.inside_mask_observations, 2)
        self.assertEqual(assoc.mask_support_ratio, 0.50)

    def test_background_candidate_classification(self):
        # Point inside image (u=700, v=700) but outside mask (mask is in [400:600, 400:600])
        # With Z=5.0: X = (700-500)*5/1000 = 1.0, Y = (700-500)*5/1000 = 1.0
        p3d = (1.0, 1.0, 5.0)
        observations = [
            {"image_id": cid, "image_name": f"cam_{cid}.png", "coords": [700.0, 700.0]}
            for cid in range(4)
        ]

        assoc = self.associator.evaluate_point(
            point_id=2,
            track_id=12,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.classification, "BACKGROUND_CANDIDATE")
        self.assertEqual(assoc.inside_mask_observations, 0)
        self.assertEqual(assoc.mask_support_ratio, 0.0)

    def test_insufficient_evidence_and_missing_cameras(self):
        # Only 1 observation available
        p3d = (0.0, 0.0, 5.0)
        observations = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},
            {"image_id": 99, "image_name": "unregistered.png", "coords": [500.0, 500.0]},  # Unregistered cam
        ]

        assoc = self.associator.evaluate_point(
            point_id=3,
            track_id=13,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.classification, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(assoc.masked_observations, 1)

    def test_threshold_boundary_behavior(self):
        # Exactly 3 inside out of 5 masked views -> ratio = 3/5 = 0.60
        # Should be STRONG_OBJECT_CANDIDATE because ratio >= 0.60 and inside >= 3 and masked >= 3
        cams_5 = {cid: self.cams[0] for cid in range(5)}
        masks_5 = {f"cam_{cid}_mask.png": self.mask for cid in range(5)}
        statuses_5 = {f"cam_{cid}.png": "VALID" for cid in range(5)}

        observations = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 1, "image_name": "cam_1.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 2, "image_name": "cam_2.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 3, "image_name": "cam_3.png", "coords": [100.0, 100.0]},  # Outside
            {"image_id": 4, "image_name": "cam_4.png", "coords": [100.0, 100.0]},  # Outside
        ]

        assoc = self.associator.evaluate_point(
            point_id=4,
            track_id=14,
            point_3d=(0.0, 0.0, 5.0),
            observations=observations,
            registered_cameras=cams_5,
            camera_intrinsics_k=self.K,
            masks=masks_5,
            frame_statuses=statuses_5,
        )

        self.assertEqual(assoc.classification, "STRONG_OBJECT_CANDIDATE")
        self.assertAlmostEqual(assoc.mask_support_ratio, 0.60, places=2)

    def test_missing_masks_excluded_from_denominator(self):
        """Audit 5.3.2.1: Missing masks must not count in denominator or as outside-mask observations."""
        p3d = (0.0, 0.0, 5.0)
        # Cam 2 has no mask in masks dict
        observations = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 1, "image_name": "cam_1.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 2, "image_name": "cam_missing.png", "coords": [500.0, 500.0]},  # Missing mask
        ]

        assoc = self.associator.evaluate_point(
            point_id=5,
            track_id=15,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,  # Contains cam_0, cam_1, but NOT cam_missing_mask.png
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.total_observations, 3)
        self.assertEqual(assoc.valid_projections, 3)
        self.assertEqual(assoc.masked_observations, 2)  # Denominator excludes missing mask
        self.assertEqual(assoc.inside_mask_observations, 2)
        self.assertEqual(assoc.outside_mask_observations, 0)  # Missing is NOT outside
        self.assertAlmostEqual(assoc.mask_support_ratio, 1.0, places=3)

    def test_invalid_projections_excluded_from_denominator(self):
        """Audit 5.3.2.1: Invalid projections (e.g. behind camera) must not count in support denominator."""
        p3d = (0.0, 0.0, 5.0)
        cams_with_behind = {
            0: self.cams[0],
            1: self.cams[1],
            2: {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [0.0, 0.0, -10.0],  # Puts point at Z_c = 5 - 10 = -5 (behind)
                "is_registered": True,
            },
        }
        observations = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},
            {"image_id": 1, "image_name": "cam_1.png", "coords": [500.0, 500.0]},
            {"image_id": 2, "image_name": "cam_2.png", "coords": [500.0, 500.0]},
        ]

        assoc = self.associator.evaluate_point(
            point_id=6,
            track_id=16,
            point_3d=p3d,
            observations=observations,
            registered_cameras=cams_with_behind,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.total_observations, 3)
        self.assertEqual(assoc.valid_projections, 2)
        self.assertEqual(assoc.masked_observations, 2)  # Denominator excludes invalid projection
        self.assertEqual(assoc.inside_mask_observations, 2)
        self.assertEqual(assoc.outside_mask_observations, 0)
        self.assertAlmostEqual(assoc.mask_support_ratio, 1.0, places=3)

    def test_duplicate_camera_observations_not_increasing_support(self):
        """Audit 5.3.2.1: Duplicate observations from the same camera must not inflate support."""
        p3d = (0.0, 0.0, 5.0)
        observations = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Cam 0 (Inside)
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Duplicate Cam 0!
            {"image_id": 1, "image_name": "cam_1.png", "coords": [500.0, 500.0]},  # Cam 1 (Inside)
        ]

        assoc = self.associator.evaluate_point(
            point_id=7,
            track_id=17,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.total_observations, 3)
        # Duplicate should be ignored, giving exactly 2 distinct masked views
        self.assertEqual(assoc.masked_observations, 2)
        self.assertEqual(assoc.inside_mask_observations, 2)
        self.assertEqual(len(assoc.supporting_camera_ids), 2)
        self.assertEqual(assoc.supporting_camera_ids, [0, 1])

    def test_zero_inside_mask_observations(self):
        """Audit 5.3.2.1: Point with zero inside-mask observations is classified as BACKGROUND_CANDIDATE."""
        p3d = (1.0, 1.0, 5.0)
        observations = [
            {"image_id": cid, "image_name": f"cam_{cid}.png", "coords": [100.0, 100.0]}
            for cid in range(4)
        ]

        assoc = self.associator.evaluate_point(
            point_id=8,
            track_id=18,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.masked_observations, 4)
        self.assertEqual(assoc.inside_mask_observations, 0)
        self.assertEqual(assoc.outside_mask_observations, 4)
        self.assertEqual(assoc.mask_support_ratio, 0.0)
        self.assertEqual(assoc.classification, "BACKGROUND_CANDIDATE")

    def test_three_valid_masked_observations_with_zero_mask_support(self):
        """Audit 5.3.2.1: Exactly 3 valid masked observations, all outside mask."""
        p3d = (1.0, 1.0, 5.0)
        observations = [
            {"image_id": cid, "image_name": f"cam_{cid}.png", "coords": [100.0, 100.0]}
            for cid in range(3)
        ]

        assoc = self.associator.evaluate_point(
            point_id=9,
            track_id=19,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.masked_observations, 3)
        self.assertEqual(assoc.inside_mask_observations, 0)
        self.assertEqual(assoc.mask_support_ratio, 0.0)
        self.assertEqual(assoc.classification, "BACKGROUND_CANDIDATE")

    def test_three_valid_masked_observations_at_threshold(self):
        """Audit 5.3.2.1: Exactly 3 valid masked observations with support at thresholds."""
        p3d = (0.0, 0.0, 5.0)
        # 3/3 inside -> ratio 1.0 >= 0.60, inside >= 3 -> STRONG
        observations_3 = [
            {"image_id": cid, "image_name": f"cam_{cid}.png", "coords": [500.0, 500.0]}
            for cid in range(3)
        ]
        assoc_3 = self.associator.evaluate_point(
            point_id=10,
            track_id=20,
            point_3d=p3d,
            observations=observations_3,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )
        self.assertEqual(assoc_3.classification, "STRONG_OBJECT_CANDIDATE")

        # 2 inside, 1 outside out of 3 masked views -> ratio = 2/3 = 0.667
        # inside_obs = 2 < 3 (min_inside is 3), so fails Strong, but satisfies Weak (inside >= 2, ratio >= 0.35)
        observations_2_of_3 = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 1, "image_name": "cam_1.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 2, "image_name": "cam_2.png", "coords": [100.0, 100.0]},  # Outside
        ]
        assoc_2_of_3 = self.associator.evaluate_point(
            point_id=11,
            track_id=21,
            point_3d=p3d,
            observations=observations_2_of_3,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )
        self.assertEqual(assoc_2_of_3.classification, "WEAK_OBJECT_CANDIDATE")

    def test_weak_classification_with_only_one_inside_mask_observation(self):
        """Audit 5.3.2.1: Weak candidate with 1 inside and 1 outside view (masked=2, inside=1, ratio=0.50)."""
        p3d = (0.0, 0.0, 5.0)
        observations = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 1, "image_name": "cam_1.png", "coords": [100.0, 100.0]},  # Outside
        ]

        assoc = self.associator.evaluate_point(
            point_id=12,
            track_id=22,
            point_3d=p3d,
            observations=observations,
            registered_cameras=self.cams,
            camera_intrinsics_k=self.K,
            masks=self.masks,
            frame_statuses=self.statuses,
        )

        self.assertEqual(assoc.masked_observations, 2)
        self.assertEqual(assoc.inside_mask_observations, 1)
        self.assertEqual(assoc.mask_support_ratio, 0.50)
        # Evaluates as WEAK_OBJECT_CANDIDATE under current rule (inside >= 1 and masked == 2 and ratio >= 0.50)
        self.assertEqual(assoc.classification, "WEAK_OBJECT_CANDIDATE")

    def test_classification_precedence_and_mutually_exclusive_classes(self):
        """Audit 5.3.2.1: Tests precedence rules and mutual exclusivity across all categories."""
        # 1 masked obs -> INSUFFICIENT_EVIDENCE regardless of inside status
        p3d = (0.0, 0.0, 5.0)
        obs_1 = [{"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]}]
        assoc_1 = self.associator.evaluate_point(13, 23, p3d, obs_1, self.cams, self.K, self.masks, self.statuses)
        self.assertEqual(assoc_1.classification, "INSUFFICIENT_EVIDENCE")

        # 4 masked obs, 1 inside -> ratio = 1/4 = 0.25 -> AMBIGUOUS (band [0.25, 0.60))
        obs_ambig = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 1, "image_name": "cam_1.png", "coords": [100.0, 100.0]},  # Outside
            {"image_id": 2, "image_name": "cam_2.png", "coords": [100.0, 100.0]},  # Outside
            {"image_id": 3, "image_name": "cam_3.png", "coords": [100.0, 100.0]},  # Outside
        ]
        assoc_ambig = self.associator.evaluate_point(14, 24, p3d, obs_ambig, self.cams, self.K, self.masks, self.statuses)
        self.assertEqual(assoc_ambig.classification, "AMBIGUOUS")

        # 5 masked obs, 1 inside -> ratio = 1/5 = 0.20 -> BACKGROUND_CANDIDATE (ratio < 0.25)
        cams_5 = {cid: self.cams[0] for cid in range(5)}
        masks_5 = {f"cam_{cid}_mask.png": self.mask for cid in range(5)}
        statuses_5 = {f"cam_{cid}.png": "VALID" for cid in range(5)}
        obs_bg = [
            {"image_id": 0, "image_name": "cam_0.png", "coords": [500.0, 500.0]},  # Inside
            {"image_id": 1, "image_name": "cam_1.png", "coords": [100.0, 100.0]},  # Outside
            {"image_id": 2, "image_name": "cam_2.png", "coords": [100.0, 100.0]},  # Outside
            {"image_id": 3, "image_name": "cam_3.png", "coords": [100.0, 100.0]},  # Outside
            {"image_id": 4, "image_name": "cam_4.png", "coords": [100.0, 100.0]},  # Outside
        ]
        assoc_bg = self.associator.evaluate_point(15, 25, p3d, obs_bg, cams_5, self.K, masks_5, statuses_5)
        self.assertEqual(assoc_bg.classification, "BACKGROUND_CANDIDATE")

    def test_camera_id_differing_from_keyframe_index(self):
        """Audit 5.3.2.1: Camera ID does not match keyframe index."""
        # Camera ID = 10, registered image_name = 'keyframe_0045_orig_00600.png'
        cams_mapped = {
            10: {
                "image_id": 10,
                "image_name": "keyframe_0045_orig_00600.png",
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [0.0, 0.0, 0.0],
                "is_registered": True,
            },
            11: {
                "image_id": 11,
                "image_name": "keyframe_0046_orig_00612.png",
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [0.0, 0.0, 0.0],
                "is_registered": True,
            },
        }
        masks_mapped = {
            "keyframe_0045_orig_00600_mask.png": self.mask,
            "keyframe_0046_orig_00612_mask.png": self.mask,
        }
        statuses_mapped = {
            "keyframe_0045_orig_00600.png": "VALID",
            "keyframe_0046_orig_00612.png": "VALID",
        }
        observations = [
            {"image_id": 10, "coords": [500.0, 500.0]},  # Cam ID 10, no explicit image_name in obs
            {"image_id": 11, "coords": [500.0, 500.0]},  # Cam ID 11, no explicit image_name in obs
        ]

        assoc = self.associator.evaluate_point(
            point_id=16,
            track_id=26,
            point_3d=(0.0, 0.0, 5.0),
            observations=observations,
            registered_cameras=cams_mapped,
            camera_intrinsics_k=self.K,
            masks=masks_mapped,
            frame_statuses=statuses_mapped,
        )

        self.assertEqual(assoc.masked_observations, 2)
        self.assertEqual(assoc.inside_mask_observations, 2)
        self.assertEqual(assoc.supporting_camera_ids, [10, 11])

    def test_correct_original_frame_to_mask_mapping(self):
        """Audit 5.3.2.1: Keyframe with original frame pattern correctly resolves mask file."""
        cams = {
            0: {
                "image_name": "keyframe_0024_orig_00300.png",
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [0.0, 0.0, 0.0],
                "is_registered": True,
            },
            1: {
                "image_name": "keyframe_0025_orig_00312.png",
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [0.0, 0.0, 0.0],
                "is_registered": True,
            },
        }
        masks = {
            "keyframe_0024_orig_00300_mask.png": self.mask,
            "keyframe_0025_orig_00312_mask.png": self.mask,
        }
        statuses = {
            "keyframe_0024_orig_00300.png": "VALID",
            "keyframe_0025_orig_00312.png": "VALID",
        }
        obs = [
            {"image_id": 0, "image_name": "keyframe_0024_orig_00300.png", "coords": [500.0, 500.0]},
            {"image_id": 1, "image_name": "keyframe_0025_orig_00312.png", "coords": [500.0, 500.0]},
        ]
        assoc = self.associator.evaluate_point(17, 27, (0.0, 0.0, 5.0), obs, cams, self.K, masks, statuses)
        self.assertEqual(assoc.masked_observations, 2)
        self.assertEqual(assoc.inside_mask_observations, 2)
        self.assertEqual(assoc.mask_support_ratio, 1.0)


class TestObjectPointFilterAndMetrics(unittest.TestCase):
    """Tests point cloud partitioning, PLY writing, and metrics aggregation."""

    def test_partition_and_spatial_statistics(self):
        assoc1 = PointMaskAssociation(
            point_id=0, track_id=1, point_3d=(1.0, 2.0, 3.0),
            total_observations=3, valid_projections=3, masked_observations=3,
            inside_mask_observations=3, outside_mask_observations=0, proj_inside_mask_observations=3,
            mask_support_ratio=1.0, mean_reprojection_error_px=1.2, median_reprojection_error_px=1.2,
            min_depth=2.0, max_depth=3.0, classification="STRONG_OBJECT_CANDIDATE",
            classification_reason="Strong support", supporting_camera_ids=[0, 1, 2],
        )
        assoc2 = PointMaskAssociation(
            point_id=1, track_id=2, point_3d=(4.0, 5.0, 6.0),
            total_observations=3, valid_projections=3, masked_observations=3,
            inside_mask_observations=0, outside_mask_observations=3, proj_inside_mask_observations=0,
            mask_support_ratio=0.0, mean_reprojection_error_px=1.5, median_reprojection_error_px=1.5,
            min_depth=4.0, max_depth=6.0, classification="BACKGROUND_CANDIDATE",
            classification_reason="Background", supporting_camera_ids=[],
        )

        partitions = ObjectPointFilter.partition_points([assoc1, assoc2])
        self.assertEqual(len(partitions["STRONG_OBJECT_CANDIDATE"]), 1)
        self.assertEqual(len(partitions["BACKGROUND_CANDIDATE"]), 1)

        stats = ObjectPointFilter.compute_spatial_statistics([assoc1, assoc2])
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["centroid"], [2.5, 3.5, 4.5])

        summary = ObjectAssociationMetricsCalculator.calculate_summary([assoc1, assoc2], stats)
        self.assertEqual(summary.strong_candidate_count, 1)
        self.assertEqual(summary.background_candidate_count, 1)
        self.assertAlmostEqual(summary.reprojection_error_mean_strong_px, 1.2, places=2)


if __name__ == "__main__":
    unittest.main()
