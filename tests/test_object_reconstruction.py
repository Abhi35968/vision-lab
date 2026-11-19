"""
Unit Tests for Phase 5.5 Object-Specific 3D Reconstruction.
Validates mesh geometry, face indices, normal calculations, silhouette projections,
IoU evaluation, missing-mask handling, and optimization reproducibility.
"""

import unittest
from pathlib import Path
import cv2
import numpy as np

from src.object_reconstruction.bottle_models import (
    BottleParameters,
    BottleProfileSample,
    Mesh3D,
    SilhouetteViewMetric,
)
from src.object_reconstruction.parametric_bottle import ParametricBottle
from src.object_reconstruction.silhouette_projector import SilhouetteProjector
from src.object_reconstruction.bottle_fitter import BottleFitter


class TestParametricBottleGeometry(unittest.TestCase):
    """Tests for parametric profile and 3D surface-of-revolution mesh generation."""

    def setUp(self):
        self.params = BottleParameters(
            cap_center_world=[-1.5, -2.6, 6.2],
            total_height=7.0,
            body_radius=1.2,
            axis_tilt_x_deg=0.0,
            axis_tilt_z_deg=0.0,
            axial_resolution=30,
            angular_resolution=24,
        )
        self.bottle = ParametricBottle(self.params)

    def test_profile_positive_radii(self):
        profile = self.bottle.generate_profile()
        self.assertEqual(len(profile), self.params.axial_resolution)
        for sample in profile:
            self.assertGreater(sample.radius, 0.0, f"Radius at z={sample.z_axial} must be positive")
            self.assertIn(sample.region, {"base", "body", "shoulder", "neck", "cap"})

    def test_profile_smooth_transitions(self):
        profile = self.bottle.generate_profile()
        radii = [s.radius for s in profile]
        # Maximum radius should equal body radius
        self.assertAlmostEqual(max(radii), self.params.body_radius, places=3)
        # Neck radius should equal neck_radius_ratio * body_radius
        neck_samples = [s.radius for s in profile if s.region == "neck"]
        self.assertGreater(len(neck_samples), 0)
        expected_neck_r = self.params.body_radius * self.params.neck_radius_ratio
        for r in neck_samples:
            self.assertAlmostEqual(r, expected_neck_r, places=3)

    def test_mesh_construction_validity(self):
        mesh = self.bottle.build_mesh()
        self.assertIsInstance(mesh, Mesh3D)
        self.assertGreater(mesh.num_vertices, 0)
        self.assertGreater(mesh.num_faces, 0)

        # Expected vertex count: n_axial * n_angular + 2 (top & bottom center vertices)
        expected_verts = self.params.axial_resolution * self.params.angular_resolution + 2
        self.assertEqual(mesh.num_vertices, expected_verts)

        # Expected face count: 2 * (n_axial - 1) * n_angular + 2 * n_angular (fans)
        expected_faces = 2 * (self.params.axial_resolution - 1) * self.params.angular_resolution + 2 * self.params.angular_resolution
        self.assertEqual(mesh.num_faces, expected_faces)

    def test_face_indices_in_bounds(self):
        mesh = self.bottle.build_mesh()
        self.assertTrue(np.all(mesh.faces >= 0))
        self.assertTrue(np.all(mesh.faces < mesh.num_vertices))

    def test_no_nan_or_inf_coordinates(self):
        mesh = self.bottle.build_mesh()
        self.assertFalse(np.isnan(mesh.vertices).any())
        self.assertFalse(np.isinf(mesh.vertices).any())
        self.assertFalse(np.isnan(mesh.normals).any())
        self.assertFalse(np.isinf(mesh.normals).any())

    def test_normal_vectors_normalized(self):
        mesh = self.bottle.build_mesh()
        norms = np.linalg.norm(mesh.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_watertight_euler_characteristic(self):
        """For a closed 2-manifold disk (sphere topology), V - E + F = 2."""
        mesh = self.bottle.build_mesh()
        # Count unique edges
        edges = set()
        for f in mesh.faces:
            for i in range(3):
                e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                edges.add(e)
        V = mesh.num_vertices
        E = len(edges)
        F = mesh.num_faces
        euler_char = V - E + F
        self.assertEqual(euler_char, 2, f"Closed manifold mesh must satisfy V - E + F = 2, got {euler_char}")

    def test_deterministic_reproducibility(self):
        mesh1 = self.bottle.build_mesh()
        mesh2 = ParametricBottle(self.params).build_mesh()
        np.testing.assert_array_equal(mesh1.vertices, mesh2.vertices)
        np.testing.assert_array_equal(mesh1.faces, mesh2.faces)


class TestSilhouetteProjector(unittest.TestCase):
    """Tests for projection math, back-face culling, and IoU evaluation."""

    def setUp(self):
        self.K = np.array([
            [1300.0, 0.0, 540.0],
            [0.0, 1300.0, 960.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.projector = SilhouetteProjector(intrinsic_matrix=self.K, image_shape=(1920, 1080))
        self.params = BottleParameters(
            cap_center_world=[0.0, -2.0, 5.0],
            total_height=6.0,
            body_radius=1.0,
            axis_tilt_x_deg=0.0,
            axis_tilt_z_deg=0.0,
            axial_resolution=20,
            angular_resolution=16,
        )
        self.mesh = ParametricBottle(self.params).build_mesh()
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros((3, 1), dtype=np.float64)

    def test_rasterization_binary_output(self):
        mask, is_valid = self.projector.project_and_rasterize(self.mesh, self.R, self.t)
        self.assertTrue(is_valid)
        self.assertEqual(mask.shape, (1920, 1080))
        unique_vals = np.unique(mask)
        self.assertTrue(set(unique_vals).issubset({0, 255}))
        self.assertGreater(np.count_nonzero(mask > 0), 1000)

    def test_cheirality_behind_camera(self):
        # Move camera far in front so object is behind camera: t_z = -10.0 => Z_cam = -5.0
        t_behind = np.array([[0.0], [0.0], [-10.0]])
        mask, is_valid = self.projector.project_and_rasterize(self.mesh, self.R, t_behind)
        self.assertFalse(is_valid)
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_evaluate_view_perfect_match(self):
        # Render a mask and evaluate it against itself
        rend_mask, _ = self.projector.project_and_rasterize(self.mesh, self.R, self.t)
        metric = self.projector.evaluate_view(
            camera_id=0,
            image_name="test.png",
            is_train_split=True,
            mesh=self.mesh,
            R=self.R,
            t=self.t,
            observed_mask=rend_mask,
        )
        self.assertIsNotNone(metric)
        self.assertAlmostEqual(metric.iou, 1.0, places=4)
        self.assertAlmostEqual(metric.precision, 1.0, places=4)
        self.assertAlmostEqual(metric.recall, 1.0, places=4)

    def test_missing_mask_handling(self):
        # Empty mask returns None (graceful exclusion, not zero IoU)
        empty_mask = np.zeros((1920, 1080), dtype=np.uint8)
        metric = self.projector.evaluate_view(
            camera_id=1,
            image_name="empty.png",
            is_train_split=False,
            mesh=self.mesh,
            R=self.R,
            t=self.t,
            observed_mask=empty_mask,
        )
        self.assertIsNone(metric)

    def test_downsampled_rasterization_consistency(self):
        mask_full, ok1 = self.projector.project_and_rasterize(self.mesh, self.R, self.t, downsample_factor=1)
        mask_ds, ok2 = self.projector.project_and_rasterize(self.mesh, self.R, self.t, downsample_factor=4)
        self.assertTrue(ok1 and ok2)
        self.assertEqual(mask_ds.shape, (480, 270))
        # Area ratio should be roughly 4^2 = 16
        area_full = np.count_nonzero(mask_full > 0)
        area_ds = np.count_nonzero(mask_ds > 0)
        ratio = area_full / area_ds
        self.assertAlmostEqual(ratio, 16.0, delta=2.5)


class TestMeshExport(unittest.TestCase):
    """Tests for PLY and OBJ mesh file serialization."""

    def setUp(self):
        self.params = BottleParameters(
            cap_center_world=[0.0, 0.0, 5.0],
            total_height=5.0,
            body_radius=1.0,
            axis_tilt_x_deg=0.0,
            axis_tilt_z_deg=0.0,
            axial_resolution=10,
            angular_resolution=12,
        )
        self.mesh = ParametricBottle(self.params).build_mesh()
        self.test_dir = Path("outputs/test_mesh_export")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_export_ply(self):
        ply_file = self.test_dir / "test_bottle.ply"
        self.mesh.export_ply(ply_file)
        self.assertTrue(ply_file.exists())
        self.assertGreater(ply_file.stat().st_size, 100)

        with open(ply_file, "r") as f:
            header = [f.readline() for _ in range(16)]
        header_text = "".join(header)
        self.assertIn("element vertex", header_text)
        self.assertIn("element face", header_text)
        self.assertIn("end_header", header_text)

    def test_export_obj(self):
        obj_file = self.test_dir / "test_bottle.obj"
        self.mesh.export_obj(obj_file)
        self.assertTrue(obj_file.exists())
        self.assertGreater(obj_file.stat().st_size, 100)

        with open(obj_file, "r") as f:
            lines = f.readlines()
        v_lines = [l for l in lines if l.startswith("v ")]
        vn_lines = [l for l in lines if l.startswith("vn ")]
        f_lines = [l for l in lines if l.startswith("f ")]
        self.assertEqual(len(v_lines), self.mesh.num_vertices)
        self.assertEqual(len(vn_lines), self.mesh.num_vertices)
        self.assertEqual(len(f_lines), self.mesh.num_faces)


if __name__ == "__main__":
    unittest.main()
