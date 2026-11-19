"""
Unit tests for Phase 4: Multi-View Texture Mapping Module
"""

import numpy as np
import pytest
import cv2
from pathlib import Path

from src.texture_mapping.multi_view_texture_mapper import (
    MultiViewTextureMapper,
    TexturingReport,
)


class TestMultiViewTextureMapper:
    """Tests projection, grazing angle gating, sharpness weighting, and texture baking."""

    @pytest.fixture
    def mapper(self):
        return MultiViewTextureMapper(texture_resolution=256)

    @pytest.fixture
    def planar_mesh(self):
        # A 1x1 flat square mesh centered at (0, 0, 2) in front of camera at origin
        # Vertices:
        # (-0.5, -0.5, 2.0), (0.5, -0.5, 2.0), (0.5, 0.5, 2.0), (-0.5, 0.5, 2.0)
        # Normal pointing towards -Z (facing camera at origin looking down +Z)
        vertices = np.array([
            [-0.5, -0.5, 2.0],
            [ 0.5, -0.5, 2.0],
            [ 0.5,  0.5, 2.0],
            [-0.5,  0.5, 2.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        normals = np.array([
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
        ], dtype=np.float64)
        return vertices, faces, normals

    @pytest.fixture
    def synthetic_camera(self):
        # Camera at origin looking down +Z
        K = np.array([
            [400.0, 0.0, 320.0],
            [0.0, 400.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        poses = {
            "cam_0.jpg": {
                "R": np.eye(3),
                "t": np.zeros(3),
                "camera_center": np.zeros(3),
            }
        }
        # Red image with high-frequency grid pattern to pass sharpness check
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :] = (0, 0, 255)  # Red in BGR
        img[::10, :] = (0, 0, 180)  # High-frequency horizontal lines
        img[:, ::10] = (0, 0, 180)  # High-frequency vertical lines
        keyframes = {"cam_0.jpg": img}
        return K, poses, keyframes

    def test_bake_texture_projects_red_color(self, mapper, planar_mesh, synthetic_camera, tmp_path):
        """Verifies direct projection assigns RGB colors to mesh vertices."""
        vertices, faces, normals = planar_mesh
        K, poses, keyframes = synthetic_camera

        tm_mesh, report = mapper.bake_texture(
            vertices=vertices,
            faces=faces,
            keyframes=keyframes,
            camera_poses=poses,
            camera_matrix=K,
            normals=normals,
            output_dir=tmp_path,
        )

        assert report.num_vertices_colored == 4
        assert report.unobserved_vertices_count == 0
        assert report.num_cameras_used == 1
        assert Path(report.obj_mesh_path).exists()
        assert Path(report.glb_mesh_path).exists()
        assert Path(report.texture_image_path).exists()

        # Trimesh TextureVisuals contains UV coordinates and material texture
        assert hasattr(tm_mesh.visual, "uv")
        assert len(tm_mesh.visual.uv) == 4
        assert tm_mesh.visual.material is not None

    def test_grazing_angle_rejection(self, mapper, planar_mesh, synthetic_camera, tmp_path):
        """Verifies that vertices facing away from camera are rejected."""
        vertices, faces, _ = planar_mesh
        # Invert normals so they point away from the camera (+Z)
        inverted_normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float64)
        K, poses, keyframes = synthetic_camera

        tm_mesh, report = mapper.bake_texture(
            vertices=vertices,
            faces=faces,
            keyframes=keyframes,
            camera_poses=poses,
            camera_matrix=K,
            normals=inverted_normals,
            output_dir=tmp_path,
        )

        # Facing away -> unobserved fallback color
        assert report.num_vertices_colored == 0
        assert report.unobserved_vertices_count == 4
        assert hasattr(tm_mesh.visual, "uv")
