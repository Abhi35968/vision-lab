"""
Unit tests for Phase 5: GLB Export and Mobile Sharing QR Code Module
"""

import numpy as np
import pytest
import trimesh
from pathlib import Path
from PIL import Image

from src.export.glb_exporter import GLBExporter, ExportResult


class TestGLBExporter:
    """Tests GLB binary generation, texture embedding, and strict QR code validation."""

    @pytest.fixture
    def exporter(self, tmp_path):
        return GLBExporter(default_output_dir=tmp_path)

    @pytest.fixture
    def simple_trimesh(self):
        return trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    def test_export_basic_glb(self, exporter, simple_trimesh):
        """Verifies binary GLB export for a Trimesh object."""
        res = exporter.export(
            mesh_or_trimesh=simple_trimesh,
            filename="test_cube.glb",
            share_url="http://192.168.1.50:8501",
        )

        assert isinstance(res, ExportResult)
        assert Path(res.glb_path).exists()
        assert res.file_size_kb > 0.1
        assert res.num_vertices == len(simple_trimesh.vertices)
        assert res.num_faces == len(simple_trimesh.faces)
        assert res.qr_code_image_path is not None
        assert Path(res.qr_code_image_path).exists()

    def test_export_scene_object(self, exporter, simple_trimesh):
        """Verifies exporting a trimesh.Scene unpacks geometry into GLB."""
        scene = trimesh.Scene(geometry=simple_trimesh)
        res = exporter.export(mesh_or_trimesh=scene, filename="from_scene.glb")
        assert Path(res.glb_path).exists()
        assert res.file_size_kb > 0.1

    def test_qr_code_valid_network_url(self, exporter):
        """Verifies QR code generation for reachable network URLs."""
        img, path = exporter.generate_qr_code("http://192.168.1.100:8501/model.glb")
        assert isinstance(img, Image.Image)
        assert path.exists()
        assert path.suffix.lower() == ".png"

    def test_qr_code_rejects_local_filesystem_path(self, exporter):
        """Strict Rule: QR codes must never point to inaccessible local paths."""
        with pytest.raises(ValueError, match="QR codes must point to reachable network HTTP/HTTPS"):
            exporter.generate_qr_code("file:///C:/Users/test/model.glb")

        with pytest.raises(ValueError, match="QR codes must point to reachable network HTTP/HTTPS"):
            exporter.generate_qr_code("C:\\projects\\mesh.glb")

        with pytest.raises(ValueError, match="QR codes must point to reachable network HTTP/HTTPS"):
            exporter.generate_qr_code("/home/user/mesh.glb")

    def test_host_network_ip_discovery(self):
        """Verifies discovery of host LAN IP address."""
        ip = GLBExporter.get_host_network_ip()
        assert isinstance(ip, str)
        # IP should be valid dot-decimal format
        parts = ip.split(".")
        assert len(parts) == 4
        assert all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)

