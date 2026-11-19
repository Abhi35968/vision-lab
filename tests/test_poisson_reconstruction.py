"""
Unit tests for Phase 3: Screened Poisson Surface Reconstruction Module
"""

import numpy as np
import pytest
import open3d as o3d
from typing import Tuple

from src.mesh_reconstruction.poisson_reconstructor import (
    PoissonReconstructor,
    PoissonConfig,
    PoissonReport,
)
from src.universal_reconstruction.universal_reconstructor import (
    UniversalReconstructor,
    ReconstructionMethod,
)


def create_synthetic_sphere_points(num_points: int = 400, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generates evenly distributed points and normals on a unit sphere (Fibonacci lattice)."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    points = np.column_stack([x, y, z])
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    return points, normals


class TestPoissonReconstructor:
    """Tests Screened Poisson reconstruction, octree depth scaling, and trimming."""

    @pytest.fixture
    def reconstructor(self):
        return PoissonReconstructor()

    def test_poisson_reconstruction_depth_7_and_8(self, reconstructor):
        """Verifies Poisson reconstruction runs successfully at depths 7 and 8."""
        pts, nls = create_synthetic_sphere_points(num_points=300)

        # Depth 7
        cfg7 = PoissonConfig(octree_depth=7, density_trim_quantile=0.05)
        mesh7, rep7 = reconstructor.reconstruct(pts, nls, config_override=cfg7)
        assert rep7.octree_depth == 7
        assert rep7.final_vertices_count > 50
        assert rep7.final_faces_count > 50

        # Depth 8 (higher octree resolution, expects >= depth 7 vertices before trimming)
        cfg8 = PoissonConfig(octree_depth=8, density_trim_quantile=0.05)
        mesh8, rep8 = reconstructor.reconstruct(pts, nls, config_override=cfg8)
        assert rep8.octree_depth == 8
        assert rep8.raw_vertices_count >= rep7.raw_vertices_count

    def test_density_trimming(self, reconstructor):
        """Verifies density trimming trims hallucinated low-density vertices."""
        pts, nls = create_synthetic_sphere_points(num_points=250)
        cfg_trimmed = PoissonConfig(octree_depth=7, density_trim_quantile=0.15)
        mesh, rep = reconstructor.reconstruct(pts, nls, config_override=cfg_trimmed)

        assert rep.trimmed_vertices_count > 0
        assert rep.final_vertices_count < rep.raw_vertices_count

    def test_fallback_on_tiny_point_cloud(self, reconstructor):
        """Verifies fallback on point cloud with fewer than 10 points."""
        pts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=np.float64)
        mesh, rep = reconstructor.reconstruct(pts)
        assert len(rep.limitation_notes) > 0
        assert "fewer than 10 points" in rep.limitation_notes[0]

    def test_universal_reconstructor_poisson_integration(self):
        """Verifies UniversalReconstructor dispatches POISSON method cleanly."""
        pts, _ = create_synthetic_sphere_points(num_points=200)
        ur = UniversalReconstructor()
        res = ur.reconstruct(target_class="generic_mesh", points_3d=pts, method=ReconstructionMethod.POISSON)
        assert res is not None
        assert res.method_used == "poisson"
        assert res.num_vertices > 20
        assert res.num_faces > 20
