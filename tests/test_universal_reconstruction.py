"""
Unit tests for Universal 3D Object Reconstruction Module.
Verifies multi-class mesh reconstruction across all 4 geometric methods:
- Surface of revolution (axisymmetric objects: bottle, cup, vase)
- Convex hull (general organic/polyhedral objects: chair, car)
- Alpha shape (concave objects)
- Oriented Bounding Box (cuboid objects: laptop, book)
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.universal_reconstruction.universal_reconstructor import (
    UniversalReconstructor,
    ReconstructionMethod,
    ReconstructionResult,
)


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def reconstructor(temp_output_dir):
    return UniversalReconstructor(output_dir=temp_output_dir)


@pytest.fixture
def sample_3d_points():
    # Synthetic point cloud representing an arbitrary object in 3D
    np.random.seed(42)
    return np.random.uniform(low=[-0.5, -1.0, 4.0], high=[0.5, 1.0, 5.0], size=(30, 3))


class TestUniversalReconstructor:
    def test_reconstruct_axisymmetric_bottle(self, reconstructor):
        """Test parametric surface of revolution for bottle class."""
        result = reconstructor.reconstruct(
            target_class="bottle",
            method=ReconstructionMethod.AUTO,
        )
        assert isinstance(result, ReconstructionResult)
        assert result.method_used == ReconstructionMethod.REVOLUTION.value
        assert result.num_vertices > 1000
        assert result.num_faces > 2000
        assert result.is_watertight is True
        assert result.ply_path.exists()
        assert result.obj_path.exists()

    def test_reconstruct_axisymmetric_cup(self, reconstructor):
        """Test parametric surface of revolution for cup class."""
        result = reconstructor.reconstruct(
            target_class="cup",
            method=ReconstructionMethod.REVOLUTION,
        )
        assert result.method_used == ReconstructionMethod.REVOLUTION.value
        assert result.num_vertices > 1000
        assert result.is_watertight is True

    def test_reconstruct_axisymmetric_vase(self, reconstructor):
        """Test parametric surface of revolution for vase class."""
        result = reconstructor.reconstruct(
            target_class="vase",
            method=ReconstructionMethod.REVOLUTION,
        )
        assert result.method_used == ReconstructionMethod.REVOLUTION.value
        assert result.num_vertices > 1000
        assert result.is_watertight is True

    def test_reconstruct_convex_hull(self, reconstructor, sample_3d_points):
        """Test convex hull reconstruction for generic object class (chair)."""
        result = reconstructor.reconstruct(
            target_class="chair",
            points_3d=sample_3d_points,
            method=ReconstructionMethod.CONVEX_HULL,
        )
        assert result.method_used == ReconstructionMethod.CONVEX_HULL.value
        assert result.num_vertices >= 4
        assert result.num_faces >= 4
        assert result.is_watertight is True
        assert result.volume > 0.0

    def test_reconstruct_bounding_box(self, reconstructor, sample_3d_points):
        """Test bounding box reconstruction for cuboid object (laptop)."""
        result = reconstructor.reconstruct(
            target_class="laptop",
            points_3d=sample_3d_points,
            method=ReconstructionMethod.BOUNDING_BOX,
        )
        assert result.method_used == ReconstructionMethod.BOUNDING_BOX.value
        assert result.num_vertices == 8
        assert result.num_faces == 12
        assert result.is_watertight is True
        assert result.volume > 0.0

    def test_reconstruct_alpha_shape(self, reconstructor, sample_3d_points):
        """Test alpha shape concave hull reconstruction."""
        result = reconstructor.reconstruct(
            target_class="general_object",
            points_3d=sample_3d_points,
            method=ReconstructionMethod.ALPHA_SHAPE,
        )
        assert result.method_used in [ReconstructionMethod.ALPHA_SHAPE.value, ReconstructionMethod.CONVEX_HULL.value]
        assert result.num_vertices >= 4
        assert result.num_faces >= 4

    def test_auto_dispatch_by_class_name(self, reconstructor, sample_3d_points):
        """Verify that AUTO method correctly routes based on class categories."""
        # 1. Bottle -> Revolution
        res_bottle = reconstructor.reconstruct(target_class="wine glass")
        assert res_bottle.method_used == ReconstructionMethod.REVOLUTION.value

        # 2. Book -> Bounding Box
        res_book = reconstructor.reconstruct(target_class="book", points_3d=sample_3d_points)
        assert res_book.method_used == ReconstructionMethod.BOUNDING_BOX.value

        # 3. Person / Car -> Convex Hull
        res_car = reconstructor.reconstruct(target_class="car", points_3d=sample_3d_points)
        assert res_car.method_used == ReconstructionMethod.CONVEX_HULL.value

    def test_fallback_on_empty_points(self, reconstructor):
        """Verify graceful fallback when 3D points are None or empty."""
        res = reconstructor.reconstruct(
            target_class="unknown_object",
            points_3d=None,
            method=ReconstructionMethod.CONVEX_HULL,
        )
        assert res.num_vertices >= 4
        assert res.num_faces >= 4
        assert res.is_watertight is True

