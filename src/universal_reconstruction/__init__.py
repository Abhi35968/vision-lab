"""
Universal 3D Object Reconstruction Module.
Supports both parametric models (for axisymmetric objects: bottles, cups, vases)
and non-parametric models (alpha shapes, convex hulls, oriented bounding boxes)
for arbitrary 3D objects from multi-view video and YOLO segmentation.
"""

from src.universal_reconstruction.universal_reconstructor import (
    UniversalReconstructor,
    ReconstructionResult,
    ReconstructionMethod,
)

__all__ = [
    "UniversalReconstructor",
    "ReconstructionResult",
    "ReconstructionMethod",
]

