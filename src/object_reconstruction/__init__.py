"""
Object Reconstruction Package (Phase 5.5).
Parametric surface-of-revolution bottle modeling, multi-view silhouette projection,
bounded optimization, and 3D mesh export.
"""

from src.object_reconstruction.bottle_models import (
    BottleParameters,
    BottleProfileSample,
    Mesh3D,
    SilhouetteViewMetric,
)
from src.object_reconstruction.parametric_bottle import ParametricBottle
from src.object_reconstruction.silhouette_projector import SilhouetteProjector
from src.object_reconstruction.bottle_fitter import BottleFitter, FittingSummary
from src.object_reconstruction.reconstruction_visualizer import ReconstructionVisualizer
from src.object_reconstruction.reconstruction_pipeline import ObjectReconstructionPipeline

__all__ = [
    "BottleParameters",
    "BottleProfileSample",
    "Mesh3D",
    "SilhouetteViewMetric",
    "ParametricBottle",
    "SilhouetteProjector",
    "BottleFitter",
    "FittingSummary",
    "ReconstructionVisualizer",
    "ObjectReconstructionPipeline",
]

