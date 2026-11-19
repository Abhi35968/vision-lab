"""
2D-to-3D Association Subpackage
Provides:
1. Phase 4.2 Bounding-box-to-track association.
2. Phase 5.3.2 Multi-view segmentation mask association, projection validation,
   point partitioning, and spatial diagnostics.
"""

# Phase 4.2 components
from src.association.association_models import (
    AssociationStatus,
    ObservationAssociation,
    TrackAssociation,
    ObjectPointGroup,
    AssociationSummary,
)
from src.association.observation_matcher import ObservationMatcher
from src.association.track_associator import TrackAssociator, AssociationConfig
from src.association.association_visualizer import AssociationVisualizer

# Phase 5.3.2 components
from src.association.mask_projection import (
    CameraProjector,
    ProjectionResult,
    MaskSampleResult,
)
from src.association.multiview_mask_associator import (
    ObservationMaskEvidence,
    PointMaskAssociation,
    MultiViewAssociatorConfig,
    MultiViewMaskAssociator,
)
from src.association.object_point_filter import ObjectPointFilter
from src.association.object_association_metrics import (
    ObjectAssociationSummary,
    ObjectAssociationMetricsCalculator,
)
from src.association.object_association_visualizer import ObjectAssociationVisualizer
from src.association.real_video_object_association import RealVideoObjectAssociationPipeline

__all__ = [
    # Phase 4.2
    "AssociationStatus",
    "ObservationAssociation",
    "TrackAssociation",
    "ObjectPointGroup",
    "AssociationSummary",
    "ObservationMatcher",
    "TrackAssociator",
    "AssociationConfig",
    "AssociationVisualizer",
    # Phase 5.3.2
    "CameraProjector",
    "ProjectionResult",
    "MaskSampleResult",
    "ObservationMaskEvidence",
    "PointMaskAssociation",
    "MultiViewAssociatorConfig",
    "MultiViewMaskAssociator",
    "ObjectPointFilter",
    "ObjectAssociationSummary",
    "ObjectAssociationMetricsCalculator",
    "ObjectAssociationVisualizer",
    "RealVideoObjectAssociationPipeline",
]
