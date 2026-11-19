"""
Feasibility Investigation Package
Modules for feature density analysis, matching quality verification,
dataset limitations evaluation, and MVS paradigm comparisons.
"""

from src.feasibility.feature_density_analyzer import (
    FeatureDensityAnalyzer,
    FrameFeatureDensityRecord,
    SilhouetteDistribution,
    TrackSurvivalSummary,
)
from src.feasibility.matching_quality_analyzer import (
    MatchingQualityAnalyzer,
    PairwiseMatchingRecord,
)
from src.feasibility.dataset_limitations_analyzer import (
    DatasetLimitationsAnalyzer,
    DatasetLimitationsSummary,
    CameraOrbitRecord,
)
from src.feasibility.mvs_comparator import (
    MVSComparator,
    ReconstructionMethodEvaluation,
)
from src.feasibility.feasibility_visualizer import FeasibilityVisualizer
from src.feasibility.feasibility_investigation_pipeline import FeasibilityInvestigationPipeline

__all__ = [
    "FeatureDensityAnalyzer",
    "FrameFeatureDensityRecord",
    "SilhouetteDistribution",
    "TrackSurvivalSummary",
    "MatchingQualityAnalyzer",
    "PairwiseMatchingRecord",
    "DatasetLimitationsAnalyzer",
    "DatasetLimitationsSummary",
    "CameraOrbitRecord",
    "MVSComparator",
    "ReconstructionMethodEvaluation",
    "FeasibilityVisualizer",
    "FeasibilityInvestigationPipeline",
]

