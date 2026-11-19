"""
Structure from Motion (SfM) Subpackage
Provides 2-view baseline, incremental multi-view SfM, feature tracking, and bundle adjustment.
"""

from src.sfm.feature_extractor import FeatureExtractor, ExtractedFeatures
from src.sfm.feature_matcher import FeatureMatcher, MatchResult
from src.sfm.geometry import GeometricVerifier, GeometricVerificationResult
from src.sfm.pose_estimator import PoseEstimator, PoseEstimationResult
from src.sfm.triangulator import Triangulator, TriangulationResult
from src.sfm.two_view_sfm import TwoViewSfM
from src.sfm.feature_tracks import FeatureTrackSystem, FeatureTrack, TrackObservation
from src.sfm.multi_view_matcher import MultiViewMatcher, VerifiedPairMatch
from src.sfm.initialization import InitialPairSelector, InitializationResult
from src.sfm.camera_registration import CameraRegistrar, CameraPose
from src.sfm.multi_view_triangulator import MultiViewTriangulator, MultiViewTriangulationResult
from src.sfm.bundle_adjustment import BundleAdjuster, BundleAdjustmentResult
from src.sfm.reconstruction_metrics import ReconstructionMetrics, ReconstructionDiagnostics
from src.sfm.outlier_diagnostics import OutlierDiagnostics, ObservationErrorRecord
from src.sfm.visualization import SfMVisualizer
from src.sfm.incremental_sfm import IncrementalSfM, IncrementalSfMConfig
from src.sfm.real_video_sfm_experiment import RealVideoSfMExperiment, FocalHypothesis

__all__ = [
    "FeatureExtractor",
    "ExtractedFeatures",
    "FeatureMatcher",
    "MatchResult",
    "GeometricVerifier",
    "GeometricVerificationResult",
    "PoseEstimator",
    "PoseEstimationResult",
    "Triangulator",
    "TriangulationResult",
    "TwoViewSfM",
    "FeatureTrackSystem",
    "FeatureTrack",
    "TrackObservation",
    "MultiViewMatcher",
    "VerifiedPairMatch",
    "InitialPairSelector",
    "InitializationResult",
    "CameraRegistrar",
    "CameraPose",
    "MultiViewTriangulator",
    "MultiViewTriangulationResult",
    "BundleAdjuster",
    "BundleAdjustmentResult",
    "ReconstructionMetrics",
    "ReconstructionDiagnostics",
    "OutlierDiagnostics",
    "ObservationErrorRecord",
    "SfMVisualizer",
    "IncrementalSfM",
    "IncrementalSfMConfig",
    "RealVideoSfMExperiment",
    "FocalHypothesis",
]

