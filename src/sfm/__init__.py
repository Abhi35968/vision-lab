"""
Structure from Motion (SfM) Subpackage
Provides 2-view baseline, incremental multi-view SfM, feature tracking, and bundle adjustment.
"""

try:
    from src.sfm.feature_extractor import FeatureExtractor, ExtractedFeatures
except Exception:
    pass
try:
    from src.sfm.feature_matcher import FeatureMatcher, MatchResult
except Exception:
    pass
try:
    from src.sfm.geometry import GeometricVerifier, GeometricVerificationResult
except Exception:
    pass
try:
    from src.sfm.pose_estimator import PoseEstimator, PoseEstimationResult
except Exception:
    pass
try:
    from src.sfm.triangulator import Triangulator, TriangulationResult
except Exception:
    pass
try:
    from src.sfm.two_view_sfm import TwoViewSfM
except Exception:
    pass
try:
    from src.sfm.feature_tracks import FeatureTrackSystem, FeatureTrack, TrackObservation
except Exception:
    pass
try:
    from src.sfm.multi_view_matcher import MultiViewMatcher, VerifiedPairMatch
except Exception:
    pass
try:
    from src.sfm.initialization import InitialPairSelector, InitializationResult
except Exception:
    pass
try:
    from src.sfm.camera_registration import CameraRegistrar, CameraPose
except Exception:
    pass
try:
    from src.sfm.multi_view_triangulator import MultiViewTriangulator, MultiViewTriangulationResult
except Exception:
    pass
try:
    from src.sfm.bundle_adjustment import BundleAdjuster, BundleAdjustmentResult
except Exception:
    pass
try:
    from src.sfm.reconstruction_metrics import ReconstructionMetrics, ReconstructionDiagnostics
except Exception:
    pass
try:
    from src.sfm.outlier_diagnostics import OutlierDiagnostics, ObservationErrorRecord
except Exception:
    pass
try:
    from src.sfm.visualization import SfMVisualizer
except Exception:
    pass
try:
    from src.sfm.incremental_sfm import IncrementalSfM, IncrementalSfMConfig
except Exception:
    pass
try:
    from src.sfm.real_video_sfm_experiment import RealVideoSfMExperiment, FocalHypothesis
except Exception:
    pass

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

