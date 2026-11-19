"""
Preprocessing Module Package
Exposes video ingestion, image quality assessment, frame filtering, keyframe selection,
and real-video validation tools.
"""

from src.preprocessing.video_reader import VideoReader, VideoMetadata
from src.preprocessing.image_quality import ImageQualityAssessor, ImageQualityMetrics
from src.preprocessing.frame_filter import FrameFilter, SimilarityComparison
from src.preprocessing.keyframe_selector import KeyframeSelector, KeyframeDecision
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.camera_calibration import CameraCalibrator
from src.preprocessing.real_video_validator import RealVideoValidator, RealVideoConfig
from src.preprocessing.capture_quality_assistant import (
    VideoCaptureQualityAssistant,
    QualityAssessmentSummary,
    FrameQualityRecord,
)

__all__ = [
    "VideoReader",
    "VideoMetadata",
    "ImageQualityAssessor",
    "ImageQualityMetrics",
    "FrameFilter",
    "SimilarityComparison",
    "KeyframeSelector",
    "KeyframeDecision",
    "PreprocessingPipeline",
    "CameraCalibrator",
    "RealVideoValidator",
    "RealVideoConfig",
    "VideoCaptureQualityAssistant",
    "QualityAssessmentSummary",
    "FrameQualityRecord",
]

