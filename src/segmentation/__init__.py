"""
Object Segmentation Module for 3DCV Project.
Provides instance segmentation, mask processing, target tracking,
and multi-panel visual diagnostics.
"""

from src.segmentation.segmentation_models import (
    InstanceMask,
    FrameSegmentation,
    SegmentationSummary,
)
from src.segmentation.mask_processor import MaskProcessor
from src.segmentation.instance_tracker import InstanceTracker, InstanceTrackerConfig
from src.segmentation.segmenter import YOLOSegmenter, SegmenterConfig
from src.segmentation.segmentation_visualizer import SegmentationVisualizer

__all__ = [
    "InstanceMask",
    "FrameSegmentation",
    "SegmentationSummary",
    "MaskProcessor",
    "InstanceTracker",
    "InstanceTrackerConfig",
    "YOLOSegmenter",
    "SegmenterConfig",
    "SegmentationVisualizer",
]

