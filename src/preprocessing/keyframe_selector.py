"""
Keyframe Selector Module
Executes rule-based selection to filter out blurred, redundant, and feature-deficient frames.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.preprocessing.image_quality import ImageQualityAssessor, ImageQualityMetrics
from src.preprocessing.frame_filter import FrameFilter


@dataclass
class KeyframeDecision:
    """Detailed audit record for a frame evaluated by KeyframeSelector."""
    frame_index: int
    timestamp_seconds: float
    quality_metrics: dict
    is_selected: bool
    rejection_reason: Optional[str] = None
    similarity_with_last_keyframe: Optional[float] = None
    hamming_distance: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


class KeyframeSelector:
    """
    Selects optimal keyframes for Structure from Motion using quality and diversity rules.
    
    Decision Rules:
    1. Quality Gate: The frame must pass blur, exposure, contrast, and resolution thresholds.
    2. Minimum Baseline / Diversity Gate: The frame must not be a near-duplicate of the
       last selected keyframe (ensures camera has translated sufficiently to provide parallax).
    3. Maximum Temporal Gap Gate: If the camera motion has been subtle for an extended period,
       a high-quality frame is accepted to avoid track breakage in incremental SfM.
    """

    def __init__(
        self,
        quality_assessor: Optional[ImageQualityAssessor] = None,
        frame_filter: Optional[FrameFilter] = None,
        min_time_delta_seconds: float = 0.25,      # Enforce minimal temporal spacing
        max_time_gap_seconds: float = 2.0,         # Maximum allowable gap between keyframes
        target_keyframe_range: Tuple[int, int] = (50, 150),
    ):
        self.quality_assessor = quality_assessor or ImageQualityAssessor()
        self.frame_filter = frame_filter or FrameFilter()
        self.min_time_delta = min_time_delta_seconds
        self.max_time_gap = max_time_gap_seconds
        self.target_range = target_keyframe_range

    def process_sequence(
        self,
        frames: List[Tuple[int, float, np.ndarray]],  # List of (frame_idx, timestamp_sec, bgr_image)
    ) -> Tuple[List[int], List[KeyframeDecision]]:
        """
        Processes a sequence of candidate frames and returns selected frame indices and full decisions.
        
        Returns:
            Tuple of:
                - selected_indices: List[int] (original frame indices selected)
                - decisions: List[KeyframeDecision] (audit log for all evaluated frames)
        """
        selected_indices: List[int] = []
        decisions: List[KeyframeDecision] = []

        last_selected_time: Optional[float] = None
        last_selected_hash: Optional[int] = None

        for frame_idx, timestamp_sec, frame in frames:
            # 1. Image Quality Assessment
            metrics = self.quality_assessor.evaluate(frame)

            if not metrics.quality_passed:
                reason = "; ".join(metrics.rejection_reasons)
                decisions.append(
                    KeyframeDecision(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        quality_metrics=metrics.to_dict(),
                        is_selected=False,
                        rejection_reason=f"Quality failure: {reason}",
                    )
                )
                continue

            current_hash = self.frame_filter.compute_dhash(frame)

            # 2. First valid frame is unconditionally accepted as initial anchor keyframe
            if last_selected_time is None or last_selected_hash is None:
                selected_indices.append(frame_idx)
                last_selected_time = timestamp_sec
                last_selected_hash = current_hash
                decisions.append(
                    KeyframeDecision(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        quality_metrics=metrics.to_dict(),
                        is_selected=True,
                        rejection_reason=None,
                        similarity_with_last_keyframe=1.0,
                        hamming_distance=0,
                    )
                )
                continue

            # 3. Temporal pacing check
            time_delta = timestamp_sec - last_selected_time
            if time_delta < self.min_time_delta:
                decisions.append(
                    KeyframeDecision(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        quality_metrics=metrics.to_dict(),
                        is_selected=False,
                        rejection_reason=f"Temporal delta ({time_delta:.3f}s) < min ({self.min_time_delta:.3f}s)",
                    )
                )
                continue

            # 4. Near-duplicate / viewpoint diversity check
            sim_comp = self.frame_filter.compare_hashes(last_selected_hash, current_hash)

            # Check if camera was static (near-duplicate) and maximum gap has not been exceeded
            if sim_comp.near_duplicate and time_delta < self.max_time_gap:
                decisions.append(
                    KeyframeDecision(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        quality_metrics=metrics.to_dict(),
                        is_selected=False,
                        rejection_reason=(
                            f"Near-duplicate of last keyframe (dist={sim_comp.perceptual_distance} <= "
                            f"{self.frame_filter.max_hamming_distance})"
                        ),
                        similarity_with_last_keyframe=sim_comp.similarity_score,
                        hamming_distance=sim_comp.perceptual_distance,
                    )
                )
                continue

            # 5. Keyframe Accepted
            selected_indices.append(frame_idx)
            last_selected_time = timestamp_sec
            last_selected_hash = current_hash
            decisions.append(
                KeyframeDecision(
                    frame_index=frame_idx,
                    timestamp_seconds=timestamp_sec,
                    quality_metrics=metrics.to_dict(),
                    is_selected=True,
                    rejection_reason=None,
                    similarity_with_last_keyframe=sim_comp.similarity_score,
                    hamming_distance=sim_comp.perceptual_distance,
                )
            )

        return selected_indices, decisions
