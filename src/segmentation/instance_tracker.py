"""
Instance Tracker Module
Implements spatiotemporal tracking, class aliasing resolution (bottle vs vase vs cup),
spatial continuity scoring, and ambiguity resolution for isolating the tabletop object.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from src.segmentation.segmentation_models import InstanceMask
from src.segmentation.mask_processor import MaskProcessor


@dataclass
class InstanceTrackerConfig:
    """Configuration parameters for instance tracking across sequential keyframes."""
    primary_classes: Tuple[str, ...] = ("bottle", "vase", "cup")
    preferred_class: str = "bottle"
    min_confidence: float = 0.15
    min_height_px: float = 400.0
    min_aspect_ratio: float = 1.3  # height / width >= 1.3
    min_area_pct: float = 1.5      # At least 1.5% of total frame area
    max_area_pct: float = 60.0     # Reject full-frame misdetections
    max_centroid_shift_pct: float = 0.40  # Max jump relative to max(H, W)
    spatial_iou_duplicate_thresh: float = 0.65  # Merge overlapping dual-class predictions
    ambiguity_score_margin: float = 0.12        # Margin between top two competing candidates
    min_valid_track_score: float = 0.35


class InstanceTracker:
    """
    Tracks the physical target tabletop container across consecutive keyframes.
    Resolves multi-class aliasing (COCO bottle/vase/cup) and maintains spatial continuity.
    """

    def __init__(self, config: Optional[InstanceTrackerConfig] = None):
        self.config = config or InstanceTrackerConfig()
        self.previous_target: Optional[InstanceMask] = None
        self.tracking_history: List[InstanceMask] = []
        self.frame_records: Dict[int, Optional[InstanceMask]] = {}

    def reset(self):
        """Resets tracker state."""
        self.previous_target = None
        self.tracking_history.clear()
        self.frame_records.clear()

    def filter_and_deduplicate_candidates(
        self,
        instances: List[InstanceMask],
        frame_shape: Tuple[int, int],  # (height, width)
    ) -> List[InstanceMask]:
        """
        Filters raw instance detections to valid container candidates and merges
        overlapping predictions that represent the same physical object (e.g. bottle + vase).

        Args:
            instances: Raw instance masks from detector.
            frame_shape: (height, width) of the image.

        Returns:
            List of filtered, deduplicated candidate InstanceMasks.
        """
        img_h, img_w = frame_shape
        valid_cands: List[InstanceMask] = []

        # Step 1: Geometric and semantic filtering
        for inst in instances:
            # Check class
            if inst.class_name.lower() not in self.config.primary_classes:
                continue

            # Check confidence
            if inst.confidence < self.config.min_confidence:
                continue

            # Check height
            if inst.height < self.config.min_height_px:
                continue

            # Check aspect ratio (tall container)
            if inst.aspect_ratio < self.config.min_aspect_ratio:
                continue

            # Check area percentage
            if inst.mask_area_pct < self.config.min_area_pct or inst.mask_area_pct > self.config.max_area_pct:
                continue

            valid_cands.append(inst)

        if not valid_cands:
            return []

        # Step 2: Merge / deduplicate co-located predictions (e.g., bottle and vase on identical pixels)
        # Sort by confidence descending
        valid_cands.sort(key=lambda x: x.confidence, reverse=True)
        merged_cands: List[InstanceMask] = []
        suppressed_indices: Set[int] = set()

        for i, c_i in enumerate(valid_cands):
            if i in suppressed_indices:
                continue

            best_match = c_i
            for j in range(i + 1, len(valid_cands)):
                if j in suppressed_indices:
                    continue

                c_j = valid_cands[j]
                iou = MaskProcessor.compute_bbox_iou(c_i.bbox, c_j.bbox)
                if iou >= self.config.spatial_iou_duplicate_thresh:
                    # Co-located detection!
                    suppressed_indices.add(j)
                    # Preference: if one is preferred_class ("bottle"), prioritize it
                    if c_j.class_name.lower() == self.config.preferred_class.lower():
                        # If the preferred class confidence is within 0.20 of top confidence, adopt it
                        if c_i.confidence - c_j.confidence <= 0.20:
                            best_match = c_j

            merged_cands.append(best_match)

        return merged_cands

    def compute_candidate_score(
        self,
        candidate: InstanceMask,
        previous_target: Optional[InstanceMask],
        frame_shape: Tuple[int, int],
    ) -> float:
        """
        Computes a matching score S in [0, 1] for a candidate instance.

        Weights:
        - Temporal bbox IoU (if previous target exists)
        - Spatial centroid displacement
        - Detection confidence
        - Class preference (bonus for preferred_class 'bottle')
        - Proximity to frame center (if initial frame)
        """
        img_h, img_w = frame_shape
        max_dim = max(img_h, img_w)

        # Baseline confidence score component [0, 0.35]
        conf_score = min(1.0, candidate.confidence) * 0.35

        # Class preference bonus [0, 0.15]
        class_bonus = 0.15 if candidate.class_name.lower() == self.config.preferred_class.lower() else 0.05

        if previous_target is not None:
            # Temporal IoU component [0, 0.30]
            t_iou = MaskProcessor.compute_bbox_iou(candidate.bbox, previous_target.bbox)
            iou_score = t_iou * 0.30

            # Centroid displacement component [0, 0.20]
            dx = candidate.centroid[0] - previous_target.centroid[0]
            dy = candidate.centroid[1] - previous_target.centroid[1]
            dist = np.sqrt(dx * dx + dy * dy)
            norm_dist = min(1.0, dist / (max_dim * self.config.max_centroid_shift_pct))
            dist_score = (1.0 - norm_dist) * 0.20

            total_score = conf_score + class_bonus + iou_score + dist_score
        else:
            # Prior for initial / unanchored frame: central proximity component [0, 0.50]
            center_x, center_y = img_w / 2.0, img_h / 2.0
            dx = candidate.centroid[0] - center_x
            dy = candidate.centroid[1] - center_y
            dist = np.sqrt(dx * dx + dy * dy)
            norm_dist = min(1.0, dist / max_dim)
            center_score = (1.0 - norm_dist) * 0.50

            total_score = conf_score + class_bonus + center_score

        return float(np.clip(total_score, 0.0, 1.0))

    def track_frame(
        self,
        frame_idx: int,
        all_instances: List[InstanceMask],
        frame_shape: Tuple[int, int],  # (height, width)
        is_registered_sfm: bool = True,
        is_excluded: bool = False,
    ) -> Tuple[Optional[InstanceMask], str, List[InstanceMask]]:
        """
        Processes a single keyframe, identifying the target instance and assigning status.

        Args:
            frame_idx: Sequential keyframe index.
            all_instances: All segmented instances in this frame.
            frame_shape: (height, width) of the image.
            is_registered_sfm: Whether the frame has registered camera pose in SfM.
            is_excluded: Whether the frame is explicitly excluded.

        Returns:
            Tuple of:
              - target_instance: Optional[InstanceMask] (the tracked target if found)
              - target_status: str ("VALID", "AMBIGUOUS", "NO_TARGET", "EXCLUDED")
              - all_instances: List[InstanceMask] (with is_target / is_ambiguous flags updated)
        """
        if is_excluded:
            return None, "EXCLUDED", all_instances

        candidates = self.filter_and_deduplicate_candidates(all_instances, frame_shape)

        if not candidates:
            # No candidate container found
            return None, "NO_TARGET", all_instances

        # Score all candidates
        scored_candidates: List[Tuple[float, InstanceMask]] = []
        for cand in candidates:
            score = self.compute_candidate_score(cand, self.previous_target, frame_shape)
            cand.tracking_score = round(score, 4)
            scored_candidates.append((score, cand))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_score, top_cand = scored_candidates[0]

        # Check for ambiguity: multiple distinct candidates with close scores
        if len(scored_candidates) > 1:
            second_score, second_cand = scored_candidates[1]
            inter_iou = MaskProcessor.compute_bbox_iou(top_cand.bbox, second_cand.bbox)
            score_diff = abs(top_score - second_score)

            # If distinct spatial objects (IoU < 0.3) and score difference is small
            if inter_iou < 0.30 and score_diff <= self.config.ambiguity_score_margin:
                top_cand.is_ambiguous = True
                top_cand.ambiguity_reason = (
                    f"Competing candidate with score {second_score:.3f} "
                    f"(delta={score_diff:.3f} <= {self.config.ambiguity_score_margin})"
                )
                second_cand.is_ambiguous = True
                second_cand.ambiguity_reason = f"Competes with top candidate {top_score:.3f}"
                return top_cand, "AMBIGUOUS", all_instances

        # Single or clearly dominant candidate
        if top_score < self.config.min_valid_track_score:
            top_cand.is_ambiguous = True
            top_cand.ambiguity_reason = (
                f"Tracking score {top_score:.3f} below minimum threshold {self.config.min_valid_track_score}"
            )
            return top_cand, "AMBIGUOUS", all_instances

        top_cand.is_target = True
        self.previous_target = top_cand
        self.tracking_history.append(top_cand)
        self.frame_records[frame_idx] = top_cand

        return top_cand, "VALID", all_instances

