"""
Object Association Metrics Module
Computes rigorous quantitative diagnostics, multi-view consensus telemetry,
and comparative statistical distributions for 2D-to-3D mask association.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.association.multiview_mask_associator import PointMaskAssociation


@dataclass
class ObjectAssociationSummary:
    """Master telemetry summary for Phase 5.3.2 2D-to-3D mask association."""
    total_reconstructed_points: int
    points_with_valid_tracks: int
    points_with_sufficient_masked_views: int  # >= 3 masked views
    strong_candidate_count: int
    weak_candidate_count: int
    ambiguous_count: int
    background_candidate_count: int
    insufficient_evidence_count: int
    total_observations_evaluated: int
    total_valid_projections: int
    total_invalid_projections: int
    reprojection_error_mean_all_px: float
    reprojection_error_mean_strong_px: float
    reprojection_error_median_strong_px: float
    reprojection_error_mean_bg_px: float
    mask_support_ratio_mean: float
    mask_support_ratio_median: float
    mask_support_ratio_p90: float
    supporting_views_distribution: Dict[str, int] = field(default_factory=dict)
    spatial_extents: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes summary to JSON-compatible dictionary."""
        tot = max(1, self.total_reconstructed_points)
        return {
            "reconstruction_counts": {
                "total_reconstructed_points": self.total_reconstructed_points,
                "points_with_valid_tracks": self.points_with_valid_tracks,
                "points_with_sufficient_masked_views": self.points_with_sufficient_masked_views,
                "percent_with_sufficient_masked_views": round((self.points_with_sufficient_masked_views / tot) * 100.0, 2),
            },
            "classification_breakdown": {
                "strong_object_candidates": {
                    "count": self.strong_candidate_count,
                    "percent": round((self.strong_candidate_count / tot) * 100.0, 2),
                },
                "weak_object_candidates": {
                    "count": self.weak_candidate_count,
                    "percent": round((self.weak_candidate_count / tot) * 100.0, 2),
                },
                "ambiguous_points": {
                    "count": self.ambiguous_count,
                    "percent": round((self.ambiguous_count / tot) * 100.0, 2),
                },
                "background_candidates": {
                    "count": self.background_candidate_count,
                    "percent": round((self.background_candidate_count / tot) * 100.0, 2),
                },
                "insufficient_evidence": {
                    "count": self.insufficient_evidence_count,
                    "percent": round((self.insufficient_evidence_count / tot) * 100.0, 2),
                },
            },
            "projections_and_observations": {
                "total_observations_evaluated": self.total_observations_evaluated,
                "total_valid_projections": self.total_valid_projections,
                "total_invalid_projections": self.total_invalid_projections,
                "valid_projection_rate_percent": round(
                    (self.total_valid_projections / max(1, self.total_observations_evaluated)) * 100.0, 2
                ),
            },
            "reprojection_errors_px": {
                "mean_all_points": round(self.reprojection_error_mean_all_px, 3),
                "mean_strong_candidates": round(self.reprojection_error_mean_strong_px, 3),
                "median_strong_candidates": round(self.reprojection_error_median_strong_px, 3),
                "mean_background": round(self.reprojection_error_mean_bg_px, 3),
            },
            "mask_support_ratio_distribution": {
                "mean": round(self.mask_support_ratio_mean, 4),
                "median": round(self.mask_support_ratio_median, 4),
                "p90": round(self.mask_support_ratio_p90, 4),
            },
            "supporting_views_distribution": self.supporting_views_distribution,
            "spatial_extents_arbitrary_units": self.spatial_extents,
        }


class ObjectAssociationMetricsCalculator:
    """Calculates statistical distributions and builds the master summary."""

    @classmethod
    def calculate_summary(
        cls,
        associations: List[PointMaskAssociation],
        spatial_extents: Optional[Dict[str, Any]] = None,
    ) -> ObjectAssociationSummary:
        """
        Computes the complete statistical summary across all point associations.
        """
        total_pts = len(associations)
        if total_pts == 0:
            return ObjectAssociationSummary(
                total_reconstructed_points=0,
                points_with_valid_tracks=0,
                points_with_sufficient_masked_views=0,
                strong_candidate_count=0,
                weak_candidate_count=0,
                ambiguous_count=0,
                background_candidate_count=0,
                insufficient_evidence_count=0,
                total_observations_evaluated=0,
                total_valid_projections=0,
                total_invalid_projections=0,
                reprojection_error_mean_all_px=0.0,
                reprojection_error_mean_strong_px=0.0,
                reprojection_error_median_strong_px=0.0,
                reprojection_error_mean_bg_px=0.0,
                mask_support_ratio_mean=0.0,
                mask_support_ratio_median=0.0,
                mask_support_ratio_p90=0.0,
            )

        sufficient_masked = sum(1 for a in associations if a.masked_observations >= 3)
        strong_pts = [a for a in associations if a.classification == "STRONG_OBJECT_CANDIDATE"]
        weak_pts = [a for a in associations if a.classification == "WEAK_OBJECT_CANDIDATE"]
        ambig_pts = [a for a in associations if a.classification == "AMBIGUOUS"]
        bg_pts = [a for a in associations if a.classification == "BACKGROUND_CANDIDATE"]
        insuff_pts = [a for a in associations if a.classification == "INSUFFICIENT_EVIDENCE"]

        tot_obs = sum(a.total_observations for a in associations)
        tot_valid_proj = sum(a.valid_projections for a in associations)
        tot_invalid_proj = sum(len(a.observations) - a.valid_projections for a in associations)

        # Reprojection errors
        all_errs = [a.mean_reprojection_error_px for a in associations if a.valid_projections > 0]
        strong_errs = [a.mean_reprojection_error_px for a in strong_pts if a.valid_projections > 0]
        bg_errs = [a.mean_reprojection_error_px for a in bg_pts if a.valid_projections > 0]

        mean_all_err = float(np.mean(all_errs)) if all_errs else 0.0
        mean_strong_err = float(np.mean(strong_errs)) if strong_errs else 0.0
        med_strong_err = float(np.median(strong_errs)) if strong_errs else 0.0
        mean_bg_err = float(np.mean(bg_errs)) if bg_errs else 0.0

        # Mask support ratios
        ratios = [a.mask_support_ratio for a in associations]
        mean_ratio = float(np.mean(ratios)) if ratios else 0.0
        med_ratio = float(np.median(ratios)) if ratios else 0.0
        p90_ratio = float(np.percentile(ratios, 90)) if ratios else 0.0

        # Supporting views distribution
        view_counts: Dict[str, int] = {}
        for a in associations:
            n_views = len(a.supporting_camera_ids)
            bucket = f"{n_views}_views"
            view_counts[bucket] = view_counts.get(bucket, 0) + 1

        # Sort view counts by number
        sorted_views = dict(sorted(view_counts.items(), key=lambda x: int(x[0].split("_")[0])))

        return ObjectAssociationSummary(
            total_reconstructed_points=total_pts,
            points_with_valid_tracks=total_pts,
            points_with_sufficient_masked_views=sufficient_masked,
            strong_candidate_count=len(strong_pts),
            weak_candidate_count=len(weak_pts),
            ambiguous_count=len(ambig_pts),
            background_candidate_count=len(bg_pts),
            insufficient_evidence_count=len(insuff_pts),
            total_observations_evaluated=tot_obs,
            total_valid_projections=tot_valid_proj,
            total_invalid_projections=tot_invalid_proj,
            reprojection_error_mean_all_px=mean_all_err,
            reprojection_error_mean_strong_px=mean_strong_err,
            reprojection_error_median_strong_px=med_strong_err,
            reprojection_error_mean_bg_px=mean_bg_err,
            mask_support_ratio_mean=mean_ratio,
            mask_support_ratio_median=med_ratio,
            mask_support_ratio_p90=p90_ratio,
            supporting_views_distribution=sorted_views,
            spatial_extents=spatial_extents or {},
        )

