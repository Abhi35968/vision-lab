"""
Data Models and Configuration for Object-Specific Point Cloud Refinement (Phase 5.3.3)
Defines structured data representations for per-observation reprojection,
triangulation geometry, spatial clustering, and multi-criteria refined classifications.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ObservationReprojectionInfo:
    """Reprojection and geometric details for a single feature observation."""
    camera_id: int
    image_name: str
    obs_coords: Tuple[float, float]
    proj_coords: Optional[Tuple[float, float]]
    error_px: float
    depth: float
    camera_distance: float
    is_inside_mask: bool
    is_outlier: bool = False
    outlier_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "image_name": self.image_name,
            "obs_coords": [round(c, 2) for c in self.obs_coords],
            "proj_coords": [round(c, 2) for c in self.proj_coords] if self.proj_coords else None,
            "error_px": round(self.error_px, 3),
            "depth": round(self.depth, 4),
            "camera_distance": round(self.camera_distance, 4),
            "is_inside_mask": self.is_inside_mask,
            "is_outlier": self.is_outlier,
            "outlier_reason": self.outlier_reason,
        }


@dataclass
class PointGeometricProfile:
    """Comprehensive geometric evaluation and refined classification for a 3D point."""
    point_id: int
    track_id: int
    point_3d: Tuple[float, float, float]
    original_classification: str
    original_mask_support_ratio: float
    total_observations: int
    masked_observations: int
    inside_mask_observations: int
    num_distinct_cameras: int
    observed_camera_ids: List[int]
    supporting_camera_ids: List[int]

    # Reprojection error statistics
    mean_reproj_error_px: float
    median_reproj_error_px: float
    std_reproj_error_px: float
    p90_reproj_error_px: float
    p95_reproj_error_px: float
    max_reproj_error_px: float
    count_err_gt_2px: int
    count_err_gt_4px: int
    count_err_gt_10px: int
    num_outlier_observations: int
    filtered_mean_reproj_error_px: float
    filtered_median_reproj_error_px: float

    # Depth & camera distance statistics
    min_camera_distance: float
    max_camera_distance: float
    mean_camera_distance: float
    relative_distance_std: float  # std(distance) / mean(distance)
    has_positive_depth_all_views: bool

    # Triangulation geometry statistics
    max_triangulation_angle_deg: float
    median_triangulation_angle_deg: float
    min_triangulation_angle_deg: float
    max_camera_baseline: float
    baseline_to_depth_ratio: float
    is_triangulation_degenerate: bool  # max angle < min_angle_threshold

    # Spatial clustering diagnostics
    nn_distance_1: float
    nn_distance_3: float
    is_spatial_outlier: bool
    cluster_id: int  # -1 for noise, >=0 for cluster

    # Refined Classification
    # HIGH_GEOMETRIC_CONFIDENCE, MODERATE_GEOMETRIC_CONFIDENCE, LOW_GEOMETRIC_CONFIDENCE, GEOMETRIC_OUTLIER, INSUFFICIENT_GEOMETRIC_EVIDENCE
    refined_status: str
    refinement_rationale: str

    observations: List[ObservationReprojectionInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "track_id": self.track_id,
            "point_3d": [round(c, 4) for c in self.point_3d],
            "original_classification": self.original_classification,
            "original_mask_support_ratio": round(self.original_mask_support_ratio, 4),
            "total_observations": self.total_observations,
            "masked_observations": self.masked_observations,
            "inside_mask_observations": self.inside_mask_observations,
            "num_distinct_cameras": self.num_distinct_cameras,
            "observed_camera_ids": self.observed_camera_ids,
            "supporting_camera_ids": self.supporting_camera_ids,
            "reprojection_statistics": {
                "mean_px": round(self.mean_reproj_error_px, 3),
                "median_px": round(self.median_reproj_error_px, 3),
                "std_px": round(self.std_reproj_error_px, 3),
                "p90_px": round(self.p90_reproj_error_px, 3),
                "p95_px": round(self.p95_reproj_error_px, 3),
                "max_px": round(self.max_reproj_error_px, 3),
                "count_gt_2px": self.count_err_gt_2px,
                "count_gt_4px": self.count_err_gt_4px,
                "count_gt_10px": self.count_err_gt_10px,
                "num_outlier_observations": self.num_outlier_observations,
                "filtered_mean_px": round(self.filtered_mean_reproj_error_px, 3),
                "filtered_median_px": round(self.filtered_median_reproj_error_px, 3),
            },
            "depth_and_distance": {
                "min_camera_distance": round(self.min_camera_distance, 4),
                "max_camera_distance": round(self.max_camera_distance, 4),
                "mean_camera_distance": round(self.mean_camera_distance, 4),
                "relative_distance_std": round(self.relative_distance_std, 4),
                "has_positive_depth_all_views": self.has_positive_depth_all_views,
            },
            "triangulation_geometry": {
                "max_angle_deg": round(self.max_triangulation_angle_deg, 2),
                "median_angle_deg": round(self.median_triangulation_angle_deg, 2),
                "min_angle_deg": round(self.min_triangulation_angle_deg, 2),
                "max_camera_baseline": round(self.max_camera_baseline, 4),
                "baseline_to_depth_ratio": round(self.baseline_to_depth_ratio, 4),
                "is_triangulation_degenerate": self.is_triangulation_degenerate,
            },
            "spatial_clustering": {
                "nn_distance_1": round(self.nn_distance_1, 4),
                "nn_distance_3": round(self.nn_distance_3, 4),
                "is_spatial_outlier": self.is_spatial_outlier,
                "cluster_id": self.cluster_id,
            },
            "refinement": {
                "refined_status": self.refined_status,
                "rationale": self.refinement_rationale,
            },
        }

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "track_id": self.track_id,
            "x_world": round(self.point_3d[0], 4),
            "y_world": round(self.point_3d[1], 4),
            "z_world": round(self.point_3d[2], 4),
            "original_classification": self.original_classification,
            "original_mask_support_ratio": round(self.original_mask_support_ratio, 3),
            "total_observations": self.total_observations,
            "masked_observations": self.masked_observations,
            "inside_mask_observations": self.inside_mask_observations,
            "num_distinct_cameras": self.num_distinct_cameras,
            "observed_cameras": ";".join(str(c) for c in self.observed_camera_ids),
            "supporting_cameras": ";".join(str(c) for c in self.supporting_camera_ids),
            "mean_reproj_err_px": round(self.mean_reproj_error_px, 3),
            "median_reproj_err_px": round(self.median_reproj_error_px, 3),
            "std_reproj_err_px": round(self.std_reproj_error_px, 3),
            "p90_reproj_err_px": round(self.p90_reproj_error_px, 3),
            "max_reproj_err_px": round(self.max_reproj_error_px, 3),
            "count_gt_2px": self.count_err_gt_2px,
            "count_gt_4px": self.count_err_gt_4px,
            "count_gt_10px": self.count_err_gt_10px,
            "num_outlier_observations": self.num_outlier_observations,
            "filtered_mean_err_px": round(self.filtered_mean_reproj_error_px, 3),
            "filtered_median_err_px": round(self.filtered_median_reproj_error_px, 3),
            "mean_camera_distance": round(self.mean_camera_distance, 3),
            "relative_distance_std": round(self.relative_distance_std, 3),
            "max_triangulation_angle_deg": round(self.max_triangulation_angle_deg, 2),
            "median_triangulation_angle_deg": round(self.median_triangulation_angle_deg, 2),
            "max_camera_baseline": round(self.max_camera_baseline, 3),
            "baseline_to_depth_ratio": round(self.baseline_to_depth_ratio, 3),
            "nn_distance_1": round(self.nn_distance_1, 3),
            "nn_distance_3": round(self.nn_distance_3, 3),
            "cluster_id": self.cluster_id,
            "is_spatial_outlier": self.is_spatial_outlier,
            "refined_status": self.refined_status,
            "refinement_rationale": self.refinement_rationale,
        }


@dataclass
class RefinementConfig:
    """Configurable thresholds for geometric refinement and confidence grading."""
    # Reprojection thresholds
    high_conf_max_median_err_px: float = 2.0
    high_conf_max_p90_err_px: float = 4.0
    mod_conf_max_median_err_px: float = 3.0
    mod_conf_max_p90_err_px: float = 6.0
    outlier_median_err_thresh_px: float = 5.0
    outlier_single_obs_thresh_px: float = 12.0
    extreme_outlier_err_thresh_px: float = 25.0

    # View & Mask thresholds
    high_conf_min_views: int = 3
    high_conf_min_inside: int = 3
    high_conf_min_mask_ratio: float = 0.60
    mod_conf_min_views: int = 2
    mod_conf_min_inside: int = 2
    mod_conf_min_mask_ratio: float = 0.50

    # Triangulation thresholds
    min_triangulation_angle_degenerate_deg: float = 1.5
    high_conf_min_triangulation_angle_deg: float = 3.0
    mod_conf_min_triangulation_angle_deg: float = 1.8

    # Spatial clustering
    spatial_sor_k: int = 5
    spatial_sor_alpha: float = 2.0
    dbscan_min_samples: int = 3
    # If None, automatically derived from k-NN 75th percentile
    dbscan_eps: Optional[float] = None

