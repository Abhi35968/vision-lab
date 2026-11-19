"""
Point Cloud Refiner and Geometric Confidence Classifier (Phase 5.3.3)
Integrates reprojection error filtering, triangulation geometry analysis,
and 3D spatial clustering to grade candidate points into explicit confidence tiers.
Exports categorized .PLY point clouds, comprehensive CSVs, and master JSON telemetry.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.refinement.refinement_models import (
    ObservationReprojectionInfo,
    PointGeometricProfile,
    RefinementConfig,
)
from src.refinement.reprojection_filter import ReprojectionFilter
from src.refinement.triangulation_analyzer import TriangulationAnalyzer
from src.refinement.spatial_clusterer import SpatialClusterer


class PointRefiner:
    """Master engine for geometric refinement of candidate 3D object points."""

    # Vertex RGB colors for PLY export
    COLOR_HIGH = (50, 205, 50)       # Lime Green
    COLOR_MODERATE = (0, 210, 255)   # Cyan / Sky Blue
    COLOR_LOW = (255, 200, 0)        # Gold / Yellow
    COLOR_OUTLIER = (220, 20, 60)    # Crimson / Red
    COLOR_INSUFFICIENT = (160, 32, 240) # Purple

    def __init__(self, config: Optional[RefinementConfig] = None):
        self.config = config or RefinementConfig()
        self.reproj_filter = ReprojectionFilter(self.config)
        self.tri_analyzer = TriangulationAnalyzer(self.config)
        self.spatial_clusterer = SpatialClusterer(self.config)

    def classify_point_confidence(
        self,
        orig_class: str,
        orig_ratio: float,
        num_masked: int,
        num_inside: int,
        num_cams: int,
        reproj_stats: Dict[str, Any],
        tri_stats: Dict[str, Any],
        is_spatial_outlier: bool,
    ) -> Tuple[str, str]:
        """
        Assigns an explicit refined status based on multi-criteria geometric evidence.
        """
        raw_med_err = reproj_stats["raw"]["median"]
        raw_p90_err = reproj_stats["raw"]["p90"]
        raw_max_err = reproj_stats["raw"]["max"]
        filt_med_err = reproj_stats["filtered"]["median"]
        max_ang = tri_stats["max_angle_deg"]
        has_pos_depth = tri_stats["has_positive_depth_all_views"]
        is_degenerate = tri_stats["is_degenerate"]

        # Case 1: Insufficient Geometric Evidence
        if num_cams < 2 or not has_pos_depth:
            reason = (
                f"Insufficient geometric constraints: observed in {num_cams} camera(s) "
                f"(minimum 2 required), positive depth valid={has_pos_depth}."
            )
            return "INSUFFICIENT_GEOMETRIC_EVIDENCE", reason

        # Case 2: Severe Geometric Outlier
        # Triggered by extreme reprojection error, extreme single-observation drift with no consensus,
        # or gross spatial detachment from the object cluster
        if is_spatial_outlier and (raw_med_err > 3.0 or num_cams <= 2):
            reason = (
                f"Spatial outlier: located far from main object cluster and lacks strong multi-view anchoring "
                f"(cams={num_cams}, med_err={raw_med_err:.2f}px)."
            )
            return "GEOMETRIC_OUTLIER", reason

        if raw_med_err > self.config.outlier_median_err_thresh_px or raw_max_err > self.config.extreme_outlier_err_thresh_px:
            reason = (
                f"Reprojection outlier: residual error exceeds acceptable tolerance "
                f"(med_err={raw_med_err:.2f}px, max_err={raw_max_err:.2f}px)."
            )
            return "GEOMETRIC_OUTLIER", reason

        if is_degenerate and max_ang < 1.0:
            reason = f"Degenerate ray intersection: maximum triangulation parallax {max_ang:.2f}° is near zero."
            return "GEOMETRIC_OUTLIER", reason

        # Case 3: High Geometric Confidence
        # Observed in >=3 views, >=3 inside, support ratio >=0.60, median err <=2.0px, p90 <=4.0px, angle >=3.0 deg, not spatial outlier
        if (
            num_cams >= self.config.high_conf_min_views
            and num_inside >= self.config.high_conf_min_inside
            and orig_ratio >= self.config.high_conf_min_mask_ratio
            and raw_med_err <= self.config.high_conf_max_median_err_px
            and raw_p90_err <= self.config.high_conf_max_p90_err_px
            and max_ang >= self.config.high_conf_min_triangulation_angle_deg
            and not is_spatial_outlier
        ):
            reason = (
                f"High confidence: {num_inside}/{num_masked} views inside mask (ratio={orig_ratio:.2f}), "
                f"strong parallax ({max_ang:.1f}°), tightly bounded reprojection (med={raw_med_err:.2f}px, p90={raw_p90_err:.2f}px)."
            )
            return "HIGH_GEOMETRIC_CONFIDENCE", reason

        # Case 4: Moderate Geometric Confidence
        # >=2 views, >=2 inside, support ratio >=0.50, median err <=3.0px, p90 <=6.0px, angle >=1.8 deg
        if (
            num_cams >= self.config.mod_conf_min_views
            and num_inside >= self.config.mod_conf_min_inside
            and orig_ratio >= self.config.mod_conf_min_mask_ratio
            and (raw_med_err <= self.config.mod_conf_max_median_err_px or filt_med_err <= self.config.mod_conf_max_median_err_px)
            and raw_p90_err <= self.config.mod_conf_max_p90_err_px
            and max_ang >= self.config.mod_conf_min_triangulation_angle_deg
            and not is_spatial_outlier
        ):
            reason = (
                f"Moderate confidence: {num_inside}/{num_masked} views inside mask (ratio={orig_ratio:.2f}), "
                f"parallax {max_ang:.1f}°, acceptable reprojection (med={raw_med_err:.2f}px, p90={raw_p90_err:.2f}px)."
            )
            return "MODERATE_GEOMETRIC_CONFIDENCE", reason

        # Case 5: Low Geometric Confidence
        # Weak parallax, borderline reprojection, ambiguous mask ratio, or spatial isolation
        reasons = []
        if orig_ratio < self.config.mod_conf_min_mask_ratio:
            reasons.append(f"low mask support ({orig_ratio:.2f})")
        if max_ang < self.config.mod_conf_min_triangulation_angle_deg:
            reasons.append(f"weak parallax ({max_ang:.1f}°)")
        if raw_med_err > self.config.mod_conf_max_median_err_px:
            reasons.append(f"elevated reprojection error ({raw_med_err:.2f}px)")
        if is_spatial_outlier:
            reasons.append("marginal spatial isolation")

        reason_str = ", ".join(reasons) if reasons else "marginal geometric evidence"
        return "LOW_GEOMETRIC_CONFIDENCE", f"Low confidence: {reason_str}."

    def refine_candidate_points(
        self,
        candidate_associations: List[Dict[str, Any]],
        tracks_by_id: Dict[int, Any],
        registered_cameras: Dict[int, Any],
        camera_intrinsics_k: np.ndarray,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[PointGeometricProfile]:
        """
        Executes end-to-end refinement across all candidate points.
        """
        if not candidate_associations:
            return []

        # 1. Spatial clustering on candidate coordinates
        coords = np.array([
            [float(c["x_world"]), float(c["y_world"]), float(c["z_world"])]
            for c in candidate_associations
        ], dtype=np.float64)

        cluster_res = self.spatial_clusterer.cluster_points(coords)
        cluster_labels = cluster_res["cluster_labels"]
        spatial_outliers = cluster_res["is_spatial_outlier"]
        d1 = cluster_res["nn_dists_1"]
        d3 = cluster_res["nn_dists_3"]

        refined_profiles: List[PointGeometricProfile] = []

        for idx, c_data in enumerate(candidate_associations):
            pid = int(c_data["point_id"])
            tid = int(c_data["track_id"])
            p3d = (float(c_data["x_world"]), float(c_data["y_world"]), float(c_data["z_world"]))
            orig_class = str(c_data["classification"])
            orig_ratio = float(c_data["mask_support_ratio"])
            num_masked = int(c_data["masked_observations"])
            num_inside = int(c_data["inside_mask_observations"])

            tr = tracks_by_id.get(tid, {})
            obs_list = tr.get("observations", [])

            # Reprojection analysis
            obs_infos, reproj_stats = self.reproj_filter.evaluate_point_reprojections(
                point_3d=p3d,
                observations=obs_list,
                registered_cameras=registered_cameras,
                camera_intrinsics_k=camera_intrinsics_k,
                masks=masks,
            )

            # Triangulation analysis
            observed_cams = [o.camera_id for o in obs_infos if o.error_px < 999.0]
            tri_stats = self.tri_analyzer.analyze_triangulation(
                point_3d=p3d,
                camera_ids=observed_cams,
                registered_cameras=registered_cameras,
            )

            is_spat_outlier = bool(spatial_outliers[idx]) if idx < len(spatial_outliers) else False
            c_label = int(cluster_labels[idx]) if idx < len(cluster_labels) else -1
            nn1 = float(d1[idx]) if idx < len(d1) else 0.0
            nn3 = float(d3[idx]) if idx < len(d3) else 0.0

            # Supporting cameras
            sup_cams = []
            if "supporting_cameras" in c_data and str(c_data["supporting_cameras"]).strip():
                try:
                    sup_cams = [int(c) for c in str(c_data["supporting_cameras"]).split(";") if c.strip()]
                except Exception:
                    sup_cams = []

            # Assign refined status
            ref_status, ref_rationale = self.classify_point_confidence(
                orig_class=orig_class,
                orig_ratio=orig_ratio,
                num_masked=num_masked,
                num_inside=num_inside,
                num_cams=tri_stats["num_distinct_cameras"],
                reproj_stats=reproj_stats,
                tri_stats=tri_stats,
                is_spatial_outlier=is_spat_outlier,
            )

            raw = reproj_stats["raw"]
            filt = reproj_stats["filtered"]

            profile = PointGeometricProfile(
                point_id=pid,
                track_id=tid,
                point_3d=p3d,
                original_classification=orig_class,
                original_mask_support_ratio=orig_ratio,
                total_observations=len(obs_list),
                masked_observations=num_masked,
                inside_mask_observations=num_inside,
                num_distinct_cameras=tri_stats["num_distinct_cameras"],
                observed_camera_ids=observed_cams,
                supporting_camera_ids=sup_cams,
                mean_reproj_error_px=raw["mean"],
                median_reproj_error_px=raw["median"],
                std_reproj_error_px=raw["std"],
                p90_reproj_error_px=raw["p90"],
                p95_reproj_error_px=raw["p95"],
                max_reproj_error_px=raw["max"],
                count_err_gt_2px=raw["count_gt_2px"],
                count_err_gt_4px=raw["count_gt_4px"],
                count_err_gt_10px=raw["count_gt_10px"],
                num_outlier_observations=reproj_stats["num_outliers"],
                filtered_mean_reproj_error_px=filt["mean"],
                filtered_median_reproj_error_px=filt["median"],
                min_camera_distance=tri_stats["min_camera_distance"],
                max_camera_distance=tri_stats["max_camera_distance"],
                mean_camera_distance=tri_stats["mean_camera_distance"],
                relative_distance_std=tri_stats["relative_distance_std"],
                has_positive_depth_all_views=tri_stats["has_positive_depth_all_views"],
                max_triangulation_angle_deg=tri_stats["max_angle_deg"],
                median_triangulation_angle_deg=tri_stats["median_angle_deg"],
                min_triangulation_angle_deg=tri_stats["min_angle_deg"],
                max_camera_baseline=tri_stats["max_camera_baseline"],
                baseline_to_depth_ratio=tri_stats["baseline_to_depth_ratio"],
                is_triangulation_degenerate=tri_stats["is_degenerate"],
                nn_distance_1=nn1,
                nn_distance_3=nn3,
                is_spatial_outlier=is_spat_outlier,
                cluster_id=c_label,
                refined_status=ref_status,
                refinement_rationale=ref_rationale,
                observations=obs_infos,
            )
            refined_profiles.append(profile)

        return refined_profiles

    @classmethod
    def write_ply(
        cls,
        output_path: Union[str, Path],
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ):
        """Writes 3D point cloud to ASCII PLY format."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        n_pts = len(points)
        has_color = (colors is not None and len(colors) == n_pts)

        with open(out_p, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_pts}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(n_pts):
                x, y, z = points[i]
                if has_color:
                    r, g, b = colors[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    def export_refinement_artifacts(
        self,
        profiles: List[PointGeometricProfile],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Exports categorized PLYs, CSVs, and master JSON summary.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. CSV of refined points
        csv_path = out_dir / "refined_object_points.csv"
        rows = [p.to_csv_row() for p in profiles]
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        # 2. Partition by refined status
        partitions: Dict[str, List[PointGeometricProfile]] = {
            "HIGH_GEOMETRIC_CONFIDENCE": [],
            "MODERATE_GEOMETRIC_CONFIDENCE": [],
            "LOW_GEOMETRIC_CONFIDENCE": [],
            "GEOMETRIC_OUTLIER": [],
            "INSUFFICIENT_GEOMETRIC_EVIDENCE": [],
        }
        for p in profiles:
            partitions.setdefault(p.refined_status, []).append(p)

        color_map = {
            "HIGH_GEOMETRIC_CONFIDENCE": self.COLOR_HIGH,
            "MODERATE_GEOMETRIC_CONFIDENCE": self.COLOR_MODERATE,
            "LOW_GEOMETRIC_CONFIDENCE": self.COLOR_LOW,
            "GEOMETRIC_OUTLIER": self.COLOR_OUTLIER,
            "INSUFFICIENT_GEOMETRIC_EVIDENCE": self.COLOR_INSUFFICIENT,
        }

        # 3. Export all refined candidate points colored by confidence tier
        all_pts = np.array([p.point_3d for p in profiles], dtype=np.float64) if profiles else np.zeros((0, 3))
        all_colors = np.array([color_map.get(p.refined_status, (200, 200, 200)) for p in profiles], dtype=np.uint8) if profiles else None
        self.write_ply(out_dir / "refined_object_points.ply", all_pts, all_colors)

        # 4. Tiered PLYs
        file_map = {
            "HIGH_GEOMETRIC_CONFIDENCE": "high_confidence_points.ply",
            "MODERATE_GEOMETRIC_CONFIDENCE": "moderate_confidence_points.ply",
            "LOW_GEOMETRIC_CONFIDENCE": "low_confidence_points.ply",
            "GEOMETRIC_OUTLIER": "geometric_outliers.ply",
        }

        tier_counts = {}
        for status, filename in file_map.items():
            pts_tier = partitions.get(status, [])
            tier_counts[status] = len(pts_tier)
            pts_arr = np.array([p.point_3d for p in pts_tier], dtype=np.float64) if pts_tier else np.zeros((0, 3))
            c_arr = np.tile(color_map[status], (len(pts_arr), 1)) if len(pts_arr) > 0 else None
            self.write_ply(out_dir / filename, pts_arr, c_arr)

        # 5. Master JSON summary
        summary = {
            "total_candidates_evaluated": len(profiles),
            "refined_classification_counts": {
                status: len(pts) for status, pts in partitions.items()
            },
            "reprojection_statistics": {
                "high_conf_mean_err_px": round(float(np.mean([p.mean_reproj_error_px for p in partitions["HIGH_GEOMETRIC_CONFIDENCE"]])), 3) if partitions["HIGH_GEOMETRIC_CONFIDENCE"] else 0.0,
                "high_conf_median_err_px": round(float(np.median([p.median_reproj_error_px for p in partitions["HIGH_GEOMETRIC_CONFIDENCE"]])), 3) if partitions["HIGH_GEOMETRIC_CONFIDENCE"] else 0.0,
                "mod_conf_mean_err_px": round(float(np.mean([p.mean_reproj_error_px for p in partitions["MODERATE_GEOMETRIC_CONFIDENCE"]])), 3) if partitions["MODERATE_GEOMETRIC_CONFIDENCE"] else 0.0,
                "outlier_mean_err_px": round(float(np.mean([p.mean_reproj_error_px for p in partitions["GEOMETRIC_OUTLIER"]])), 3) if partitions["GEOMETRIC_OUTLIER"] else 0.0,
            },
            "triangulation_statistics": {
                "high_conf_mean_max_angle_deg": round(float(np.mean([p.max_triangulation_angle_deg for p in partitions["HIGH_GEOMETRIC_CONFIDENCE"]])), 2) if partitions["HIGH_GEOMETRIC_CONFIDENCE"] else 0.0,
                "mod_conf_mean_max_angle_deg": round(float(np.mean([p.max_triangulation_angle_deg for p in partitions["MODERATE_GEOMETRIC_CONFIDENCE"]])), 2) if partitions["MODERATE_GEOMETRIC_CONFIDENCE"] else 0.0,
            },
            "points": [p.to_dict() for p in profiles],
        }

        json_path = out_dir / "refinement_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

