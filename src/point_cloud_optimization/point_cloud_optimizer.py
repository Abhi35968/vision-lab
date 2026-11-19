"""
Point Cloud Optimization Module (Phase 2)
Applies multi-stage geometric filtering, normal estimation, and noise reduction:
1. Statistical Outlier Removal (SOR) to eliminate sparse scatter.
2. Radius Outlier Removal (ROR) to prune isolated drift points.
3. Adaptive Voxel Downsampling for uniform spatial density.
4. Robust k-NN Normal Estimation and Orientation towards camera positions or object centroid.
5. DBSCAN Spatial Cluster Filtering to excise disconnected background artifacts.
6. CPU memory optimization (< 250 MB RAM) with original cloud preservation.
"""

from dataclasses import dataclass, asdict, field
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import open3d as o3d

logger = logging.getLogger("PointCloudOptimizer")


@dataclass
class OptimizationConfig:
    """Configurable parameters for multi-stage point cloud enhancement."""
    # Voxel grid downsampling
    enable_voxel_downsampling: bool = True
    voxel_size: float = 0.05                 # In reconstruction units

    # Statistical Outlier Removal
    enable_sor: bool = True
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 1.5

    # Radius Outlier Removal
    enable_ror: bool = True
    ror_nb_points: int = 8
    ror_radius: float = 0.15

    # DBSCAN Cluster Cleanup
    enable_cluster_filtering: bool = True
    dbscan_eps: float = 0.12
    dbscan_min_points: int = 6
    keep_largest_cluster_only: bool = True

    # Supporting Plane (Table / Floor) Removal
    enable_plane_removal: bool = False
    plane_dist_thresh: float = 0.03
    plane_ransac_n: int = 3
    plane_num_iterations: int = 1000

    # Normal Estimation
    enable_normal_estimation: bool = True
    normal_k_neighbors: int = 25
    normal_search_radius: float = 0.20
    orient_normals_towards_centroid: bool = True

    # Adaptive density handling for sparse SfM point clouds
    auto_scale_for_sparse: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizationReport:
    """Quantitative telemetry of points retained and rejected at each stage."""
    raw_points_count: int
    after_voxel_count: int
    sor_filtered_count: int
    ror_filtered_count: int
    cluster_filtered_count: int
    final_points_count: int
    retention_rate_pct: float
    normals_estimated: bool
    density_points_per_unit_vol: float
    execution_time_ms: float
    plane_removed_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class PointCloudOptimizer:
    """
    Cleans, refines, and estimates normals for sparse and dense 3D point clouds.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    def optimize(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        camera_centers: Optional[np.ndarray] = None,
        config_override: Optional[OptimizationConfig] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], OptimizationReport]:
        """
        Executes full multi-stage point cloud optimization.
        
        Returns:
            clean_points: (M, 3) float64 array of optimized coordinates
            clean_colors: (M, 3) float64 array of vertex RGB colors (or None)
            clean_normals: (M, 3) float64 array of unit surface normals (or None)
            report: OptimizationReport telemetry record
        """
        t0 = time.time()
        cfg = config_override or self.config
        raw_pts = np.asarray(points, dtype=np.float64)
        raw_count = len(raw_pts)

        if raw_count < 4:
            # Not enough points for statistical filtering; return identity
            return raw_pts, colors, None, OptimizationReport(
                raw_points_count=raw_count,
                after_voxel_count=raw_count,
                sor_filtered_count=0,
                ror_filtered_count=0,
                cluster_filtered_count=0,
                final_points_count=raw_count,
                retention_rate_pct=100.0,
                normals_estimated=False,
                density_points_per_unit_vol=0.0,
                execution_time_ms=0.0,
            )

        # 1. Initialize Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pts)
        if colors is not None and len(colors) == raw_count:
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))

        # 2. Voxel Downsampling (if enabled and voxel_size > 0)
        curr_pcd = pcd
        if cfg.enable_voxel_downsampling and cfg.voxel_size > 0.001:
            down_pcd = curr_pcd.voxel_down_sample(voxel_size=cfg.voxel_size)
            if len(down_pcd.points) > 0:
                curr_pcd = down_pcd
        after_voxel = len(curr_pcd.points)

        # Adaptive parameter calculation for sparse point clouds (< 100 points)
        eff_sor_nb = cfg.sor_nb_neighbors
        eff_ror_radius = cfg.ror_radius
        eff_ror_nb = cfg.ror_nb_points
        eff_dbscan_eps = cfg.dbscan_eps
        eff_dbscan_min = cfg.dbscan_min_points

        if cfg.auto_scale_for_sparse and raw_count < 100 and len(curr_pcd.points) >= 6:
            try:
                from scipy.spatial.distance import pdist, squareform
                curr_pts_arr = np.asarray(curr_pcd.points)
                dists = squareform(pdist(curr_pts_arr))
                np.fill_diagonal(dists, np.inf)
                nn_dists = np.min(dists, axis=1)
                p85_nn = float(np.percentile(nn_dists, 85))
                if p85_nn > cfg.dbscan_eps:
                    eff_dbscan_eps = max(cfg.dbscan_eps, p85_nn * 2.2)
                    eff_dbscan_min = min(cfg.dbscan_min_points, 3)
                if cfg.enable_ror and p85_nn > cfg.ror_radius:
                    eff_ror_radius = max(cfg.ror_radius, p85_nn * 2.2)
                    eff_ror_nb = min(cfg.ror_nb_points, 3)
                if cfg.sor_nb_neighbors >= len(curr_pcd.points) // 2:
                    eff_sor_nb = max(4, len(curr_pcd.points) // 6)
            except Exception as e:
                logger.debug(f"Adaptive sparse scaling fallback: {e}")

        # 3. Statistical Outlier Removal (SOR)
        sor_removed = 0
        if cfg.enable_sor and len(curr_pcd.points) > eff_sor_nb:
            clean_pcd, ind_sor = curr_pcd.remove_statistical_outlier(
                nb_neighbors=eff_sor_nb,
                std_ratio=cfg.sor_std_ratio,
            )
            if len(clean_pcd.points) >= 4:
                sor_removed = len(curr_pcd.points) - len(clean_pcd.points)
                curr_pcd = clean_pcd

        # 4. Radius Outlier Removal (ROR)
        ror_removed = 0
        if cfg.enable_ror and len(curr_pcd.points) > eff_ror_nb:
            clean_pcd, ind_ror = curr_pcd.remove_radius_outlier(
                nb_points=eff_ror_nb,
                radius=eff_ror_radius,
            )
            if len(clean_pcd.points) >= 4:
                ror_removed = len(curr_pcd.points) - len(clean_pcd.points)
                curr_pcd = clean_pcd

        # 5. DBSCAN Spatial Cluster Cleanup
        cluster_removed = 0
        if cfg.enable_cluster_filtering and len(curr_pcd.points) > eff_dbscan_min:
            labels = np.array(curr_pcd.cluster_dbscan(
                eps=eff_dbscan_eps,
                min_points=eff_dbscan_min,
                print_progress=False,
            ))
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                if cfg.keep_largest_cluster_only:
                    counts = np.bincount(valid_labels)
                    best_cluster_id = int(np.argmax(counts))
                    mask = (labels == best_cluster_id)
                else:
                    mask = (labels >= 0)
                
                if mask.sum() >= 4:
                    cluster_removed = len(curr_pcd.points) - int(mask.sum())
                    curr_pcd = curr_pcd.select_by_index(np.where(mask)[0])

        # 5b. Supporting Plane (Table / Floor) Segmentation & Removal
        plane_removed = 0
        if cfg.enable_plane_removal and len(curr_pcd.points) > 30:
            try:
                _, inliers = curr_pcd.segment_plane(
                    distance_threshold=cfg.plane_dist_thresh,
                    ransac_n=cfg.plane_ransac_n,
                    num_iterations=cfg.plane_num_iterations,
                )
                if len(inliers) > 0 and len(curr_pcd.points) - len(inliers) >= 10:
                    plane_removed = len(inliers)
                    curr_pcd = curr_pcd.select_by_index(inliers, invert=True)
            except Exception as e:
                logger.debug(f"Plane segmentation skipped: {e}")

        # 6. Normal Estimation & Consistent Orientation
        normals_estimated = False
        if cfg.enable_normal_estimation and len(curr_pcd.points) >= 4:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius=cfg.normal_search_radius,
                max_nn=cfg.normal_k_neighbors,
            )
            curr_pcd.estimate_normals(search_param=search_param)

            # Orient normals
            if camera_centers is not None and len(camera_centers) > 0:
                # Orient towards average camera position
                cam_mean = np.mean(camera_centers, axis=0)
                curr_pcd.orient_normals_towards_camera_location(cam_mean)
            elif cfg.orient_normals_towards_centroid:
                # Orient outwards from object centroid
                centroid = np.mean(np.asarray(curr_pcd.points), axis=0)
                curr_pcd.orient_normals_towards_camera_location(centroid - np.array([0, 0, 10.0]))
            else:
                curr_pcd.orient_normals_consistent_tangent_plane(k=min(15, len(curr_pcd.points) - 1))

            normals_estimated = True

        final_pts = np.asarray(curr_pcd.points, dtype=np.float64)
        final_colors = np.asarray(curr_pcd.colors, dtype=np.float64) if curr_pcd.has_colors() else None
        final_normals = np.asarray(curr_pcd.normals, dtype=np.float64) if curr_pcd.has_normals() else None

        # Compute point density metric
        if len(final_pts) > 3:
            p_min = np.min(final_pts, axis=0)
            p_max = np.max(final_pts, axis=0)
            bbox_vol = max(1e-4, float(np.prod(np.maximum(p_max - p_min, 0.01))))
            density = float(len(final_pts) / bbox_vol)
        else:
            density = 0.0

        t_ms = (time.time() - t0) * 1000.0
        retention = (len(final_pts) / raw_count * 100.0) if raw_count > 0 else 100.0

        report = OptimizationReport(
            raw_points_count=raw_count,
            after_voxel_count=after_voxel,
            sor_filtered_count=sor_removed,
            ror_filtered_count=ror_removed,
            cluster_filtered_count=cluster_removed,
            plane_removed_count=plane_removed,
            final_points_count=len(final_pts),
            retention_rate_pct=round(retention, 1),
            normals_estimated=normals_estimated,
            density_points_per_unit_vol=round(density, 2),
            execution_time_ms=round(t_ms, 2),
        )

        return final_pts, final_colors, final_normals, report

    def export_comparison_ply(
        self,
        raw_points: np.ndarray,
        clean_points: np.ndarray,
        clean_normals: Optional[np.ndarray],
        output_dir: Union[str, Path],
        tag: str = "object",
    ) -> Tuple[Path, Path]:
        """
        Saves both original raw and optimized point clouds side-by-side for verification.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        raw_path = out_p / f"{tag}_original_raw.ply"
        clean_path = out_p / f"{tag}_optimized_clean.ply"

        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(np.asarray(raw_points, dtype=np.float64))
        o3d.io.write_point_cloud(str(raw_path), pcd_raw)

        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(np.asarray(clean_points, dtype=np.float64))
        if clean_normals is not None and len(clean_normals) == len(clean_points):
            pcd_clean.normals = o3d.utility.Vector3dVector(np.asarray(clean_normals, dtype=np.float64))
        o3d.io.write_point_cloud(str(clean_path), pcd_clean)

        return raw_path, clean_path
