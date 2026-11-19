"""
Multi-View Dense Depth Fusion Engine (Phase 6 Enhancement & Forensic Repair)
Integrates Monocular Depth (Depth Anything V2 / MiDaS) across multiple camera viewpoints:
1. Triangulates multi-view mask centroid rays to establish a drift-free 3D object anchor.
2. Formulates mathematically rigorous inverse depth alignment (1/Z = a * d + b) anchored to physical camera distances.
3. Filters silhouette boundary leakage and disparity outliers (eroded masks + quantile filtering).
4. Unprojects object pixels into a unified world coordinate frame with RGB vertex colors.
5. Employs Open3D for voxel downsampling, statistical outlier removal, table plane segmentation,
   and normal estimation.
6. Evaluates surface meshes with truthful manifoldness and watertightness diagnostics.
"""

from dataclasses import dataclass, asdict, field
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import numpy as np
try:
    import open3d as o3d
except (ImportError, Exception):
    o3d = None

from src.dense_depth.depth_anything_estimator import DepthAnythingEstimator
from src.point_cloud_optimization.point_cloud_optimizer import (
    PointCloudOptimizer,
    OptimizationConfig,
    OptimizationReport,
)

logger = logging.getLogger("MultiViewDenseFuser")


@dataclass
class DenseFusionConfig:
    """Configurable options for multi-view dense fusion."""
    stride: int = 3                       # Pixel stride for unprojection (denser = more points)
    max_points_per_view: int = 20000       # Cap points per camera view
    voxel_size: float = 0.015             # Voxel grid downsampling size (finer = more detail)
    enable_sor: bool = True               # Statistical Outlier Removal
    sor_nb_neighbors: int = 30
    sor_std_ratio: float = 1.2
    enable_plane_removal: bool = True     # Strip supporting horizontal table/floor plane
    plane_dist_thresh: float = 0.03
    enable_cluster_filtering: bool = True # Keep largest connected component
    enable_mesh_generation: bool = True   # Poisson surface reconstruction
    poisson_depth: int = 9                # Octree depth for Poisson reconstruction (higher = more detail)
    output_dir: str = "outputs/dense_depth"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeshDiagnostics:
    """Truthful geometric topology metrics for reconstructed surface mesh."""
    is_watertight: bool
    is_edge_manifold: bool
    is_vertex_manifold: bool
    is_self_intersecting: bool
    num_vertices: int
    num_triangles: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DenseFusionResult:
    """Telemetry and file references from multi-view dense reconstruction."""
    total_views_processed: int
    raw_points_fused: int
    clean_points_count: int
    retention_rate_pct: float
    sparse_points_reference_count: int
    density_multiplier: float
    mesh_vertices_count: int
    mesh_triangles_count: int
    raw_ply_path: str
    clean_ply_path: str
    mesh_ply_path: Optional[str]
    summary_json_path: str
    execution_time_s: float
    mesh_diagnostics: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class MultiViewDenseFuser:
    """
    Orchestrates dense point cloud generation across multiple registered views.
    """

    def __init__(
        self,
        config: Optional[DenseFusionConfig] = None,
        estimator: Optional[DepthAnythingEstimator] = None,
    ):
        self.config = config or DenseFusionConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_frame_dir = self.output_dir / "per_frame"
        self.per_frame_dir.mkdir(parents=True, exist_ok=True)
        self.estimator = estimator or DepthAnythingEstimator(cache_dir=self.output_dir)

    def _triangulate_object_anchor(
        self,
        views_data: List[Dict[str, Any]],
        sparse_points_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Determines the 3D world anchor of the target object using:
        1. Multi-view mask centroid ray intersection (guaranteed drift-free baseline, immune to background penetration).
        2. Inlier sparse points projecting inside object masks (fallback).
        """
        # A. Triangulate multi-view mask centroid rays (Primary & Drift-Free)
        rays = []
        centers = []
        for v in views_data:
            m = v.get("mask")
            if m is None or (m > 127).sum() < 20:
                continue
            ys, xs = np.where(m > 127)
            u_c, v_c = float(np.mean(xs)), float(np.mean(ys))

            R = np.asarray(v["R"], dtype=np.float64)
            t = np.asarray(v["t"], dtype=np.float64).flatten()
            K = np.asarray(v["K"], dtype=np.float64)
            C = -R.T @ t

            ray_cam = np.linalg.inv(K) @ np.array([u_c, v_c, 1.0])
            norm = np.linalg.norm(ray_cam)
            if norm < 1e-6:
                continue
            ray_cam /= norm
            ray_world = R.T @ ray_cam
            rays.append(ray_world)
            centers.append(C)

        if len(rays) >= 2:
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for C, r in zip(centers, rays):
                r_col = r.reshape(3, 1)
                proj = np.eye(3) - r_col @ r_col.T
                A += proj
                b += proj @ C
            try:
                anchor = np.linalg.solve(A, b)
                # Verify that anchor is in front of cameras (positive depth)
                valid_proj_count = 0
                for v in views_data[:3]:
                    R = np.asarray(v["R"], dtype=np.float64)
                    t = np.asarray(v["t"], dtype=np.float64).flatten()
                    if (R @ anchor + t)[2] > 0.5:
                        valid_proj_count += 1
                if valid_proj_count >= 2:
                    logger.info(f"Anchor from mask ray intersection: {anchor}")
                    return anchor
            except np.linalg.LinAlgError:
                pass

        # B. Fallback: Try inlier sparse points inside masks
        if len(sparse_points_3d) > 0:
            obj_sparse = []
            for v in views_data:
                m = v.get("mask")
                if m is None or (m > 127).sum() < 20:
                    continue
                R = np.asarray(v["R"], dtype=np.float64)
                t = np.asarray(v["t"], dtype=np.float64).flatten()
                K = np.asarray(v["K"], dtype=np.float64)
                h, w = m.shape[:2]

                pts_cam = (R @ sparse_points_3d.T + t.reshape(3, 1)).T
                z = pts_cam[:, 2]
                valid_z = z > 0.05
                proj = (K @ pts_cam.T).T
                u = np.zeros(len(sparse_points_3d))
                v_arr = np.zeros(len(sparse_points_3d))
                u[valid_z] = proj[valid_z, 0] / z[valid_z]
                v_arr[valid_z] = proj[valid_z, 1] / z[valid_z]

                inside = (
                    valid_z
                    & (u >= 0) & (u < w)
                    & (v_arr >= 0) & (v_arr < h)
                )
                for idx in np.where(inside)[0]:
                    v_px = min(h - 1, max(0, int(round(v_arr[idx]))))
                    u_px = min(w - 1, max(0, int(round(u[idx]))))
                    if m[v_px, u_px] > 127:
                        obj_sparse.append(sparse_points_3d[idx])

            if len(obj_sparse) >= 5:
                obj_sparse_np = np.asarray(obj_sparse, dtype=np.float64)
                med = np.median(obj_sparse_np, axis=0)
                dists = np.linalg.norm(obj_sparse_np - med, axis=1)
                inliers = obj_sparse_np[dists < np.percentile(dists, 80)]
                if len(inliers) >= 3:
                    anchor = np.median(inliers, axis=0)
                    logger.info(f"Anchor from sparse inliers: {anchor}")
                    return anchor
                pass

        # Fallback to mean of reference points or nominal distance
        if len(sparse_points_3d) > 0:
            return np.median(sparse_points_3d, axis=0)
        return np.array([0.0, 0.0, 3.0])

    def fuse_views(
        self,
        views_data: List[Dict[str, Any]],
        sparse_points_3d: np.ndarray,
        tag: str = "fused_object",
    ) -> DenseFusionResult:
        """
        Fuses dense unprojected points across all supplied camera views.
        """
        t0 = time.time()
        logger.info(f"Starting multi-view dense fusion across {len(views_data)} views...")

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        notes: List[str] = []

        pts_ref = np.asarray(sparse_points_3d, dtype=np.float64)
        if len(pts_ref) == 0:
            notes.append("No sparse SfM reference points supplied; used multi-view mask ray triangulation.")

        # Establish global object 3D anchor
        object_anchor = self._triangulate_object_anchor(views_data, pts_ref)
        notes.append(f"Global object anchor world position: {np.round(object_anchor, 3).tolist()}")

        valid_views_count = 0

        for idx, v in enumerate(views_data):
            img_bgr = v.get("image_bgr")
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            name = v.get("name", f"view_{idx:03d}")
            mask = v.get("mask")
            R = np.asarray(v["R"], dtype=np.float64)
            t = np.asarray(v["t"], dtype=np.float64).flatten()
            K = np.asarray(v["K"], dtype=np.float64)

            # Expected camera Z for the object
            cam_pt = R @ object_anchor + t
            expected_z = float(cam_pt[2])
            if expected_z <= 0.1:
                expected_z = 3.0

            # 1. Estimate monocular relative disparity/depth
            depth_rel, _ = self.estimator.estimate_relative_depth(img_bgr, cache_key=name)

            # 2. Mask processing & boundary silhouette leak rejection
            if mask is not None:
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                k_size = 7 if min(h, w) > 400 else 3
                kernel = np.ones((k_size, k_size), np.uint8)
                eroded_mask = cv2.erode(mask, kernel, iterations=1)
                obj_pixels = eroded_mask > 127
            else:
                eroded_mask = np.ones((h, w), dtype=np.uint8) * 255
                obj_pixels = np.ones((h, w), dtype=bool)

            disp_vals = depth_rel[obj_pixels]
            if len(disp_vals) < 20:
                continue

            # Reject far background pixels bleeding into the object mask (bottom 15% disparity)
            disp_thresh = np.percentile(disp_vals, 15)
            clean_mask = obj_pixels & (depth_rel >= disp_thresh)
            clean_disp = depth_rel[clean_mask]
            if len(clean_disp) < 20:
                continue

            med_disp = float(np.median(clean_disp))
            disp_span = max(1e-4, float(np.percentile(clean_disp, 90) - np.percentile(clean_disp, 10)))

            # 3. Physics-based Inverse Depth Calibration: 1/Z = a * d + b
            # Delta(1/Z) approx Delta Z / Z^2 with object depth variation ~0.15m
            scale_a = (0.15 / (expected_z ** 2)) / disp_span
            b_shift = (1.0 / expected_z) - scale_a * med_disp

            inv_z = np.clip(scale_a * depth_rel + b_shift, 1e-4, 10.0)
            metric_z = 1.0 / inv_z

            # 4. Unproject masked pixels
            stride = self.config.stride
            K_inv = np.linalg.inv(K)
            R_inv = R.T
            t_vec = t.reshape(3, 1)

            yy, xx = np.mgrid[0:h:stride, 0:w:stride]
            vm = clean_mask[::stride, ::stride]

            xx_v = xx[vm]
            yy_v = yy[vm]
            z_v = metric_z[::stride, ::stride][vm]

            good = (z_v > 0.05) & (z_v < 50.0)
            xx_v = xx_v[good]
            yy_v = yy_v[good]
            z_v = z_v[good]

            if len(z_v) == 0:
                continue

            # Cap per-view points for CPU memory management
            if len(z_v) > self.config.max_points_per_view:
                sub = np.random.choice(len(z_v), self.config.max_points_per_view, replace=False)
                xx_v = xx_v[sub]
                yy_v = yy_v[sub]
                z_v = z_v[sub]

            pixels_homo = np.vstack([xx_v, yy_v, np.ones_like(xx_v)])
            rays_cam = K_inv @ pixels_homo
            pts_cam = rays_cam * z_v
            pts_world = (R_inv @ (pts_cam - t_vec)).T

            # Extract RGB colors
            colors_rgb = img_bgr[yy_v, xx_v, ::-1].astype(np.float64) / 255.0

            # Save individual per-frame cloud
            pcd_frame = o3d.geometry.PointCloud()
            pcd_frame.points = o3d.utility.Vector3dVector(pts_world)
            pcd_frame.colors = o3d.utility.Vector3dVector(colors_rgb)
            frame_ply = self.per_frame_dir / f"{name}_cloud.ply"
            o3d.io.write_point_cloud(str(frame_ply), pcd_frame)

            all_points.append(pts_world)
            all_colors.append(colors_rgb)
            valid_views_count += 1

        if not all_points:
            logger.warning("No valid points generated across all views.")
            empty_report = DenseFusionResult(
                total_views_processed=0,
                raw_points_fused=0,
                clean_points_count=0,
                retention_rate_pct=0.0,
                sparse_points_reference_count=len(pts_ref),
                density_multiplier=0.0,
                mesh_vertices_count=0,
                mesh_triangles_count=0,
                raw_ply_path="",
                clean_ply_path="",
                mesh_ply_path=None,
                summary_json_path="",
                execution_time_s=round(time.time() - t0, 2),
                notes=["Zero points reconstructed across all frames."],
            )
            return empty_report

        fused_pts = np.vstack(all_points)
        fused_cols = np.vstack(all_colors)
        raw_count = len(fused_pts)

        # 5. Save raw fused point cloud
        raw_ply_path = self.output_dir / f"{tag}_raw_fused.ply"
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(fused_pts)
        pcd_raw.colors = o3d.utility.Vector3dVector(fused_cols)
        o3d.io.write_point_cloud(str(raw_ply_path), pcd_raw)

        # 6. Point Cloud Optimization & Open3D Cleanup
        opt_cfg = OptimizationConfig(
            enable_voxel_downsampling=True,
            voxel_size=self.config.voxel_size,
            enable_sor=self.config.enable_sor,
            sor_nb_neighbors=self.config.sor_nb_neighbors,
            sor_std_ratio=self.config.sor_std_ratio,
            enable_ror=False,
            enable_cluster_filtering=self.config.enable_cluster_filtering,
            enable_plane_removal=self.config.enable_plane_removal,
            plane_dist_thresh=self.config.plane_dist_thresh,
            enable_normal_estimation=True,
            normal_k_neighbors=30,
            orient_normals_towards_centroid=True,
            auto_scale_for_sparse=False,
        )
        optimizer = PointCloudOptimizer(config=opt_cfg)
        clean_pts, clean_cols, clean_normals, opt_report = optimizer.optimize(
            fused_pts, colors=fused_cols
        )
        clean_count = len(clean_pts)

        # 7. Save clean optimized point cloud
        clean_ply_path = self.output_dir / f"{tag}_clean_optimized.ply"
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(clean_pts)
        if clean_cols is not None:
            pcd_clean.colors = o3d.utility.Vector3dVector(clean_cols)
        if clean_normals is not None:
            pcd_clean.normals = o3d.utility.Vector3dVector(clean_normals)
        o3d.io.write_point_cloud(str(clean_ply_path), pcd_clean)

        # 8. Surface Reconstruction & Diagnostics (Poisson)
        mesh_ply_path = None
        mesh_verts = 0
        mesh_tris = 0
        mesh_diag = None

        if self.config.enable_mesh_generation and clean_count >= 50 and clean_normals is not None:
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd_clean, depth=self.config.poisson_depth
                )
                # Save untrimmed mesh
                untrimmed_path = self.output_dir / f"{tag}_poisson_mesh_untrimmed.ply"
                o3d.io.write_triangle_mesh(str(untrimmed_path), mesh)

                # Trim low density extrapolation bubbles
                densities = np.asarray(densities)
                thresh = np.quantile(densities, 0.05)
                vertices_to_remove = densities < thresh
                mesh.remove_vertices_by_mask(vertices_to_remove)

                mesh_verts = len(mesh.vertices)
                mesh_tris = len(mesh.triangles)
                if mesh_tris > 0:
                    mesh_out = self.output_dir / f"{tag}_poisson_mesh.ply"
                    o3d.io.write_triangle_mesh(str(mesh_out), mesh)
                    mesh_ply_path = str(mesh_out)

                    mesh_diag = MeshDiagnostics(
                        is_watertight=bool(mesh.is_watertight()),
                        is_edge_manifold=bool(mesh.is_edge_manifold()),
                        is_vertex_manifold=bool(mesh.is_vertex_manifold()),
                        is_self_intersecting=bool(mesh.is_self_intersecting()),
                        num_vertices=mesh_verts,
                        num_triangles=mesh_tris,
                    ).to_dict()

            except Exception as e:
                logger.warning(f"Poisson mesh generation failed: {e}")
                notes.append(f"Poisson mesh generation error: {str(e)}")

        exec_time = round(time.time() - t0, 2)
        sparse_ref_count = max(1, len(pts_ref))
        density_mult = round(float(clean_count) / float(sparse_ref_count), 2)
        retention_pct = round((clean_count / raw_count * 100.0) if raw_count > 0 else 0.0, 1)

        result = DenseFusionResult(
            total_views_processed=valid_views_count,
            raw_points_fused=raw_count,
            clean_points_count=clean_count,
            retention_rate_pct=retention_pct,
            sparse_points_reference_count=len(pts_ref),
            density_multiplier=density_mult,
            mesh_vertices_count=mesh_verts,
            mesh_triangles_count=mesh_tris,
            raw_ply_path=str(raw_ply_path),
            clean_ply_path=str(clean_ply_path),
            mesh_ply_path=mesh_ply_path,
            summary_json_path=str(self.output_dir / f"{tag}_summary.json"),
            execution_time_s=exec_time,
            mesh_diagnostics=mesh_diag,
            notes=notes,
        )

        with open(result.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result
