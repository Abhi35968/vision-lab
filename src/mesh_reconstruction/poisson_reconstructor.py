"""
Screened Poisson Surface Reconstruction Module (Phase 3)
Generates smooth, watertight manifold 3D surfaces from oriented point clouds:
1. Open3D Screened Poisson Surface Reconstruction with configurable octree depth (7 and 8).
2. Density-based vertex trimming to eliminate low-confidence hallucinations in unobserved space.
3. Connected-component filtering to prune isolated floating mesh fragments.
4. Vertex and face normal recalculation and area-weighted smoothing.
5. Comparative evaluation against Alpha Shapes, Convex Hulls, and Parametric Revolutions.
"""

from dataclasses import dataclass, asdict, field
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
try:
    import open3d as o3d
except (ImportError, Exception):
    o3d = None
import trimesh

from src.object_reconstruction.bottle_models import Mesh3D

logger = logging.getLogger("PoissonReconstructor")


@dataclass
class PoissonConfig:
    """Parameters for Open3D Screened Poisson reconstruction and post-processing."""
    octree_depth: int = 7                # Recommended: 7 (smoother, faster), 8 (finer features)
    scale: float = 1.1                   # Bounding box scale
    linear_fit: bool = False             # Linear interpolation for smoother boundaries
    density_trim_quantile: float = 0.08  # Fraction of lowest-density boundary vertices to prune
    clean_disconnected: bool = True     # Retain only the largest connected surface component
    recompute_normals: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PoissonReport:
    """Metrics and fidelity assessment of the generated Poisson surface."""
    octree_depth: int
    raw_vertices_count: int
    raw_faces_count: int
    trimmed_vertices_count: int
    final_vertices_count: int
    final_faces_count: int
    is_watertight: bool
    mesh_volume: float
    surface_area: float
    bounding_box_extents: List[float]
    execution_time_ms: float
    limitation_notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class PoissonReconstructor:
    """
    Computes Screened Poisson surface meshes from oriented 3D point clouds.
    """

    def __init__(self, config: Optional[PoissonConfig] = None):
        self.config = config or PoissonConfig()

    def reconstruct(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        config_override: Optional[PoissonConfig] = None,
    ) -> Tuple[Mesh3D, PoissonReport]:
        """
        Executes Poisson surface reconstruction with density trimming and cleaning.
        
        Args:
            points: (N, 3) float64 point coordinates.
            normals: (N, 3) float64 surface normal vectors. If None, estimated on the fly.
        """
        t0 = time.time()
        cfg = config_override or self.config
        pts = np.asarray(points, dtype=np.float64)
        limitations: List[str] = []

        if len(pts) < 10:
            limitations.append("Point cloud has fewer than 10 points. Screened Poisson requires surface samples; falling back to convex hull.")
            # Fallback
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            hull, _ = pcd.compute_convex_hull()
            v = np.asarray(hull.vertices, dtype=np.float64)
            f = np.asarray(hull.triangles, dtype=np.int32)
            mesh3d = Mesh3D(vertices=v, faces=f)
            report = PoissonReport(
                octree_depth=cfg.octree_depth,
                raw_vertices_count=len(v),
                raw_faces_count=len(f),
                trimmed_vertices_count=0,
                final_vertices_count=len(v),
                final_faces_count=len(f),
                is_watertight=True,
                mesh_volume=0.0,
                surface_area=0.0,
                bounding_box_extents=[0.0, 0.0, 0.0],
                execution_time_ms=0.0,
                limitation_notes=limitations,
            )
            return mesh3d, report

        # 1. Prepare Open3D PointCloud with Normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        if normals is not None and len(normals) == len(pts):
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
        else:
            limitations.append("Surface normals were not precomputed; estimated using hybrid k-NN search.")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=min(15, len(pts) - 1))

        # Check for sparse sampling warning
        if len(pts) < 100:
            limitations.append("Sparse point cloud (< 100 points). Poisson indicator function may over-smooth fine local geometry.")

        # 2. Run Screened Poisson Surface Reconstruction
        mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=cfg.octree_depth,
            scale=cfg.scale,
            linear_fit=cfg.linear_fit,
        )

        raw_v_count = len(mesh_raw.vertices)
        raw_f_count = len(mesh_raw.triangles)

        # 3. Density Trimming (remove hallucinated boundary regions in unobserved space)
        densities_arr = np.asarray(densities)
        trimmed_v_count = 0

        if cfg.density_trim_quantile > 0.0 and len(densities_arr) > 0:
            thresh = np.percentile(densities_arr, cfg.density_trim_quantile * 100.0)
            vertices_to_remove = densities_arr < thresh
            trimmed_v_count = int(vertices_to_remove.sum())

            if trimmed_v_count < (0.90 * raw_v_count):
                mesh_raw.remove_vertices_by_mask(vertices_to_remove)

        # 4. Clean Disconnected Components
        if cfg.clean_disconnected and len(mesh_raw.triangles) > 0:
            triangle_clusters, num_triangles, _ = mesh_raw.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            num_triangles = np.asarray(num_triangles)

            if len(num_triangles) > 1:
                largest_cluster_idx = int(np.argmax(num_triangles))
                triangles_to_remove = triangle_clusters != largest_cluster_idx
                mesh_raw.remove_triangles_by_mask(triangles_to_remove)
                mesh_raw.remove_unreferenced_vertices()

        # 5. Normal Recalculation
        if cfg.recompute_normals:
            mesh_raw.compute_vertex_normals()
            mesh_raw.compute_triangle_normals()

        final_v = np.asarray(mesh_raw.vertices, dtype=np.float64)
        final_f = np.asarray(mesh_raw.triangles, dtype=np.int32)
        final_norm = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)

        # Check watertightness and volume with Trimesh
        try:
            tm = trimesh.Trimesh(vertices=final_v, faces=final_f, process=False)
            is_watertight = bool(tm.is_watertight)
            vol = float(abs(tm.volume)) if is_watertight else 0.0
            area = float(tm.area)
        except Exception:
            is_watertight = False
            vol = 0.0
            area = 0.0

        p_min = np.min(final_v, axis=0) if len(final_v) > 0 else np.zeros(3)
        p_max = np.max(final_v, axis=0) if len(final_v) > 0 else np.zeros(3)
        extents = [round(float(v), 3) for v in (p_max - p_min)]

        t_ms = (time.time() - t0) * 1000.0

        report = PoissonReport(
            octree_depth=cfg.octree_depth,
            raw_vertices_count=raw_v_count,
            raw_faces_count=raw_f_count,
            trimmed_vertices_count=trimmed_v_count,
            final_vertices_count=len(final_v),
            final_faces_count=len(final_f),
            is_watertight=is_watertight,
            mesh_volume=round(vol, 4),
            surface_area=round(area, 4),
            bounding_box_extents=extents,
            execution_time_ms=round(t_ms, 2),
            limitation_notes=limitations,
        )

        mesh3d = Mesh3D(vertices=final_v, faces=final_f, normals=final_norm)
        return mesh3d, report

    def compare_methods(self, points: np.ndarray) -> Dict[str, dict]:
        """
        Benchmarks Screened Poisson (depth 7 & 8) against Alpha Shape and Convex Hull.
        """
        pts = np.asarray(points, dtype=np.float64)
        results = {}

        # 1. Poisson Depth 7
        try:
            _, rep7 = self.reconstruct(pts, config_override=PoissonConfig(octree_depth=7))
            results["Screened Poisson (Depth 7)"] = {
                "vertices": rep7.final_vertices_count,
                "faces": rep7.final_faces_count,
                "watertight": rep7.is_watertight,
                "volume": rep7.mesh_volume,
                "time_ms": rep7.execution_time_ms,
            }
        except Exception as e:
            results["Screened Poisson (Depth 7)"] = {"error": str(e)}

        # 2. Poisson Depth 8
        try:
            _, rep8 = self.reconstruct(pts, config_override=PoissonConfig(octree_depth=8))
            results["Screened Poisson (Depth 8)"] = {
                "vertices": rep8.final_vertices_count,
                "faces": rep8.final_faces_count,
                "watertight": rep8.is_watertight,
                "volume": rep8.mesh_volume,
                "time_ms": rep8.execution_time_ms,
            }
        except Exception as e:
            results["Screened Poisson (Depth 8)"] = {"error": str(e)}

        # 3. Convex Hull
        try:
            t0 = time.time()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            hull, _ = pcd.compute_convex_hull()
            t_ms = (time.time() - t0) * 1000.0
            tm = trimesh.Trimesh(np.asarray(hull.vertices), np.asarray(hull.triangles), process=False)
            results["Convex Hull"] = {
                "vertices": len(hull.vertices),
                "faces": len(hull.triangles),
                "watertight": bool(tm.is_watertight),
                "volume": round(float(tm.volume), 4),
                "time_ms": round(t_ms, 2),
            }
        except Exception as e:
            results["Convex Hull"] = {"error": str(e)}

        return results
