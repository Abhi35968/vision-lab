"""
Comprehensive Comparative Evaluation of 3D Reconstruction Paradigms:
1. Baseline: Depth Anything + Depth Gradient Filtering Unprojection
2. Open3D TSDF Volume Integration
3. COLMAP CUDA PatchMatch MVS + Poisson Surface Reconstruction
"""
import json
import logging
from pathlib import Path
import numpy as np
import open3d as o3d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mvs_eval")

def evaluate_point_cloud(pcd_path: Path, name: str) -> dict:
    if not pcd_path.exists():
        logger.warning(f"File not found: {pcd_path}")
        return {"name": name, "exists": False}

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return {"name": name, "exists": True, "points": 0}

    # Bounding box
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    extent = bbox_max - bbox_min

    # Centroid & Radial circularity in X-Z plane
    centroid = np.mean(pts, axis=0)
    rel_pts = pts - centroid
    r_xz = np.sqrt(rel_pts[:, 0]**2 + rel_pts[:, 2]**2)

    # Filter out top/bottom cap outlier slices to evaluate cylinder body circularity
    y_mid = np.percentile(pts[:, 1], [25, 75])
    mid_mask = (pts[:, 1] >= y_mid[0]) & (pts[:, 1] <= y_mid[1])
    mid_r = r_xz[mid_mask] if np.any(mid_mask) else r_xz

    circularity_ratio = float(extent[2] / (extent[0] + 1e-8)) # Z / X ratio (ideal cylinder = 1.0)
    r_mean = float(np.mean(mid_r))
    r_std = float(np.std(mid_r))
    r_rel_err = float(r_std / (r_mean + 1e-8))

    return {
        "name": name,
        "exists": True,
        "points_count": len(pts),
        "bounding_box_extent": [round(float(v), 4) for v in extent],
        "height_y": round(float(extent[1]), 4),
        "width_x": round(float(extent[0]), 4),
        "depth_z": round(float(extent[2]), 4),
        "circularity_zx_ratio": round(circularity_ratio, 4),
        "radial_mean": round(r_mean, 4),
        "radial_std": round(r_std, 4),
        "radial_relative_error_pct": round(r_rel_err * 100, 2),
    }

def evaluate_mesh(mesh_path: Path, name: str) -> dict:
    if not mesh_path.exists():
        logger.warning(f"File not found: {mesh_path}")
        return {"name": name, "exists": False}

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(vertices) == 0 or len(triangles) == 0:
        return {"name": name, "exists": True, "vertices": 0, "faces": 0}

    is_watertight = mesh.is_watertight()
    surface_area = mesh.get_surface_area()

    return {
        "name": name,
        "exists": True,
        "vertices_count": len(vertices),
        "triangles_count": len(triangles),
        "is_watertight": bool(is_watertight),
        "surface_area": round(float(surface_area), 4),
    }

def main():
    root = Path(".")
    
    # Paths
    pcd_baseline = root / "outputs" / "deployment_live_validation" / "final_object_only_cloud.ply"
    mesh_baseline = root / "outputs" / "deployment_live_validation" / "poisson_mesh.ply"
    
    pcd_tsdf = root / "outputs" / "deployment_live_validation" / "tsdf_fusion" / "tsdf_volume_cloud.ply"
    mesh_tsdf = root / "outputs" / "deployment_live_validation" / "tsdf_fusion" / "tsdf_volume_mesh.ply"
    
    pcd_colmap = root / "outputs" / "colab_mvs_results" / "fused.ply"
    mesh_colmap = root / "outputs" / "colab_mvs_results" / "mvs_poisson_mesh.ply"

    eval_results = {
        "point_clouds": {
            "baseline_depth_anything": evaluate_point_cloud(pcd_baseline, "Baseline (Depth Anything + Gradient Filter)"),
            "tsdf_volume_fusion": evaluate_point_cloud(pcd_tsdf, "Open3D TSDF Volume Integration"),
            "colmap_patchmatch_mvs": evaluate_point_cloud(pcd_colmap, "COLMAP CUDA PatchMatch MVS"),
        },
        "meshes": {
            "baseline_poisson": evaluate_mesh(mesh_baseline, "Baseline Poisson Mesh"),
            "tsdf_marching_cubes": evaluate_mesh(mesh_tsdf, "TSDF Marching Cubes Mesh"),
            "colmap_poisson": evaluate_mesh(mesh_colmap, "COLMAP PatchMatch Poisson Mesh"),
        }
    }

    out_file = root / "outputs" / "colab_mvs_results" / "comparative_evaluation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Evaluation report written to {out_file}")
    print(json.dumps(eval_results, indent=2))

if __name__ == "__main__":
    main()
