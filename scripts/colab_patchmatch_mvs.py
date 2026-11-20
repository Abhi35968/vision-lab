"""
COLMAP Dense PatchMatch Multi-View Stereo (MVS) Pipeline for Google Colab (T4 GPU).
Runs GPU-accelerated PatchMatch stereo and stereo fusion on Stage 02 SfM camera poses.
"""

import argparse
import json
import logging
from pathlib import Path
import time
import cv2
import numpy as np
import open3d as o3d
import pycolmap
import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("COLMAP_PatchMatch_MVS")

def run_mvs_pipeline(
    workspace_dir: Path,
    sparse_dir: Path,
    images_dir: Path,
    max_image_size: int = 1000,
    min_num_consistent: int = 3,
):
    t0 = time.time()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    mvs_dir = workspace_dir / "mvs"
    mvs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Undistort Images
    logger.info("Step 1: Running Image Undistortion...")
    undistort_opts = pycolmap.UndistortCameraOptions()
    undistort_opts.max_image_size = max_image_size

    try:
        pycolmap.undistort_images(
            output_path=str(mvs_dir),
            input_path=str(sparse_dir),
            image_path=str(images_dir),
            output_type="COLMAP",
            undistort_options=undistort_opts,
        )
    except TypeError:
        # Fallback to positional arguments
        try:
            pycolmap.undistort_images(
                str(mvs_dir),
                str(sparse_dir),
                str(images_dir),
                [],
                "COLMAP",
                pycolmap.FileCopyType.copy,
                -1,
                undistort_opts,
            )
        except Exception as e:
            logger.warning(f"pycolmap.undistort_images failed: {e}. Trying CLI fallback...")
            import subprocess
            subprocess.run([
                "colmap", "image_undistorter",
                "--image_path", str(images_dir),
                "--input_path", str(sparse_dir),
                "--output_path", str(mvs_dir),
                "--output_type", "COLMAP",
                "--max_image_size", str(max_image_size),
            ], check=True)
    logger.info("Image undistortion complete.")

    # 2. PatchMatch Stereo (CUDA accelerated)
    stereo_dm_dir = mvs_dir / "stereo" / "depth_maps"
    if stereo_dm_dir.exists() and len(list(stereo_dm_dir.glob("*.bin"))) >= 28:
        logger.info(f"PatchMatch Stereo depth maps already exist ({len(list(stereo_dm_dir.glob('*.bin')))} files). Skipping re-computation.")
    else:
        logger.info("Step 2: Running Dense PatchMatch Stereo (CUDA)...")
        pm_opts = pycolmap.PatchMatchOptions()
        pm_opts.max_image_size = max_image_size
        pm_opts.window_radius = 5
        pm_opts.window_step = 1
        pm_opts.num_samples = 15
        pm_opts.num_iterations = 5
        pm_opts.geom_consistency = True

        pycolmap.patch_match_stereo(
            workspace_path=str(mvs_dir),
            options=pm_opts,
        )
        logger.info("PatchMatch Stereo complete.")

    # 3. Stereo Fusion
    logger.info("Step 3: Running Stereo Multi-View Fusion...")
    dense_fused_ply = mvs_dir / "fused.ply"
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.min_num_pixels = min_num_consistent # View consistency check across views

    rec = pycolmap.stereo_fusion(
        output_path=str(dense_fused_ply),
        workspace_path=str(mvs_dir),
        output_type="PLY",
        options=fusion_opts,
    )
    if not dense_fused_ply.exists() and rec is not None:
        rec.export_PLY(str(dense_fused_ply))
    logger.info(f"Stereo fusion complete. Saved to {dense_fused_ply}")

    # 4. Load Dense Point Cloud & Diagnostics
    pcd = o3d.io.read_point_cloud(str(dense_fused_ply))
    pts = np.asarray(pcd.points)
    logger.info(f"Dense MVS Points Extracted: {len(pts)}")

    # 5. Poisson Surface Reconstruction from Dense MVS Cloud
    logger.info("Step 4: Generating Poisson Surface Mesh from MVS Cloud...")
    dense_mesh_ply = mvs_dir / "mvs_poisson_mesh.ply"
    if len(pts) > 0:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        dens_np = np.asarray(densities)
        if len(dens_np) > 0:
            thresh = np.percentile(dens_np, 5)
            mesh.remove_vertices_by_mask(dens_np < thresh)
        o3d.io.write_triangle_mesh(str(dense_mesh_ply), mesh)
    else:
        logger.warning("No dense points available for Poisson meshing.")
        mesh = o3d.geometry.TriangleMesh()
        o3d.io.write_triangle_mesh(str(dense_mesh_ply), mesh)

    elapsed = round(time.time() - t0, 2)
    summary = {
        "status": "success",
        "elapsed_seconds": elapsed,
        "dense_points_count": len(pts),
        "mesh_vertices_count": len(mesh.vertices),
        "mesh_triangles_count": len(mesh.triangles),
        "dense_ply_path": str(dense_fused_ply),
        "dense_mesh_ply": str(dense_mesh_ply),
    }

    with open(workspace_dir / "mvs_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"MVS Pipeline Finished in {elapsed}s: {json.dumps(summary, indent=2)}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COLMAP PatchMatch MVS Runner")
    parser.add_argument("--workspace", type=str, default="colmap_mvs_workspace", help="Workspace dir")
    parser.add_argument("--sparse", type=str, required=True, help="Path to COLMAP sparse directory")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    args = parser.parse_args()

    run_mvs_pipeline(Path(args.workspace), Path(args.sparse), Path(args.images))
