"""
Open3D TSDF Volume Integration on VID_20261102_015654418.mp4 Registered Keyframes.
Generates continuous, watertight 3D surface mesh and point cloud using volumetric signed distance fields.
"""

import json
import logging
import os
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import trimesh
from ultralytics import YOLO

from src.dense_depth.depth_anything_estimator import DepthAnythingEstimator

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("TSDF_Integration")

VALIDATION_DIR = Path("outputs/deployment_live_validation")
KEYFRAMES_DIR = VALIDATION_DIR / "keyframes"
COLMAP_SPARSE_DIR = VALIDATION_DIR / "colmap/sparse"
TSDF_OUT_DIR = VALIDATION_DIR / "tsdf_fusion"

def get_best_reconstruction(sparse_root: Path) -> pycolmap.Reconstruction:
    best_rec = None
    max_images = -1
    for p in sparse_root.iterdir():
        if p.is_dir() and (p / "cameras.bin").exists():
            try:
                rec = pycolmap.Reconstruction(str(p))
                if len(rec.images) > max_images:
                    max_images = len(rec.images)
                    best_rec = rec
            except Exception as e:
                logger.warning(f"Error loading {p}: {e}")
    if best_rec is None:
        raise RuntimeError(f"No valid COLMAP sparse reconstruction found in {sparse_root}")
    return best_rec

def run_tsdf_reconstruction():
    TSDF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("Loading registered COLMAP cameras...")
    rec = get_best_reconstruction(COLMAP_SPARSE_DIR)
    logger.info(f"Loaded {len(rec.images)} registered views.")

    # 1. Load Depth Estimator and YOLO Segmenter
    logger.info("Initializing Depth Anything Estimator and YOLO segmenter...")
    depth_estimator = DepthAnythingEstimator()
    yolo_model = YOLO("yolo11n-seg.pt")

    # 2. Extract reliable sparse points to anchor depth
    pts_3d = []
    for pt in rec.points3D.values():
        pts_3d.append(pt.xyz)
    sparse_pts = np.asarray(pts_3d, dtype=np.float64)
    logger.info(f"Loaded {len(sparse_pts)} sparse reference points.")

    # 3. Collect views and align depth maps
    aligned_views = []
    registered_images = sorted(list(rec.images.values()), key=lambda x: x.name)
    
    # Process up to 28 views
    for idx, col_img in enumerate(registered_images):
        img_path = KEYFRAMES_DIR / col_img.name
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        cam = rec.cameras[col_img.camera_id]
        K = cam.calibration_matrix()
        rigid = col_img.cam_from_world()
        R = rigid.rotation.matrix()
        t = rigid.translation

        # YOLO instance mask
        yolo_res = yolo_model(img_bgr, verbose=False)[0]
        mask = np.zeros((h, w), dtype=np.uint8)
        if yolo_res.masks is not None and len(yolo_res.masks.data) > 0:
            m_np = yolo_res.masks.data[0].cpu().numpy()
            m_resized = cv2.resize((m_np * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            # Gentle erosion
            k_size = 5 if min(h, w) > 400 else 3
            mask = cv2.erode(m_resized, np.ones((k_size, k_size), np.uint8), iterations=1)

        # Monocular depth
        d_rel, _ = depth_estimator.estimate_relative_depth(img_bgr, cache_key=f"val_{col_img.name}")
        
        # Align with SfM
        s, b, rmse, aligned_depth = depth_estimator.align_depth_with_sfm(
            d_rel, sparse_pts, R, t, K, mode="inverse"
        )

        aligned_views.append({
            "name": col_img.name,
            "bgr": img_bgr,
            "mask": mask,
            "R": R,
            "t": t,
            "K": K,
            "aligned_z": aligned_depth,
            "h": h,
            "w": w,
        })
        logger.info(f"[{idx+1}/{len(registered_images)}] Aligned view {col_img.name} (s={s:.4f}, b={b:.4f}, rmse={rmse:.3f})")

    # 4. Compute Volume Bounding Box & Origin
    logger.info("Computing optimal TSDF Volume bounding box...")
    # Sample unprojected points from views to get true bounds
    sample_pts = []
    for v in aligned_views[::2]:
        z_map = v["aligned_z"]
        m_map = v["mask"]
        pts_view = depth_estimator.unproject_dense_cloud(
            z_map, v["R"], v["t"], v["K"], object_mask=m_map, stride=8, max_points=3000, max_depth_gradient=0.08
        )
        if len(pts_view) > 0:
            sample_pts.append(pts_view)

    if not sample_pts:
        raise RuntimeError("No sample points unprojected; check depth alignment.")

    all_samp = np.vstack(sample_pts)
    p_min = np.percentile(all_samp, 1, axis=0)
    p_max = np.percentile(all_samp, 99, axis=0)
    span = p_max - p_min
    logger.info(f"Object bounds: min={p_min.round(3)}, max={p_max.round(3)}, span={span.round(3)}")

    pad = 0.5
    vol_origin = p_min - pad
    vol_length = float(np.max(span) + 2 * pad)
    logger.info(f"Configuring UniformTSDFVolume: length={vol_length:.2f}, origin={vol_origin.round(3)}")

    # 5. Initialize UniformTSDFVolume
    # Resolution 256 gives fine sub-centimeter voxels
    resolution = 256
    voxel_size = vol_length / resolution
    sdf_trunc = max(0.04, 3.0 * voxel_size)
    logger.info(f"Voxel size: {voxel_size:.4f} units, SDF Truncation: {sdf_trunc:.4f} units")

    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=vol_length,
        resolution=resolution,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=vol_origin.reshape(3, 1),
    )

    # 6. Integrate all views
    logger.info(f"Integrating {len(aligned_views)} views into TSDF Volume...")
    for i, v in enumerate(aligned_views):
        rgb = cv2.cvtColor(v["bgr"], cv2.COLOR_BGR2RGB)
        z = v["aligned_z"].copy()
        m = v["mask"]

        # Depth gradient filtering to eliminate boundary edge bleed
        z_smooth = cv2.GaussianBlur(z.astype(np.float32), (3, 3), 0)
        gx = cv2.Sobel(z_smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(z_smooth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        rel_grad = grad_mag / np.maximum(z_smooth, 1e-4)
        
        # Zero out invalid / non-object pixels (Open3D TSDF ignores depth <= 0)
        valid = (m > 127) & (rel_grad <= 0.08) & (z > 0.1) & (z < 30.0)
        z[~valid] = 0.0

        o3d_rgb = o3d.geometry.Image(rgb)
        o3d_depth = o3d.geometry.Image(z.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_depth, depth_scale=1.0, depth_trunc=vol_length + 1.0, convert_rgb_to_intensity=False
        )

        intr = o3d.camera.PinholeCameraIntrinsic(v["w"], v["h"], v["K"][0, 0], v["K"][1, 1], v["K"][0, 2], v["K"][1, 2])
        T_cw = np.eye(4)
        T_cw[:3, :3] = v["R"]
        T_cw[:3, 3] = v["t"]

        volume.integrate(rgbd, intr, T_cw)
        logger.info(f"Integrated view {i+1}/{len(aligned_views)} ({v['name']})")

    # 7. Extract Marching Cubes Mesh and Point Cloud
    logger.info("Extracting zero-crossing surface mesh via Marching Cubes...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    pcd = volume.extract_point_cloud()
    logger.info(f"TSDF Extraction: Mesh Vertices={len(mesh.vertices)}, Triangles={len(mesh.triangles)}; PCD Points={len(pcd.points)}")

    # Clean isolated clusters if any
    if len(mesh.triangles) > 0:
        # Keep largest component
        triangle_clusters, num_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        num_triangles = np.asarray(num_triangles)
        if len(num_triangles) > 0:
            largest_cluster_idx = num_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()

    # Save assets
    tsdf_mesh_ply = TSDF_OUT_DIR / "tsdf_volume_mesh.ply"
    tsdf_mesh_obj = TSDF_OUT_DIR / "tsdf_volume_mesh.obj"
    tsdf_pcd_ply = TSDF_OUT_DIR / "tsdf_volume_point_cloud.ply"

    o3d.io.write_triangle_mesh(str(tsdf_mesh_ply), mesh)
    o3d.io.write_point_cloud(str(tsdf_pcd_ply), pcd)

    # Export OBJ
    try:
        m_tri = trimesh.load(str(tsdf_mesh_ply), process=False)
        m_tri.export(str(tsdf_mesh_obj))
    except Exception as e:
        logger.warning(f"Trimesh OBJ export warning: {e}")

    # Compute diagnostics
    pts_np = np.asarray(pcd.points)
    if len(pts_np) > 0:
        span_final = pts_np.max(axis=0) - pts_np.min(axis=0)
        aspect_ratio = span_final[1] / ((span_final[0] + span_final[2]) / 2.0)
        circ = span_final[2] / span_final[0]
    else:
        aspect_ratio, circ = 0, 0

    elapsed = round(time.time() - t0, 2)
    metrics = {
        "status": "success",
        "elapsed_seconds": elapsed,
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "point_cloud_points": len(pcd.points),
        "aspect_ratio": round(float(aspect_ratio), 4),
        "circularity": round(float(circ), 4),
        "tsdf_mesh_ply": str(tsdf_mesh_ply),
        "tsdf_mesh_obj": str(tsdf_mesh_obj),
        "tsdf_pcd_ply": str(tsdf_pcd_ply)
    }

    with open(TSDF_OUT_DIR / "tsdf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"TSDF Fusion completed in {elapsed}s! Results: {json.dumps(metrics, indent=2)}")
    return metrics

if __name__ == "__main__":
    run_tsdf_reconstruction()
