"""
Dense Monocular Depth Fusion & Multimodal Consensus Runner for Objectron & Video Runs.
Transforms sparse SfM point clouds into dense, gap-free 3D point clouds and Poisson meshes,
and applies multimodal Bayesian consensus to eliminate PointNet false classifications.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
try:
    import open3d as o3d
except (ImportError, Exception):
    o3d = None
import pycolmap
from ultralytics import YOLO

from src.dense_depth.depth_anything_estimator import DepthAnythingEstimator
from src.dense_depth.multi_view_dense_fuser import MultiViewDenseFuser, DenseFusionConfig
from src.classification_3d.pointnet_classifier import PointNetInferenceEngine
from src.classification_3d.yolo_pointnet_fusion import YOLOPointNetFusionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DenseReconstructRunner")


def reconstruct_dense(target_dir: Path, max_views: int = 12):
    t0 = time.time()
    target_dir = Path(target_dir).resolve()
    logger.info(f"Starting dense reconstruction on: {target_dir}")

    colmap_sparse_dir = target_dir / "colmap" / "sparse" / "0"
    if not colmap_sparse_dir.exists():
        logger.error(f"COLMAP sparse directory not found at {colmap_sparse_dir}")
        return False

    kf_dir = target_dir / "keyframes"
    if not kf_dir.exists():
        logger.error(f"Keyframes directory not found at {kf_dir}")
        return False

    # 1. Load COLMAP reconstruction
    rec = pycolmap.Reconstruction(str(colmap_sparse_dir))
    cam = next(iter(rec.cameras.values()))
    f, cx, cy = float(cam.params[0]), float(cam.params[1]), float(cam.params[2])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    logger.info(f"Loaded COLMAP camera: f={f:.1f}, cx={cx:.1f}, cy={cy:.1f}, images={len(rec.images)}")

    # Load sparse points
    scene_ply = target_dir / "scene_points.ply"
    if scene_ply.exists():
        pcd_sparse = o3d.io.read_point_cloud(str(scene_ply))
        sparse_pts_ref = np.asarray(pcd_sparse.points)
    else:
        sparse_pts_ref = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)

    # 2. Select evenly distributed views
    reg_images = [img for img in rec.images.values() if img.has_pose and (kf_dir / img.name).exists()]
    reg_images = sorted(reg_images, key=lambda x: x.name)
    if len(reg_images) > max_views:
        indices = np.linspace(0, len(reg_images) - 1, max_views).astype(int)
        selected_images = [reg_images[i] for i in indices]
    else:
        selected_images = reg_images

    logger.info(f"Selected {len(selected_images)} camera views for multi-view dense fusion.")

    # 3. Load YOLO model for instance segmentation masks
    yolo_model_path = Path("yolo11n-seg.pt")
    if not yolo_model_path.exists():
        yolo_model_path = Path("yolov8n-seg.pt")
    logger.info(f"Loading YOLO segmentation model from {yolo_model_path}...")
    yolo = YOLO(str(yolo_model_path))

    # Read category from metrics.json if available
    metrics_path = target_dir / "metrics.json"
    target_category = "bottle"
    if metrics_path.exists():
        try:
            m_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            target_category = m_data.get("category", "bottle")
        except Exception:
            pass

    views_data = []
    for img_info in selected_images:
        img_path = kf_dir / img_info.name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # Extract camera pose
        cfw = img_info.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        M = np.asarray(cfw.matrix())
        R = M[:, :3]
        t = M[:, 3]

        # Predict YOLO mask
        pred = yolo.predict(img_bgr, conf=0.15, imgsz=640, retina_masks=True, verbose=False)[0]
        mask = None
        if pred.masks is not None and len(pred.masks) > 0:
            masks = pred.masks.data.cpu().numpy() > 0.5
            labels = [yolo.names[int(c)] for c in pred.boxes.cls.cpu().numpy()]
            confs = pred.boxes.conf.cpu().numpy()
            
            # Match target category
            matched = [
                (m, lab, cf) for m, lab, cf in zip(masks, labels, confs)
                if lab.lower() == target_category.lower() or target_category.lower() in lab.lower()
            ]
            if matched:
                best_m = max(matched, key=lambda x: x[2])[0]
                mask = (best_m.astype(np.uint8) * 255)
            else:
                # Fallback to largest central mask
                best_m = max(masks, key=lambda m: m.sum())
                mask = (best_m.astype(np.uint8) * 255)

        if mask is None and ("cereal" in target_category.lower() or "box" in target_category.lower()):
            # Fallback for out-of-vocabulary cereal box: HSV yellow detector
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            m = cv2.inRange(hsv, np.array([14, 60, 60]), np.array([34, 255, 255]))
            m[:int(h * 0.35), :] = 0
            m[int(h * 0.85):, :] = 0
            m[:, :int(w * 0.25)] = 0
            m[:, int(w * 0.75):] = 0
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:
                    best_m = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(best_m, [c], -1, 255, -1)
                    mask = best_m

        if mask is not None and mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        views_data.append({
            "name": img_info.name.split(".")[0],
            "image_bgr": img_bgr,
            "mask": mask,
            "R": R,
            "t": t,
            "K": K,
        })

    # 4. Run Multi-View Dense Depth Fusion
    fuser_config = DenseFusionConfig(
        stride=3,
        max_points_per_view=12000,
        voxel_size=0.015,
        enable_sor=True,
        sor_nb_neighbors=25,
        sor_std_ratio=1.5,
        enable_plane_removal=True,
        plane_dist_thresh=0.03,
        enable_mesh_generation=True,
        poisson_depth=8,
        output_dir=str(target_dir),
    )
    cache_d = target_dir / "depth_cache"
    cache_d.mkdir(parents=True, exist_ok=True)
    estimator = DepthAnythingEstimator(cache_dir=str(cache_d))
    fuser = MultiViewDenseFuser(estimator=estimator, config=fuser_config)

    logger.info("Executing dense fusion across selected views...")
    res = fuser.fuse_views(views_data, sparse_pts_ref, tag="dense_object")

    # 5. Run Multimodal Bayesian Consensus
    seg_csv = target_dir / "segmentation.csv"
    pn_engine = PointNetInferenceEngine(device="cpu")
    fusion_engine = YOLOPointNetFusionEngine(pointnet_engine=pn_engine)
    clean_pcd = o3d.io.read_point_cloud(res.clean_ply_path)
    clean_pts = np.asarray(clean_pcd.points)

    logger.info("Running Multimodal Bayesian Decision Fusion...")
    fusion_res = fusion_engine.fuse(clean_pts, seg_csv)

    # Save consensus result
    consensus_path = target_dir / "classification_consensus.json"
    with open(consensus_path, "w", encoding="utf-8") as f:
        json.dump(fusion_res.to_dict(), f, indent=2)
    logger.info(f"Saved consensus classification to {consensus_path}")

    # Also update metrics.json with dense stats and consensus
    if metrics_path.exists():
        try:
            m_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            m_data["dense_points"] = res.clean_points_count
            m_data["dense_multiplier"] = res.density_multiplier
            m_data["mesh_triangles"] = res.mesh_triangles_count
            m_data["mesh_vertices"] = res.mesh_vertices_count
            top_cand = fusion_res.top_candidates[0] if fusion_res.top_candidates else None
            m_data["consensus_prediction"] = top_cand.class_name if top_cand else "unknown"
            m_data["consensus_confidence"] = float(top_cand.fused_confidence) if top_cand else 0.0
            m_data["consensus_status"] = fusion_res.status
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(m_data, f, indent=2)
            logger.info(f"Updated {metrics_path} with dense reconstruction metrics.")
        except Exception as e:
            logger.warning(f"Could not update metrics.json: {e}")

    logger.info(f"Dense reconstruction completed in {time.time() - t0:.1f} s.")
    logger.info(f"Clean dense points: {res.clean_points_count:,} | Poisson triangles: {res.mesh_triangles_count:,}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Object Reconstructor")
    parser.add_argument("--dir", type=str, default="outputs/objectron/01_bottle", help="Directory of reconstruction")
    parser.add_argument("--max-views", type=int, default=10, help="Max views to fuse")
    args = parser.parse_args()
    reconstruct_dense(Path(args.dir), max_views=args.max_views)
