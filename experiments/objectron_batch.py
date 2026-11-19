"""
Multi-object evaluation: the full pipeline on ten object-centric videos (Objectron clips).

For every video in Objectron_10_Videos/:
  1. keyframes  - sharpest frame in each of N equal time windows, resized to 960x1280
  2. SfM        - COLMAP (pycolmap, CPU, self-calibrated shared camera); the project's own
                  incremental SfM is also run once with COLMAP's focal length, for comparison
  3. 2D object  - YOLO instance segmentation; the target instance is the one whose label matches
                  the video's category when COCO has that category, otherwise the most central one
  4. 3D object  - COLMAP points whose observations fall inside the masks, 3D oriented box
  5. check      - points chosen from even keyframes are re-projected into odd keyframes
  6. recognise  - PointNet (ModelNet40) on the object points

    python -m experiments.objectron_batch
    python -m experiments.objectron_batch --only 03_cup     # one video
    python -m experiments.objectron_batch --plots-only      # rebuild summary figures
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import open3d as o3d
except Exception:
    o3d = None
import pandas as pd
import pycolmap

from experiments.common import PROJECT_ROOT, run_sfm, save_json, style_axes, C_A, C_B, C_C, C_GREY
from experiments.build_object_scene import obb_from_points, BOX_EDGES
from src.sfm.pose_validation import reliable_point_ids, validate_reconstruction
from src.classification_3d.yolo_pointnet_fusion import YOLOPointNetFusionEngine

VIDEO_DIR = PROJECT_ROOT / "Objectron_10_Videos"
OUT = PROJECT_ROOT / "outputs" / "objectron"
N_KEYFRAMES = 40
LONG_SIDE = 1280
MIN_OBS, MIN_RATIO = 3, 0.8
SEG_WEIGHTS = PROJECT_ROOT / "yolo11n-seg.pt"
SEG_CONF = 0.15

# COCO labels accepted as "the object" for each video category (None: category is not in COCO).
COCO_EQUIV = {
    "bottle": {"bottle", "vase", "cup"}, "cup": {"cup", "bowl", "vase"}, "chair": {"chair", "couch", "bench"},
    "laptop": {"laptop", "keyboard", "tv"}, "bike": {"bicycle", "motorcycle"}, "book": {"book"},
    "camera": None, "shoe": None, "cereal_box": None,
}
COCO_EXACT = {"bottle": "bottle", "cup": "cup", "chair": "chair", "laptop": "laptop", "bike": "bicycle", "book": "book"}
# ModelNet40 class that would count as a correct PointNet answer (None: no such class).
MODELNET_EQUIV = {"bottle": "bottle", "cup": "cup", "chair": "chair", "laptop": "laptop",
                  "bike": None, "book": None, "camera": None, "shoe": None, "cereal_box": None}
SUPPORT_SURFACES = {"dining table", "bed", "couch", "bench", "person"}


# ----------------------------------------------------------------------------- keyframes
def extract_keyframes(video, kf_dir, max_keyframes=N_KEYFRAMES):
    kf_dir.mkdir(parents=True, exist_ok=True)

    def resized(fr):
        h, w = fr.shape[:2]
        s = LONG_SIDE / max(h, w)
        return cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)

    # pass 1: sharpness of every frame (only the score is kept, so long videos do not fill memory)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sharp = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(fr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        sharp.append(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    cap.release()
    if len(sharp) < 2 * 8:
        raise ValueError(f"Video has only {len(sharp)} readable frames; at least 16 are needed.")
    n_kf = min(max(8, int(max_keyframes)), len(sharp) // 2)
    edges = np.linspace(0, len(sharp), n_kf + 1).astype(int)
    picked = [int(edges[k] + np.argmax(sharp[edges[k]:max(edges[k + 1], edges[k] + 1)])) for k in range(n_kf)]

    # pass 2: write the selected frames
    want = {j: k for k, j in enumerate(picked)}
    paths = [None] * n_kf
    cap = cv2.VideoCapture(str(video))
    j, w, h = 0, 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if j in want:
            fr = resized(fr)
            h, w = fr.shape[:2]
            p = kf_dir / f"kf_{want[j]:03d}_f{j:04d}.jpg"
            cv2.imwrite(str(p), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths[want[j]] = p
        j += 1
    cap.release()
    paths = [p for p in paths if p is not None]
    return paths, {"total_frames": len(sharp), "fps": round(float(fps), 1),
                   "duration_s": round(len(sharp) / max(fps, 1e-6), 1), "width": w, "height": h,
                   "keyframes": len(paths), "mean_sharpness": float(np.mean([sharp[i] for i in picked]))}


# ----------------------------------------------------------------------------- COLMAP
def run_colmap(kf_dir, work, calibration=None):
    shutil.rmtree(work, ignore_errors=True)
    (work / "sparse").mkdir(parents=True)
    db = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    # Never silently use a fake calibration.  A validated OPENCV calibration is
    # passed through to COLMAP; otherwise COLMAP self-calibrates SIMPLE_RADIAL.
    if calibration and calibration.get("status") == "calibrated":
        reader.camera_model = "OPENCV"
        reader.camera_params = ",".join(str(x) for x in calibration["colmap_params"])
    else:
        reader.camera_model = "SIMPLE_RADIAL"
    t0 = time.time()
    n_threads = min(4, os.cpu_count() or 2)
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.num_threads = n_threads
    extract_opts.max_image_size = 1600
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.num_threads = n_threads
    map_opts = pycolmap.IncrementalPipelineOptions()
    map_opts.num_threads = n_threads

    pycolmap.extract_features(db, kf_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=extract_opts, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(db, matching_options=match_opts, device=pycolmap.Device.cpu)
    maps = pycolmap.incremental_mapping(db, kf_dir, work / "sparse", options=map_opts)
    elapsed = time.time() - t0
    db.unlink(missing_ok=True)
    if not maps:
        return None, elapsed
    return max(maps.values(), key=lambda r: r.num_reg_images()), elapsed


def cam_matrix(img):
    cfw = img.cam_from_world
    cfw = cfw() if callable(cfw) else cfw
    return np.asarray(cfw.matrix())


def project(points, M, params):
    """Project for the camera models accepted by this pipeline."""
    fx, fy, cx, cy = params["fx"], params["fy"], params["cx"], params["cy"]
    k1, k2, p1, p2, k3 = (params["distortion"] + [0.0] * 5)[:5]
    pc = points @ M[:, :3].T + M[:, 3]
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u, v = pc[:, 0] / zs, pc[:, 1] / zs
    r2 = u * u + v * v
    d = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    ud = u * d + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u)
    vd = v * d + p1 * (r2 + 2.0 * v * v) + 2.0 * p2 * u * v
    return np.stack([fx * ud + cx, fy * vd + cy], axis=1), z


def camera_params(camera):
    """Normalize COLMAP camera parameters for diagnostics and reprojection."""
    p = np.asarray(camera.params, dtype=np.float64).tolist()
    model = str(camera.model).upper()
    if "SIMPLE" in model:
        f, cx, cy = p[:3]
        return {"model": model, "fx": f, "fy": f, "cx": cx, "cy": cy, "distortion": p[3:]}
    if "PINHOLE" in model:
        return {"model": model, "fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3], "distortion": []}
    if "OPENCV" in model:
        return {"model": model, "fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3], "distortion": p[4:]}
    raise ValueError(f"Unsupported COLMAP camera model for object association: {model}")


def load_calibration(path, image_size):
    """Load a real calibration, rescaled to extracted keyframes, or return None."""
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "calibrated":
        return None
    old_w, old_h = float(data["image_width"]), float(data["image_height"])
    new_w, new_h = image_size
    if old_w <= 0 or old_h <= 0:
        return None
    sx, sy = new_w / old_w, new_h / old_h
    return {
        "status": "calibrated", "source": str(Path(path).resolve()),
        "colmap_params": [
            float(data["fx"]) * sx, float(data["fy"]) * sy,
            float(data["cx"]) * sx, float(data["cy"]) * sy,
            *[float(x) for x in data.get("distortion_coefficients", [])[:5]],
        ],
        "rmse_px": float(data.get("reprojection_error_rmse", 0.0)),
    }


def in_mask(uv, z, mask):
    h, w = mask.shape
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    hit = np.zeros(len(uv), dtype=bool)
    px = np.round(uv[ok]).astype(int)
    px[:, 0] = np.clip(px[:, 0], 0, w - 1); px[:, 1] = np.clip(px[:, 1], 0, h - 1)
    hit[ok] = mask[px[:, 1], px[:, 0]]
    return hit


# ----------------------------------------------------------------------------- segmentation
def segment(paths, category, model):
    """Returns {image name: dict(mask, label, conf, mode)} for frames where a target was found."""
    if category in COCO_EQUIV:
        expected = COCO_EQUIV[category]
    elif category in set(model.names.values()):
        expected = {category}          # any YOLO class chosen by the user
    else:
        expected = None                # "auto": most central instance
    out, rows = {}, []
    for p in paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        r = model.predict(img, conf=SEG_CONF, imgsz=640, retina_masks=True, verbose=False)[0]
        best = None
        if r.masks is not None and len(r.masks) > 0:
            masks = r.masks.data.cpu().numpy() > 0.5
            labels = [model.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
            confs = r.boxes.conf.cpu().numpy()
            cands = []
            for m, lab, cf in zip(masks, labels, confs):
                area = m.sum() / (h * w)
                if area < 0.005:
                    continue
                ys, xs = np.nonzero(m)
                dist = np.hypot(xs.mean() - w / 2, ys.mean() - h / 2) / np.hypot(w, h)
                cands.append((m, lab, float(cf), float(area), float(dist)))
            if expected:
                hit = [c for c in cands if c[1] in expected and c[4] <= 0.30]
                if hit:
                    best = max(hit, key=lambda c: c[2] - c[4]) + ("category label",)
            if best is None:
                central = [c for c in cands if c[4] <= 0.20 and c[1] not in SUPPORT_SURFACES and c[3] <= 0.80]
                if central:
                    best = max(central, key=lambda c: c[2] - 2.0 * c[4]) + ("most central instance",)
        if best is not None:
            out[p.name] = {"mask": best[0], "label": best[1], "conf": best[2], "mode": best[5]}
        rows.append({"image": p.name, "detected": best is not None, "label": best[1] if best else "none",
                     "confidence": best[2] if best else 0.0, "mode": best[5] if best else "none",
                     "mask_area_pct": 100 * best[3] if best else 0.0})
    return out, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- per video
def process(video, model, engine, category=None, out_root=None, name=None, run_own=True, progress=None,
            calibration_path=None, max_keyframes=N_KEYFRAMES):
    """
    Runs the whole pipeline on one video and writes its results to <out_root>/<name>/.
    category : video category or any YOLO class name; "auto" picks the most central object.
               Defaults to the part of the file name after the first underscore.
    progress : optional callable(str) that receives a short message at each stage.
    """
    say = progress or (lambda msg: None)
    name = name or video.stem
    if category is None:
        category = name.split("_", 1)[1] if "_" in name else "auto"
    vdir = (out_root or OUT) / name
    kf_dir = vdir / "keyframes"
    shutil.rmtree(vdir, ignore_errors=True)
    t_start = time.time()
    say("Selecting keyframes")
    paths, meta = extract_keyframes(video, kf_dir, max_keyframes=max_keyframes)
    W, H = meta["width"], meta["height"]
    calibration = load_calibration(calibration_path, (W, H))
    m = {"video": name, "category": category, **meta,
         "input_video": str(Path(video).resolve()),
         "camera_calibration": calibration or {"status": "auto_estimated", "source": None}}

    # --- COLMAP
    say(f"Structure from Motion on {len(paths)} keyframes (COLMAP, the slowest step)")
    rec, t_colmap = run_colmap(kf_dir, vdir / "colmap", calibration=calibration)
    m["colmap_time_s"] = round(t_colmap, 1)
    if rec is None:
        m.update({"colmap_registered": 0, "colmap_points": 0, "status": "COLMAP produced no reconstruction"})
        save_json(m, vdir / "metrics.json")
        return m
    params = camera_params(next(iter(rec.cameras.values())))
    pose_report = validate_reconstruction(
        rec, total_images=len(paths), output_dir=vdir / "diagnostics",
        calibration_source=calibration["source"] if calibration else "COLMAP auto-estimated",
    )
    m.update({"colmap_registered": int(rec.num_reg_images()), "colmap_points": int(rec.num_points3D()),
              "colmap_reproj_mean_px": float(rec.compute_mean_reprojection_error()),
               "colmap_track_length": float(rec.compute_mean_track_length()), "colmap_focal_px": float(params["fx"]),
               "pose_validation": pose_report})

    # --- own incremental SfM, given COLMAP's focal length so only the pipeline differs
    if run_own:
        say("Structure from Motion with the project's own pipeline")
        K = np.array([[params["fx"], 0, params["cx"]], [0, params["fy"], params["cy"]], [0, 0, 1.0]])
        own = run_sfm(paths, vdir / "_own_sfm", seed=0, k_mat=K, image_size=(W, H))
        shutil.rmtree(vdir / "_own_sfm", ignore_errors=True)
        m.update({"own_registered": own["registered_cameras"], "own_points": own["points_3d"],
                  "own_reproj_median_px": own["reproj_median_px"], "own_time_s": own["runtime_s"]})

    # --- segmentation
    say("Segmenting the object in every keyframe (YOLO)")
    seg, seg_df = segment(paths, category, model)
    seg_df.to_csv(vdir / "segmentation.csv", index=False)
    det = seg_df[seg_df.detected]
    exact = COCO_EXACT.get(category, category if category in set(model.names.values()) else None)
    m.update({"frames_with_target": int(len(det)), "detection_rate_pct": 100.0 * len(det) / len(seg_df),
              "category_in_coco": exact is not None,
              "exact_label_rate_pct": float(100.0 * (det.label == exact).mean()) if exact and len(det) else np.nan,
              "label_counts": det.label.value_counts().to_dict(),
              "selection_modes": det["mode"].value_counts().to_dict(),
              "mean_confidence": float(det.confidence.mean()) if len(det) else 0.0})

    # --- 2D -> 3D association on COLMAP points
    say("Lifting the object into 3D and checking it in unseen views")
    views = {}
    for iid, img in rec.images.items():
        if img.has_pose and img.name in seg:
            views[iid] = {"name": img.name, "idx": int(img.name.split("_")[1]), "M": cam_matrix(img),
                          "mask": seg[img.name]["mask"], "img": img}
    sel = {i for i, v in views.items() if v["idx"] % 2 == 0}
    held = sorted((i for i, v in views.items() if v["idx"] % 2 == 1), key=lambda i: views[i]["idx"])
    reliable_ids = reliable_point_ids(rec)
    pids = [pid for pid in rec.points3D if pid in reliable_ids]
    m["track_filter"] = {"accepted_points": len(pids), "rejected_points": int(rec.num_points3D()) - len(pids),
                         "max_reprojection_error_px": 3.0, "min_track_length": 3}
    if not pids:
        m.update({"status": "insufficient_reliable_tracks"})
        save_json(m, vdir / "metrics.json")
        return m
    xyz = np.array([rec.points3D[p].xyz for p in pids])
    rgb = np.array([rec.points3D[p].color for p in pids], dtype=np.float64) / 255.0
    cnt = {t: np.zeros(len(pids), int) for t in ("sel_obs", "sel_in", "all_obs", "all_in")}
    for k, p in enumerate(pids):
        for el in rec.points3D[p].track.elements:
            v = views.get(el.image_id)
            if v is None:
                continue
            x, y = v["img"].points2D[el.point2D_idx].xy
            inside = v["mask"][int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))]
            cnt["all_obs"][k] += 1; cnt["all_in"][k] += inside
            if el.image_id in sel:
                cnt["sel_obs"][k] += 1; cnt["sel_in"][k] += inside

    def pick(tag, min_obs):
        keep = (cnt[f"{tag}_obs"] >= min_obs) & (cnt[f"{tag}_in"] / np.maximum(cnt[f"{tag}_obs"], 1) >= MIN_RATIO)
        if keep.sum() >= 8:  # drop gross 3D outliers caused by mask leaks onto the background
            c = np.median(xyz[keep], axis=0)
            d = np.linalg.norm(xyz - c, axis=1)
            mad = np.median(np.abs(d[keep] - np.median(d[keep]))) + 1e-9
            keep &= d <= np.median(d[keep]) + 4.0 * 1.4826 * mad
        return keep

    obj_sel, obj_all = pick("sel", 2), pick("all", MIN_OBS)
    m.update({"views_with_mask_and_pose": len(views), "object_points": int(obj_all.sum()),
              "object_points_pct_of_scene": 100.0 * obj_all.sum() / max(len(xyz), 1)})

    # --- held-out consistency
    ho = []
    if obj_sel.sum() >= 5:
        for iid in held:
            v = views[iid]
            uo, zo = project(xyz[obj_sel], v["M"], params)
            us, zs = project(xyz, v["M"], params)
            ho.append((100 * in_mask(uo, zo, v["mask"]).mean(), 100 * in_mask(us, zs, v["mask"]).mean()))
    m.update({"held_out_views": len(ho), "object_points_selection": int(obj_sel.sum()),
              "held_out_in_mask_pct": float(np.mean([a for a, _ in ho])) if ho else np.nan,
              "held_out_in_mask_min_pct": float(np.min([a for a, _ in ho])) if ho else np.nan,
              "baseline_in_mask_pct": float(np.mean([b for _, b in ho])) if ho else np.nan})

    # --- gravity-aligned frame (cameras are held roughly upright): up = -mean(camera y axis)
    Ms = [cam_matrix(i) for i in rec.images.values() if i.has_pose]
    up = -np.mean([M[1, :3] for M in Ms], axis=0); up /= np.linalg.norm(up)
    a = np.cross(up, [1.0, 0, 0]); a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([0, 0, 1.0])
    R_align = np.stack([np.cross(a, up), up, a])  # rows: x, y(up), z  (right-handed)
    xyz_a = xyz @ R_align.T
    centres_a = np.array([-(M[:, :3].T @ M[:, 3]) for M in Ms]) @ R_align.T

    # --- object cloud, box, PointNet
    obj = xyz[obj_all]
    corners = None
    if len(obj) >= 4:
        if o3d is not None:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(obj); pc.colors = o3d.utility.Vector3dVector(rgb[obj_all])
            o3d.io.write_point_cloud(str(vdir / "object_points.ply"), pc, write_ascii=True)
        else:
            try:
                import trimesh
                pc = trimesh.PointCloud(vertices=obj, colors=rgb[obj_all])
                pc.export(str(vdir / "object_points.ply"))
            except Exception:
                pass
        centre, R, extent, corners = obb_from_points(obj, up_vector=up)
        save_json({"units": "arbitrary COLMAP units", "num_points": int(len(obj)), "center": centre,
                   "rotation_matrix_columns_are_axes": R, "extent": extent, "corners": corners},
                  vdir / "object_bounding_box.json")
        m["obb_extent"] = np.sort(extent)[::-1]
        cover = []
        for v in views.values():
            uv, z = project(obj, v["M"], params)
            hit = in_mask(uv, z, v["mask"])
            ys = np.nonzero(v["mask"].any(axis=1))[0]
            if hit.sum() >= 3 and len(ys) > 1:
                cover.append((uv[hit, 1].max() - uv[hit, 1].min()) / (ys[-1] - ys[0]))
        m["mask_height_covered_pct"] = float(100 * np.median(cover)) if cover else np.nan
        np.random.seed(0)
        res = engine.classify_point_cloud(obj @ R_align.T, top_k=3).to_dict()
        # Keep each modality and the conservative final decision separate.
        # The raw PointNet result above remains in metrics for diagnosis.
        fusion = YOLOPointNetFusionEngine(pointnet_engine=engine).fuse(
            obj @ R_align.T,
            [{"class": item["label"], "confidence": item["conf"]} for item in seg.values()],
        ).to_dict()
        save_json(fusion, vdir / "classification.json")
        target = MODELNET_EQUIV.get(category, category if category in engine.classes else None)
        m.update({"pointnet_prediction": res["predicted_class"], "pointnet_confidence": res["predicted_confidence"],
                  "pointnet_top3": [(p["class_name"], p["confidence"]) for p in res["top_predictions"]],
                   "classification_decision": fusion["predicted_class"],
                   "classification_status": fusion["status"],
                   "classification_is_uncertain": fusion["is_uncertain"],
                  "category_in_modelnet": target is not None,
                  "pointnet_correct": (res["predicted_class"] == target) if target else None})
    rec.export_PLY(str(vdir / "scene_points.ply"))

    # --- overlay tile: masks, object points and the 3D box in four held-out views
    show = held if len(held) >= 4 else sorted(views, key=lambda i: views[i]["idx"])
    tiles = []
    for iid in [show[int(round(t))] for t in np.linspace(0, len(show) - 1, 4)] if show else []:
        v = views[iid]
        img = cv2.imread(str(kf_dir / v["name"]))
        cnts, _ = cv2.findContours(v["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 255, 255), 3)
        if corners is not None:
            uvc, zc = project(corners, v["M"], params)
            if np.all(zc > 0):
                for i, j in BOX_EDGES:
                    cv2.line(img, tuple(np.round(uvc[i]).astype(int)), tuple(np.round(uvc[j]).astype(int)), (214, 120, 42), 3)
            uv, z = project(obj, v["M"], params)
            for p in uv[z > 0]:
                cv2.circle(img, tuple(np.round(p).astype(int)), 6, (0, 0, 0), -1)
                cv2.circle(img, tuple(np.round(p).astype(int)), 4, (80, 220, 255), -1)
        tiles.append(cv2.resize(img, (360, 480)))
    if tiles:
        cv2.imwrite(str(vdir / "overlay.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])

    np.savez_compressed(vdir / "scene_aligned.npz", scene=xyz_a.astype(np.float32), is_object=obj_all,
                        cameras=centres_a.astype(np.float32))
    m["status"] = "ok"
    m["total_time_s"] = round(time.time() - t_start, 1)
    save_json(m, vdir / "metrics.json")
    return m


# ----------------------------------------------------------------------------- summary
def summarise():
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*/metrics.json"))]
    df = pd.DataFrame(rows)
    flat = df.drop(columns=[c for c in ("label_counts", "selection_modes", "pointnet_top3", "obb_extent") if c in df])
    flat.to_csv(OUT / "summary.csv", index=False)
    ok = df[df.status == "ok"]
    overall = {
        "videos": int(len(df)), "videos_reconstructed": int((df.colmap_registered > 0).sum()),
        "colmap_registration_rate_pct_mean": float((100 * df.colmap_registered / df.keyframes).mean()),
        "own_registration_rate_pct_mean": float((100 * ok.own_registered / ok.keyframes).mean()),
        "colmap_points_mean": float(df.colmap_points.mean()),
        "colmap_reproj_mean_px_mean": float(ok.colmap_reproj_mean_px.mean()),
        "detection_rate_pct_mean": float(ok.detection_rate_pct.mean()),
        "videos_with_object_cloud": int((ok.object_points >= 4).sum()),
        "held_out_in_mask_pct_mean": float(ok.held_out_in_mask_pct.mean()),
        "baseline_in_mask_pct_mean": float(ok.baseline_in_mask_pct.mean()),
        "pointnet_evaluable": int(ok.pointnet_correct.notna().sum()) if "pointnet_correct" in ok else 0,
        "pointnet_correct": int((ok.pointnet_correct == True).sum()) if "pointnet_correct" in ok else 0,  # noqa: E712
    }
    save_json({"overall": overall, "videos": rows}, OUT / "summary.json")

    names = [v.replace("_", " ", 1) for v in df.video]
    x = np.arange(len(df)); w = 0.38
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    ax = axs[0, 0]
    ax.bar(x - w / 2, 100 * df.colmap_registered / df.keyframes, w, color=C_B, label="COLMAP")
    ax.bar(x + w / 2, 100 * df.own_registered.fillna(0) / df.keyframes, w, color=C_A, label="Own SfM (one run)")
    ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Keyframes registered (%)")
    ax.set_title("Camera registration"); ax.legend(frameon=False, ncol=2, loc="upper left")
    ax = axs[0, 1]
    b = ax.bar(x, df.colmap_points, 0.6, color=C_B)
    for bi, n in zip(b, df.object_points.fillna(0).astype(int)):
        ax.text(bi.get_x() + bi.get_width() / 2, bi.get_height(), f"{int(bi.get_height())}\n({n} on object)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, df.colmap_points.max() * 1.22); ax.set_ylabel("COLMAP sparse points")
    ax.set_title("Scene points, and points assigned to the object")
    ax = axs[1, 0]
    ax.bar(x, df.detection_rate_pct.fillna(0), 0.6, color=[C_C if c else C_GREY for c in df.category_in_coco.fillna(False)])
    for xi, r in zip(x, df.itertuples()):
        lab = max(r.label_counts, key=r.label_counts.get) if isinstance(r.label_counts, dict) and r.label_counts else "-"
        ax.text(xi, (r.detection_rate_pct or 0) + 1.5, lab, ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Keyframes with a target mask (%)")
    ax.set_title("2D segmentation (green: category exists in COCO; text: most frequent label)")
    ax = axs[1, 1]
    ax.bar(x - w / 2, df.held_out_in_mask_pct.fillna(0), w, color=C_A, label="Object points, held-out views")
    ax.bar(x + w / 2, df.baseline_in_mask_pct.fillna(0), w, color=C_GREY, label="All scene points (baseline)")
    for xi, v in zip(x, df.held_out_in_mask_pct):
        ax.text(xi - w / 2, (v if np.isfinite(v) else 0) + 1.5, f"{v:.0f}" if np.isfinite(v) else "n/a", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Points projecting inside the mask (%)")
    ax.set_title("Object localisation consistency"); ax.legend(frameon=False, ncol=2, loc="upper left")
    for ax in axs.ravel():
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); style_axes(ax)
    plt.suptitle("Pipeline results on ten object videos", fontsize=14)
    plt.tight_layout(); plt.savefig(OUT / "summary_metrics.png", dpi=150); plt.close()

    # top-down scenes
    fig, axs = plt.subplots(2, 5, figsize=(20, 8.4))
    for ax, r in zip(axs.ravel(), df.itertuples()):
        f = OUT / r.video / "scene_aligned.npz"
        ax.set_title(f"{r.video.replace('_', ' ', 1)}", fontsize=11)
        if not f.exists():
            ax.text(0.5, 0.5, "no reconstruction", ha="center", transform=ax.transAxes); ax.axis("off"); continue
        d = np.load(f)
        sc, io, cams = d["scene"], d["is_object"], d["cameras"]
        lo, hi = np.percentile(sc, 2, axis=0), np.percentile(sc, 98, axis=0)
        keep = np.all((sc >= lo) & (sc <= hi), axis=1) & ~io
        ax.scatter(sc[keep, 0], sc[keep, 2], s=1.2, c=C_GREY, alpha=0.5)
        ax.scatter(sc[io, 0], sc[io, 2], s=7, c=C_C)
        ax.plot(cams[:, 0], cams[:, 2], "o", color=C_B, ms=2.5)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{r.colmap_registered}/{r.keyframes} cameras, {r.colmap_points} points, "
                      f"{int(r.object_points)} on object", fontsize=9)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C_GREY, label="Scene points"),
                        plt.Line2D([], [], marker="o", ls="", color=C_C, label="Object points"),
                        plt.Line2D([], [], marker="o", ls="", color=C_B, label="Cameras")],
               loc="lower center", ncol=3, frameon=False)
    plt.suptitle("COLMAP reconstructions in an approximate top view, with the object separated", fontsize=13)
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUT / "scenes_topdown.png", dpi=140); plt.close()

    # gallery of overlays
    tiles = []
    for r in df.itertuples():
        f = OUT / r.video / "overlay.jpg"
        img = cv2.imread(str(f)) if f.exists() else np.full((480, 1440, 3), 40, np.uint8)
        for col, th in (((0, 0, 0), 7), ((255, 255, 255), 3)):
            cv2.putText(img, r.video.replace("_", " ", 1), (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, th)
        tiles.append(img)
    rows_img = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(OUT / "overlay_gallery.jpg"), cv2.resize(np.vstack(rows_img), None, fx=0.75, fy=0.75),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return df, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="process one video (stem, e.g. 03_cup)")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.plots_only:
        from ultralytics import YOLO
        from src.classification_3d.pointnet_classifier import PointNetInferenceEngine
        model = YOLO(str(SEG_WEIGHTS))
        engine = PointNetInferenceEngine(device="cpu")
        videos = sorted(VIDEO_DIR.glob("*.MOV")) + sorted(VIDEO_DIR.glob("*.mp4"))
        if args.only:
            videos = [v for v in videos if v.stem == args.only]
        for v in videos:
            try:
                m = process(v, model, engine)
            except Exception as e:  # keep the batch going; the failure is recorded
                import traceback
                traceback.print_exc()
                m = {"video": v.stem, "category": v.stem.split("_", 1)[1], "status": f"failed: {type(e).__name__}: {e}"}
                (OUT / v.stem).mkdir(parents=True, exist_ok=True)
                save_json(m, OUT / v.stem / "metrics.json")
            print(f"[INFO] {v.stem}: " + ", ".join(f"{k}={m.get(k)}" for k in (
                "status", "colmap_registered", "colmap_points", "own_registered", "detection_rate_pct",
                "object_points", "held_out_in_mask_pct", "baseline_in_mask_pct", "pointnet_prediction",
                "total_time_s")), flush=True)

    df, overall = summarise()
    print(json.dumps(overall, indent=1))


if __name__ == "__main__":
    main()
