"""
Experiment 6 - Object-level 3D localisation on the COLMAP reconstruction.

Uses the YOLO segmentation masks to pick out the COLMAP 3D points that belong to the object,
builds the object point cloud and its 3D bounding box, and measures how consistently those
points land on the object in views that were NOT used to select them.

  Selection views : even keyframe indices with a valid mask
  Held-out views  : odd keyframe indices with a valid mask

A point is labelled "object" if it is observed in at least MIN_OBS selection views and at
least MIN_RATIO of those observations fall inside the mask.

    python -m experiments.exp6_colmap_object      (run exp5_colmap_baseline first)
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import pycolmap

from experiments.common import (
    PROJECT_ROOT, SEG_DIR, KEYFRAMES_DIR, EXPERIMENTS_OUT, IMAGE_W, IMAGE_H,
    save_json, style_axes, C_A, C_B, C_C, C_GREY,
)
from experiments.build_object_scene import obb_from_points, aabb_corners, BOX_EDGES

COLMAP_DIR = PROJECT_ROOT / "outputs" / "colmap"
OUT = COLMAP_DIR / "object"
MIN_OBS, MIN_RATIO = 2, 0.8


def cam_matrix(img):
    cfw = img.cam_from_world
    cfw = cfw() if callable(cfw) else cfw
    return np.asarray(cfw.matrix())  # 3x4 world -> camera


def project(points, M, params):
    f, cx, cy, k = params
    pc = points @ M[:, :3].T + M[:, 3]
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u, v = pc[:, 0] / zs, pc[:, 1] / zs
    d = 1.0 + k * (u * u + v * v)
    return np.stack([f * d * u + cx, f * d * v + cy], axis=1), z


def in_mask(uv, z, mask):
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_H)
    hit = np.zeros(len(uv), dtype=bool)
    px = np.round(uv[ok]).astype(int)
    px[:, 0] = np.clip(px[:, 0], 0, IMAGE_W - 1); px[:, 1] = np.clip(px[:, 1], 0, IMAGE_H - 1)
    hit[ok] = mask[px[:, 1], px[:, 0]] > 127
    return hit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(COLMAP_DIR / "sparse" / "0"))
    params = np.asarray(next(iter(rec.cameras.values())).params, dtype=np.float64)
    seg = pd.read_csv(SEG_DIR / "segmentation_summary.csv").set_index("image_name")

    views = {}  # image_id -> dict
    for iid, img in rec.images.items():
        if not img.has_pose or img.name not in seg.index or seg.loc[img.name, "target_detected"] != 1:
            continue
        mask = cv2.imread(str(PROJECT_ROOT / seg.loc[img.name, "mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        views[iid] = {"name": img.name, "frame": int(seg.loc[img.name, "frame_idx"]), "M": cam_matrix(img),
                      "mask": mask, "img": img}
    sel = {i for i, v in views.items() if v["frame"] % 2 == 0}
    held = {i for i, v in views.items() if v["frame"] % 2 == 1}

    pids = list(rec.points3D.keys())
    xyz = np.array([rec.points3D[p].xyz for p in pids])
    n_obs = {"sel": np.zeros(len(pids), int), "all": np.zeros(len(pids), int)}
    n_in = {"sel": np.zeros(len(pids), int), "all": np.zeros(len(pids), int)}
    for k, p in enumerate(pids):
        for el in rec.points3D[p].track.elements:
            if el.image_id not in views:
                continue
            v = views[el.image_id]
            x, y = v["img"].points2D[el.point2D_idx].xy
            xi, yi = int(np.clip(round(x), 0, IMAGE_W - 1)), int(np.clip(round(y), 0, IMAGE_H - 1))
            inside = v["mask"][yi, xi] > 127
            n_obs["all"][k] += 1; n_in["all"][k] += inside
            if el.image_id in sel:
                n_obs["sel"][k] += 1; n_in["sel"][k] += inside

    def pick(tag):
        ratio = n_in[tag] / np.maximum(n_obs[tag], 1)
        m = (n_obs[tag] >= MIN_OBS) & (ratio >= MIN_RATIO)
        # drop gross 3D outliers (mask leaks onto background): robust distance to the median
        if m.sum() >= 5:
            c = np.median(xyz[m], axis=0)
            d = np.linalg.norm(xyz - c, axis=1)
            mad = np.median(np.abs(d[m] - np.median(d[m]))) + 1e-9
            m &= d <= np.median(d[m]) + 4.0 * 1.4826 * mad
        return m

    obj_sel = pick("sel")   # used for the held-out consistency test
    obj_all = pick("all")   # final object cloud (all masked views)

    # --- held-out consistency
    rows = []
    for iid in sorted(held, key=lambda i: views[i]["frame"]):
        v = views[iid]
        uv_o, z_o = project(xyz[obj_sel], v["M"], params)
        uv_s, z_s = project(xyz, v["M"], params)
        rows.append({"frame": v["frame"],
                     "object_points_in_mask_pct": 100 * in_mask(uv_o, z_o, v["mask"]).mean(),
                     "scene_points_in_mask_pct": 100 * in_mask(uv_s, z_s, v["mask"]).mean(),
                     "mask_area_pct": float(seg.loc[v["name"], "mask_area_pct"])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "held_out_localization.csv", index=False)

    # --- object cloud + boxes
    obj = xyz[obj_all]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(obj)
    cols = np.array([rec.points3D[p].color for p, keep in zip(pids, obj_all) if keep], dtype=np.float64) / 255.0
    pc.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(OUT / "object_points_colmap.ply"), pc, write_ascii=True)
    centre, R, extent, corners = obb_from_points(obj)
    mn, mx = obj.min(axis=0), obj.max(axis=0)
    save_json({"units": "arbitrary COLMAP units", "num_points": int(len(obj)),
               "axis_aligned": {"min": mn, "max": mx, "extent": mx - mn, "corners": aabb_corners(mn, mx)},
               "oriented": {"center": centre, "rotation_matrix_columns_are_axes": R, "extent": extent,
                            "corners": corners}}, OUT / "object_bounding_box_colmap.json")

    # how much of the object's silhouette height do the points cover, per view?
    cover = []
    for iid, v in views.items():
        uv, z = project(obj, v["M"], params)
        hit = in_mask(uv, z, v["mask"])
        if hit.sum() >= 3:
            r = seg.loc[v["name"]]
            cover.append((uv[hit, 1].max() - uv[hit, 1].min()) / (r.bbox_y2 - r.bbox_y1))
    height_cover = float(np.median(cover)) if cover else 0.0

    own = EXPERIMENTS_OUT / "exp3_object_localization" / "per_view_localization.csv"
    own_df = pd.read_csv(own) if own.exists() else None
    summary = {
        "colmap_points_total": int(len(xyz)), "views_with_mask": len(views),
        "selection_views": len(sel), "held_out_views": len(held),
        "selection_rule": f">= {MIN_OBS} masked observations and >= {MIN_RATIO:.0%} inside the mask",
        "object_points_from_selection_views": int(obj_sel.sum()),
        "object_points_final_all_views": int(obj_all.sum()),
        "held_out_object_points_in_mask_pct": {"mean": res.object_points_in_mask_pct.mean(),
                                               "median": res.object_points_in_mask_pct.median(),
                                               "min": res.object_points_in_mask_pct.min()},
        "held_out_scene_points_in_mask_pct_baseline_mean": res.scene_points_in_mask_pct.mean(),
        "object_obb_extent": extent, "median_share_of_bbox_height_covered_by_points": height_cover,
        "own_sfm_comparison": None if own_df is None else {
            "object_points": 29, "mean_in_mask_pct_all_views": own_df.object_points_in_mask_pct.mean()},
    }
    save_json(summary, OUT / "summary.json")

    # --- figure 1: consistency curves
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axs[0]
    ax.plot(res.frame, res.object_points_in_mask_pct, "-o", color=C_A, lw=2, ms=5,
            label=f"COLMAP object points, held-out views (mean {res.object_points_in_mask_pct.mean():.0f}%)")
    if own_df is not None:
        ax.plot(own_df.camera_id, own_df.object_points_in_mask_pct, "-^", color=C_B, lw=2, ms=5,
                label=f"Own SfM object points (mean {own_df.object_points_in_mask_pct.mean():.0f}%)")
    ax.plot(res.frame, res.scene_points_in_mask_pct, "-s", color=C_GREY, lw=1.5, ms=4,
            label=f"All COLMAP scene points, baseline (mean {res.scene_points_in_mask_pct.mean():.0f}%)")
    ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Keyframe index"); ax.set_ylabel("Points projecting inside the object mask (%)")
    ax.set_title("Object localisation consistency across frames"); ax.legend(frameon=False, loc="upper left", fontsize=9)
    style_axes(ax)

    ax = axs[1]
    centres = np.array([v["img"].projection_center() for v in sorted(
        (dict(img=i) for i in rec.images.values() if i.has_pose), key=lambda d: d["img"].name)])
    lo, hi = np.percentile(xyz, 3, axis=0), np.percentile(xyz, 97, axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1) & ~obj_all
    ax.scatter(xyz[keep, 0], xyz[keep, 2], s=1.5, c=C_GREY, alpha=0.45, label=f"Scene points ({len(xyz) - obj_all.sum()})")
    ax.scatter(obj[:, 0], obj[:, 2], s=9, c=C_C, label=f"Object points ({len(obj)})")
    ax.plot(centres[:, 0], centres[:, 2], "-o", color=C_B, ms=3, lw=1.2, label=f"Cameras ({len(centres)})")
    for k, (i, j) in enumerate(BOX_EDGES):
        ax.plot(corners[[i, j], 0], corners[[i, j], 2], color=C_A, lw=1.6, label="3D bounding box" if k == 0 else None)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title("COLMAP scene, top-down, with the object separated"); ax.legend(frameon=False, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(OUT / "colmap_object_localization.png", dpi=150); plt.close()

    # --- figure 2: object points and 3D box re-projected into six held-out views
    hv = sorted(held, key=lambda i: views[i]["frame"])
    pick_ids = [hv[int(round(t))] for t in np.linspace(0, len(hv) - 1, 6)]
    tiles = []
    for iid in pick_ids:
        v = views[iid]
        img = cv2.imread(str(KEYFRAMES_DIR / v["name"]))
        cnts, _ = cv2.findContours((v["mask"] > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 255, 255), 3)
        uvc, zc = project(corners, v["M"], params)
        if np.all(zc > 0):
            for i, j in BOX_EDGES:
                cv2.line(img, tuple(np.round(uvc[i]).astype(int)), tuple(np.round(uvc[j]).astype(int)), (214, 120, 42), 4)
        uv, z = project(obj, v["M"], params)
        for p in uv[z > 0]:
            cv2.circle(img, tuple(np.round(p).astype(int)), 8, (0, 0, 0), -1)
            cv2.circle(img, tuple(np.round(p).astype(int)), 5, (80, 220, 255), -1)
        for col, th in (((0, 0, 0), 9), ((255, 255, 255), 4)):
            cv2.putText(img, f"frame {v['frame']} (held out)", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.7, col, th)
        tiles.append(cv2.resize(img, (360, 640)))
    cv2.imwrite(str(OUT / "colmap_object_projection_contact_sheet.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"[INFO] Views with mask: {len(views)} (selection {len(sel)}, held-out {len(held)})")
    print(f"[INFO] Object points: {obj_sel.sum()} from selection views; {obj_all.sum()} final (all views) of {len(xyz)}")
    print(f"[INFO] Held-out in-mask: mean {res.object_points_in_mask_pct.mean():.1f}%, min {res.object_points_in_mask_pct.min():.1f}%; "
          f"baseline {res.scene_points_in_mask_pct.mean():.1f}%")
    print(f"[INFO] OBB extent {np.round(extent, 3)}; points cover a median {height_cover:.0%} of the box height")
    print(f"[INFO] Results written to {OUT}")


if __name__ == "__main__":
    main()
