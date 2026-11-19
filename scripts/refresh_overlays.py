"""
Refreshes object 3D bounding boxes to be strictly upright (gravity-aligned)
and updates overlay.jpg and overlay_gallery.jpg.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import pandas as pd

from experiments.objectron_batch import cam_matrix, project, camera_params, BOX_EDGES, in_mask
from experiments.build_object_scene import obb_from_points

ROOT = Path(__file__).resolve().parent.parent
OBJ_DIR = ROOT / "outputs" / "objectron"
SUMMARY_CSV = OBJ_DIR / "summary.csv"


def refresh_overlays():
    if not SUMMARY_CSV.exists():
        print("summary.csv not found.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    tiles = []

    for r in df.itertuples():
        vdir = OBJ_DIR / r.video
        kf_dir = vdir / "keyframes"
        obj_ply = vdir / "object_points.ply"
        colmap_dir = vdir / "colmap" / "sparse" / "0"
        seg_csv = vdir / "segmentation.csv"

        if not (colmap_dir.exists() and kf_dir.exists()):
            continue

        rec = pycolmap.Reconstruction(str(colmap_dir))
        params = camera_params(next(iter(rec.cameras.values())))
        Ms = [cam_matrix(i) for i in rec.images.values() if i.has_pose]
        up = -np.mean([M[1, :3] for M in Ms], axis=0)
        up /= np.linalg.norm(up)

        corners = None
        if obj_ply.exists():
            pcd = o3d.io.read_point_cloud(str(obj_ply))
            obj = np.asarray(pcd.points)
            if len(obj) >= 4:
                centre, R, extent, corners = obb_from_points(obj, up_vector=up)
                box_json = vdir / "object_bounding_box.json"
                with open(box_json, "w", encoding="utf-8") as f:
                    json.dump({
                        "units": "arbitrary COLMAP units",
                        "num_points": int(len(obj)),
                        "center": centre.tolist(),
                        "rotation_matrix_columns_are_axes": R.tolist(),
                        "extent": extent.tolist(),
                        "corners": corners.tolist(),
                        "upright_constrained": True
                    }, f, indent=2)

        # Load segmentations
        masks = {}
        if seg_csv.exists():
            sdf = pd.read_csv(seg_csv)
            for _, srow in sdf.iterrows():
                if bool(srow.get("detected", False)):
                    m_path = vdir / "masks" / f"{Path(srow['image']).stem}.png"
                    # If mask images exist, load them; else draw from bounding box
                    if m_path.exists():
                        masks[srow["image"]] = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)

        # Pick 4 evenly spaced views
        reg_imgs = sorted([img for img in rec.images.values() if img.has_pose and (kf_dir / img.name).exists()],
                          key=lambda x: x.name)
        if len(reg_imgs) >= 4:
            picked_indices = np.linspace(0, len(reg_imgs) - 1, 4).astype(int)
            picked_imgs = [reg_imgs[i] for i in picked_indices]
        else:
            picked_imgs = reg_imgs

        sub_tiles = []
        for img_info in picked_imgs:
            img_path = kf_dir / img_info.name
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            M = cam_matrix(img_info)

            # Draw 3D points if available
            if obj_ply.exists() and len(obj) >= 4:
                uv, z = project(obj, M, params)
                ok = z > 0
                for pt in np.round(uv[ok]).astype(int):
                    if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                        cv2.circle(img, (pt[0], pt[1]), 2, (42, 175, 27), -1)

            # Draw Upright 3D Bounding Box
            if corners is not None:
                uvc, zc = project(corners, M, params)
                if np.all(zc > 0):
                    for i, j in BOX_EDGES:
                        pt1 = (int(round(uvc[i, 0])), int(round(uvc[i, 1])))
                        pt2 = (int(round(uvc[j, 0])), int(round(uvc[j, 1])))
                        cv2.line(img, pt1, pt2, (230, 120, 42), 2)

            sub_tiles.append(cv2.resize(img, (240, 320)))

        if sub_tiles:
            tile_row = np.hstack(sub_tiles)
            overlay_file = vdir / "overlay.jpg"
            cv2.imwrite(str(overlay_file), tile_row, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Label for gallery
            labeled_tile = tile_row.copy()
            cv2.putText(labeled_tile, r.video.replace("_", " "), (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            tiles.append(labeled_tile)

    if tiles:
        # Build gallery: 2 columns per row
        row_imgs = []
        for i in range(0, len(tiles), 2):
            if i + 1 < len(tiles):
                row_imgs.append(np.hstack([tiles[i], tiles[i + 1]]))
            else:
                blank = np.zeros_like(tiles[i])
                row_imgs.append(np.hstack([tiles[i], blank]))
        gallery = np.vstack(row_imgs)
        gallery_resized = cv2.resize(gallery, None, fx=0.75, fy=0.75)
        out_gallery = OBJ_DIR / "overlay_gallery.jpg"
        cv2.imwrite(str(out_gallery), gallery_resized, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"Updated overlay gallery saved to: {out_gallery}")


if __name__ == "__main__":
    refresh_overlays()
