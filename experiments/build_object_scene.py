"""
Object-level 3D outputs and scene visualisation (plan Stage 4 and Stage 5 visualisation).

Writes to outputs/objects_3d/:
    bottle_object_points.ply        validated object points (high + moderate confidence)
    bottle_candidate_points.ply     all 47 candidates, coloured by confidence tier
    bottle_bounding_boxes.json      axis-aligned and oriented 3D boxes (arbitrary SfM units)
    scene_with_object.png           scene cloud, camera trajectory, object points, 3D boxes
    bbox_projection_contact_sheet.jpg   3D boxes re-projected into six keyframes
    scene_interactive.html          rotatable Plotly version of the scene

    python -m experiments.build_object_scene
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import open3d as o3d
except (ImportError, Exception):
    o3d = None
import pandas as pd
import plotly.graph_objects as go

from experiments.common import (
    PROJECT_ROOT, SFM_DIR, SEG_DIR, REFINED_CSV, BOTTLE_MESH, KEYFRAMES_DIR, K,
    load_registered_cameras, project, save_json, C_A, C_B, C_C, C_GREY,
)

OUT = PROJECT_ROOT / "outputs" / "objects_3d"
TIER_COLORS = {
    "HIGH_GEOMETRIC_CONFIDENCE": "#1baf7f",
    "MODERATE_GEOMETRIC_CONFIDENCE": "#2a78d6",
    "LOW_GEOMETRIC_CONFIDENCE": "#eda100",
    "GEOMETRIC_OUTLIER": "#d03b3b",
}
BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def hex_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def aabb_corners(mn, mx):
    return np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])


def obb_from_points(pts, up_vector=None):
    """PCA oriented box. When up_vector is supplied, constructs an upright box constrained to gravity."""
    c = pts.mean(axis=0)
    if up_vector is not None:
        up = np.asarray(up_vector, dtype=np.float64)
        up_norm = np.linalg.norm(up)
        if up_norm > 1e-6:
            up = up / up_norm
            z_coords = (pts - c) @ up
            pts_proj = (pts - c) - np.outer(z_coords, up)
            _, _, vt = np.linalg.svd(pts_proj, full_matrices=False)
            ax0 = vt[0] - np.dot(vt[0], up) * up
            ax0_norm = np.linalg.norm(ax0)
            if ax0_norm > 1e-6:
                ax0 = ax0 / ax0_norm
                ax1 = np.cross(ax0, up)
                ax1 = ax1 / (np.linalg.norm(ax1) + 1e-9)
                R = np.column_stack([ax0, up, ax1])
                local = (pts - c) @ R
                mn, mx = local.min(axis=0), local.max(axis=0)
                centre = c + R @ ((mn + mx) / 2.0)
                extent = mx - mn
                corners = aabb_corners(-extent / 2.0, extent / 2.0) @ R.T + centre
                return centre, R, extent, corners

    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    R = vt.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    local = (pts - c) @ R
    mn, mx = local.min(axis=0), local.max(axis=0)
    centre = c + R @ ((mn + mx) / 2.0)
    extent = mx - mn
    corners = aabb_corners(-extent / 2.0, extent / 2.0) @ R.T + centre
    return centre, R, extent, corners


def box_record(name, source, pts):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    centre, R, extent, corners = obb_from_points(pts)
    return {
        "name": name, "source": source, "num_points": int(len(pts)), "units": "arbitrary SfM units",
        "axis_aligned": {"min": mn, "max": mx, "center": (mn + mx) / 2.0, "extent": mx - mn,
                         "volume": float(np.prod(mx - mn)), "corners": aabb_corners(mn, mx)},
        "oriented": {"center": centre, "rotation_matrix_columns_are_axes": R, "extent": extent,
                     "volume": float(np.prod(extent)), "corners": corners},
    }


def write_ply(path, pts, colors):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pc, write_ascii=True)


def draw_box(ax, corners, color, dims, lw=1.8, label=None):
    for k, (i, j) in enumerate(BOX_EDGES):
        seg = corners[[i, j]][:, dims]
        ax.plot(*seg.T, color=color, lw=lw, label=label if k == 0 else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REFINED_CSV)
    xyz = df[["x_world", "y_world", "z_world"]].values.astype(np.float64)
    status = df["refined_status"].values
    valid = np.isin(status, ["HIGH_GEOMETRIC_CONFIDENCE", "MODERATE_GEOMETRIC_CONFIDENCE"])
    obj = xyz[valid]

    scene = np.load(SFM_DIR / "sparse_points.npy")[:, :3].astype(np.float64)
    cams = load_registered_cameras()
    centres = np.array([c["C"] for c in cams.values()])
    mesh = o3d.io.read_triangle_mesh(str(BOTTLE_MESH))
    mesh_v = np.asarray(mesh.vertices)

    # --- point clouds
    write_ply(OUT / "bottle_object_points.ply", obj, [hex_rgb(TIER_COLORS[s]) for s in status[valid]])
    write_ply(OUT / "bottle_candidate_points.ply", xyz, [hex_rgb(TIER_COLORS[s]) for s in status])

    # --- boxes: measured (sparse points) versus model (fitted parametric mesh)
    boxes = {
        "sparse_points_box": box_record(
            "bottle (validated sparse points)",
            "29 validated SfM points (high + moderate confidence). Measured, but the points sit mostly "
            "on the cap and upper body, so this box does not cover the whole bottle.", obj),
        "fitted_model_box": box_record(
            "bottle (fitted parametric mesh)",
            "Vertices of the silhouette-fitted surface-of-revolution mesh. Covers the whole bottle, "
            "but its shape below the cap comes from the bottle prior, not from triangulated points.", mesh_v),
    }
    save_json(boxes, OUT / "bottle_bounding_boxes.json")
    sp_c = np.asarray(boxes["sparse_points_box"]["oriented"]["corners"])
    md_c = np.asarray(boxes["fitted_model_box"]["axis_aligned"]["corners"])

    # --- static scene figure (robust crop so far-away outliers do not flatten the view)
    lo, hi = np.percentile(scene, 3, axis=0), np.percentile(scene, 97, axis=0)
    keep = np.all((scene >= lo) & (scene <= hi), axis=1)
    sc = scene[keep]

    fig = plt.figure(figsize=(17, 6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(sc[:, 0], sc[:, 2], -sc[:, 1], s=2, c=C_GREY, alpha=0.45, label=f"Scene points ({len(scene)})")
    ax.plot(centres[:, 0], centres[:, 2], -centres[:, 1], "-o", color=C_B, ms=3, lw=1.2,
            label=f"Cameras ({len(centres)})")
    for s, col in TIER_COLORS.items():
        m = status == s
        if s in ("HIGH_GEOMETRIC_CONFIDENCE", "MODERATE_GEOMETRIC_CONFIDENCE"):
            ax.scatter(xyz[m, 0], xyz[m, 2], -xyz[m, 1], s=28, c=col, edgecolors="black", linewidths=0.4,
                       label=f"Object: {s.split('_')[0].lower()} ({m.sum()})")
    for corners, col, lab in ((md_c, C_C, "Box: fitted model"), (sp_c, C_A, "Box: sparse points")):
        for k, (i, j) in enumerate(BOX_EDGES):
            p = corners[[i, j]]
            ax.plot(p[:, 0], p[:, 2], -p[:, 1], color=col, lw=1.8, label=lab if k == 0 else None)
    ax.set_xlabel("X"); ax.set_ylabel("Z (depth)"); ax.set_zlabel("-Y (up)")
    ax.set_title("3D scene with object and bounding boxes"); ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=18, azim=-62)

    for k, (dims, xl, yl, title, flip) in enumerate([
        ((0, 2), "X", "Z (depth)", "Top-down view", False),
        ((0, 1), "X", "Y (image-down)", "Front view", True),
    ]):
        a = fig.add_subplot(1, 3, k + 2)
        a.scatter(sc[:, dims[0]], sc[:, dims[1]], s=2, c=C_GREY, alpha=0.45)
        a.plot(centres[:, dims[0]], centres[:, dims[1]], "-o", color=C_B, ms=3, lw=1.2)
        for s, col in TIER_COLORS.items():
            m = status == s
            if s in ("HIGH_GEOMETRIC_CONFIDENCE", "MODERATE_GEOMETRIC_CONFIDENCE"):
                a.scatter(xyz[m, dims[0]], xyz[m, dims[1]], s=26, c=col, edgecolors="black", linewidths=0.4)
        draw_box(a, md_c, C_C, list(dims)); draw_box(a, sp_c, C_A, list(dims))
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(title); a.set_aspect("equal", adjustable="datalim")
        if flip:
            a.invert_yaxis()
        a.grid(True, linestyle=":", alpha=0.5)
    plt.suptitle("Sparse reconstruction, camera trajectory and object-level localisation (arbitrary SfM units)",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "scene_with_object.png", dpi=150); plt.close()

    # --- re-project both boxes into six keyframes
    seg = pd.read_csv(SEG_DIR / "segmentation_summary.csv").set_index("frame_idx")
    cam_ids = [i for i in cams if i in seg.index and seg.loc[i, "target_detected"] == 1]
    pick = [cam_ids[int(round(t))] for t in np.linspace(0, len(cam_ids) - 1, 6)]
    tiles = []
    for cid in pick:
        cam = cams[cid]
        img = cv2.imread(str(KEYFRAMES_DIR / cam["name"]))
        mask = cv2.imread(str(PROJECT_ROOT / seg.loc[cid, "mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 255, 255), 3)
        for corners, bgr in ((md_c, (127, 175, 27)), (sp_c, (214, 120, 42))):
            uv, z = project(corners, cam["R"], cam["t"])
            if np.all(z > 0):
                for i, j in BOX_EDGES:
                    cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[j]).astype(int)), bgr, 4)
        uv, z = project(obj, cam["R"], cam["t"])
        for p in uv[z > 0]:
            cv2.circle(img, tuple(np.round(p).astype(int)), 9, (0, 0, 0), -1)
            cv2.circle(img, tuple(np.round(p).astype(int)), 6, (80, 220, 255), -1)
        cv2.putText(img, f"camera {cid}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9)
        cv2.putText(img, f"camera {cid}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        tiles.append(cv2.resize(img, (360, 640)))
    sheet = np.hstack(tiles)
    cv2.imwrite(str(OUT / "bbox_projection_contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # --- interactive scene
    traces = [
        go.Scatter3d(x=sc[:, 0], y=sc[:, 2], z=-sc[:, 1], mode="markers", name="Scene points",
                     marker=dict(size=1.5, color=C_GREY, opacity=0.5)),
        go.Scatter3d(x=centres[:, 0], y=centres[:, 2], z=-centres[:, 1], mode="lines+markers",
                     name="Camera trajectory", marker=dict(size=3, color=C_B), line=dict(color=C_B, width=3)),
        go.Mesh3d(x=mesh_v[:, 0], y=mesh_v[:, 2], z=-mesh_v[:, 1],
                  i=np.asarray(mesh.triangles)[:, 0], j=np.asarray(mesh.triangles)[:, 1],
                  k=np.asarray(mesh.triangles)[:, 2], color=C_C, opacity=0.25, name="Fitted bottle model",
                  showlegend=True),
    ]
    for s, col in TIER_COLORS.items():
        m = status == s
        traces.append(go.Scatter3d(x=xyz[m, 0], y=xyz[m, 2], z=-xyz[m, 1], mode="markers",
                                   name=f"Object points: {s.replace('_', ' ').lower()} ({m.sum()})",
                                   marker=dict(size=4, color=col, line=dict(color="black", width=0.5))))
    for corners, col, lab in ((md_c, C_C, "3D box: fitted model"), (sp_c, C_A, "3D box: sparse points")):
        xs, ys, zs = [], [], []
        for i, j in BOX_EDGES:
            xs += [corners[i, 0], corners[j, 0], None]
            ys += [corners[i, 2], corners[j, 2], None]
            zs += [-corners[i, 1], -corners[j, 1], None]
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=lab, line=dict(color=col, width=5)))
    fig = go.Figure(traces)
    fig.update_layout(title="Reconstructed scene, camera trajectory and object (arbitrary SfM units)",
                      scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Z (depth)", zaxis_title="-Y (up)"),
                      margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(str(OUT / "scene_interactive.html"), include_plotlyjs=True)

    sp, md = boxes["sparse_points_box"], boxes["fitted_model_box"]
    print(f"[INFO] Object points: {len(obj)} validated of {len(xyz)} candidates; scene points: {len(scene)}; cameras: {len(centres)}")
    print(f"[INFO] Sparse-point OBB extent: {np.round(sp['oriented']['extent'], 3)}")
    print(f"[INFO] Fitted-model AABB extent: {np.round(md['axis_aligned']['extent'], 3)}")
    print(f"[INFO] Outputs written to {OUT}")


if __name__ == "__main__":
    main()
