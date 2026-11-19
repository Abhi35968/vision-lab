"""
Experiment 3 - How consistently is the object localised across frames?

Three measurements on the real video:
  A. 3D -> 2D consistency: the validated object points are re-projected into EVERY registered
     camera that has a segmentation mask (not only the views that triangulated them). The share
     that lands inside the object mask is compared with the same share for all scene points.
  B. Stability of the projected object centroid inside the detected 2D box (normalised 0..1).
  C. 2D detector stability: confidence and predicted class per keyframe.

    python -m experiments.exp3_object_localization
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (
    PROJECT_ROOT, SFM_DIR, SEG_DIR, REFINED_CSV, IMAGE_W, IMAGE_H,
    load_registered_cameras, project, out_dir, save_json, style_axes, C_A, C_B, C_C, C_GREY,
)

CLASS_COLORS = {"bottle": C_A, "vase": C_B, "cup": C_C, "none": "#d03b3b"}


def inside_mask_fraction(points, cam, mask):
    uv, z = project(points, cam["R"], cam["t"])
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_H)
    if ok.sum() == 0:
        return np.nan, 0
    px = np.round(uv[ok]).astype(int)
    px[:, 0] = np.clip(px[:, 0], 0, IMAGE_W - 1); px[:, 1] = np.clip(px[:, 1], 0, IMAGE_H - 1)
    hits = mask[px[:, 1], px[:, 0]] > 127
    # points that project outside the image count as "not on the object"
    return hits.sum() / len(points), int(ok.sum())


def main():
    out = out_dir("exp3_object_localization")
    cams = load_registered_cameras()
    seg = pd.read_csv(SEG_DIR / "segmentation_summary.csv")
    df = pd.read_csv(REFINED_CSV)
    valid = df["refined_status"].isin(["HIGH_GEOMETRIC_CONFIDENCE", "MODERATE_GEOMETRIC_CONFIDENCE"])
    obj = df.loc[valid, ["x_world", "y_world", "z_world"]].values.astype(np.float64)
    scene = np.load(SFM_DIR / "sparse_points.npy")[:, :3].astype(np.float64)
    centroid = obj.mean(axis=0, keepdims=True)

    rows = []
    for _, r in seg.iterrows():
        cid = int(r.frame_idx)
        if cid not in cams or r.target_detected != 1:
            continue
        mask = cv2.imread(str(PROJECT_ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        cam = cams[cid]
        f_obj, _ = inside_mask_fraction(obj, cam, mask)
        f_scene, _ = inside_mask_fraction(scene, cam, mask)
        uv, z = project(centroid, cam["R"], cam["t"])
        bw, bh = r.bbox_x2 - r.bbox_x1, r.bbox_y2 - r.bbox_y1
        rows.append({
            "camera_id": cid, "object_points_in_mask_pct": 100 * f_obj,
            "scene_points_in_mask_pct": 100 * f_scene, "mask_area_pct": r.mask_area_pct,
            "centroid_u_px": uv[0, 0], "centroid_v_px": uv[0, 1],
            "centroid_u_norm": (uv[0, 0] - r.bbox_x1) / bw, "centroid_v_norm": (uv[0, 1] - r.bbox_y1) / bh,
            "centroid_inside_bbox": bool(0 <= (uv[0, 0] - r.bbox_x1) / bw <= 1 and 0 <= (uv[0, 1] - r.bbox_y1) / bh <= 1),
            "bbox_diag_px": float(np.hypot(bw, bh)),
        })
    res = pd.DataFrame(rows)
    res.to_csv(out / "per_view_localization.csv", index=False)

    # centroid stability, expressed relative to the size of the detected box
    du = res.centroid_u_norm - res.centroid_u_norm.mean()
    dv = res.centroid_v_norm - res.centroid_v_norm.mean()
    det = seg.copy()
    det["cls"] = np.where(det.target_detected == 1, det.target_class.fillna("none"), "none")
    cls_counts = det.cls.value_counts().to_dict()
    detected = det[det.target_detected == 1]

    summary = {
        "views_evaluated": int(len(res)),
        "validated_object_points": int(len(obj)),
        "scene_points": int(len(scene)),
        "object_points_in_mask_pct": {"mean": res.object_points_in_mask_pct.mean(),
                                      "median": res.object_points_in_mask_pct.median(),
                                      "min": res.object_points_in_mask_pct.min(),
                                      "std": res.object_points_in_mask_pct.std()},
        "scene_points_in_mask_pct_baseline": {"mean": res.scene_points_in_mask_pct.mean()},
        "mean_mask_area_pct": res.mask_area_pct.mean(),
        "centroid_inside_bbox_pct": 100 * res.centroid_inside_bbox.mean(),
        "centroid_norm_position_mean": [res.centroid_u_norm.mean(), res.centroid_v_norm.mean()],
        "centroid_norm_position_std": [res.centroid_u_norm.std(), res.centroid_v_norm.std()],
        "centroid_rms_deviation_fraction_of_bbox": float(np.sqrt((du ** 2 + dv ** 2).mean())),
        "detector_2d": {
            "keyframes": int(len(det)), "detected": int(len(detected)),
            "detection_rate_pct": 100 * len(detected) / len(det),
            "class_label_counts": cls_counts,
            "most_common_label_share_pct": 100 * detected.target_class.value_counts().iloc[0] / len(detected),
            "confidence_mean": detected.confidence.mean(), "confidence_std": detected.confidence.std(),
        },
    }
    save_json(summary, out / "summary.json")

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax = axs[0, 0]
    ax.plot(res.camera_id, res.object_points_in_mask_pct, "-o", color=C_A, lw=2, ms=5,
            label=f"Validated object points (mean {res.object_points_in_mask_pct.mean():.0f}%)")
    ax.plot(res.camera_id, res.scene_points_in_mask_pct, "-s", color=C_GREY, lw=2, ms=4,
            label=f"All scene points, baseline (mean {res.scene_points_in_mask_pct.mean():.0f}%)")
    ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Registered camera id"); ax.set_ylabel("Points projecting inside the object mask (%)")
    ax.set_title("A. 3D object points re-projected into every view"); ax.legend(frameon=False, loc="upper left")
    style_axes(ax)

    ax = axs[0, 1]
    sc = ax.scatter(res.centroid_u_norm, res.centroid_v_norm, c=res.camera_id, cmap="viridis", s=45,
                    edgecolors="black", linewidths=0.4)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.5, color="black"))
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(1.1, -0.1); ax.set_aspect("equal")
    ax.set_xlabel("Horizontal position inside detected box (0 = left, 1 = right)")
    ax.set_ylabel("Vertical position (0 = top, 1 = bottom)")
    ax.set_title(f"B. Projected 3D centroid inside the 2D box "
                 f"(std {res.centroid_u_norm.std():.2f} / {res.centroid_v_norm.std():.2f})")
    plt.colorbar(sc, ax=ax, label="Camera id"); ax.grid(True, linestyle=":", alpha=0.5)

    ax = axs[1, 0]
    for cls, col in CLASS_COLORS.items():
        d = det[det.cls == cls]
        if len(d):
            y = d.confidence.fillna(0) if cls != "none" else np.zeros(len(d))
            ax.bar(d.frame_idx, y if cls != "none" else np.full(len(d), 0.02), color=col,
                   label=f"{cls} ({len(d)})", width=0.8)
    ax.set_xlabel("Keyframe index"); ax.set_ylabel("Detection confidence")
    ax.set_title("C. 2D detector output per keyframe (label and confidence)")
    ax.set_ylim(0, 0.85); ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center"); style_axes(ax)

    ax = axs[1, 1]
    names = list(cls_counts.keys())
    bars = ax.bar(names, [cls_counts[n] for n in names], color=[CLASS_COLORS.get(n, C_GREY) for n in names], width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}", ha="center", va="bottom")
    ax.set_ylabel("Keyframes"); ax.set_title("Label given to the same physical bottle across 47 keyframes")
    style_axes(ax)
    plt.suptitle("Object localisation consistency across frames", fontsize=13)
    plt.tight_layout(); plt.savefig(out / "localization_consistency.png", dpi=150); plt.close()

    print(pd.Series({k: v for k, v in summary.items() if not isinstance(v, dict)}).to_string())
    print("object in mask:", {k: round(v, 2) for k, v in summary["object_points_in_mask_pct"].items()})
    print("detector:", summary["detector_2d"])
    print(f"[INFO] Results written to {out}")


if __name__ == "__main__":
    main()
