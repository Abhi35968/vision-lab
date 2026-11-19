"""
Experiment 5 - COLMAP as the reference reconstruction engine (via pycolmap, CPU).

Runs COLMAP's feature extraction, exhaustive matching and incremental mapping on the same
47 keyframes, with one shared camera whose focal length COLMAP estimates itself, and compares
the result with the project's own OpenCV-based incremental SfM.

    python -m experiments.exp5_colmap_baseline
"""

import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycolmap

from experiments.common import (
    PROJECT_ROOT, KEYFRAMES_DIR, SFM_DIR, FOCAL_PX, keyframe_paths, save_json, style_axes, C_A, C_B, C_GREY,
)

OUT = PROJECT_ROOT / "outputs" / "colmap"
OWN_STORED = {"registered": 32, "points": 2077, "mean_reproj_px": 3.118, "mean_track_length": 3.41}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "database.db"
    sparse = OUT / "sparse"
    if db.exists():
        db.unlink()
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir(parents=True)
    n_images = len(keyframe_paths())
    timings = {}

    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"  # f, cx, cy, k1 - estimated by COLMAP, shared by all frames
    pycolmap.extract_features(db, KEYFRAMES_DIR, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, device=pycolmap.Device.cpu)
    timings["feature_extraction_s"] = round(time.time() - t0, 1)
    print(f"[INFO] COLMAP feature extraction: {timings['feature_extraction_s']} s", flush=True)

    t0 = time.time()
    pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    timings["matching_s"] = round(time.time() - t0, 1)
    print(f"[INFO] COLMAP exhaustive matching: {timings['matching_s']} s", flush=True)

    t0 = time.time()
    maps = pycolmap.incremental_mapping(db, KEYFRAMES_DIR, sparse)
    timings["mapping_s"] = round(time.time() - t0, 1)
    print(f"[INFO] COLMAP incremental mapping: {timings['mapping_s']} s, {len(maps)} model(s)", flush=True)
    if not maps:
        save_json({"status": "COLMAP produced no reconstruction", "timings": timings}, OUT / "colmap_summary.json")
        print("[WARNING] COLMAP produced no reconstruction")
        return

    models = []
    for mid, rec in maps.items():
        cam = next(iter(rec.cameras.values()))
        names = sorted(img.name for img in rec.images.values() if img.has_pose)
        models.append({
            "model_id": int(mid), "registered_images": int(rec.num_reg_images()),
            "points_3d": int(rec.num_points3D()),
            "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
            "mean_track_length": float(rec.compute_mean_track_length()),
            "camera_model": str(cam.model).split(".")[-1], "camera_params": np.asarray(cam.params).tolist(),
            "estimated_focal_px": float(cam.params[0]),
            "first_image": names[0] if names else None, "last_image": names[-1] if names else None,
            "registered_keyframe_indices": [int(n.split("_")[1]) for n in names],
        })
    models.sort(key=lambda m: -m["registered_images"])
    best = models[0]
    rec = maps[best["model_id"]]
    rec.export_PLY(str(OUT / "colmap_sparse_points.ply"))

    summary = {
        "pycolmap_version": pycolmap.__version__, "images": n_images, "timings": timings,
        "total_time_s": round(sum(timings.values()), 1), "models": models, "largest_model": best,
        "own_sfm_stored_run": OWN_STORED, "assumed_focal_px_in_own_sfm": FOCAL_PX,
    }
    save_json(summary, OUT / "colmap_summary.json")

    # --- figure: COLMAP reconstruction (top-down) and comparison with the project's own SfM
    pts = np.array([p.xyz for p in rec.points3D.values()])
    centres = np.array([img.projection_center() for img in sorted(rec.images.values(), key=lambda i: i.name)
                        if img.has_pose])
    lo, hi = np.percentile(pts, 3, axis=0), np.percentile(pts, 97, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    ax = axs[0]
    ax.scatter(pts[keep, 0], pts[keep, 2], s=1.5, c=C_GREY, alpha=0.5, label=f"Points ({len(pts)})")
    ax.plot(centres[:, 0], centres[:, 2], "-o", color=C_B, ms=3, lw=1.2, label=f"Cameras ({len(centres)})")
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title("COLMAP reconstruction, top-down"); ax.legend(frameon=False, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    labels = ["Own SfM\n(stored run)", "COLMAP"]
    for ax, title, vals, fmt in (
        (axs[1], f"Registered cameras (of {n_images})", [OWN_STORED["registered"], best["registered_images"]], "{:.0f}"),
        (axs[2], "Sparse 3D points", [OWN_STORED["points"], best["points_3d"]], "{:.0f}"),
        (axs[3], "Mean reprojection error (px)", [OWN_STORED["mean_reproj_px"], best["mean_reprojection_error_px"]], "{:.2f}"),
    ):
        bars = ax.bar(labels, vals, color=[C_A, C_B], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom")
        ax.set_title(title); style_axes(ax)
    plt.suptitle(f"COLMAP baseline versus the project's own incremental SfM "
                 f"(COLMAP estimated focal length: {best['estimated_focal_px']:.0f} px; own SfM assumed {FOCAL_PX:.0f} px)",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "colmap_vs_own_sfm.png", dpi=150); plt.close()

    pd.DataFrame([
        {"engine": "Own incremental SfM (stored run)", "registered": OWN_STORED["registered"],
         "points_3d": OWN_STORED["points"], "mean_reproj_px": OWN_STORED["mean_reproj_px"],
         "mean_track_length": OWN_STORED["mean_track_length"], "focal_px": FOCAL_PX, "focal_source": "assumed"},
        {"engine": "COLMAP (pycolmap, CPU)", "registered": best["registered_images"], "points_3d": best["points_3d"],
         "mean_reproj_px": round(best["mean_reprojection_error_px"], 3),
         "mean_track_length": round(best["mean_track_length"], 2),
         "focal_px": round(best["estimated_focal_px"], 1), "focal_source": "self-calibrated"},
    ]).to_csv(OUT / "colmap_vs_own_sfm.csv", index=False)
    print(f"[INFO] COLMAP largest model: {best['registered_images']}/{n_images} images, {best['points_3d']} points, "
          f"mean reprojection error {best['mean_reprojection_error_px']:.3f} px, "
          f"focal {best['estimated_focal_px']:.1f} px, params {np.round(best['camera_params'], 3).tolist()}")
    print(f"[INFO] Models: {[(m['registered_images'], m['points_3d']) for m in models]}; timings {timings}")
    print(f"[INFO] Results written to {OUT}")


if __name__ == "__main__":
    main()
