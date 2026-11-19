"""
Experiment 2b - Reconstruction versus number of frames, with COLMAP as the engine.

Same uniform frame subsets as exp2_frame_count, reconstructed with COLMAP (pycolmap, CPU),
and plotted next to the project's own SfM results from exp2.

    python -m experiments.exp2b_colmap_frame_count      (run exp2_frame_count first)
"""

import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycolmap

from experiments.common import KEYFRAMES_DIR, keyframe_paths, out_dir, style_axes, C_A, C_B, C_GREY
from experiments.exp2_frame_count import FRAME_COUNTS


def colmap_run(names, work):
    shutil.rmtree(work, ignore_errors=True)
    (work / "sparse").mkdir(parents=True)
    db = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    t0 = time.time()
    pycolmap.extract_features(db, KEYFRAMES_DIR, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    maps = pycolmap.incremental_mapping(db, KEYFRAMES_DIR, work / "sparse")
    elapsed = time.time() - t0
    if not maps:
        return {"registered_cameras": 0, "points_3d": 0, "reproj_mean_px": np.nan, "mean_track_length": np.nan,
                "focal_px": np.nan, "models": 0, "runtime_s": round(elapsed, 1)}
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    return {"registered_cameras": int(rec.num_reg_images()), "points_3d": int(rec.num_points3D()),
            "reproj_mean_px": float(rec.compute_mean_reprojection_error()),
            "mean_track_length": float(rec.compute_mean_track_length()),
            "focal_px": float(next(iter(rec.cameras.values())).params[0]), "models": len(maps),
            "runtime_s": round(elapsed, 1)}


def main():
    out = out_dir("exp2_frame_count")
    paths = keyframe_paths()
    work = out / "_colmap_work"
    rows = []
    for n in FRAME_COUNTS:
        idx = np.unique(np.round(np.linspace(0, len(paths) - 1, n)).astype(int))
        row = {"frames_used": len(idx), **colmap_run([paths[i].name for i in idx], work)}
        row["registration_rate_pct"] = round(100.0 * row["registered_cameras"] / len(idx), 1)
        rows.append(row)
        print(f"[INFO] COLMAP N={n}: {row}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    cm = pd.DataFrame(rows)
    cm.to_csv(out / "colmap_frame_count.csv", index=False)

    own = pd.read_csv(out / "frame_count_summary.csv")
    x = cm.frames_used.values
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))

    ax = axs[0]
    ax.plot(x, x, "--", color=C_GREY, lw=1.2, label="All frames registered")
    ax.plot(x, cm.registered_cameras, "-o", color=C_B, lw=2, ms=7, label="COLMAP")
    ax.errorbar(own.frames_used, own.registered_median,
                yerr=[own.registered_median - own.registered_min, own.registered_max - own.registered_median],
                fmt="-o", color=C_A, lw=2, ms=7, capsize=4, label="Own SfM (median, min-max over seeds)")
    ax.set_title("Registered cameras"); ax.legend(frameon=False, fontsize=8)

    ax = axs[1]
    ax.plot(x, cm.points_3d, "-o", color=C_B, lw=2, ms=7, label="COLMAP")
    ax.errorbar(own.frames_used, own.points_median,
                yerr=[own.points_median - own.points_min, own.points_max - own.points_median],
                fmt="-o", color=C_A, lw=2, ms=7, capsize=4, label="Own SfM")
    for xi, yi in zip(x, cm.points_3d):
        ax.annotate(f"{yi}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_title("Sparse 3D points"); ax.legend(frameon=False, fontsize=8)

    ax = axs[2]
    ax.plot(x, cm.reproj_mean_px, "-o", color=C_B, lw=2, ms=7, label="COLMAP (mean)")
    ax.plot(own.frames_used, own.reproj_mean_px, "-o", color=C_A, lw=2, ms=7, label="Own SfM (mean)")
    ax.set_title("Mean reprojection error (px)"); ax.legend(frameon=False, fontsize=8)

    ax = axs[3]
    ax.plot(x, cm.focal_px, "-o", color=C_B, lw=2, ms=7, label="COLMAP self-calibrated")
    ax.axhline(1300, color=C_A, lw=1.5, ls="--", label="Assumed by own SfM (1300)")
    for xi, yi in zip(x, cm.focal_px):
        if np.isfinite(yi):
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_title("Focal length (px)"); ax.legend(frameon=False, fontsize=8)

    for ax in axs:
        ax.set_xlabel("Frames given to SfM"); ax.set_xticks(list(x)); style_axes(ax)
    plt.suptitle("Reconstruction versus number of frames: COLMAP and the project's own SfM", fontsize=13)
    plt.tight_layout(); plt.savefig(out / "frame_count_colmap_vs_own.png", dpi=150); plt.close()

    # refresh the seed-stability figure legend placement from the saved runs
    runs = pd.read_csv(out / "all_runs.csv")
    full = runs[runs.frames_used == len(paths)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    bars = axs[0].bar(full.seed.astype(str), full.registered_cameras, color=C_A, width=0.6)
    for b, a, c in zip(bars, full.first_registered, full.last_registered):
        axs[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{int(b.get_height())}\n({a}-{c})",
                    ha="center", va="bottom", fontsize=8)
    axs[0].axhline(len(paths), color=C_GREY, ls="--", lw=1.2, label="47 keyframes (COLMAP registers all)")
    axs[0].axhline(32, color=C_B, lw=1.5, label="Stored main reconstruction (32)")
    axs[0].set_ylim(0, 64); axs[0].set_yticks(range(0, 51, 10))
    axs[0].set_xlabel("RANSAC seed"); axs[0].set_ylabel("Registered cameras")
    axs[0].set_title("Registered cameras, with the registered frame range")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left"); style_axes(axs[0])
    axs[1].bar(full.seed.astype(str), full.points_3d, color=C_A, width=0.6)
    axs[1].axhline(2077, color=C_B, lw=1.5, label="Stored main reconstruction (2,077)")
    axs[1].set_ylim(0, 2600); axs[1].set_xlabel("RANSAC seed"); axs[1].set_ylabel("Sparse 3D points")
    axs[1].set_title("Sparse 3D points"); axs[1].legend(frameon=False, fontsize=8, loc="upper left"); style_axes(axs[1])
    plt.suptitle("Own SfM run-to-run stability: 47 keyframes, identical settings, only the random seed changes",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(out / "seed_stability.png", dpi=150); plt.close()

    print(cm.to_string(index=False))


if __name__ == "__main__":
    main()
